import torch
import torch.nn as nn
from models.network.net_stage1 import net_stage1
from util import read_yaml

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, reduction_factor: int , attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // reduction_factor),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_model // reduction_factor, d_model),
            nn.Dropout(0.5)
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FAFormer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, reduction_factor: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, reduction_factor, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        out = {}
        for idx, layer in enumerate(self.resblocks.children()):
            x = layer(x)
            out['layer'+str(idx)] = x[0] # shape:LND. choose cls token feature
        return out, x


class ProbeFormer(nn.Module):
    """Multi-Probe Attention Module for forensics artifact detection.

    Uses N learnable probes to actively probe CLIP layer features,
    replacing the single CLS token approach.
    """
    def __init__(self, width: int, layers: int, heads: int, reduction_factor: int,
                 num_probes: int = 8, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.num_probes = num_probes
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, reduction_factor, attn_mask) for _ in range(layers)])

        # Initialize learnable probes: [1, num_probes, width]
        self.probes = nn.Parameter(torch.randn(1, num_probes, width) * 0.02)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: CLIP features, shape [L_clip, B, D] (e.g., [24, Batch, Dim])
        Returns:
            x: Probes after transformer interaction, shape [num_probes, B, D]
        """
        L, B, D = x.shape

        # 1. Expand Probes: [1, num_probes, D] -> [num_probes, B, D]
        probes = self.probes.permute(1, 0, 2).repeat(1, B, 1)

        # 2. Concat: [num_probes + L_clip, B, D]
        x = torch.cat([probes, x], dim=0)

        # 3. Transformer Interaction
        for layer in self.resblocks:
            x = layer(x)

        # 4. Return Probes Only: [num_probes, B, D]
        return x[:self.num_probes]

class net_stage2(nn.Module):
    def __init__(self, opt, dim=768, drop_rate=0.5, output_dim=1, train=True):
        super(net_stage2, self).__init__()

        self.use_probeformer = getattr(opt, 'use_probeformer', False)

        self.backbone = net_stage1()
        if train:
            model_load = torch.load(opt.intermediate_model_path)
            self.backbone.load_state_dict(model_load['model_state_dict'])
            print(f"LOAD {opt.intermediate_model_path}!!!!!!")

        params = []
        for name, p in self.backbone.named_parameters():
            if name == "fc.weight" or name == "fc.bias":
                params.append(p)
            else:
                p.requires_grad = False
        # print(params)

        if self.use_probeformer:
            print(f"Using ProbeFormer with {opt.num_probes} probes")
            self.transformer = ProbeFormer(dim, layers=opt.FAFormer_layers, heads=opt.FAFormer_head,
                                            reduction_factor=opt.FAFormer_reduction_factor,
                                            num_probes=opt.num_probes)
            self.cls_token = None  # Not used in ProbeFormer
        else:
            print("Using FAFormer with CLS token")
            self.transformer = FAFormer(dim, layers=opt.FAFormer_layers, heads=opt.FAFormer_head,
                                         reduction_factor=opt.FAFormer_reduction_factor)
            self.cls_token = nn.Parameter(torch.zeros([dim]))

        self.ln_post = nn.LayerNorm(dim)

        self.fc = nn.Sequential(
            nn.Dropout(drop_rate),
            torch.nn.Linear(dim, output_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize cls_token for FAFormer
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

        for m in self.transformer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.size()
        _, cls_tokens, _ = self.backbone(x)

        # Each cls_tokens[i] is [L, B, D] (LND format, L=257: 256 patches + 1 CLS)
        # Extract CLS token (index 0 in sequence): [L, B, D] -> [B, D]
        cls_tokens = [t[0, :, :] for t in cls_tokens]

        # Stack: [24, B, D]
        x = torch.stack(cls_tokens, dim=0)

        if self.use_probeformer:
            # ProbeFormer: x is already [24, B, D] in LND format
            x = self.transformer(x)
            # Mean Pooling over probes: [num_probes, B, D] -> [B, D]
            x = x.mean(dim=0)
        else:
            # FAFormer with CLS token: concat in NLD first, then convert to LND
            x = x.permute(1, 0, 2)  # [B, 24, D] NLD format
            cls = self.cls_token.view(1, 1, -1).repeat(B, 1, 1)  # [1, B, D] -> [B, 1, D]
            x = torch.cat([cls, x], dim=1)  # [B, 25, D]
            x = x.permute(1, 0, 2)  # LND: [25, B, D]
            out, x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD: [B, 25, D]
            x = x[:, 0, :]  # Take CLS token

        x = self.ln_post(x)
        result = self.fc(x)
        return result


