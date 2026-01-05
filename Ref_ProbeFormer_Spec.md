# 任务说明：将 net_stage2 从 FAFormer 升级为 ProbeFormer (Multi-Probe 机制)

## 1. 背景
当前的 `net_stage2.py` 使用了一个简单的 Self-Attention 模块 (`FAFormer`) 来聚合 CLIP 的 24 层特征。这种方式依赖于一个手动初始化的 CLS Token，容易被 CLIP 的强语义信息淹没，导致对微弱伪造痕迹（Artifacts）的检测能力不足。

我们决定引入 **ForensicsAdapter** 的核心思想——**"Probing Mechanism" (探针机制)**。
即：不再使用 1 个 CLS Token，而是使用 **N 个可学习的探针 (Probes)** 主动去探测 CLIP 的层级特征。

## 2. 核心变更点

### 2.1 架构图示
* **Old (FAFormer)**:
    Input: `[CLS_Token, Layer_1, ..., Layer_24]` (Sequence Length = 1 + 24)
    Output: 取 `CLS_Token` 做分类。
* **New (ProbeFormer)**:
    Input: `[Probe_1, ..., Probe_8, Layer_1, ..., Layer_24]` (Sequence Length = 8 + 24)
    Output: 取 `[Probe_1 ... Probe_8]` 的平均值做分类。

### 2.2 关键参数
* **`num_probes`**: 初始设置为 **8** (后续可扩展至 16, 32)。
* **输入维度**: CLIP 特征维度 (例如 768 或 1024)。

### 2.3 类结构修改要求
请在 `models/network/net_stage2.py` 中：

1.  **定义 `ProbeFormer` 类**：
    * 替代原有的 `FAFormer`。
    * 在 `__init__` 中初始化 `self.probes`，形状为 `[1, num_probes, width]`。
    * 在 `forward` 中，将 `self.probes` 扩展并拼接到输入特征的前面。
    * 经过 Transformer 层处理后，**只截取前 `num_probes` 个 token 返回**。
    * 注意处理 `nn.MultiheadAttention` 默认的 `(L, N, E)` 维度格式。

2.  **修改 `net_stage2` 类**：
    * 删除 `self.cls_token` (不再需要)。
    * 将 `self.transformer` 的实例化从 `FAFormer` 改为 `ProbeFormer`。
    * 在 `forward` 函数中：
        * 不再手动拼接 `cls_token`。
        * 调用 `self.transformer` 得到 `[Num_Probes, Batch, Dim]` 的输出。
        * 对探针输出进行 **Mean Pooling (平均池化)**，得到 `[Batch, Dim]`。
        * 送入 `self.fc` 进行分类。

## 3. 代码实现细节参考

```python
# ProbeFormer 的核心逻辑示意
class ProbeFormer(nn.Module):
    def __init__(self, width, layers, heads, num_probes=8):
        super().__init__()
        # 初始化探针：正态分布初始化
        self.probes = nn.Parameter(torch.randn(1, num_probes, width) * 0.02)
        # 复用原有的 ResidualAttentionBlock
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, ...) for _ in range(layers)])

    def forward(self, x):
        # x input shape: [L_clip, B, D] (24, Batch, Dim)
        L, B, D = x.shape
        
        # 1. Expand Probes: [1, Num, D] -> [Num, B, D]
        probes = self.probes.permute(1, 0, 2).repeat(1, B, 1)
        
        # 2. Concat: [Num + L_clip, B, D]
        x = torch.cat([probes, x], dim=0)
        
        # 3. Transformer Interaction
        for layer in self.resblocks:
            x = layer(x)
            
        # 4. Return Probes Only: [Num, B, D]
        return x[:self.num_probes]