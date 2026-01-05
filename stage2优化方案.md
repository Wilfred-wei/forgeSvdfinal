# Stage 2 深度优化方案：Dual-Stream ProbeFormer & Diversity Learning

## 1. 核心改进理论
根据 ICLR 2026 论文 *Beyond the Final Layer* 和探针学习理论，我们对 `net_stage2` 进行两大关键升级：

### A. 双流特征输入 (Dual-Stream Input)
**痛点**：当前模型仅提取 Backbone 每一层的 `CLS` token。论文指出，Deepfake/AIGC 检测严重依赖纹理和结构异常，而这些信息主要保留在 Spatial Tokens 中，容易被 `CLS` 聚合时丢失。
**改进**：
* 不仅提取 `CLS` (Semantic Stream)。
* 同时计算每一层 Spatial Tokens 的 **Global Average Pooling (AP)** (Texture/Structure Stream)。
* 输入序列长度从 24 (仅CLS) 翻倍为 48 (CLS + AP)。
* **收益**：让探针同时“看到”语义逻辑错误和底层纹理伪影。

### B. 探针多样性约束 (Diversity Regularization)
**痛点**：实验发现探针数量多于 4 个时效果饱和，原因是探针发生了“模式坍缩” (Mode Collapse)，即多个探针学到了重复的特征。
**改进**：
* 在训练时引入 **正交损失 (Orthogonal Loss)**。
* 强制 `Probe_1` 到 `Probe_4` 的特征向量两两互不相关（余弦相似度趋近于 0）。
* **收益**：迫使每个探针成为不同领域的专家（如：一个看高频噪点，一个看五官逻辑），最大化 4 个探针的效能。

---

## 2. 代码修改规范

### 修改目标 1: `models/network/net_stage2.py`
1.  **ProbeFormer 类**：无需大改，只需确认能处理长度为 48 的序列（Transformer 本身支持变长，通常无需修改结构，只需注意输入维度）。
2.  **net_stage2.forward()**：
    * **输入处理升级**：从 `backbone` 返回的 `cls_tokens` (实际包含 full patches) 中，分别提取 `CLS` (`index 0`) 和 `AP` (`index 1: mean`).
    * **序列堆叠**：将 24 层 CLS 和 24 层 AP 拼接或交替堆叠，形成 `[B, 48, Dim]` 的张量。
    * **返回值变更**：除了返回分类 logits `result`，还必须返回 **`probes` 特征** (Shape: `[B, Num_Probes, Dim]`)，以便 Trainer 计算多样性 Loss。

### 修改目标 2: `models/trainer_stage2.py`
1.  **新增 Loss 函数**：`diversity_loss(probes)`，计算 Gram 矩阵与单位矩阵的差异。
2.  **训练循环 (`train_epoch`)**：
    * 接收模型返回的 `output` 和 `probes`。
    * 计算 `total_loss = cls_loss + lambda * div_loss` (推荐 lambda=0.1)。
    * 在进度条或日志中打印 `div_loss` 以监控探针是否“分家”成功。