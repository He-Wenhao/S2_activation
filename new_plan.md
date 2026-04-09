# S2 Activation 非线性与采样交互研究计划

## 1. 研究动机

S2Activation 的核心流程：

```
输入系数 c_lm (l ≤ l_max)
    → 在采样点上展开: f(x) = Σ c_lm Y_lm(x)
    → 逐点施加非线性: g(x) = σ(f(x))
    → 积分回系数: ĉ_lm = ∫ g(x) Y_lm(x) dΩ
输出系数 ĉ_lm (l ≤ l_max)
```

**核心问题**：σ(f(x)) 不再局限于 l ≤ l_max 的球面调和空间。非线性会产生所有阶数的分量，但输出被截断到 l_max，导致：
- **截断误差**：l > l_max 的分量被丢弃
- **混叠误差（aliasing）**：如果采样点不足以分辨高阶分量，它们会 alias 回低阶

**核心假设**：越尖锐的非线性函数（如 ReLU）→ 产生越多高阶分量 → 更严重的截断和混叠 → 输出系数误差更大。

---

## 2. 研究问题

### 研究问题1：非线性函数的频谱特性

**目标**：量化不同非线性函数对球面信号频谱的影响

**方法**：
1. 生成输入信号 f(x) = Σ_{l≤l_max} c_lm Y_lm(x)
2. 在球面上施加非线性 g(x) = σ(f(x))
3. 用极高分辨率（l_max_ref >> l_max）计算 g(x) 的完整频谱
4. 分析功率谱 P(l) = Σ_m |c_lm|²

**测量指标**：
- 功率谱 P(l) 随 l 的衰减曲线
- 截断比 R = Σ_{l>l_max} P(l) / Σ_l P(l)（泄露到高阶的能量比例）
- 有效带宽 l_eff：P(l) 衰减到某阈值时的 l 值

**非线性函数列表**：
- 尖锐类：ReLU, abs(x), LeakyReLU
- 中等光滑：GELU, SiLU/Swish, ELU
- 光滑类：tanh, sigmoid, Softplus(β)
- 参数化：Softplus(x, β) — β从小到大连续控制尖锐度（β→∞ 趋近 ReLU）
- 其他：sin(x), x², gate (e3nn 中使用的 gating 机制)

**预期结果**：
- ReLU 的 P(l) 衰减最慢（类似 ~1/l 或更慢），截断比最大
- 光滑激活的 P(l) 指数衰减，截断比很小
- Softplus 的 β 参数提供连续的过渡

### 研究问题2：采样策略 × 非线性的交互

**目标**：理解采样方法和非线性的组合如何影响 S2Activation 的输出精度

**方法**：
1. 对每种（采样方法, 非线性）组合执行完整的 S2Activation
2. 计算 ground truth：极高分辨率采样 + 截断到 l_max 的输出系数
3. 比较输出系数与 ground truth 的误差

**采样方法**：
- Gauss-Legendre（e3nn 默认）
- Lebedev quadrature
- 均匀网格
- 各方法在不同分辨率/点数下

**关键问题**：
- 当非线性光滑时，采样方法的差异是否不重要？
- 当非线性尖锐时，更多采样点是否能有效减少 aliasing？
- Lebedev 的"精确积分"优势在非线性后是否仍然有意义？（因为 σ(f) 的 degree 远超采样精度）

### 研究问题3：过采样（Oversampling）作为缓解策略

**目标**：研究用更高分辨率的网格（l_max_grid > l_max）做采样是否能有效减少 aliasing

**方法**：
- 固定输入/输出 l_max
- 变化采样网格的分辨率（从 l_max 到 2*l_max, 3*l_max, ...）
- 测量输出精度随过采样比的变化

**预期**：过采样能部分缓解 aliasing，但无法消除截断误差

### 研究问题4：对实际网络性能的影响

**目标**：验证频谱分析的发现是否反映在实际任务性能上

**方法**：
- 在相同架构下，切换不同的（非线性, 采样方法, 分辨率）组合
- 比较训练速度、最终精度、数值稳定性

---

## 3. 文献调研

需要收集以下经典模型中 S2Activation 的具体设置：

| 模型 | l_max | 非线性函数 | 采样方式 | 采样分辨率 | Aliasing 讨论 |
|------|-------|-----------|---------|-----------|-------------|
| e3nn (default) | 用户指定 | 用户指定 (常用 ReLU/tanh) | Kostelec & Rockmore 等角网格 (半像素偏移避极点) | res_beta=2*(lmax+1), res_alpha=2*lmax+1 (最小分辨率，无过采样) | 无 |
| SCN (NeurIPS 2022) | 6 | SiLU | 加权球面 Fibonacci 采样 | 128 个点 | 无 |
| eSCN (ICML 2023) | 6 | SiLU | 输出层: 加权 Fibonacci (128点); S2Act: e3nn GL 网格 | 14×15=210 点 (GL网格) | 无 (后续 eSEN 指出网格离散化破坏能量守恒) |
| EquiformerV2 (ICLR 2024) | 6, m_max=3 | SiLU (Separable S2 Activation) | e3nn GL 网格 (ToS2Grid/FromS2Grid) | 默认14×7=98点; 配置中 grid_resolution=18 → 18×18=324点 | 无 |
| eSEN (arXiv 2502.12147, ICML 2025) | 3 (生产30M), 2 (消融) | **不使用 S2 Act** — SiLU gated nonlinearity 在系数空间；SiLU 作用于标量(m=0)后 gate 高阶分量 | 无（不投影到网格）| 无 | **核心论点**: 网格离散化引入超出 Nyquist 频率的高频信号，破坏严格等变性和能量守恒；消融实验 res=6/10/14 显示离散版本都破坏能量守恒 |
| UMA (arXiv 2506.23971, 2025) | 2/4/6 (S/M/L) | **混合**: 卷积层用 gated nonlinearity (SiLU gate); FFN 层用 grid-based S2 激活 (SiLU on grid) | SO3_Grid (可配置 grid_resolution) | 配置决定 | 无讨论 aliasing；FFN 层仍然投影到网格做 MLP |
| Esteves et al. 2018 (ECCV) | - | ReLU | S2 网格 | - | 指出"pointwise nonlinearity does not preserve bandlimit and causes equivariance errors" |
| CG Nets (Kondor et al. NeurIPS 2018) | - | **不使用 S2 Act** — 用 CG tensor product 作为非线性 | 无 | 无 | 避免 S2 Act，因为 FFT 来回变换"costly and source of numerical errors" |
| Xu et al. 2022 (ICML) | - | 统一框架 | 理论分析 | - | 提供了 S2 activation 的傅里叶空间统一理论框架 |
| Cohen et al. 2018 (Spherical CNNs) | - | ReLU | 等角网格 | - | **有**: ReLU 在 1x 分辨率等变误差~0.37, 2x→~0.10, 8x→~0.01 |

### 不使用 S2 Activation 的模型（使用 gated nonlinearity）

| 模型 | 非线性方式 |
|------|-----------|
| TFN (Thomas et al. 2018) | 仅对标量(l=0)施加非线性，或使用 norm-based gating |
| NequIP (2022) | SiLU (偶宇称标量) + tanh (奇宇称标量) gating 高阶特征 |
| MACE (2022) | SiLU 仅在标量 MLP 中；CG product 提供 irreps 耦合 |
| Allegro (2023) | SiLU 仅在标量 latent MLP 中 |
| BOTNet (2022) | 仅标量非线性 |
| SEGNNs (Brandstetter et al. 2022) | "steerable nonlinear convolutions"，不用 S2 Act |

**关键观察**：
1. **所有现代模型（SCN/eSCN/EquiformerV2）都使用 SiLU**，没有用 ReLU —— 可能正是因为 SiLU 更光滑，产生的高阶泄露更少
2. **都用 l_max=6**，采样点数在 100-324 范围
3. **几乎没有讨论 aliasing 问题**，只有 Cohen et al. 2018 定量分析了 ReLU 的等变误差
4. **没有过采样**：e3nn 默认分辨率是精确重构带限信号的最小值，非线性产生的高阶分量完全被 alias
5. SCN 使用 Fibonacci 采样（非精确求积），eSCN/EquiformerV2 切换到了 GL 网格
6. **eSEN 完全放弃了 S2 Activation**：改用 gated nonlinearity 在系数空间操作，避免网格投影。核心论点是网格离散化引入 Nyquist 以上的高频信号，破坏能量守恒。消融实验显示即使 res=14 也无法完全消除守恒律破坏。这说明 aliasing 问题在实际应用中确实造成了可测量的后果。

---

## 4. 实验设计

核心研究链条：**激活函数光滑度 / 积分采样方式 → 频谱泄露 → 等变误差 → 实际性能**

四个实验分别对应这条链上的每一环。

### 实验 A：频谱泄露分析（光滑度 → 频谱）

**目标**：量化不同激活函数在球面上施加后产生的高阶分量

**方法**：
1. 生成带限输入 f(x) = Σ_{l≤l_max} c_lm Y_lm(x)
2. 施加非线性 g(x) = σ(f(x))
3. 用极高分辨率 GL 求积（l_max_ref >> l_max）计算 g(x) 的**完整频谱**
4. 分析功率谱 P(l) 的衰减行为

```python
l_max_values = [3, 6, 10]          # 输入带宽（6 是实际模型常用值）
l_max_ref = 40                      # 参考阶数，足够观察衰减

nonlinearities = {
    # 尖锐类 (C^0)
    'ReLU': torch.relu,
    'abs': torch.abs,
    'LeakyReLU': nn.LeakyReLU(0.1),
    # 光滑类 (C^∞)
    'SiLU': torch.nn.SiLU(),        # 现代模型标配
    'GELU': torch.nn.GELU(),
    'tanh': torch.tanh,              # e3nn 示例默认
    # 参数化光滑度 (Softplus_β → ReLU as β→∞)
    'Softplus_1': nn.Softplus(beta=1),
    'Softplus_3': nn.Softplus(beta=3),
    'Softplus_10': nn.Softplus(beta=10),
    'Softplus_30': nn.Softplus(beta=30),
    # 其他
    'x^2': lambda x: x**2,
    'sin': torch.sin,
}

num_random_inputs = 20

# 输出指标：
#   P(l) = Σ_m |c_lm|^2                          功率谱
#   R = Σ_{l>l_max} P(l) / Σ_l P(l)              截断泄露比
#   l_eff: P(l) < ε * P(0) 时的 l                 有效带宽
#   衰减拟合: log P(l) ~ -α·l 或 log P(l) ~ -k·log(l)  （指数 vs 幂律）

# 图表：
#   图A1: P(l) vs l（log-log），每条线一个激活函数，固定 l_max=6
#   图A2: 截断比 R vs 激活函数（bar chart）
#   图A3: Softplus_β 的 R vs β（连续过渡曲线）
```

### 实验 B：S2Activation 输出误差（频谱泄露 × 采样 → 系数误差）

**目标**：测量完整 S2Activation 流程的输出误差，分离截断误差和混叠误差

**方法**：
1. 对每组 (激活函数, 采样方法, 采样分辨率) 执行 S2Activation
2. Ground truth: 用 l_max_ref=40 的 GL 求积做 S2Activation，取前 l_max 个系数
3. 分离两种误差：
   - **截断误差** = ground truth 与"完美无非线性"的差 → 不依赖采样
   - **混叠误差** = 实际输出与 ground truth 的差 → 依赖采样方式

```python
sampling_configs = {
    'GL_1x':  ('gauss_legendre', l_max),        # 最小精确分辨率
    'GL_2x':  ('gauss_legendre', 2*l_max),       # 2x 过采样
    'GL_3x':  ('gauss_legendre', 3*l_max),       # 3x 过采样
    'Leb_min': ('lebedev', degree=2*l_max+1),     # Lebedev 最小精确
    'Leb_2x':  ('lebedev', degree=4*l_max+1),     # Lebedev 2x
    'Uniform_100': ('uniform', resolution=100),
}

# 输出指标：
#   ||c_out - c_gt||_2 / ||c_gt||_2      总相对误差
#   逐 l 的误差分解                       看哪些阶被污染最多
#   截断误差 vs 混叠误差的比例

# 图表：
#   图B1: 误差 vs 采样点数，按激活函数分组
#   图B2: 热力图 (激活函数 × 采样方法)，颜色表示误差
#   图B3: 误差随过采样比的衰减曲线
```

### 实验 C：等变误差测量（系数误差 → 等变性破坏）

**目标**：直接测量 S2Activation 在旋转下的等变误差

**方法**：
等变性要求 S2Act(D·x) = D·S2Act(x)，其中 D 是 Wigner-D 旋转矩阵。
1. 随机生成输入系数 x
2. 随机生成旋转 R
3. 计算 path A: 先旋转再激活 → S2Act(D(R)·x)
4. 计算 path B: 先激活再旋转 → D(R)·S2Act(x)
5. 等变误差 = ||path_A - path_B|| / ||path_A||

```python
num_rotations = 50
num_inputs = 20

# 对每组 (激活函数, 采样方法, 分辨率)：
#   测量平均等变误差

# 图表：
#   图C1: 等变误差 vs 激活函数（按采样方法分组）
#   图C2: 等变误差 vs 过采样比（按激活函数分组）
#   图C3: 等变误差 vs 频谱泄露比 R（散点图，验证相关性）
```

### 实验 D：实际任务性能（等变误差 → 模型性能）

**目标**：验证频谱/等变分析是否反映在实际下游任务中

**方法**：
在相同模型架构下，切换不同 (激活函数, 采样配置)，比较：
- 分类/回归精度
- 训练稳定性
- 能量守恒（MD 相关任务）

```python
# 任务选择（按难度递进）：
# 1. 球面信号去噪（合成任务，直接体现频谱效果）
# 2. Spherical MNIST 分类
# 3. 简单分子性质预测（如果可行）

# 对每组配置训练模型，比较：
#   最终精度 vs 训练时间
#   最终精度 vs 等变误差（散点图，验证相关性）
```

---

## 5. 预期结论与故事线

整条链：**光滑度 → 频谱泄露 → 等变误差 → 性能**

1. **实验A** 建立：激活函数光滑度决定频谱衰减速率
   - C^∞（SiLU/tanh）→ 指数衰减，泄露比 R < 1%
   - C^0（ReLU）→ 幂律衰减，泄露比 R ~ 10-50%
   - Softplus(β) 提供连续过渡

2. **实验B** 建立：频谱泄露通过 aliasing 机制污染低阶系数
   - 泄露越多 + 采样越少 → 系数误差越大
   - 过采样能减少 aliasing 但不能消除截断误差
   - **关键发现**：对光滑激活，最小分辨率已够用；对尖锐激活，即使大量过采样也有残余误差

3. **实验C** 建立：系数误差直接导致等变性破坏
   - 等变误差与频谱泄露比 R 强相关
   - 这解释了 eSEN 为什么要放弃 S2 Activation

4. **实验D** 验证：等变误差影响下游性能
   - 对需要严格等变性的任务（MD 能量守恒），影响显著
   - 对分类任务，影响可能较小（网络可以学会补偿）

**实用建议**：
- 如果用 S2 Activation，选 SiLU/GELU + 适度过采样（2x）是最佳权衡
- 如果需要严格等变性，考虑 gated nonlinearity（牺牲表达力）
- Softplus(β) 可作为可调参数，在训练中从小 β（光滑）逐步增大（curriculum）
