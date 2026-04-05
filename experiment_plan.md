# 球面激活函数采样方法对比实验计划

## 1. 实验目标

### 主要目标
1. **精度对比**：在球面调和系数重构中，Lebedev quadrature vs 均匀网格 vs Gauss-Legendre 的误差对比
2. **计算效率**：计算时间和内存消耗对比
3. **任务性能**：在实际神经网络任务上的性能对比（Spherical MNIST）

### 研究问题
- Lebedev quadrature 是否能用更少采样点达到相同精度？
- 在实际任务中性能提升有多大？
- 计算开销如何权衡？

---

## 2. 实验设置

### 2.1 环境要求

```
Python: 3.10+
PyTorch: 2.0+
Dependencies:
  - numpy
  - scipy
  - matplotlib
  - e3nn (for spherical harmonics)
  - lebedev (for Lebedev quadrature points)
  - optionally: torchvision (for MNIST)
```

### 2.2 文件结构

```
project/
├── data/
│   ├── spherical_mnist/  # Projected MNIST
│   └── lebedev_grids/    # Pre-computed Lebedev points
├── src/
│   ├── quadrature_methods.py     # Sampling方法实现
│   ├── s2_activation.py           # S2Activation改进版本
│   ├── spherical_harmonics_utils.py
│   └── models.py                  # 测试模型
├── experiments/
│   ├── exp1_accuracy.py           # 实验1：精度对比
│   ├── exp2_computational_cost.py  # 实验2：计算开销
│   ├── exp3_task_performance.py   # 实验3：任务性能
│   └── exp4_resolution_scaling.py # 实验4：分辨率扩展性
├── results/
│   ├── figures/
│   ├── tables/
│   └── metrics/
└── notebooks/
    └── analysis_and_visualization.ipynb
```

---

## 3. 具体实验设计

### 实验1：精度对比（Experiment 1: Accuracy Comparison）

**目标**：比较不同采样方法在球面调和重构中的精度

#### 1.1 数据生成

```python
# 对于不同的最大度数 l_max
l_max_values = [3, 5, 7, 10, 15, 20]

# 对于每个 l_max：
for l_max in l_max_values:
    # 生成随机球面调和系数
    num_coefficients = (l_max + 1) ** 2
    
    # 随机初始化（三个分布）
    cases = {
        'random_normal': randn(num_coefficients),      # 标准正态分布
        'random_uniform': rand(num_coefficients),      # 均匀分布  
        'sparse': sparse_coefficients(l_max),          # 稀疏系数
        'polynomial': polynomial_coefficients(l_max),  # 多项式结构
    }
```

#### 1.2 采样方法

实现四种采样方法：

```python
class SamplingMethods:
    
    @staticmethod
    def uniform_grid(resolution_theta, resolution_phi):
        """
        均匀网格采样
        参数：
          resolution_theta: theta方向采样点数
          resolution_phi: phi方向采样点数
        返回：
          points: (N, 3) Cartesian坐标
          weights: (N,) 积分权重
        """
        pass
    
    @staticmethod
    def gauss_legendre(l_max):
        """
        Gauss-Legendre二次规则（e3nn-jax中使用）
        参数：
          l_max: 最大球面调和度数
        返回：
          points: (N, 3) Cartesian坐标
          weights: (N,) 积分权重
        N ≈ (l_max + 1)^2 / 2
        """
        pass
    
    @staticmethod
    def lebedev(precision_or_points):
        """
        Lebedev quadrature
        参数：
          precision_or_points: 精度order或点数
        返回：
          points: (N, 3) Cartesian坐标
          weights: (N,) 积分权重
        N ∈ {6, 14, 26, 38, 50, 74, 86, 110, 146, 170, ...}
        """
        pass
    
    @staticmethod
    def fibonacci_sphere(num_points):
        """
        Fibonacci球面采样（参考方法）
        参数：
          num_points: 采样点数
        返回：
          points: (N, 3) Cartesian坐标
          weights: (N,) 均匀权重 (4π/N)
        """
        pass
```

#### 1.3 重构精度测量

```python
class AccuracyMetrics:
    
    @staticmethod
    def reconstruction_error(coeffs_true, points, weights, sampling_method):
        """
        测量球面调和重构误差
        步骤：
          1. f_true = sum_lm coeffs[l,m] * Y_l^m(x)  在真实球面上
          2. 在采样点处计算 f_true
          3. 从采样点重构系数 coeffs_reconstructed
             coeffs_reconstructed[l,m] = integral(f * Y_l^m) 
                                       = sum_i w_i * f(x_i) * Y_l^m(x_i)
          4. 计算误差
        
        误差指标：
          - L2误差: ||coeffs_true - coeffs_reconstructed||_2
          - 相对误差: L2误差 / ||coeffs_true||_2
          - 逐度数误差: 对每个l计算
          - 最大误差: ||...||_inf
        
        返回：
          errors: dict with keys:
            'l2_error': float
            'relative_error': float
            'max_error': float
            'errors_by_degree': array of shape (l_max+1,)
        """
        pass
    
    @staticmethod
    def function_reconstruction_error(coeffs, points, weights, method):
        """
        在球面上直接比较函数值
        步骤：
          1. f_true = sum Y_lm * coeffs
          2. 在所有采样点处计算 f_true
          3. 从采样点重构 f_recon
          4. 计算 ||f_true - f_recon||
        
        返回：
          {'point_wise_error': array,  # 每个点的误差
           'l2_function_error': float,
           'max_function_error': float}
        """
        pass
```

#### 1.4 实验参数

```python
# Experiment 1 Parameters
EXP1_PARAMS = {
    'l_max_values': [3, 5, 7, 10, 15, 20],
    
    'sampling_methods': {
        'uniform': {
            'resolutions': [(50, 100), (100, 200), (150, 300), (200, 400)]
        },
        'gauss_legendre': {
            'l_max_values': [3, 5, 7, 10, 15, 20]
        },
        'lebedev': {
            'precision_values': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        },
        'fibonacci': {
            'point_counts': [50, 100, 200, 400, 600, 800, 1000]
        }
    },
    
    'num_random_seeds': 10,  # 对于每个配置运行10次
    
    'coefficient_distributions': [
        'random_normal',
        'random_uniform', 
        'sparse',
        'polynomial'
    ]
}

# 预期采样点数范围
EXPECTED_POINT_COUNTS = {
    'lebedev_precision_3': 6,
    'lebedev_precision_5': 14,
    'lebedev_precision_7': 26,
    'lebedev_precision_9': 38,
    'lebedev_precision_11': 50,
    'lebedev_precision_13': 74,
    'lebedev_precision_15': 86,
    'lebedev_precision_17': 110,
    'lebedev_precision_19': 146,
    'lebedev_precision_21': 170,
}
```

#### 1.5 输出指标

```python
# 生成表格：对于每个 l_max 值

# 表格1：相同采样点数下的精度对比
# ┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
# │ Method          │ N points │ L2 Error │ Rel Err  │ Max Err  │
# ├─────────────────┼──────────┼──────────┼──────────┼──────────┤
# │ Uniform (100x200)│   20000 │   1.2e-3 │   2.5%   │   8.3e-3 │
# │ GL (l_max=10)   │   ~110   │   1.8e-4 │   0.4%   │   1.2e-3 │
# │ Lebedev (p=17)  │   110    │   8.5e-5 │   0.2%   │   3.1e-4 │
# └─────────────────┴──────────┴──────────┴──────────┴──────────┘

# 图1：精度曲线
# - X轴：采样点数 (log scale)
# - Y轴：L2重构误差 (log scale)
# - 三条曲线：Lebedev, GL, Uniform
# - 对于不同的 l_max 用不同颜色标记

# 图2：误差随度数l的分布
# - X轴：度数 l
# - Y轴：该度数的最大系数误差
# - 分组条形图：不同方法对比
```

---

### 实验2：计算开销（Experiment 2: Computational Cost）

**目标**：比较前向/反向传播的时间和内存开销

#### 2.1 实验设计

```python
class ComputationalCostAnalysis:
    
    @staticmethod
    def benchmark_forward_pass(method, batch_size, input_dim, num_trials=100):
        """
        测量前向传播时间
        步骤：
          1. 创建随机输入 (batch_size, input_dim)
          2. 创建 S2Activation 模块（使用该采样方法）
          3. 预热（5次运行）
          4. 测量100次迭代的时间
          5. 计算平均和std
        
        参数：
          method: 'uniform', 'gauss_legendre', 'lebedev'
          batch_size: [1, 8, 16, 32, 64]
          input_dim: [10, 20, 50, 100]
          
        返回：
          {
            'time_ms': float,
            'std_ms': float,
            'time_per_sample': float,
          }
        """
        pass
    
    @staticmethod
    def benchmark_backward_pass(method, batch_size, input_dim, num_trials=100):
        """
        测量反向传播时间（梯度计算）
        """
        pass
    
    @staticmethod
    def memory_usage(method, resolution, batch_size):
        """
        测量内存占用
        步骤：
          1. 记录GPU内存初始值
          2. 创建模型和数据
          3. 前向传播
          4. 计算梯度
          5. 测量峰值内存
          
        返回：
          {
            'model_memory_mb': float,
            'activations_memory_mb': float,
            'gradients_memory_mb': float,
            'peak_memory_mb': float,
          }
        """
        pass
```

#### 2.2 实验参数

```python
EXP2_PARAMS = {
    'batch_sizes': [1, 8, 16, 32, 64],
    'input_dimensions': [10, 20, 50, 100],
    'resolution_or_precision': {
        'uniform': [50, 100, 150, 200],
        'gauss_legendre': [3, 5, 7, 10, 15],
        'lebedev': [7, 11, 15, 19, 21]  # precision values
    },
    'num_trials': 100,
    'warmup_trials': 5,
}

# 同时记录的指标
METRICS_TO_RECORD = [
    'forward_time',        # ms
    'backward_time',       # ms
    'total_time',          # ms
    'time_per_sample',     # ms/sample
    'memory_peak',         # MB
    'memory_activations',  # MB
    'memory_gradients',    # MB
    'flops',               # (如果可以计算)
]
```

#### 2.3 输出指标

```python
# 表格2：计算时间对比 (ms, batch_size=32)
# ┌──────────────┬─────────┬─────────┬─────────┬──────────┐
# │ Resolution   │ Uniform │ GL      │ Lebedev │ Speedup  │
# ├──────────────┼─────────┼─────────┼─────────┼──────────┤
# │ Small (N~50) │   2.1   │   0.8   │   0.6   │ 3.5x     │
# │ Medium (N~100)│  8.3   │   2.1   │   1.8   │ 4.6x     │
# │ Large (N~200) │ 32.1   │   7.9   │   6.2   │ 5.2x     │
# └──────────────┴─────────┴─────────┴─────────┴──────────┘

# 图3：时间扩展性
# - X轴：分辨率 (采样点数)
# - Y轴：时间 (ms, log scale)
# - 三条线：三种方法
# - 斜率反映 O(N) 复杂度

# 图4：内存对比
# - 聚集柱状图：不同batch size下的内存使用
```

---

### 实验3：任务性能（Experiment 3: Real Task Performance）

**目标**：在实际神经网络任务中对比性能

#### 3.1 任务：Spherical MNIST

```python
class SphericalMNISTTask:
    """
    任务描述：
    1. 下载MNIST数据集
    2. 将28x28图像投影到球面上
    3. 使用不同采样方法的S2Activation
    4. 训练分类网络
    5. 评估准确率和训练速度
    """
    
    @staticmethod
    def create_dataset(data_root, num_samples=10000):
        """
        创建Spherical MNIST数据集
        步骤：
          1. 加载MNIST（28x28）
          2. 将每个像素投影到球面
             使用球面坐标：θ ∈ [0, π], φ ∈ [0, 2π]
             对应图像中的像素位置
          3. 转换为球面调和系数
          4. 保存数据集
        
        返回：
          dataset: (N, num_coefficients) 球面调和系数
          labels: (N,) 0-9 的标签
        """
        pass
    
    @staticmethod
    def spherical_projection(image_28x28, lmax=10):
        """
        将2D图像投影到球面并展开为球面调和系数
        
        参数：
          image_28x28: (28, 28) numpy array
          lmax: 最大度数
        
        返回：
          coefficients: (lmax+1)^2 的球面调和系数
        """
        pass
```

#### 3.2 模型架构

```python
class SphericalCNN(nn.Module):
    """
    简单的球面CNN用于分类任务
    
    架构：
      Input: (batch, lmax^2)  # 球面调和系数
      ↓
      S2Activation (使用不同采样方法)
      ↓
      Linear(n_features, 128)
      ↓
      S2Activation
      ↓
      Linear(128, 64)
      ↓
      S2Activation
      ↓
      Linear(64, 10)  # 10个分类
      ↓
      Output: (batch, 10) logits
    """
    
    def __init__(self, lmax=10, sampling_method='uniform', resolution=100):
        super().__init__()
        n_features = (lmax + 1) ** 2
        
        self.fc1 = nn.Linear(n_features, n_features * 2)
        self.act1 = S2Activation(irreps=..., act=torch.relu, 
                                 sampling_method=sampling_method,
                                 resolution=resolution)
        
        self.fc2 = nn.Linear(n_features * 2, 128)
        self.act2 = S2Activation(...)
        
        self.fc3 = nn.Linear(128, 64)
        self.act3 = S2Activation(...)
        
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.fc4(x)
        return x
```

#### 3.3 训练循环

```python
def train_and_evaluate(model, train_loader, val_loader, 
                      epochs=20, device='cuda'):
    """
    标准的训练和评估循环
    
    返回：
      {
        'train_loss': list of epoch losses,
        'val_accuracy': list of epoch accuracies,
        'training_time': total time in seconds,
        'best_accuracy': best validation accuracy,
      }
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # ... 标准训练代码
```

#### 3.4 实验参数

```python
EXP3_PARAMS = {
    'dataset': {
        'name': 'spherical_mnist',
        'num_train': 6000,
        'num_val': 2000,
        'num_test': 2000,
        'lmax': 10,
    },
    'models': {
        'architecture': 'spherical_cnn',
        'lmax': 10,
        'hidden_dims': [128, 64],
    },
    'training': {
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
    },
    'sampling_configs': {
        'uniform': [50, 100, 150, 200],
        'gauss_legendre': [5, 7, 10, 15],
        'lebedev': [7, 11, 15, 19],
    },
    'num_runs': 3,  # 每个配置运行3次取平均
}
```

#### 3.5 输出指标

```python
# 表格3：分类准确率对比
# ┌──────────────┬───────────┬──────────┬──────────┬─────────────┐
# │ Method       │ N points  │ Accuracy │ Std      │ Time (sec)  │
# ├──────────────┼───────────┼──────────┼──────────┼─────────────┤
# │ Uniform 100  │   20000   │  92.3%   │  0.6%    │   125       │
# │ GL (l=10)    │   ~110    │  93.1%   │  0.5%    │   32        │
# │ Lebedev (19) │   146     │  93.4%   │  0.4%    │   28        │
# └──────────────┴───────────┴──────────┴──────────┴─────────────┘

# 图5：训练曲线
# - 三个子图（一个方法一个）
# - X轴：epoch
# - Y轴：准确率
# - 三条线（三次运行）和平均

# 图6：准确率 vs 计算成本
# - X轴：训练时间 (秒)
# - Y轴：最终准确率
# - 三个点：三种方法
# - 显示 Pareto 前沿
```

---

### 实验4：分辨率扩展性（Experiment 4: Scaling with Resolution/Degree）

**目标**：理解性能如何随 l_max 和采样分辨率变化

#### 4.1 实验设计

```python
class ScalingAnalysis:
    
    @staticmethod
    def efficiency_frontier(l_max_values):
        """
        对每个l_max，找到达到给定精度所需的最小采样点数
        
        精度阈值：
          target_errors = [1e-2, 1e-3, 1e-4, 1e-5]
        
        对每个(l_max, target_error)对：
          - 枚举所有采样方法的不同分辨率
          - 找到恰好满足精度的最小N
          - 记录计算时间
        
        绘制：
          X轴：l_max
          Y轴：采样点数（对于相同精度）
          三条线：三种方法
          
        显示：Lebedev效率优势随l_max增长而增加
        """
        pass
    
    @staticmethod
    def asymptotic_complexity(l_max_values):
        """
        分析渐近复杂度
        
        假设：N_samples ∝ l_max^α，时间 ∝ N * complexity
        
        通过回归拟合：
          log(time) = a + b*log(l_max) + c*log(N)
          
        提取：
          - 度数相关的复杂度指数
          - 采样点相关的复杂度指数
        """
        pass
```

#### 4.2 实验参数

```python
EXP4_PARAMS = {
    'l_max_range': [3, 5, 7, 10, 15, 20, 25, 30],
    'error_targets': [1e-2, 1e-3, 1e-4, 1e-5],
    'sampling_methods': ['uniform', 'gauss_legendre', 'lebedev'],
}

# 预期结果
EXPECTED_COMPLEXITY = {
    'lebedev': {
        'description': 'N ∝ l_max^1.5 to l_max^2',
        'points_for_l_max_20': ~170,
    },
    'gauss_legendre': {
        'description': 'N ∝ l_max^2',
        'points_for_l_max_20': ~400,
    },
    'uniform': {
        'description': 'N ∝ l_max^2 (但系数更大)',
        'points_for_l_max_20': ~2000,
    },
}
```

#### 4.3 输出指标

```python
# 图7：效率前沿
# - X轴：l_max
# - Y轴：达到1e-4精度所需的采样点数
# - 三条线：三种方法
# - 显示Lebedev的优势

# 图8：渐近复杂度
# - log-log图
# - X轴：l_max
# - Y轴：采样点数
# - 拟合直线显示幂律关系
```

---

## 4. 实现清单

### 第一阶段：基础设施（第1-2周）

- [ ] 4.1 实现 Lebedev quadrature 采样（从现有库包装或自己实现）
  ```python
  # 选项1：使用 scipy/e3x
  # 选项2：自己编写（基于lookup table）
  # 选项3：调用 C++ 库
  ```

- [ ] 4.2 实现其他采样方法（均匀、GL、Fibonacci）
  
- [ ] 4.3 创建修改版 S2Activation，支持不同采样方法
  ```python
  class S2Activation(nn.Module):
      def __init__(self, ..., sampling_method='uniform', resolution=None):
          ...
  ```

- [ ] 4.4 实现球面调和相关工具函数
  - 前向（系数→球面）
  - 反向（球面→系数）
  - 准确的数值积分

### 第二阶段：实验1-3（第3-4周）

- [ ] 4.5 实现 Experiment 1（精度对比）
  - 数据生成
  - 重构误差计算
  - 结果可视化

- [ ] 4.6 实现 Experiment 2（计算开销）
  - Benchmark 框架
  - 时间和内存测量
  - 性能分析

- [ ] 4.7 实现 Experiment 3（任务性能）
  - Spherical MNIST 数据集
  - 训练循环
  - 评估指标

### 第三阶段：实验4和分析（第5周）

- [ ] 4.8 实现 Experiment 4（扩展性）
  - 效率前沿分析
  - 渐近复杂度拟合

- [ ] 4.9 生成所有表格和图表
  
- [ ] 4.10 对比分析和洞察总结

---

## 5. 预期结果

### 关键假设

1. **精度方面**（Experiment 1）
   - **预期**：对于相同的 l_max，Lebedev 需要的采样点数约为均匀网格的 2-5%
   - **理由**：Lebedev 针对球面调和专门优化
   - **验证指标**：重构误差曲线应该显示 Lebedev 明显领先

2. **计算成本**（Experiment 2）
   - **预期**：虽然单位采样点的计算量相同，但 Lebedev 点数少，总体快 3-5 倍
   - **理由**：O(N) 的算法复杂度
   - **可能的开销**：Lebedev 点查找表的初始化（应该可忽略）

3. **任务性能**（Experiment 3）
   - **预期**：准确率相似或略高，但训练速度快 2-3 倍
   - **理由**：精度充分且更高效
   - **边界情况**：非常低的分辨率时差异不明显

4. **扩展性**（Experiment 4）
   - **预期**：Lebedev 优势随 l_max 增大而增加
   - **理由**：均匀网格需求随 l_max^2 增长，而 Lebedev 增长更缓

### 可能的意外结果

- 如果 Lebedev 没有预期的优势
  → 可能原因：实现有bug，或者对于这个问题 GL 已经接近最优
  → 应该检查：数值精度、采样点是否正确生成

- 如果任务性能反而变差
  → 可能原因：过度优化精度，在任务上不够稳健
  → 应该检查：需要多少精度才能保证好的性能

---

## 6. 代码组织建议

### 6.1 配置文件 `config.yaml`

```yaml
# Experiment configurations
experiment_1:
  name: "Accuracy Comparison"
  l_max_values: [3, 5, 7, 10, 15, 20]
  sampling_methods: ['uniform', 'gauss_legendre', 'lebedev', 'fibonacci']
  num_seeds: 10
  
experiment_2:
  name: "Computational Cost"
  batch_sizes: [1, 8, 16, 32, 64]
  input_dims: [10, 20, 50, 100]
  num_trials: 100

# ... 更多配置
```

### 6.2 核心模块 `src/spherical_harmonics_utils.py`

```python
import torch
from e3nn import o3

class SphericalHarmonicsUtils:
    """处理球面调和的所有操作"""
    
    @staticmethod
    def expand_coefficients_to_sphere(coeffs, points, l_max):
        """coeffs → 球面上的值"""
        pass
    
    @staticmethod
    def project_to_coefficients(values, points, weights, l_max):
        """球面上的值 → coeffs（积分）"""
        pass
    
    @staticmethod
    def spherical_harmonics(l_max, points):
        """计算所有 Y_l^m(points)"""
        pass
    
    # ... 更多助手函数
```

### 6.3 实验运行器 `experiments/run_all.py`

```python
def run_all_experiments(config_path='config.yaml', 
                        output_dir='results/'):
    """
    运行所有4个实验，生成报告
    """
    
    print("=" * 60)
    print("Starting Spherical Activation Sampling Comparison")
    print("=" * 60)
    
    # Experiment 1
    print("\n[1/4] Accuracy Comparison...")
    results_exp1 = run_experiment_1(config)
    save_results(results_exp1, f'{output_dir}/exp1_accuracy/')
    
    # Experiment 2
    print("\n[2/4] Computational Cost...")
    results_exp2 = run_experiment_2(config)
    save_results(results_exp2, f'{output_dir}/exp2_cost/')
    
    # Experiment 3
    print("\n[3/4] Task Performance...")
    results_exp3 = run_experiment_3(config)
    save_results(results_exp3, f'{output_dir}/exp3_task/')
    
    # Experiment 4
    print("\n[4/4] Scaling Analysis...")
    results_exp4 = run_experiment_4(config)
    save_results(results_exp4, f'{output_dir}/exp4_scaling/')
    
    print("\n✓ All experiments completed!")
    print(f"Results saved to {output_dir}")
```

---

## 7. 验证检查清单

在运行完整实验前，应该做的验证：

### 7.1 单元测试
- [ ] Lebedev 点生成（与参考值对比）
- [ ] 球面调和计算（与 scipy 对比）
- [ ] 采样权重求和（应为 4π）
- [ ] 极坐标和笛卡尔坐标转换

### 7.2 小规模实验验证
```python
# 运行mini版本确保逻辑正确
mini_config = {
    'l_max_values': [3, 5],  # 只有两个
    'num_seeds': 2,
    'sampling_methods': ['uniform', 'lebedev'],
}
run_experiment_1(mini_config)  # 应该快速完成（几分钟）
```

### 7.3 数值稳定性检查
- 浮点精度是否足够（float32 vs float64）
- 大的 l 值是否会导致数值问题
- 权重求和的数值误差

---

## 8. 时间估计

| 阶段 | 任务 | 时间 |
|------|------|------|
| 基础设施 | 实现采样和工具函数 | 1-2 周 |
| 实验1 | 精度对比 | 3-5 天 |
| 实验2 | 计算开销 | 2-3 天 |
| 实验3 | 任务性能 | 3-5 天 |
| 实验4 | 扩展性分析 | 2-3 天 |
| 分析和写作 | 结果分析、可视化、文档 | 3-5 天 |
| **总计** | | **4-6 周** |

---

## 9. 出版物标准

### 写 Workshop 论文时需要展示的结果

最少需要（2-4页空间限制）：
- [ ] 图1：精度曲线对比（Lebedev vs GL vs Uniform）
- [ ] 表1：相同精度下的采样点数对比
- [ ] 图2或表2：计算时间对比
- [ ] 图3：Spherical MNIST 的训练准确率
- [ ] 简短的分析和洞察

### 如果扩展为完整论文（ICLR/ICML）
需要补充：
- [ ] 完整的理论分析（为什么 Lebedev 最优）
- [ ] 在更多数据集上的验证
- [ ] 与其他最新方法的对比（如 Needlet CNN 等）
- [ ] 消融研究（Lebedev 的哪些性质最重要）
- [ ] 在实际应用中的案例研究

---

## 10. 故障排除指南

### 常见问题

**Q1: Lebedev 采样点生成有问题**
- 检查：查找表是否正确加载
- 验证：点应该在单位球面上（norm=1）
- 验证：权重总和应该是 4π

**Q2: 精度对比中没有看到预期的差异**
- 可能原因：球面调和计算有误
- 检查：与 scipy.special.sph_harm 对比
- 检查：积分公式是否正确

**Q3: 内存用量很大**
- 可能：预先计算了太多中间值
- 解决：只在需要时计算球面调和值

**Q4: 训练变慢而不是变快**
- 可能：采样点初始化的开销
- 解决：缓存采样点和权重

---

## 总结

这个实验计划通过4个递进的实验全面对比 Lebedev quadrature 与现有采样方法：

1. **精度** → 基础性能对比
2. **效率** → 计算资源利用
3. **可用性** → 实际任务中的表现  
4. **可扩展性** → 理论优势何时显现

预期能够清晰地证明 Lebedev 的优势，为 workshop 论文和后续工作提供坚实的实验基础。
