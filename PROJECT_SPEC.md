# 🧠 BCI 动态闭环控制系统仿真平台 —— 工程规格说明书

## 项目概述
基于 Yerkes-Dodson 定律（倒U型曲线）建立人脑注意力数学模型，并在此模型上实验四种闭环控制算法的追踪控制效果。系统由 Python 后端引擎 + Streamlit 前端面板构成。

---

## 1. 核心数学模型 —— 受控对象 (Plant)

### 1.1 专注度响应函数
```
Attention(D, t) = exp( -(D - Optimal(t))^2 / C ) + Noise
```
| 参数 | 含义 | 默认值 |
|------|------|--------|
| D | 当前游戏难度 [0, 100] | - |
| t | 仿真时间步 | 0..N |
| C | 倒U曲线宽度常数 | 400 |
| Noise | 高斯白噪声 N(0, σ²) | σ=0.02 |

### 1.2 最佳难度时变函数（学习效应）
```
Optimal(t) = Optimal_0 + Δmax * (1 - exp(-λ * t))
```
| 参数 | 含义 | 默认值 |
|------|------|--------|
| Optimal_0 | 初始最佳难度 | 40 |
| Δmax | 最大技能提升幅度 | 40 |
| λ | 学习率 | 0.05 |

---

## 2. 四种闭环控制算法

### 2.1 PID 控制器 (连续型)
- 标准 Parallel-form PID: u(t) = Kp*e(t) + Ki*∫e dt + Kd*de/dt
- 抗积分饱和 (Clamping anti-windup)
- 死区 (Deadband)，输出限幅 [0, 100]
- 可调参数: Kp, Ki, Kd, deadband, output_min, output_max

### 2.2 离散/阶梯 PID 控制器
- 基于连续 PID 输出进行量化
- 输出量化为 N 个固定档位（如 10 档）
- 可调参数: Kp, Ki, Kd, quantization_levels

### 2.3 模糊控制器
- 输入: 误差 e(t) 和误差变化率 Δe(t)
- 模糊集: {NB, NS, ZO, PS, PB}
- 隶属度函数: 三角形/梯形
- 去模糊化: 重心法 (Centroid)
- 可调参数: 模糊规则表中的增益系数

### 2.4 贝叶斯优化控制器 (UCB)
- 使用 Upper Confidence Bound 策略
- GP 代理模型 (使用简单的核函数近似)
- 每步选择 D = argmax(μ(D) + κ*σ(D))
- 平衡探索(Exploration)与利用(Exploitation)
- 可调参数: κ (探索系数), window_size

---

## 3. 仿真引擎 (SimulationEngine)
- 将 `SimulatedBrain` 与所选 Controller 连接
- 支持 step-by-step 逐帧运行
- 记录完整历史: [t, D, Attention, Target, error, Optimal_t]
- 支持重置和参数热更新

---

## 4. 前端控制面板 (Streamlit)
- 控制器选择下拉菜单
- 各控制器参数滑块
- 大脑模型参数设置 (λ, C, σ, Optimal_0, Δmax)
- 仿真控制按钮 (开始/暂停/重置/步进)
- 四个可视化面板:
  1. 时域响应图 (双Y轴: 专注度 + 控制量)
  2. 倒U曲线动态图 (标注当前工作点)
  3. 评价指标卡片 (稳态误差、超调量、抖振)
  4. 实时数据表格

---

## 5. Agent 团队分工

### Agent 1: 算法与后端工程师
**文件**: `brain_model.py`, `controllers.py`, `simulation_engine.py`
- 实现 SimulatedBrain 类
- 实现四种 Controller 类
- 实现 SimulationEngine 类
- 提供 get_state() / step() API

### Agent 2: 前端UI工程师
**文件**: `app.py` (Streamlit)
- 控制面板布局
- 四种可视化图表 (Plotly)
- 评价指标实时计算与展示
- 与后端 API 对接

### Agent 3: 系统测试与优化工程师
- 边界条件测试 (λ=0, λ=1.0, σ=0, σ=0.2)
- 控制器稳定性验证
- 前后端集成测试
- 反馈修改意见给 Agent 1/2
