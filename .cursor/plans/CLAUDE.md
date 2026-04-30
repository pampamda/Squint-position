# Squint — 代码架构文档

## 项目概述

**Squint** 是一个面向机器人 sim-to-real 的视觉强化学习框架，目标是在仿真中**快速收敛**（2-9 分钟），然后迁移到真实的 SO-101 / SO-100 机械臂。

核心算法：**SAC + 分布式 Q 学习（C51）+ 视觉编码器（CNN）**，配合大规模并行仿真（1024 环境）和 torch.compile / CUDA Graphs 加速。

---

## 目录结构与文件职责

```
squint/
├── train_squint.py          # 主训练脚本（神经网络结构 + 训练循环）
├── train_squint_pos.py      # _pos 系列：位置感知训练入口（新）
├── deploy.py                # 真实机器人部署
├── deploy_pos.py            # _pos 系列：位置感知部署（新）
├── utils.py                 # 工具包装器：图像下采样、色彩抖动
├── envs/
│   ├── __init__.py          # 注册所有 gym 环境
│   ├── base_random_env.py   # 基础环境（领域随机化 + 相机系统）
│   ├── lift.py              # Lift 任务
│   ├── lift_position.py     # _pos 系列：带位置观测的 Lift 任务（新）
│   ├── reach.py             # Reach 任务
│   ├── place.py             # Pick-and-place 任务
│   ├── stack.py             # Stack 任务
│   └── robot/
│       ├── so101.py         # SO101 机械臂 agent（含抓取判断）
│       └── so100.py         # SO100 机械臂 agent
└── deploy_utils/
    ├── robot_config.py      # 硬件配置（串口、相机、标定文件）
    ├── manipulator.py       # LeRobotRealAgent：真实机器人控制接口
    ├── position_provider.py # _pos 系列：位置检测 provider（新）
    ├── tune_camera.py       # 相机标定交互工具
    └── test_cam.py          # 相机测试工具
```

### `_pos` 后缀说明

以下是用户**后续新增**的位置感知相关逻辑，与原始视觉策略并列独立：

| 文件 | 功能 |
|------|------|
| `train_squint_pos.py` | 封装 `train_squint.py`，使用 `SO101LiftCubePos-v1` 环境 |
| `envs/lift_position.py` | 在 Lift 任务基础上，将 `(物体位置估计, tcp→物体向量, 位置有效标志)` 注入 state 观测 |
| `deploy_pos.py` | 部署时注入 position provider 的位置特征到 state |
| `deploy_utils/position_provider.py` | 定义两类 provider：`SimGTPositionProvider`（仿真真值+噪声）和 `RealDetectorPositionProvider`（占位，未来用 RGB-D 检测） |

---

## 仿真训练架构

### 1. 环境层（ManiSkill3 + SAPIEN）

**环境继承链：**
```
ManiSkillBaseEnv
  └── BaseRandomEnv          (领域随机化基础 + 绿幕叠加)
        └── ThirdCameraEnv / WristCameraEnv  (相机类型，当前默认 wrist)
              └── LiftEnv / ReachEnv / PlaceEnv / StackEnv
```

**BaseRandomEnv（`envs/base_random_env.py`）：**
- `RandomizationConfig` 数据类集中管理所有随机化参数范围（关节噪声、光照、夹爪刚度、相机偏移等）
- 绿幕叠加：用 segmentation mask 将背景替换为 `black_overlay.png`，提升 sim-to-real 鲁棒性
- `CAMERA_TYPE = "wrist"` 控制使用腕部相机还是第三视角相机

**WristCameraEnv：**
- 相机固定在 `gripper_link`（SO101 的 `finger1_tip`）上
- 每个控制步随机化位置偏移和旋转，模拟真实安装误差
- 基础偏移：`(0, 0.056, -0.06)`

**领域随机化（每 episode 或每步）：**
- 机器人关节初始位置噪声
- 夹爪刚度 / 阻尼
- 机器人颜色、光照强度
- 相机位置 / 旋转 / FOV
- 物体几何形状、摩擦系数、密度
- 背景叠加图像

**Lift 任务奖励（`envs/lift.py`）：**
```
reaching  = 1 - tanh(5 * dist_to_item)        # 接近物体
grasping  = +1  if grasped
lifting   = exp(-2 * dist_to_rest_qpos)        # 抬起 + 回归姿态
penalty   = -3  if table_contact
           -1   per step not lifted
```

### 2. 神经网络架构（`train_squint.py`）

**图像处理流水线：**
```
128×128 RGB (sim 渲染)
    ↓ ColorJitter（apply_jitter）
    ↓ 下采样到 16×16（utils.py 中的 DownsampleWrapper）
    → 存入 Replay Buffer
```

**CNN 编码器（`SACEncoder`）：**
```python
Input: (N, H=16, W=16, C)  # channels-last
Normalize: → [-0.5, 0.5]
NHWC → NCHW
Conv2d(C, 32, kernel=4, stride=2)   → (N, 32, 7, 7)
Conv2d(32, 64, kernel=4, stride=1)  → (N, 64, 4, 4)
Flatten                              → (N, 1024)
```

**投影层（`SACProjection`）：**
```python
rgb_proj:   Linear(1024) → LayerNorm → Tanh → 50-dim
state_proj: Linear(state_dim) → LayerNorm → ReLU → 256-dim
concat: → 306-dim 表示
```

**Actor（策略网络）：**
```python
Projection(306) → FC(256)→LN→ReLU → FC(256)→LN→ReLU → FC(256)→LN→ReLU
→ mean, log_std → tanh squashing → action ∈ [-1, 1]
```

**Critic（分布式 Q，C51 Ensemble）：**
- 2 个独立 Q 网络（vmap 并行化）
- 101 个原子，支撑 [-20, 20]
- 输出 logits → softmax → 期望 Q 值
- 目标分布用 Bellman 方程的 C51 投影计算

### 3. SAC 训练循环（`train_squint.py:847-951`）

```
初始化 1024 个并行 env + 16 个 eval env
↓
循环（total_timesteps = 1,500,000 步）：
  │
  ├─ [每 eval_freq 步] 评估：
  │    16 个 eval env 运行确定性策略
  │    记录 success_rate、return，保存 checkpoint
  │
  ├─ [Rollout] 环境交互：
  │    前 learning_starts 步：随机动作
  │    之后：encoder_detach + actor_detach 推理
  │    存储 (obs, next_obs, action, reward, done) 到 Replay Buffer
  │
  └─ [每步 num_updates=256 次] 梯度更新：
       采样 batch_size=512 条转移
       │
       ├─ Critic 更新（update_main）：
       │    编码 obs + next_obs
       │    计算 C51 目标分布（Bellman 投影）
       │    交叉熵损失
       │    自动调整熵系数 alpha（autotune）
       │
       ├─ Actor 更新（update_actor，每 policy_frequency 步）：
       │    梯度：maximize E[Q] - alpha*log_pi
       │
       └─ 目标网络软更新：
            θ_target ← (1-tau)*θ_target + tau*θ_online  (tau=0.01)
```

**推理副本（weight-sharing，无额外内存开销）：**
- `encoder_detach` / `actor_detach`：Rollout 时用，不参与梯度
- `encoder_eval` / `actor_eval`：评估时用（eval mode）
- 通过 `tensordict.from_module()` 实现参数共享

### 4. 工程加速

| 技术 | 效果 |
|------|------|
| `torch.compile()` | 图追踪，消除 Python overhead |
| CUDA Graphs | 内核融合，减少 CPU-GPU 同步 |
| 1024 并行环境 | 充分利用 GPU 并行度 |
| channels-last 内存格式 | Conv 运算更高效 |
| vmap（Q 网络 ensemble）| 并行化多头 Q 网络 |
| bfloat16 混合精度 | 前向传播加速 |

---

## 关键超参数（默认值）

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `num_envs` | 1024 | 并行训练环境数 |
| `total_timesteps` | 1,500,000 | 总训练步数 |
| `buffer_size` | 1,000,000 | Replay Buffer 大小 |
| `batch_size` | 512 | 每次采样数 |
| `num_updates` | 256 | 每环境步的梯度更新次数 |
| `learning_starts` | 5,000 | 开始训练前的随机探索步数 |
| `gamma` | 0.9 | 折扣因子 |
| `tau` | 0.01 | 目标网络软更新系数 |
| `num_atoms` | 101 | C51 分布原子数 |
| `render_size` | 128 | 仿真渲染分辨率 |
| `image_size` | 16 | 网络输入分辨率 |

---

## 部署（真实机器人）

**`deploy.py` 主流程：**
1. 初始化真实机器人（LeRobot，串口 `/dev/ttyACM0`）
2. 同时创建仿真环境（供可视化/调试）
3. 加载 checkpoint（本地路径或 W&B）
4. 主循环：获取摄像头图像 → `DeployAgent` 推理 → 动作缩放 → 执行
5. 键盘控制：`s` 跳过，`q` 退出
6. 可选：异步录制视频（`AsyncRecorder`）

**`DeployAgent`（定义在 `train_squint.py`）：**
- 封装 encoder + actor
- 输入：128×128 RGB → 下采样 16×16 → 推理 → 确定性动作

**`deploy_pos.py`（位置感知部署）：**
- 调用 `SimGTPositionProvider` 从仿真获取物体位置（加噪声）
- 将位置特征注入 state 观测，其余同 `deploy.py`
- `RealDetectorPositionProvider` 为占位符，未来接 RGB-D 检测器

---

## 收敛基准

| 任务 | 收敛时间（wall clock ~15分钟内） |
|------|-------------------------------|
| ReachCube / ReachCan | ~2 分钟 |
| LiftCube | ~3 分钟 |
| LiftCan | ~4 分钟 |
| PlaceCube | ~5 分钟 |
| PlaceCan | ~6 分钟 |
| StackCube / StackCan | ~6-9 分钟 |
