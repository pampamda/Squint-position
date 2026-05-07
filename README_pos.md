# Squint — 位置感知原语系列（`_pos`）

本文档说明带 `_pos` 后缀的训练、部署、调用流程。

---

## 什么是 `_pos` 系列

标准视觉策略（`SO101LiftCube-v1`）只看腕部相机图像；`_pos` 系列在此基础上额外注入一个**物体位置估计** $(x,y,z)$，让策略知道目标在哪里，而不只是靠视觉猜测。

State 末尾多 7 个维度：

```
[ ...原始 state... | object_pos_est(3) | tcp_to_object(3) | position_valid(1) ]
```

- `position_valid=1`：位置有效，正常使用
- `position_valid=0`：检测失败，这 7 维全归零，策略学会在不确定时停止运动

位置来源有三种，训练和部署分开使用：

| Provider | 用途 | 位置来源 |
|----------|------|----------|
| `SimGT`（仿真默认）| 训练 + 仿真验证 | 仿真真值 + 高斯噪声 |
| `Fixed`（dummy）| 真机初步测试 | 手动指定固定坐标 |
| `RealDetector`（待实现）| 真实部署 | RGB-D 桌面相机检测 |

---

## 1. 训练

### 当前支持的环境

| 环境 ID | 任务 | 脚本 | 输出路径 |
|---------|------|------|---------|
| `SO101LiftCubePos-v1` | 抓取 cube 并抬至 rest | `train_squint_pos.py` | `runs/lift_pos__YYYYMMDD/` |

### 训练命令

**有 wandb 追踪（推荐）：**
```bash
python train_squint_pos.py \
    --num_envs=512 \
    --num_eval_envs=8 \
    --render_size=96 \
    --track \
    --wandb_entity=YOUR_WANDB_USERNAME
```

**纯本地训练：**
```bash
python train_squint_pos.py \
    --num_envs=512 \
    --num_eval_envs=8 \
    --render_size=96
```

训练约 3-5 分钟收敛（LiftCubePos）。`eval/success_at_end > 0.7` 视为合格。

Checkpoint 保存于：
```
runs/lift_pos__20260430/ckpt.pt
```

---

## 2. 仿真验证（部署前）

训练完成后，先在仿真中用 SimGT provider 验证策略是否有效：

```bash
python deploy_pos.py \
    --checkpoint runs/lift_pos__20260430/ckpt.pt \
    --provider sim_gt \
    --episodes 10
```

观察成功率是否与训练 eval 指标一致。

---

## 3. 真实机器人部署

### 3a. 用 `deploy_pos.py`（命令行方式）

**方式一：Fixed provider（dummy 固定坐标，首次验证用）**

把方块放在桌面已知位置，手动量出坐标后传入：

```bash
python deploy_pos.py \
    --checkpoint runs/lift_pos__20260430/ckpt.pt \
    --provider fixed \
    --fixed_pos_x 0.30 \
    --fixed_pos_y 0.00 \
    --fixed_pos_z 0.013
```

坐标系：机器人基座坐标系（原点 = base link 中心）。
训练时物品典型范围：`X∈[0.20, 0.40]`，`Y∈[-0.10, 0.10]`，`Z≈0.013 m`。

**方式二：SimGT provider（仿真位置同步，调试用）**

仅限仿真 + 真实对齐时使用（`--debug` 模式），不适合独立真实部署：

```bash
python deploy_pos.py \
    --checkpoint runs/lift_pos__20260430/ckpt.pt \
    --provider sim_gt \
    --episodes 5
```

### 3b. 用 `primitive_executor.py`（Python 接口方式）

适合从另一个脚本调用（上层规划器、测试脚本等）。

**直接运行测试：**

```bash
python primitive_executor.py \
    --checkpoint runs/lift_pos__20260430/ckpt.pt \
    --x 0.30 --y 0.00 --z 0.013 \
    --episodes 3
```

**从代码调用（推荐方式）：**

```python
from primitive_executor import PrimitiveExecutor

# 启动时加载（慢，只做一次）
executor = PrimitiveExecutor(
    pick_checkpoint="runs/lift_pos__20260430/ckpt.pt",
    control_freq=15,
    action_scale=0.15,
)

# --- 当前阶段：dummy 检测（手动指定坐标）---
pos = executor.detect_dummy([0.30, 0.00, 0.013])
success = executor.pick(pos)
print("pick", "OK" if success else "FAILED")

# --- 未来：接入真实 RGB-D 检测器 ---
# pos = your_detector.get_object_position()
# success = executor.pick(pos)

executor.close()
```

### 上层规划器调用示例

```python
executor = PrimitiveExecutor(pick_checkpoint="...")

# 高层逻辑：检测 → 抓取 → 放置（place 待实现）
obj_pos = executor.detect_dummy([0.30, 0.00, 0.013])

if executor.pick(obj_pos):
    print("抓取成功，等待 place 原语实现...")
else:
    print("抓取失败，重试或跳过")

executor.close()
```

---

## 4. 参数说明

### `deploy_pos.py` 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | - | checkpoint 路径，或 `wandb` |
| `--provider` | `sim_gt` | `sim_gt` / `fixed` / `real_detector` |
| `--fixed_pos_x/y/z` | 0.30/0.00/0.013 | Fixed provider 的目标坐标（米） |
| `--action_scale` | 0.15 | 动作缩放，值越小动作越慢越安全 |
| `--control_freq` | 30 | 控制频率（Hz），建议真机用 15 |
| `--episodes` | 5 | 运行的 episode 数量 |
| `--max_episode_steps` | 120 | 每 episode 最大步数 |

### `PrimitiveExecutor` 构造参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pick_checkpoint` | 必填 | lift_pos ckpt.pt 路径 |
| `control_freq` | 15 | 建议真机用 15 Hz |
| `action_scale` | 0.15 | 动作缩放 |
| `max_episode_steps` | 120 | 每次 pick 最多执行步数 |

---

## 5. 典型使用流程

```
Step 1: 训练
        python train_squint_pos.py --num_envs=512
        → runs/lift_pos__YYYYMMDD/ckpt.pt

Step 2: 仿真验证
        python deploy_pos.py --checkpoint runs/lift_pos__YYYYMMDD/ckpt.pt
                             --provider sim_gt --episodes 10
        → success_rate > 70% 再进行真机测试

Step 3: 真机 Fixed provider 测试
        把方块放在 (0.30, 0.00) 处
        python deploy_pos.py --checkpoint ... --provider fixed
                             --fixed_pos_x 0.30 --fixed_pos_y 0.00

Step 4: 调用接口测试
        python primitive_executor.py --checkpoint ... --x 0.30 --y 0.00 --z 0.013

Step 5 (后续): 接入真实 RGB-D 检测
        实现 RealDetectorPositionProvider（深度点云分割）
        在 PrimitiveExecutor 中替换 detect_dummy()
```

---

## 6. 文件索引

| 文件 | 职责 |
|------|------|
| `train_squint_pos.py` | 训练入口，保存到 `runs/lift_pos__DATE/` |
| `envs/lift_position.py` | `SO101LiftCubePos-v1` 环境，注入位置观测 |
| `deploy_pos.py` | 命令行部署脚本，支持三种 provider |
| `deploy_utils/position_provider.py` | SimGT / Fixed / RealDetector provider |
| `primitive_executor.py` | Python 类接口，供上层脚本调用 |
