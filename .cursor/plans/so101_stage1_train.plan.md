# 桌面整理机器人 — 基础原语规划（最终版 v3）

## Context（背景）

目标：为机械臂训练一套**位置感知基础动作原语**，供上层规划器以坐标接口调用，最终实现桌面整理。
上层（场景理解、决策哪先抓哪放哪）由队友负责；本阶段只负责：给定坐标 → 可靠执行。

**已确认设计决策：**
- Place 目标：动态坐标传入（上层运行时指定 x,y,z）
- Pick→Place 衔接：Place 始终从 rest_qpos 持物出发（两个独立策略）
- Stack：需要训练，桌面整理复杂场景会用到
- home()：必须的复位原语，脚本式关节插值实现，无需 RL 训练
- 相机分工：腕部相机 → RL 策略视觉输入；桌面相机 → 位置检测（稍后实现）
- 检测时机：先完成 RL 训练，`RealDetectorPositionProvider` 作为独立任务稍后开展
- 接口形式：Python 类接口为主，可选 FastAPI 包装供跨进程调用
- 检测方案：改用深度点云分割（非颜色分割），适配任意形状的真实物品

---

## 最终原语集

```
detect(rgb_d_frame)      →  object_pos(x,y,z)        [稍后实现，桌面相机，深度点云分割]
pick(object_pos)         →  rest_qpos + holding       [RL，= LiftCubePos，训练中]
place(target_pos)        →  object released at pos    [RL，待建，PlaceCubePos-v1]
stack(stack_obj_pos)     →  object placed on top      [RL，待建，StackCubePos-v1]
home()                   →  arm @ rest_qpos, empty    [脚本式关节插值，无需训练]
```

**高层调用流（队友接口）：**
```python
executor = PrimitiveExecutor(...)
obj_pos = detector.detect(target_object)   # 桌面相机 RGB-D，深度点云分割
success = executor.pick(obj_pos)
if success:
    target = planner.decide()              # 上层规划：放哪里 / 叠哪里
    executor.place(target)                 # 或 executor.stack(target)
    executor.home()                        # 每次 place/stack 后必须调用，复位到 rest_qpos
# 下一轮 pick 从 rest_qpos 出发，与 PlaceCubePos 训练初始状态对齐
```

---

## 设计决策详细说明

### Q1：PlaceCubePos 环境设计

#### 关键结论：从 DefaultCameraEnv 直接继承，不扩展 place.py

`place.py` 的任务结构与 PlaceCubePos 根本不兼容：

| 维度 | `place.py`（现有） | `PlaceCubePos`（待建） |
|------|-------------------|-----------------------|
| 放置目标 | 物理存在的 bin，有四面墙 | 虚拟坐标点，无物理约束 |
| Episode 起始状态 | item 在桌面，arm 在 rest | arm 持物在 rest_qpos |
| 成功判定 | item 在 bin XY 范围内 | item 距 target_pos < threshold，已释放，robot static |
| 奖励信号 | 依赖 `bin.pose.p` | 依赖注入的 `target_pos`（obs） |

**正确的继承路径**（和 `stack.py` 一样）：
```
DefaultCameraEnv → PlacePosition
```

**可复用的代码（约 80%）：**

| 可复用内容 | 来源 |
|-----------|------|
| `_load_agent` 整段 | `place.py:112-120` |
| `_get_obs_agent` 整段 | `place.py:405-415` |
| Domain randomization config 结构 | `place.py:23-40` |
| Position noise/dropout 注入模式 | `lift_position.py:42-79` |
| Table scene + item spawn 逻辑 | `place.py:120-165`（去掉 bin 部分） |

**需要从零写的（约 250 行）：**
- `_initialize_episode`（持物初始化 + target_pos 随机采样）
- `_get_obs_extra`（注入 target_pos 而非 bin 位置）
- `evaluate`（纯坐标距离判定，无 bin）
- `compute_dense_reward`（基于 target_pos 的 dense reward）

#### PlaceCubePos Obs 设计（7 dims 追加到 state 末尾）

与 LiftPosition 的 7-dim 追加模式对齐：

```python
# _get_obs_extra 追加内容
target_pos      (3)   # 放置目标坐标；训练时 per-episode 随机，部署时来自上层规划器
tcp_to_target   (3)   # 实时更新：target_pos - tcp_pos，给策略提供导航向量
is_item_grasped (1)   # 策略据此学习松夹爪时机
# 共 7 dims，与 LiftPosition 追加维度数一致
```

**与 LiftPosition 的关键区别：**
- `target_pos` 是确定性值（来自规划器），**不加噪声**，无 dropout
- `tcp_to_target` 实时计算，episode 内随 TCP 运动持续更新
- 不需要 `position_valid` flag，target 永远已知；若上层规划器未就绪，在 `primitive_executor.py` 层阻塞

#### target_pos 训练时如何生成

训练时，`target_pos` 在 `_initialize_episode` 里 per-episode 随机采样：
- XY：用与 item 相同的 `UniformPlacementSampler`，在桌面 spawn 区域内随机，需排除 item 初始位置附近（避免零距离 trivial solution，min_dist ≈ 8cm）
- Z：`table_height + item_half_size`（平放）；若需覆盖叠放，z 范围扩展至 `[table_height, table_height + 8cm]`（覆盖单层叠放的 ~3.6-4.9cm）

---

### Q2：PlaceCubePos 技术难点

#### 难点一：持物初始化

Place 策略从 rest_qpos 持物出发，训练时 episode 开始时物体必须已被抓住。

**推荐方案：前 N 步 Weld 固定，随后解除**
1. 设置机器人为 rest_qpos + 夹爪半闭合
2. 将物体 spawn 在 TCP 位置（加 grasp_offset_noise，见难点二）
3. 前 `warmup_steps`（建议 5 步）通过 SAPIEN `create_drive` Weld 关节固定 item 到 TCP
4. warmup 结束后解除 Weld，让物理接触正式形成
5. 若 `is_item_grasped=False` 且步数 < 10 → 强制 reset

不依赖"希望摩擦力够大"，Weld 方案物理上可靠，是硬性保证。

#### 难点二：抓握分布偏移（grasp domain gap）

**问题**：LiftCubePos 策略结束时，item 在夹爪里的实际位置未必是"完美居中"（TCP 中心）。PlaceCubePos 若训练时总是完美抓握，则部署时 pick→place 串联会有成功率下降。

**解决时机：在写 PlaceCubePos 的 `_initialize_episode` 时直接解决，不推后。**

**实现**：给 item spawn 位置添加 `grasp_offset_noise`，作为 `PlacePositionRandomizationConfig` 的可配置参数：

```python
# PlacePositionRandomizationConfig 新增字段
grasp_offset_noise_std: Sequence[float] = (0.005, 0.005, 0.003)  # XY 5mm, Z 3mm
grasp_offset_noise_clip: float = 0.015

# _initialize_episode 里
tcp_pos = self.agent.tcp_pos[env_idx]
grasp_noise = torch.randn((b, 3), device=self.device) * std
grasp_noise = torch.clamp(grasp_noise, -clip, clip)
item_init_pos = tcp_pos + grasp_noise   # item 不总在 TCP 正中心
```

噪声量与 `LiftPositionRandomizationConfig.position_noise_std` 对齐（约 5mm XY / 3mm Z），确保训练分布覆盖 LiftPos 策略的实际输出范围。Step 4 串联测试若成功率偏低，可调大此参数重训。

---

### Q3：Position Provider 接口重设计

**现有 `SimGTPositionProvider`（ObjectPositionProvider）语义：** "被抓物体在哪"  
**新增 `TargetPositionProvider` 语义：** "把物体放到哪"

两者语义截然不同，必须分开为两个独立 Protocol：

```python
# deploy_utils/position_provider.py（更新后）

class ObjectPositionProvider(Protocol):
    """LiftPos 用：感知检测被抓物体位置"""
    def get_object_position(
        self, sim_env, real_obs: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (object_pos_est, tcp_to_object, position_valid)"""

class TargetPositionProvider(Protocol):
    """PlacePos 用：由上层规划器指定放置目标坐标"""
    def get_target_position(
        self, target_pos: np.ndarray, tcp_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (target_pos_tensor, tcp_to_target)
        target_pos: 上层规划器传入的 numpy 坐标，不经过感知模块
        tcp_pos: 当前 TCP 位置，用于计算 tcp_to_target
        """
```

**已有类的归属：**
- `SimGTPositionProvider` → 实现 `ObjectPositionProvider`（用于 LiftPos）
- `RealDetectorPositionProvider` → 实现 `ObjectPositionProvider`（Phase 2 实现）
- `DirectTargetPositionProvider`（新建）→ 实现 `TargetPositionProvider`，直接将规划器坐标转为 tensor

#### 部署侧 state 注入稳定性规范（必须）

当前 `deploy_pos.py` 的 `inject_position_features` 采用“替换 state 末尾 7 维”的 MVP 约定，仅适用于 LiftPos，**不能作为 PlacePos/StackPos 的长期方案**。

为避免训练环境字段顺序变化导致部署 silent bug，统一采用**命名式 schema 注入**：

1. 每个 `_pos` 环境在 reset 后暴露 `position_feature_spec`（或同等配置），明确：
   - 需要注入的 feature 名称（如 `object_pos_est` / `tcp_to_target` / `position_valid`）
   - 每个 feature 的维度
   - 在 flatten 后 state 中的 slice 索引（start, end）
2. 部署脚本按 `position_feature_spec` 写入对应 slice，**禁止**硬编码“末尾 N 维替换”。
3. 启动时执行一次 schema 校验：
   - 模型 checkpoint 中保存 `obs_schema_hash`
   - 部署环境实时计算 `obs_schema_hash`
   - hash 不一致时拒绝启动并报错，避免错误推理。
4. LiftPos / PlacePos / StackPos 共用同一注入工具函数 `inject_features_by_spec(...)`，仅 spec 不同。

> 实施顺序：先保留 Lift 兼容路径；新增 spec 路径并默认启用；确认无回归后移除旧的“末尾 7 维”写法。

---

### Q4：home() 原语

**推荐：脚本式关节插值，不需要 RL 训练。**

```python
# primitive_executor.py
def home(self, duration: float = 3.0) -> bool:
    """空手回 rest_qpos。使用 pd_joint_target_delta_pos，
    每步 delta = clip((rest_qpos - current_qpos) * gain, -max_step, max_step)
    直到 |current - rest| < threshold 或 timeout。
    """
```

原因：
- 空手回 rest 是有确定目标的控制问题，不需要 RL
- `pd_joint_target_delta_pos` 控制模式天然支持增量关节指令
- 无需视觉输入，不依赖任何感知模块
- 确定性行为，鲁棒性优于 RL 策略

调用约束：**每次 `place()` 或 `stack()` 后必须调用 `home()`，下一次 `pick()` 才能从 `rest_qpos` 出发（与 PlaceCubePos 训练初始状态对齐）。**

---

### Q5：Stack 原语 — 优先用 PlacePos 拆解，按需再训 StackPos

#### 推荐先用 PlacePos 拆解 Stack

```
pick(itemA_pos)         ← LiftPos 策略
place(itemB_top_pos)    ← PlacePos 策略
home()                  ← 复位
```
其中 `itemB_top_pos.z = itemB_pos.z + itemB_half_height + itemA_half_height`，由检测层计算。
PlacePos 训练时 z 范围设计为 `[table_height, table_height + 8cm]`，cube 叠放高度约 3.6-4.9cm，在覆盖范围内。

#### 何时需要单独训练 StackPos

**接触动力学是关键差异：**

| 对比维度 | place on table | place on object (stack) |
|---------|---------------|------------------------|
| 支撑面积 | 无限大（桌面） | 很小（cube 顶面 ~5×5cm） |
| 释放后稳定性 | 几乎不会滑落 | XY 偏差 >1cm 就会滑落或翻倒 |
| 释放时机 | 任何时候松开都稳定 | 必须在"正上方 + 几乎静止"时才能松开 |
| 对高度敏感性 | 差几 cm 无所谓 | 差 5mm 就可能撞到已叠物体边缘 |

**触发条件：以下任何一条成立，就需要单独训练 StackPos：**

1. **实测成功率 < 50%**：PlacePos 在叠放测试（target 为 cube 顶面）中成功率低于 50%
2. **底座物体较小**（itemB 宽度 < 3cm）：支撑面积过小，PlacePos 精度不够
3. **低摩擦表面**（如光面物品叠放）：接触不稳定，需要专门学"轻放"
4. **需要叠放 2 层以上**：重心高，稳定性要求指数级上升

**StackPos 的实现方案（按需）：**

基于现有 `stack.py`，仿照 `lift_position.py` 模式注入双位置输入：
```
itemA_pos_est  (3)  + tcp_to_itemA (3) + valid_A (1) = 7 dims  ← 抓取目标
itemB_pos_est  (3)  + itemA_to_itemB (3) + valid_B (1) = 7 dims  ← 叠放基准
```
共 14 dims 附加到 state 末尾，其余结构与 LiftPos/PlacePos 一致。
注册环境 ID：`SO101StackCubePos-v1`。

---

### Q6：多物品场景兼容性

**结论：有条件可行，有明确限制。**

**能工作的情况：**
- 物品间距 > 10cm，上层系统提供目标坐标，策略受位置信号强引导
- 腕部相机视野窄（只看夹爪附近），其他物品在画面外时几乎无干扰

**会失败的情况：**
- 物品密集排列：手臂运动轨迹可能撞到旁边物品（策略没有学过避障）
- 多物品出现在腕相机画面中：可能轻微干扰策略的视觉判断

**后续优化（不在本次范围）：** 训练时随机添加 1-3 个干扰物（distractor objects），策略会学会忽略它们。

---

### Q7：上层系统调用接口

**主接口：Python 类（同进程直接调用）**

```python
# primitive_executor.py（待建）
class PrimitiveExecutor:
    def pick(self, object_pos: np.ndarray, timeout: float = 15.0) -> bool:
        """执行抓取，结束时 arm @ rest_qpos + holding"""
    def place(self, target_pos: np.ndarray, timeout: float = 15.0) -> bool:
        """执行放置，结束时 arm 停在 target 附近，item 已释放"""
    def stack(self, target_pos: np.ndarray, timeout: float = 15.0) -> bool:
        """执行叠放"""
    def home(self, duration: float = 3.0) -> bool:
        """空手回 rest_qpos，脚本式插值，每次 place/stack 后调用"""
```

**可选包装：FastAPI（跨进程/跨机器）**
```
POST /pick  {"position": [x, y, z]}  →  {"success": true}
POST /place {"position": [x, y, z]}  →  {"success": false, "reason": "timeout"}
POST /stack {"position": [x, y, z]}  →  {"success": true}
POST /home  {}                        →  {"success": true}
```

不引入 ROS2，对当前阶段过重。

---

### Q7.5：坐标系规范（新增，强制执行）

#### 推荐方案：以机器人基座坐标系（`base_frame`）作为全链路唯一真值坐标系

`pick(object_pos)` 与 `place(target_pos)` 的输入坐标统一定义为 **robot base frame（米）**，原因：

1. 控制执行层（IK/关节控制）天然工作在 base frame，减少运行时坐标转换链长度。
2. 检测层（桌面相机）只需一次外参变换 `camera_frame -> base_frame`，接口边界清晰。
3. 上层规划可在任意局部坐标（桌面平面/table frame）工作，但在调用原语前统一转换到 base frame，避免多模块混用 frame。
4. sim2real 一致性最好：仿真中物体 pose 与 tcp pose 本就易映射到机器人基座语义。

#### 约束与接口

- `ObjectPositionProvider` 输出：`object_pos_est_base`, `tcp_to_object_base`, `position_valid`
- `TargetPositionProvider` 输入：`target_pos_base`
- `PrimitiveExecutor.pick/place/stack` 的对外参数一律为 `np.ndarray([x, y, z])` in base frame
- 文档和日志统一标注单位为米（m），禁止混入毫米（mm）裸数

#### 必需标定与验收指标（部署前门槛）

1. 相机内参标定完成；
2. 桌面相机到机器人基座外参标定完成（`T_base_camera`）；
3. 静态点位复投影误差：
   - XY 均值误差 < 1cm
   - Z 均值误差 < 1.5cm
4. 若超过阈值，禁止进入真实抓放联调，只能进行仿真/回放验证。

> 备注：可在规划层维护 `table_frame` 便于任务表达，但执行层的唯一输入标准仍是 `base_frame`。

---

### Q8：真实物品适配 + 夹爪开合自适应

#### 夹爪开合：物理自适应，无需训练特定尺寸

控制模式 `pd_joint_target_delta_pos` 的物理特性决定夹爪天然自适应：
```
策略输出"关闭夹爪"→ PD 控制器持续施力 → 碰到物体表面 → 接触力平衡 → 自然停止
2cm 橡皮：夹爪关得更紧 | 10cm 鼠标：夹爪停在较开位置 | 5cm cube：居中
```
真实机器人同理（电机电流限制防压碎）。**与训练时的物品尺寸无关。**

#### 实际限制

| 问题 | 性质 | 说明 |
|------|------|------|
| 夹爪开合量 | ✅ 自适应 | 物理接触决定，与训练尺寸无关 |
| 物品宽度 > SO-101 最大开口（约 8-10cm） | ❌ 硬件限制 | 上层规划器应排除过大物品 |
| 物品极小（2cm 橡皮） | ⚠️ 精度问题 | 夹爪能闭合，但定位偏差 5mm 就可能抓空 |
| 物品较高（直立杯子） | ⚠️ 轨迹问题 | 检测层需给出正确抓取高度 |

#### 检测层的责任

检测系统提供的应是**抓取点**，而不只是物体几何中心：
- 扁平物品（橡皮、便利贴）：物体中心高度
- 中等高度（笔、U盘）：物体中心高度
- 较高物品（直立笔、杯子）：物体中心高度的一半处

这个逻辑放在检测层，RL 策略无需修改。

#### 检测方案：深度点云分割（物品形状无关）

具体流程：桌面平面检测 → 聚类 → 每个聚类中心 = 物品位置 → 配合 RGB 做物品身份识别（由上层负责）。

---

## 本次需要新建/修改的文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `envs/place_position.py` | `SO101PlaceCubePos-v1`，继承 `DefaultCameraEnv`（**不是** `Place`） |
| 新建 | `envs/stack_position.py` | `SO101StackCubePos-v1`，继承 `DefaultCameraEnv` |
| 新建 | `train_squint_place_pos.py` | 训练入口（薄包装） |
| 新建 | `train_squint_stack_pos.py` | 训练入口（薄包装） |
| 新建 | `deploy_place_pos.py` | 部署脚本 |
| 新建 | `deploy_stack_pos.py` | 部署脚本 |
| 新建 | `primitive_executor.py` | 统一调用接口类，含 `home()` 脚本插值 |
| 修改 | `envs/__init__.py` | 注册新环境 |
| 修改 | `deploy_utils/position_provider.py` | 重设计：`ObjectPositionProvider` + `TargetPositionProvider` 两个独立 Protocol；新增 `DirectTargetPositionProvider` |

**参考实现：**
- `envs/lift_position.py` → PlacePos/StackPos 的 7-dim obs 注入模式、noise/dropout 写法
- `deploy_pos.py` → `deploy_place_pos` / `deploy_stack_pos` 整体结构
- `envs/stack.py` → `stack_position` 的奖励函数和成功条件
- `envs/place.py`（仅参考）→ `_load_agent`、`_get_obs_agent`、item spawn 逻辑

---

## PlaceCubePos 实现速查

### _initialize_episode 关键逻辑

```python
def _initialize_episode(self, env_idx, options):
    super()._initialize_episode(env_idx, options)
    b = len(env_idx)

    # 1. 机器人初始化为 rest_qpos + 夹爪半闭合
    self.agent.robot.set_qpos(self.rest_qpos + noise)

    # 2. 随机采样 target_pos（per-episode）
    sampler = UniformPlacementSampler(bounds=region, ...)
    target_xy = sampler.sample(min_dist_from_item, 100)
    self.target_pos[env_idx, :2] = spawn_center + target_xy
    self.target_pos[env_idx, 2] = table_height + item_half_size

    # 3. 持物初始化：item spawn 在 TCP 附近（加 grasp_offset_noise）
    tcp_pos = self.agent.tcp_pos[env_idx]
    grasp_noise = clamp(randn * grasp_offset_noise_std, clip)
    item_init_pos = tcp_pos + grasp_noise
    self.item.set_pose(Pose.create_from_pq(item_init_pos, ...))

    # 4. Weld 固定前 warmup_steps 步（warmup_steps=5）
    self._attach_item_to_tcp(env_idx)   # SAPIEN create_drive / fixed joint
    self._warmup_steps_remaining[env_idx] = self.warmup_steps
```

### _get_obs_extra 关键逻辑

```python
def _get_obs_extra(self, info):
    obs = {}
    if self.obs_mode_struct.state:
        target = self.target_pos           # (B, 3)，确定性，不加噪声
        tcp_to_target = target - self.agent.tcp_pos
        obs.update(
            qvel=..., tcp_pos=..., dist_to_rest_qpos=...,
            target_pos=target,             # 3 dims
            tcp_to_target=tcp_to_target,   # 3 dims（实时更新）
            is_item_grasped=info["is_item_grasped"],  # 1 dim
        )
    return obs
```

### evaluate 关键逻辑

```python
def evaluate(self):
    item_pos = self.item.pose.p
    dist_to_target = norm(item_pos - self.target_pos, dim=1)
    is_at_target = dist_to_target < 0.03          # 3cm 阈值
    is_released = ~self.agent.is_touching(self.item)
    is_robot_static = self.agent.is_static()
    success = is_at_target & is_released & is_robot_static
    ...
```

---

## 训练顺序与验证方式

```
Step 1: LiftCubePos 训练完成（进行中）
         验证：eval/success_at_end > 0.7
         ↓
Step 2: 实现 PlaceCubePos 环境（envs/place_position.py）
         重点：持物初始化 Weld 方案 + grasp_offset_noise + target_pos obs
         ↓
Step 3: 训练 PlaceCubePos
         python train_squint_place_pos.py --num_envs=512 --track
         验证：eval/success_at_end > 0.6
         ↓
Step 4: 训练 StackCubePos（按需，先用 PlacePos 拆解）
         python train_squint_stack_pos.py --num_envs=512 --track
         验证：eval/success_at_end > 0.5（Stack 最难）
         ↓
Step 5: 实现 primitive_executor.py（含 home() 插值逻辑）
         实现 DirectTargetPositionProvider
         实现 position_feature_spec + inject_features_by_spec
         启动时 obs_schema_hash 一致性校验
         ↓
Step 6: 全链路仿真测试（pick → place → home → pick 循环）
         若 place 成功率 < 50% for stack 目标 → 触发 StackPos 训练
         若 place 成功率低于预期 → 调大 grasp_offset_noise_std 重训
         ↓
Step 7: 真实机器人联调（各策略 deploy + SimGT/Direct provider）
         ↓
Step 8（后续）: 实现 RealDetectorPositionProvider（深度点云分割）
Step 9（后续）: 封装 FastAPI，供队友跨进程调用
```

---

## 稍后实现：真实位置检测

**方案：深度点云分割**
1. 桌面相机拍 RGB-D 帧
2. 深度平面检测（RANSAC）提取桌面平面
3. 过滤出平面以上的点 → 聚类（DBSCAN 或欧氏距离聚类）
4. 每个聚类重心 → 物品 3D 位置（机器人基座坐标系）
5. 失败重试（点数过少、聚类异常）→ 最多 `retry_steps` 次

**需要的标定数据（存于 `deploy_utils/`）：**
- 桌面相机内参（`cv2.calibrateCamera`）
- 桌面相机 → 机器人基座外参（手眼标定）
