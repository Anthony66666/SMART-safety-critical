# 遮挡感知的 nuPlan 安全关键闭环 Benchmark

## Context

**要解决的问题.** 所有闭环 planning 评测都给 planner 发送周围 agent 的**真值框**（nuPlan 的 `DetectionsTracks`）。现实中不存在这种感知——停靠车辆、卡车、巴士会挡住视线，而真实事故报告里视线遮挡是高频成因。这个假设从来没被动过，因此"planner 在遮挡下会怎样"这个问题目前**没有任何数据**。

**为什么现在做.** 邻域刚被两篇工作点亮，且都没有占这块地：

| 工作 | 做了什么 | 留下的空位 |
|---|---|---|
| 2510.14677（2025.10） | 把 nuPlan 的 IDM 换成 SMART，证明 agent 模型会改变 planner 排名 | 只换 agent 模型，planner 仍全可观 |
| CRAFT（2606.31844） | 修**仿真器**的可观性失配（训练时局部、部署时全局） | 对象是交通模型，不是 planner |
| SaFeR | 在 NTP 定义的真实行为空间内搜最危险且**可避撞**的 token | criticality 与 feasibility 均定义在**全状态**上 |
| nuPlan-R（2511.10403） | 用扩散式反应 agent 替换 nuPlan 的 IDM，重实现一批 planner，加 Success Rate / All-Core Pass Rate 两个指标 | 同样只换 agent 模型，planner 仍全可观 |
| interPlan（2404.07569） | 手工编辑 nuPlan 场景造 80 个边缘案例 | 手工、无遮挡、规模小 |

空位是：**planner 自身的可观性假设**。三个独立团队都在问"nuPlan 闭环哪里不对"，**没有一个碰 planner 的可观性**——空位是真的，但邻域很热，不会一直空着。而且它同时打穿 SaFeR 的两个支柱——遮挡下 criticality 看不见（威胁藏在遮挡物后，全状态 TTC 正常），feasibility 也变了（全知 ego 可避的，盲区 ego 可能不可避）。

**目标产出.** 一个遮挡感知的 nuPlan 闭环 benchmark + 场景生成方法，主张是：现有 planner 的分数建立在一个错误的感知假设上，去掉它之后排名会变，而这个新的能力维度可以被生成、被测量、被训练。

**主线是遮挡**，贯穿全部四个阶段：LLM 从事故报告抽**视线构型** → 生成遮挡诱发的危险场景 → 遮挡观测下评测 planner → 用生成场景训练。不是三个模块拼贴。

---

## 分支现状（已核实）

`nuplan-safety-bench` = 干净的 main，已推送到 `origin`，`checkpoints/` 已忽略。

- `smart/safety/` **不存在**——M0–M4 的 42 个文件 / 3596 行（含 15 个测试文件）只在 `safety-critical` 分支上
- `beam_size` 仍硬编码：[agent_decoder.py:107](smart/modules/agent_decoder.py:107)
- 无似然记账、无 ego 注入、无倾斜采样
- `av_mask = data["agent"]["av_index"]` 已定义但从未使用（[agent_decoder.py:379](smart/modules/agent_decoder.py:379)）
- 磁盘：`/dev/sdd` 1007G，剩 **107G**
- 现有 checkpoint：`checkpoints/epoch=31.ckpt`（WOMD 全量 487k 训练）
- 词表：`smart/tokens/cluster_frame_5_2048.pkl`，2048 token，shift=5 @ 10Hz

**S0 和 S1 不需要 `safety-critical` 的任何代码**，保持干净起步。S2 开工时再选择性移植。

---

## 已锁定的决策

| 项 | 决定 | 理由 |
|---|---|---|
| 旧代码 | S2 时按需移植 tilting/likelihood/objectives/judge/tokenization + 其测试 | S0/S1 用不到；避免在 WOMD 专用代码里工作 |
| 遮挡保真度 | 2D 光线遮挡，遮挡物 = 车辆多边形 | 覆盖真实事故主因，确定性，**零自由参数**，nuPlan 一定支持 |
| 目标记忆 | benchmark 提供统一跟踪缓冲 | 无缓冲则所有 planner 因接口冲击而崩，测不出区分度 |
| SMART 迁移 | 零样本 → 微调 → 重训，**三档递进** | README 的 SMART-zeroshot 0.7210 说明架构扛跨域；别一上来付最贵的档 |
| ego 历史 | 生成时**不倾斜**（β=∞），只倾斜对抗者 | 否则是让 planner 为它没造成的烂摊子背锅 |

---

## 实测结论

### 遮挡是中远场现象，记忆只挽回约 20%（2026-08-28）

11 个 WOMD demo 场景全时长，2D 地面视线模型，遮挡物 = 车辆多边形，跟踪缓冲 3s 等速外推。
复现：`PYTHONPATH=. python scripts/occlusion_stats.py`

| 距离带 (m) | hidden | partial | visible | hidden% | 缓冲挽回 | unknown | **unknown%** |
|---|---|---|---|---|---|---|---|
| 0–10 | 185 | 651 | 2676 | 5.3% | 93 | 92 | **2.6%** |
| 10–20 | 2057 | 2013 | 1898 | 34.5% | 480 | 1577 | **26.4%** |
| 20–30 | 2852 | 1324 | 908 | 56.1% | 574 | 2278 | **44.8%** |
| 30–40 | 2624 | 1432 | 940 | 52.5% | 403 | 2221 | **44.5%** |
| 40–55 | 4339 | 1542 | 1131 | 61.9% | 886 | 3453 | **49.2%** |
| 全部 | 12057 | 6962 | 7553 | 45.4% | 2436 | 9621 | **36.2%** |

**`unknown` 才是 benchmark 真正立足的数**——ego 对这些 agent 没有任何信念，不是"记得但看不见"。

**记忆只挽回被遮挡目标的约 20%**（2436/12057）。这条很关键：benchmark **不会被"加个跟踪器"破解**，剩下 36.2% 是硬难度。近场 unknown 仅 2.6%，说明模型没有荒谬地悲观。

**这个形状正是 benchmark 需要的。** 近场 94% 可见说明模型没有荒谬地悲观——planner 仍看得见紧邻的东西。而 10–30m 带 35–57% 被遮挡，恰好是**预判**所依赖的距离：一辆 25m 外正在接近的车，看不见就只剩紧急制动。聚合的 45.8% 被远场稀释（40–55m 带 agent 最多、遮挡率最高，但对避撞最不重要），**按与规划的相关性加权后效应更温和，却集中在预判带**。

### 遮挡模型无法用数据集经验校准（2026-08-28）

原本设想用 WOMD 的 `valid_mask` 当真实感知丢失的代理来标定模型，**此路不通**：轨迹内部空洞只占 0.62%（212 / 34139 跟踪步），数据集已把感知间隙插值填平。

两个后果：
1. 遮挡模型只能靠几何论证，没有数据集内的经验校准可用。论文中必须如实说明这是**建模选择**而非拟合结果。
2. 反过来强化立论：**全可观假设深到连数据本身都不记录"AV 当时到底看见了什么"**。

⚠️ 已知的保守偏置：真实 lidar 在约 2m 车顶高度，能越过部分车顶；感知还会跨短暂遮挡做时序跟踪。2D 地面模型比真实传感器**更悲观**。S1.2 的跟踪缓冲正是补这一块，其收益应当被单独量化。

---

## S0：nuPlan 接口 + 伪影基线

### S0.1 存储与数据（动手前的前置）
确认 A800 服务器可用存储；选定子集（maps + 足量 train logs + val14 评测集，**不需要全量**）；确认预处理缓存建好后可删原始 db。107G 本地装不下 nuPlan，这是硬约束。

### S0.2 nuPlan → SMART schema 转换器
**新建** `smart/nuplan/converter.py`
- 参照 [smart/datasets/preprocess.py](smart/datasets/preprocess.py) 与 [smart/preprocess/preprocess.py](smart/preprocess/preprocess.py) 的 WOMD 预处理产出同构的 `HeteroData`
- 20Hz → 10Hz 下采样；地图 polyline 需匹配 `smart/tokens/map_traj_token5.pkl` 的 token 化方式
- **复用** [smart/datasets/preprocess.py:195](smart/datasets/preprocess.py:195) 的 `tokenize_agent` 投影逻辑，不要另写

### S0.3 三档迁移测试
用 `checkpoints/epoch=31.ckpt` 零样本跑转换后的 nuPlan 场景。
**验收**：运动学合理性（最大加速度中位数，参照 M4 教训的 3 m/s² 量级）、越界率、碰撞率。达标则**跳过重训与重聚词表**；不达标才升级到微调，最后才考虑重训。

### S0.4 反应式 agent 接入 `AbstractObservation`
⚠️ **这一格已被两篇占了**（2510.14677 用 SMART，nuPlan-R 用扩散 agent）。**不作为贡献主张**，只作为基础设施采纳。

**首选路径：等/跟踪 nuPlan-R 开源**（论文明写 "We will open-source"）。若释出，直接拿到 `AbstractObservation` 接线、interaction-aware agent selection（算力效率）、以及**已重实现的规则型/学习型/混合型 planner 集合**——等于 S0.4 全部 + S1.4 的大半。

**回退路径**：自建 `smart/nuplan/observation.py`，SMART 接入 `AbstractObservation`。

无论哪条路径，**批量化模型服务必须在这一步做完**：1000 场景 × N planner × 2 观测条件 × 前后两轮 = 数千次完整 benchmark 跑，不能每个仿真进程各自加载模型。

**由此产生的定位选择（需决策）**：SMART 的不可替代性其实只在 S2——精确 `log p` 与显式 token 几何是扩散模型给不了的。S0/S1 用任何像样的反应式 agent 都行。若 nuPlan-R 释出，可以让**它做仿真基座、SMART 只做场景生成**：S2 产出场景规格（初始条件 + 对抗者轨迹 + 关键帧），由 nuPlan-R 的 agent 跑闭环。这样成本更低，且定位从"和它们竞争"变成"在社区最新标准上加一根新轴"。代价是两个模型的一致性需要论证。

### S0.5 伪影基线（硬门）
CRAFT 报告在未经处理的 AR 交通模型上，其修正能降碰撞 31.2%、降违规 33.2%——反读即**约 30% 的碰撞与违规是模型自身伪影**。若不先测掉，S1 分不清"planner 失败"和"背景车发疯"。

**测法**：纯常规场景（无对抗、无遮挡），ego 设为日志回放，测量纯由背景 agent 引起的碰撞率与违规率，并与 IDM agents 基线对照。
先查 nuPlan-R 的 **Behavior Plausibility** 指标——它可能已给出现成的定义与基线数字，能省掉自己定标准。
**验收**：伪影基线被量化并记录；它是后续所有数字的本底噪声，必须在论文中报出。**不过此门不得进入 S1。**

---

## S1：遮挡层 + planner 评测

### S1.1 可见性计算
**新建** `smart/occlusion/visibility.py`
- 从 ego 传感器原点向每个 agent 包围盒角点做 2D 光线检测，被其他车辆多边形遮挡则判为不可见
- **复用** [agent_decoder.py:20](smart/modules/agent_decoder.py:20) 的 `cal_polygon_contour` 和 [smart/utils/geometry.py](smart/utils/geometry.py)
- 确定性、无自由参数
- 需确认 nuPlan 地图是否含建筑几何；**大概率没有**，则明确把范围限定为车-车遮挡并在论文中声明

### S1.2 遮挡观测包装器 + 跟踪缓冲
**新建** `smart/nuplan/occluded_observation.py`
- 包装 S0.4 的 observation，按可见性过滤 `DetectionsTracks`
- **统一跟踪缓冲**：最后一次观测的位姿/速度 + 已失观时长，对每个 planner 暴露完全相同的接口
- 这是 benchmark 强加的感知假设，论文中必须明说

### S1.3 逐 agent 可见性掩码（CRAFT 假设，标记为假设）
CRAFT 的修法是输出端补丁；**源头修法**是恢复模型训练时的可观性条件。插入点：[agent_decoder.py:268](smart/modules/agent_decoder.py:268) `build_interaction_edge` 在 `radius_graph` 之后按 (source, target) 对过滤 `edge_index_a2a` 的列——小手术，不动架构。

⚠️ **这不是已知的解**。CRAFT 描述的失配是场景范围/轨迹完整性问题，不是逐 agent 视线遮挡；方向对但不等价。
**实验**：逐 agent 掩码 / CRAFT 式输出重加权 / 什么都不做，三者对 S0.5 伪影基线的改善量。不论结果如何都是有效结果。

### S1.4 planner suite + 评测
- PDM-Closed（nuPlan 挑战赛冠军，规则型）、IDM、一个学习型、条件允许再加一个端到端/VLM
- **主指标是 gap，不是绝对分**：现有 planner 全部在全可观假设下调参，去掉信息后会集体掉分；绝对分测的是"抽掉信息有多疼"，gap 才测遮挡鲁棒性
- 分析排名是否翻转
- ⚠️ SMART 系 planner 与背景 agent 同源，有不公平优势，需单列或排除

### S1.5 场景三带分级
有了全可观与遮挡下**两个**可行域，其差集自然分带：

| 带 | 含义 | 处理 |
|---|---|---|
| 全盲也能避 | 太简单 | sanity check |
| **全可观可避、遮挡下不可避** | 必须因看不见而主动减速才能避 | **benchmark 核心** |
| 全信息下也不可避 | 不公平 | 剔除 |

中间带测的是防御性驾驶与主动信息获取——遮挡感知 planning 文献二十年在造的能力，但从无 benchmark。且"必撞场景不公平"问题由此**被定义解决**，不靠启发式过滤。

---

## S2：LLM 规格 + 关键帧条件生成

### S2.0 从 safety-critical 移植
`tilting.py` / `likelihood.py` / `objectives.py` / `tokenization.py` / `judge.py` / `scoring.py` / `splits.py` + 对应测试；以及 `agent_decoder.py` 的可配 `beam_size`、`log_p`/`log_q` 记账、`forced_tokens`、`temperature`。
⚠️ 一并带上那条工程约束：`sample_pt_pred` 的地图点掩码是训练增强，**必须设种子**，否则似然有 9e-2 量级的静默偏差。

### S2.1 开篇实验：覆盖缺口（不依赖本方法完成即可做）
跑全可观的危险倾斜生成（SaFeR 式），展示它**产不出**真实事故里哪些遮挡成因类别。全状态危险度对藏在遮挡物后的威胁失明——这是 S2 的 Fig. 1，一击命中。

### S2.2 LLM + RAG：事故叙述 → 视线构型规格
- 语料：NMVCCS / CIREN / GIDAS 的自然语言事故叙述
- 输出：参数化关键帧（相对位姿、速度、所需路口拓扑、**所需遮挡物位置**）
- 与 LD-Scene 等的区分点：那些工作的 LLM 产出**行为**规格、服务于全可观世界；这里产出**视线几何**，现有工作一个都产不出来，因为其下游不建模遮挡
- **LLM 输出必须验证，不得信任**：每份规格过几何/物理可行性检查，报出通过率
- **对照组**：手写 ~10 类碰撞构型 taxonomy。实验主张是"LLM 从真实报告生成的规格，覆盖度更高 / 分布更接近真实事故频次"——而不是"我们用了 LLM"

### S2.3 关键帧条件采样
- **历史 = 向关键帧约束的 twisted SMC**，不是反向模型。Adv-BMT 式反向模型采出的历史在前向真实分布下似然未知，与本项目"真实性必须可测"的立论自相矛盾
- **未来 = 前向 rollout**，定义的是**世界**（其他 agent 怎么动），不是 ego 的答案
- **专家 = 特权规划器**（已知其他 agent 完整未来的采样式 MPC），产出即**可解性证书**；找不到解的场景标记为不可避、剔除
- ego 历史不倾斜（β=∞），只倾斜对抗者
- **接管时刻是免费的难度轴**：早接管测预见能力，晚接管测紧急响应。同一关键帧扫接管时刻得到曲线而非单点
- 所有倾斜必须带 `tilt_topk`（M4 教训：全支撑采样产生 59 m/s² 量级的抖动运动，连不倾斜都抖）
- 诚实补丁：特权 ego 偏离后其他 agent 未来严格说不再一致；做 1–2 轮迭代重规划，并在论文中写明

### S2.4 背景车插入
**复用** [mh0797/interPlan](https://github.com/mh0797/interPlan)——SMART 的 agent 集合在 t=0 固定，加背景车是结构上做不到的；interPlan 已实现 nuPlan 场景的 agent/障碍物插入与导航目标修改。背景车密度控制由此获得。

---

## S3：有用性验证

- 用生成场景 A 组微调现有 planner，在**留出的**危险场景 B 组上测试
- **两组必须在 log/地图层级不相交**，不是场景 ID 层级——nuPlan 的 log 反复经过同一批路口，共享路口则提升是记忆而非泛化
- **强制对照：回 Val14 重测常规性能**。用危险场景微调会造成灾难性遗忘（危险分涨、正常驾驶变畏缩）。没有这个对照，"我们在自己的 benchmark 上提升了"不成立
- RL 后训练**不做**（SMART 当环境时 RL 会学会"硬挤对方就会让"，且 1000 场景对 RL 远远不够）

---

## Verification

每阶段的可断言验收，不靠肉眼：

| 阶段 | 验收 |
|---|---|
| S0.2 转换器 | nuPlan 日志回放经转换后能被 SMART 前向打分，不 NaN；地图 token 覆盖率达标 |
| S0.3 迁移 | 零样本最大加速度中位数在 3 m/s² 量级；越界率、碰撞率与 WOMD 上同量级 |
| S0.4 接口 | 日志回放 observation 复现原始 nuPlan 场景（逐帧断言） |
| **S0.5 伪影基线** | **纯常规、ego 回放下的背景车碰撞率与违规率被量化并记录；与 IDM 基线对照** |
| S1.1 可见性 | 与独立实现的射线-多边形求交在随机样本上一致；确定性（同输入同输出） |
| S1.2 缓冲 | 全可观模式下缓冲不改变任何 planner 的输出（退化断言） |
| S1.4 评测 | 全可观下的 CLS-R 复现文献区间（PDM-Closed 尤其） |
| S1.5 分级 | 三带互斥且穷尽；Band C 场景在特权规划器下确实无解 |
| S2.0 移植 | `safety-critical` 的 15 个测试文件全部通过 |
| S2.2 规格 | 可行性通过率被报出；与手写 taxonomy 的覆盖度对比有数字 |
| S2.3 生成 | β→∞ 时退化为 `p`（分布距离断言）；生成场景的关键帧确实落在指定构型集 A 内 |
| S3 | 危险分提升 **且** Val14 常规分不塌，两个条件同时成立才算数 |

---

## 风险

1. **S0.5 伪影基线过不了.** 若背景 agent 的伪影碰撞率压不下来，整个 benchmark 的信噪比不成立。缓解：S1.3 的三种修法；最坏情况退回 IDM agents（牺牲真实性换干净）。
2. **所有 planner 在遮挡下集体崩溃，benchmark 失去区分度.** 缓解：gap 作为主指标 + 统一跟踪缓冲 + 三带分级剔除不公平场景。
3. **nuPlan 地图无建筑几何**，遮挡只能做车-车。缓解：真实事故主因本就是车辆遮挡；明确声明范围。
4. ~~nuPlan-R 未核查~~ **已核查（2026-08-28）**：扩散式反应 agent 换 IDM，**不做遮挡、不做安全关键生成**。S1 的遮挡轴不受影响；S0.4 被占，已降级为基础设施采纳。新风险：**邻域三篇密集落地（2025.10、2025.11、2026.06），遮挡这一格空着但不会久空**。缓解：S2.1 的覆盖缺口实验不依赖完整方法即可做，应尽早拿到。
5. **存储.** 本地仅剩 107G，nuPlan 装不下，全程依赖服务器存储。
6. **吞吐量.** nuPlan 闭环本就是单场景 CPU 重的框架，每步塞进 transformer 后需批量化服务设计。这是工程问题但会吃掉数周。
7. **LLM 部分被判为"撒料".** 缓解：手写 taxonomy 作为对照组，主张落在覆盖度与分布匹配的数字上，而非"我们用了 LLM"。

---

## 建议的起手

S0.1（确认服务器存储 + 选定 nuPlan 子集）与 S0.2/S0.3 第一档（转换器 + 零样本迁移测试）。零样本这一档的结论会反过来影响后续所有阶段的成本估计——如果 `epoch=31.ckpt` 直接可用，重训与重聚词表的代价整个消失。

### nuPlan 实测：密集市区里记忆几乎无效（2026-08-29）

服务器 `smart` 环境**没装 nuplan-devkit**，且链路仅约 280 KB/s。但 nuPlan log 就是 SQLite，
`lidar_box` + `ego_pose` 已含全部几何 —— 遮挡几何的开发完全不需要 devkit（仿真才需要）。
`scripts/extract_nuplan_fixture.py` 直读 SQLite，输出与 WOMD 同构，
因此 visibility / stats / renderer 三个脚本一行不改就能跑 nuPlan。

两个 nuPlan 特有坑已在抽取层处理：`ego_pose` 存**后轴**（Pacifica 偏 1.461 m，会让所有视线起点系统性偏移）；
log 为 20 Hz，默认降采样到 WOMD 的 10 Hz。

**选 log 不能按文件大小**：最小的 log = 最空的场景（1.6 boxes/帧，测遮挡等于没测）。
服务器端扫密度找到 Las Vegas Strip 达 234 boxes/帧，差两个数量级。

目标类型需过滤：nuPlan 把 0.4 m 路边碎物记为 `generic_object`（fixture 里 201 个 agent 中占 70 个），
挡住一块碎石不构成对 planner 的判断。

|                | WOMD demo | nuPlan Vegas |
|----------------|-----------|--------------|
| hidden         | 45.4%     | **59.8%**    |
| unknown        | 36.2%     | **58.4%**    |
| 记忆挽回       | 20.2%     | **2.3%**     |

**⚠️ 关于记忆那一行的更正（2026-08-31）。** 曾据此写下"密集市区里记忆几乎无效"，
**这是过度推广**：2.3% 是那一条 log / 那个 9.1 秒窗口的性质，不是密集场景的普遍规律。
在 nuPlan mini 另一条同样密集的 Vegas log 上，两条独立流水线一致给出 **35–42% 的挽回率**
（devkit 41.9%，SQLite 34.8%，同 log、同 55m、同只算真实 agent）。
真实结论是：**记忆的收益方差极大，取决于遮挡是持续的还是短暂的**，
单场景数字不能当结论用——正式数字必须跨足够多场景聚合。

按类型看，**车辆是最容易被遮挡的（70.7%），高于行人（51.3%）**——车在路面上被同类挡住，
行人在路侧视线更通。方向合理，是模型正确性的一个佐证。

渲染验证而非假设：ego 两侧邻车的阴影扇形覆盖整个西侧，隔一条车道的车确实返回 hidden。

**阻塞**：服务器装 nuplan-devkit 仍是 S0.2 前置。nuPlan 地图 638 MB（las_vegas），按当前链路约 38 分钟且易断。

### S0.3 零样本迁移：通过，且生成比日志更平滑（2026-08-31）

`epoch=31.ckpt`（纯 WOMD 训练）直接跑 12 个转换后的 Las Vegas 场景，**不微调、不重训、不重聚词表**。

| | 预测 | nuPlan 日志 |
|---|---|---|
| 速度 p50 / p95 | 0.6 / 10.2 m/s | 0.6 / 12.4 m/s |
| **加速度 p50** | **0.15** | 18.75 m/s² |
| **加速度 p95** | **4.68** | 25.77 m/s² |
| 加速度 p99 | 11.48 | 50.00 m/s² |
| 碰撞率（粗略圆形判据） | 31.4% | 19.4% |
| 车道中心线距离 p50 | 0.7 m | 3.0 m |

**日志的加速度不是物理基线。** 决定性证据：9.1 秒内位移不足 1 米的**停放**车辆，加速度 p95 达 25 m/s²。停着的车不可能加速——那是 nuPlan 感知框的位置抖动被二次微分放大。因此"生成轨迹比日志更平滑"是真的，而不是指标口径问题。

**结论：跳过微调与重训**，按方案的三档递进直接停在零样本档。可视化（`nuplan_zeroshot.png`）显示红线（生成）处处贴合蓝线（日志），路口转弯几何正确，无退化堆叠。

⚠️ 保留项：碰撞率 31.4% vs 19.4% 仍偏高。但判据是"半车长圆形"粗略近似，对并排长车过报严重；两侧同判据，差值可比、绝对值不可信。真正的框重叠判据应在 S0.5 伪影基线时做。

### 转换器的三个静默 bug（2026-08-31）

都不报错，都会毁掉结果，按发现顺序：

1. **UTM 存 float32**。北向坐标约 4×10⁶，float32 ulp = **0.25 m**，车道与车辆位置全被量化到四分之一米。WOMD 是几千量级局部坐标，从没暴露过。修法：float64 下以自车重定心再转 float32。
2. **无效槽位不为零**。`position` 零初始化、只填有效帧，然后对**全部**条目减原点——无效槽变成 `-origin`。而 SMART 预处理用 `position[:, current_step, 0] != 0` 判断"无效但有位置"并据此插值，于是垃圾灌进 token 化并扩散到有效 agent。修法：减完原点后把无效槽显式归零。
3. **锥桶/护栏被当成 agent**。映射到 type 3（background），但 SMART 只有 veh/ped/cyc 三个 token embedding，`tokenize_agent` 的掩码只认 0/1/2——**type 3 被静默跳过**，token 保持默认值，解码成压在车道中心线上的退化轨迹。Vegas 场景里这类物体比真实 agent 还多（34529 vs 14301 样本），直接主导碰撞率。修法：非 agent 类型不进 agent 集合；它们在遮挡层仍然有效，那层读原始 nuPlan 对象。

修复前后：碰撞率 66.7% → 31.4%，加速度 p99 25 → 11.5。

### 改用官方代码计算指标（2026-08-31，用户纠正）

原先我在张量上重写了 nuPlan 的碰撞判定与责任分类。**这是把问题复杂化了**：benchmark 的说服力来自
"官方仿真 + 官方 planner + 官方指标，唯一变量是观测"。自造指标一旦与官方数字有出入，对比就无法引用。
已删除 `smart/metrics/`（`collision.py` / `at_fault.py`）及 `scripts/artefact_baseline.py`。

保留的自有代码只有真正属于本工作的部分：`smart/occlusion/`（遮挡几何 + 记忆）与
`smart/nuplan/occluded_observation.py`（观测包装器）。

`scripts/run_benchmark.py` 走完整官方链路：`SimulationSetup` + `SimulationRunner` +
`PerfectTrackingController` + 官方 `EgoAtFaultCollisionStatistics` / `DrivableAreaCompliance` /
`EgoProgressAlongExpertRoute` / `SpeedLimitCompliance`。

### 场景必须按官方 scenario_tag 选（2026-08-31）

随意挑起始帧会挑到静止场景——mini 里 `stationary` 标签 62375 帧，远超
`traversing_intersection` 的 20213。我最初选中的场景里 **expert 全程只移动 3.8 米**，
任何感知假设都不可能改变结果。

`smart/nuplan/scenarios.py` 按官方 tag 选场景（已发表 nuPlan 结果也是按类型报的）。
遮挡相关类型：`traversing_intersection`、`traversing_traffic_light_intersection`、
`near_pedestrian_on_crosswalk`、`high_magnitude_speed`、`following_lane_with_lead`。
改选后 expert 移动 107–196 米。

### 关键不变量：遮挡观测必须是全观测的子集（2026-08-31）

跟踪缓冲无法区分"被遮挡"与"底层观测不再报告"（目标驶出检测范围）。不加约束时，
**planner 在遮挡条件下收到的目标（215.7/步）比全观测条件（180.0/步）还多**——记忆把已消失的目标
当幽灵保留了 3 秒。这会让遮挡条件反而信息更多，整个实验倒转。

修法：输出只保留底层观测仍在报告的目标。记忆只桥接**遮挡**间隙，不桥接**检测范围**间隙。
修复后：180.0 提供 → 167.9 交付（149.6 当前可见 + 18.3 记忆），**扣留 6.7%**，越界目标 0。
已加测试固化。

### ⚠️ IDM planner 对遮挡完全不敏感（2026-08-31）

4 个 `traversing_intersection` + `near_pedestrian_on_crosswalk` 场景，全部官方指标 **delta 恒为 0**。
进一步验证：**两个条件下 ego 轨迹最大偏差 0.000000 米**，299–300 步逐点相同。

原因是结构性的，不是 bug：`IDMPlanner` 只在自己规划路径上取最近障碍物作为 lead agent，
而**最近前车恰恰是全场最不可能被遮挡的目标**——你和它之间没有别的东西。它从不看横向交通。

**对 S1 的直接后果**：planner suite 的选择不是可选项而是前提。IDM 无论遮挡多严重都会给出零效应，
不能作为 benchmark 的主力。需要使用更广场景信息的 planner：PDM-Closed、学习型、或端到端。
这也印证了方案 S1.4 里"主指标是 gap"的判断——但前提是 planner 本身对信息量敏感。

### planner suite 的答案：Flow Planner（2026-08-31）

IDM 的零效应是结构性的，所以 benchmark 需要会用全场信息的 planner。选定
**Flow Planner**（NeurIPS 2025，Val14 NR **90.43**，源码 `~/SimAgentJEPA/external/Flow-Planner`，
权重 HuggingFace `ttwhy/flow-planner`）。它从 `history_buffer.observation_buffer` 取
32 个邻居 agent 和静态物体——**正是遮挡包装器替换的地方，不会绕过**。地图仍走 `map_api`，
这是对的：地图是先验知识，不该被遮挡。

三个发布物的坑，都不是猜的：

1. HF 上的 `model_config.yaml` 是从训练配置树里摘出来的，仍插值到没带过来的分支
   （`data.dataset.train.*`、`train.epoch`）。值取自仓库自己的 `nuplan_data.yaml`，
   合并到副本，原文件保持不动。
2. checkpoint 是**已导出的扁平 EMA 权重**（338 个 `module.` 前缀张量），不是带
   `ema_state_dict` 的训练检查点 → `enable_ema=false`。hydra 里要用 `+` 追加，
   因为他们的 yaml 里没这个键。
3. devkit 的 `hydra-core==1.1.0rc1` 依赖已从 PyPI 撤下的 `omegaconf==2.1.0.rc1`，
   照装必失败。Flow Planner 要 1.3.2，丢掉旧 pin 即可。

首批结果（6 个 `traversing_intersection`）：**ego 轨迹平均偏离 1.413 m，最大 2.109 m**
（对照 IDM 的 0.000000 m），超速违规 4.5 → 10.0。两个条件均无碰撞，**n=6 不构成结论**。

### 遮挡默认逐帧，不带记忆（2026-08-31，用户决定）

用户判断：Flow Planner 训练时没考虑遮挡，自然没有记忆功能，测试应逐帧判定。
查证后更彻底——`filter_agents_tensor(reverse=True)` 只保留**当前帧**存在的 agent，
`pad_agent_states` 注释明写"只在过去出现的 agent 会被丢弃"。**被遮挡目标连同全部历史消失。**

所以外加跟踪缓冲不是补齐感知，而是凭空赋予 planner 一个它不具备的能力，且是全套里
唯一的自由参数。实测缓冲把扣留率从 **16.9% 压到 6.7%**，抹掉约六成效应。

默认 `memory_horizon: 0.0`，带记忆版留作显式消融 `occluded_box_observation_with_memory`。
**部分遮挡仍按可见处理**——5 个采样点（4 角 + 质心）有 1 个通视即算看见，且透传真实完整框。
从局部观测补全完整框是感知的职责，不该算作遮挡。

### 可视化的一个教训：单向验证不算验证（2026-08-31）

动图里出现"阴影中的车却是绿色"。第一反应是遮挡算错，实际不是。完整账目：

| | 数量 | 解释 |
|---|---|---|
| 被给出但完全被挡 | 561 | |
| ├ 记忆供给 | 520 | 跟踪缓冲，设计如此（这批 log 用旧默认 3s） |
| └ ego 滞后一帧 | 41 | 已知；按包装器实际用的位姿算是 **0** |
| 无法解释 | **0** | |

**教训**：之前只验证了"被扣留的是否真的可见"（零违例），从没验证反方向
"被给出的是否真的不可见"——问题全在那一侧。**不变量必须双向查。**

另一个被推翻的假设：我先归因于"两次仿真 ego 分叉导致视点错配"，但那两次是 IDM、
轨迹逐点相同，换成单次自洽算法后数字一模一样，假设当场作废。

### val14 全量评测（进行中）

必须走**官方 `run_simulation.py`**，不能用 `scripts/run_benchmark.py`——后者是手写循环、
用 `PerfectTrackingController`，而官方闭环用 `two_stage_controller`（LQR），复现不了已发表分数。
遮挡观测通过 `configs/nuplan/observation/occluded_box_observation.yaml` 接入官方 runner，
工厂函数解决 hydra 不给嵌套节点注入 scenario 的问题。

**验收线：baseline 必须接近 90.43。对不上则 harness 不可信，遮挡差值无意义。**

性能：瓶颈在 CPU（`observation_adapter` 约 296 ms/步，nuPlan 的 GPKG 地图查询），
不在 GPU（实测 CPU 70% / GPU 25%）。**遮挡本身只加约 5%**（每步 23.9 ms）——
这个数字对论文有用：遮挡层的计算成本可忽略，别人复现没有算力门槛。

`scripts/server/score.py` 逐项报 8 个官方子指标而非只报总分：nuPlan 的闭环分是
乘性惩罚 × 加权平均，**遮挡该打中的碰撞和 TTC 会被限速、舒适性稀释**。
要盯 `no_ego_at_fault_collisions` 和 `time_to_collision_within_bound`。

---

## 备选的第二根轴：地图陈旧（施工 / 封道 / 几何变更）

**尚未开工，先记录分析。**

nuPlan 的 tracked objects 含 `TRAFFIC_CONE` / `BARRIER` / `CZONE_SIGN`，而地图是静态的
——**这个失配数据里天然存在**。mini 的 64 个 log **全部**含有这些物体，且有官方 tag：
`near_trafficcone_on_driveable`（3691）、`near_construction_zone_sign`（1457）、
`near_barrier_on_driveable`（438）。

**注入点和遮挡不同，且可组合**：遮挡在 observation 侧；地图在 planner 侧。
`Simulation.__init__` 里指标用 `SimulationHistory(scenario.map_api, ...)`，
`Simulation.initialize()` 里 planner 拿 `PlannerInitialization(route_roadblock_ids, mission_goal, map_api)`
——**两者是分开的对象**，可以只污染 planner 一侧而指标保持真值。实现为 planner 包装器，
不是仿真包装器。`route_roadblock_ids` 最便宜（Flow Planner 每步调 `route_roadblock_correction`）。

**realism 不需要训练生成模型。** 从真实锥桶实测的布局统计（6 个 log 最密帧，302 个锥桶）：

| | 中位数 | p10 / p90 |
|---|---|---|
| 相邻间距 | 1.94 m | 0.41 / 7.96 |
| 每簇数量 | 4 | p90 = 23，最大 33 |
| 簇长度 | 9.0 m | p90 = 31.7 m |

封道本质是车道图上的四五个参数（roadblock / lane / 起止 / 渐变），用神经网络拟合这个
是杀鸡用牛刀，且缺乏训练信号（有锥桶框，无"施工区范围"标注）。**学习应放在"提出 spec"**：
搜索 + planner 在环（现在就能做）→ LLM 从事故叙述抽 spec（S2.2 天然的另一种输出）→
学习型代理加速搜索（仅当搜索成瓶颈）。

**⚠️ 主要阻塞：背景车是日志回放的。** 放一段封道，日志里的车会直接开过去——它们被录制时
那里没有东西。场景当场失真，还会产生假碰撞。**地图修改在依赖上位于反应式 agent（S0.4）之下，
不是独立的轴。**

两个必须声明的自由参数：多少锥桶算封了一条道；封道后是否仍留可行路径（不留就落进
S1.5 要剔除的"全信息下也不可避"带）。

**零成本的第一步**：`score.py --by-type` 看那三个 tag 是否显著更差。若是，它就是这条路的
动机证据——现有 planner 的 val14 分数把这些场景平均掉了，没人单独报过。

---

## ★ 第一个真结果：val14 全量，遮挡 gap（2026-09-01）

1118 个场景，官方仿真 + 官方 Flow Planner + 官方指标，**唯一变量是 observation**。逐帧语义（`memory_horizon: 0`）。

| | baseline | occluded | delta |
|---|---|---|---|
| **score** | 88.25 | 86.58 | **−1.67** |
| **no_ego_at_fault_collisions** | 94.77 | 93.65 | **−1.12** |
| **time_to_collision_within_bound** | 89.27 | 87.92 | **−1.34** |
| ego_progress_along_expert_route | 93.19 | 92.20 | −0.99 |
| ego_is_comfortable | 94.45 | 93.56 | −0.89 |
| drivable_area_compliance | 98.12 | 97.67 | −0.45 |
| driving_direction_compliance | 99.46 | 99.60 | +0.13 |
| ego_is_making_progress | 99.64 | 99.73 | +0.09 |
| speed_limit_compliance | 97.75 | 97.79 | +0.04 |

**掉得最多的是两个安全指标**，而遮挡不该碰的合规项基本不动。效应落在机制预测的位置。

### 噪声底：确定性，不是 2 分

baseline 复现出 88.25 而非已发表的 90.43（差 2.18），一度担心 run-to-run 方差会淹掉 1.67 的 gap。
**配对分析否定了这个担心**：1118 个场景里 **423 个变化精确为 0.00**——输入相同时流水线是确定性的，
噪声是 0 不是 2 分。所以那 2.18 是**系统性偏差**（devkit 版本 / 配置差异），gap 是真信号。

### 分布形状才是发现

```
mean -1.67   sd 18.16   se 0.54  -> 3.1 个标准误
worse / same / better   408 / 423 / 287
p05 -12.59   p50 +0.00   p95 +2.32
```

**遮挡不是把所有场景拉低一点，而是大部分毫发无损、少数被打得很惨。** 中位数为 0，
尾巴单边（p05 −12.6 vs p95 +2.3）。这正是遮挡 benchmark 该有的形状——多数路况没有重要
东西被挡住；一小部分变得真正困难。**若效应均匀摊开反而可疑。**

### 按场景类型（前后各三）

| 类型 | n | delta |
|---|---|---|
| following_lane_with_lead | **15** | **−19.85** ⚠️ 见下 |
| changing_lane | 70 | **−5.56** |
| traversing_pickup_dropoff | 99 | −3.37 |
| waiting_for_pedestrian_to_cross | 53 | −2.52 |
| … | | |
| starting_straight_traffic_light_intersection | 98 | **+0.00** |
| high_lateral_acceleration | 96 | **+0.01** |

`changing_lane` 最可解释：变道要看相邻车道，而挡住视线的正是旁边那辆车。
`waiting_for_pedestrian_to_cross` 是行人从车后走出，遮挡事故的典型形态。
**直行过灯口和大横向加速度几乎为零（各约百个样本），是有力的阴性对照。**

⚠️ **`following_lane_with_lead` 的 −19.85 暂不采用。** n 只有 15，且**与机制矛盾**——
跟车时前车是全场最不可能被遮挡的目标（同一论证解释了 IDM 为何对遮挡免疫）。
要么背后有真实原因（前车的前车、横向切入），要么是 bug。**这 15 个场景需逐个查证**，
结论出来之前不引用这个数字。

### 待办

- 逐个查那 15 个 `following_lane_with_lead`
- 挖 p05 尾部场景 → 这正是 S1.5 中间带（全可观可避、遮挡下不可避）的候选
- val14 反应式两轮（对照 83.31）
