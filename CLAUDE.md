# 遮挡感知的 nuPlan 闭环 benchmark

在 `nuplan-safety-bench` 分支上。完整研究方案在 [`docs/PLAN.md`](docs/PLAN.md)——**开工前先读它**，下面只写方案里没有的操作性内容。方案是活文档，有新结论就往里加。

（`~/.claude/plans/tidy-questing-shell.md` 是它的历史出处，已不再维护；以仓库内这份为准。）

一句话概括：所有闭环 planning 评测都给 planner 发周围 agent 的**真值框**。这个工作把它拿掉，只留 ego 真正能看见的，然后用**官方 nuPlan 仿真 + 官方 planner + 官方指标**测排名怎么变。

## 三个 conda 环境，别用错

| 环境 | 用途 |
|---|---|
| `flow_planner` | **默认用这个。** torch 2.3 + nuplan-devkit + Flow Planner。跑 benchmark、测试、渲染都在这里 |
| `nuplan` | torch 1.9，devkit 能用但跑不了 Flow Planner。基本弃用 |
| `smart` | 原 SMART 仓库的环境，WOMD 那条线才需要 |

`flow_planner` 里已补装 `torch_geometric` + `torch_scatter` + `torch_cluster`（SMART 要用）。
代理封了 `data.pyg.org`，预编译轮子拿不到，是用 conda 环境 `build_tools` 里的 gcc 从 pypi
源码编的，**只有 CPU 版**。服务器上要装 CUDA 版（那边能连 data.pyg.org 就直接 pip）。

一律用绝对路径调解释器，**不要 `conda activate`**——非交互 shell 里 conda 初始化不可靠，这个坑踩过：

```bash
PYTHONPATH=. ~/anaconda3/envs/flow_planner/bin/python -m pytest tests/ -q
```

## 数据和 checkpoint

- nuPlan mini：`/mnt/e/nuplan-mini/`（`nuplan-v1.1_mini/data/cache/mini/` 是 db，`nuplan-maps-v1.0/maps/` 是地图）
- nuplan-devkit 源码：`~/nuplan-devkit`
- Flow Planner 源码：`~/SimAgentJEPA/external/Flow-Planner`
- Flow Planner 权重：`checkpoints/flow_planner/`（gitignore，从 HuggingFace `ttwhy/flow-planner` 取）

## 网络

外网要先开代理，`proxy_on` 是 `.bashrc` 里的函数，非交互 shell 加载不到，手动设：

```bash
WH=$(ip route show | awk '/default/ {print $3; exit}')
export HTTPS_PROXY="http://${WH}:7897" HTTP_PROXY="$HTTPS_PROXY" https_proxy="$HTTPS_PROXY" http_proxy="$HTTPS_PROXY"
```

`git push` 经常撞 TLS 握手失败，设了代理重试即可。

**但代理开着就连不上服务器。** Clash Verge 的 TUN 模式（`enable_tun_mode: true`）在网络层截
TCP，去 `172.18.37.114` 这种内网地址的流量也一并截走，而代理出不去内网。症状具有迷惑性：
`ping` 是 ICMP，走另一条路径，**照样通**。要么 `proxy_off` 并关掉 TUN，要么在 Clash 里给
`172.18.0.0/16` 加直连规则。拉外网和连服务器这两件事得分开做。

## 服务器（val14 全量评测）

`ssh l40s`，4×L40S + 192 CPU，`$HOME=/lab/haoq_lab/12432702`。**SSH 很不稳，经常超时，重试几次**。

- 工作区 `~/occlusion-bench`，仓库 `~/SMART-safety-critical`
- 数据 `/hqlab/dataset_nas3/nuplan/raw/`
- 连不上 huggingface.co，用 `hf-mirror.com`
- 环境 `~/miniforge3/envs/flow_planner`

**手动调 `run_val14.sh` 一定要带 `VAL_SPLIT`。** val14 在 NAS 上读极慢，本地副本在
`~/occlusion-bench/val14_local`（328 个 .db，87G）。`sweep_val14.sh` 有默认值所以看不出问题，
但直接调 `run_val14.sh` 时它会回退到探测，探到 NAS 路径就照跑不误——**不报错，只是慢到不可用**。
症状是进程 `Dl` 状态、`/proc/PID/wchan` 显示 `rpc_wait_bit_killable`，几十分钟零产出。
同一个 4 场景冒烟：NAS 上 24 分钟没跑完，本地副本 6 分钟结束。

```bash
export VAL_SPLIT=$HOME/occlusion-bench/val14_local   # 先设这个，否则会走 NAS

bash scripts/server/setup_val14.sh                              # 一次性
CHECK_ONLY=1 bash scripts/server/sweep_val14.sh                 # 预检，几秒钟看清缺什么
CUDA_VISIBLE_DEVICES=3 LIMIT=4 bash scripts/server/run_val14.sh baseline   # 冒烟
bash scripts/server/sweep_val14.sh                              # 全量，出总表
python scripts/server/score.py ~/occlusion-bench/exp --by-type
```

跑之前先 `CHECK_ONLY=1` 过一遍预检。它用 `run_val14.sh` 的 `DRY_RUN=1` 解析每个组合、
检查要打开的文件，缺权重几秒钟就报出来，而不是排队几小时后在启动时失败。

服务器上没有 `tmux` 和 `rsync`。后台用 `setsid nohup ... < /dev/null &`，传目录用
`tar czf - dir | ssh l40s "tar xzf - -C /path"`。

**验收线：baseline 必须接近 Flow Planner 已发表的 90.43（Val14 非反应式）。对不上就别看遮挡的数字**——harness 不可信的话，差值没有意义。

## 我们自己的代码

只有三块是这个工作的贡献，其余全部用官方的：

- `smart/occlusion/visibility.py`——2D 视线几何，**零自由参数**
- `smart/occlusion/tracking.py`——跨遮挡的目标记忆，**不依赖 nuPlan**，可用合成序列测
- `smart/nuplan/occluded_observation.py`——包装任意 `AbstractObservation`，过滤 `DetectionsTracks`
- `smart/nuplan/smart_agents.py`——用 SMART 驱动背景车流替代 IDM。参考了 Bosch 的
  `interactive-closed-loop`（AGPL-3.0，**只读不抄**，代码自己写），只共用他们的 nuPlan checkpoint

指标、碰撞判定、责任归属**一律用 devkit 的**。曾经自己重写过一套，已删除——自造指标和官方数字对不上，整个对比就无法引用。

## 已经踩过的坑，别重蹈

**遮挡观测必须是全观测的严格子集。** 跟踪缓冲分不清"被遮挡"和"底层不再报告"（目标驶出检测范围），不加约束会把已消失的目标当幽灵保留，导致**遮挡条件比全观测条件信息更多**，实验倒转。已有测试锁住这个不变量。

**部分遮挡按可见处理。** 5 个采样点（4 角 + 质心）有 1 个通视就算看见，且透传真实完整框——从局部观测补全是感知该做的事。

**默认逐帧，`memory_horizon: 0`。** Flow Planner 的 `filter_agents_tensor(reverse=True)` 只保留当前帧存在的 agent，被遮挡目标连同历史一起丢弃——它本来就没有记忆能力。外加缓冲等于凭空赋予它不具备的能力。带记忆的版本留作显式消融。

**IDM planner 对遮挡完全免疫**，两个条件下轨迹逐点相同（0.000000 m）。它只取自己路径上最近的障碍物，而最近前车恰恰最不可能被遮挡。**不能用 IDM 做主力**，需要 Flow Planner 这类会用全场信息的。

**验证不变量要双向查。** 曾经只查"被扣留的是否真的不可见"（零违例），漏了反方向"被给出的是否真的可见"——问题全在那一侧。

**选场景必须按官方 `scenario_tag`。** 随意挑起始帧会挑到静止场景（mini 里 `stationary` 标签 62375 帧），expert 全程只动 3.8 米，任何感知假设都不可能改变结果。见 `smart/nuplan/scenarios.py`。

**ego_pose 存的是后轴不是车体中心**，Pacifica 差 1.461 m。

**转换器的大问题是坐标重心化，不是地图。已修。** `convert_scenario(recentre=False)`，
`SMARTAgents` 默认走这条。同场景同 checkpoint，按真实 agent 算 next-token（必须排除背景物体，
否则静止的锥桶白送准确率）：

| | top-1 | top-10 |
|---|---|---|
| 重心化（改前） | 7.49% | 32.65% |
| **全局 UTM（改后）** | **23.62%** | **70.69%** |
| Bosch 全套 | 25.70% | 75.28% |

28 个场景、两个 scenario_tag、21345 tokens，**28/28 全胜**。补回了 92% 的差距。

**为什么反直觉：** 重心化在数值上更好（UTM northing 在 float32 下量化到约 0.5 m），
当初就是为这个加的。但 Bosch 的 checkpoint 就是在**没重心化、带着量化**的数据上训的，
干净的局部坐标对它是分布外输入——这一项压过了数值条件。**坐标系要匹配 checkpoint，不是匹配数值。**
WOMD 训的模型（`epoch=31`）用局部坐标是对的，所以 `recentre` 默认仍为 True。

配套条件：`MAP_POINT_SPACING` 必须保持 1 米。nuPlan 原生 0.25 m 在 UTM 量级的 float32 下
会让相邻点塌陷，`interplating_polyline` 直接抛异常。这才是那个稀疏化改动的真正价值。

**归因错过一次，值得记：** 中间做过"换他们的地图 → 21.27%"的实验，据此得出"差距在地图侧"。
那个实验同时改了坐标系，我把增益全算到地图头上了。补上缺的对照格（我们的地图 + 全局 UTM）
立刻就看出来了。**一次只改一个变量，或者把对照表填满。**

以下全部实测无影响，别再试：地图半径、点密度、车道边界、agent 筛选、地图类型编号、
polygon 结构（一车道一 polygon）、局部坐标系偏移 1 万米。

**跑他们的转换器做对比要 `av2`**：`pip install --no-deps av2 universal-pathlib pathlib_abc`，
再把 `cv2.typing` 用 sys.modules 桩掉（那只是类型标注，别升级环境里的 opencv 4.5.1）。
他们的模块用 `nuplan.planning.training...` 命名空间互相 import，按文件路径加载后注册进 sys.modules 即可。
**代码是 AGPL-3.0，只在 scratchpad 里跑，不进仓库；ckpt 也不要提交**（`.gitignore` 已挡）——
AGPL 不限制"使用"，但抄代码会让我们从 Apache-2.0 变成 AGPL，对要给别人跑的 benchmark 是自断采用率。

**Bosch 的 checkpoint 用的是 Argoverse 风格的地图类型编号，和原版 SMART(WOMD) 不一样。**
他们走 nuPlan→argo→SMART 两段式，`_point_types` 表更短且顺序不同：CENTERLINE 是 11 不是 16，
CROSSWALK 是 10 不是 15，车道边界是 0(DASHED_WHITE)/8(NONE)。checkpoint 本身能证实——
`type_pt_emb` 只有第 0、8、11 行范数约 2.3，其余全在 0.11（初始化没动过）。
我们的转换器按 WOMD 编号发 12/15/16，**全部打在未训练的行上**。
`converter.to_nuplan_checkpoint_semantics()` 做重映射，`SMARTAgents` 默认开启。

**但这不是精度差距的原因——假设被自己的数据推翻了**（真正的原因见上面的 polygon 结构）。 重映射把 type embedding 平均范数
从 0.109 提到 2.240（确认生效），next-token 准确率在 17816 个 token 上从 6.81%→6.65% (top-1)、
30.07%→30.02% (top-10)，逐场景两个方向都是噪声。**地图类型 embedding 根本不承载信息，几何才承载。**
改还是要改（给 checkpoint 喂没见过的索引是隐患），但别当成性能修复。

**他们完全没对齐真实车道线语义。** `argo_vector_utils.py:148` 只用 `is_intersection` 一个布尔量：
路口内→NONE，路口外→DASHED_WHITE。nuPlan 的真实标线类型一个都没读。

**SMART 的 token 词表只覆盖前向运动。** 造反应性测试时让假 ego 沿 +y 平移、heading 却指别处，
等于在倒着开——`match_token` 匹配不上任何模板，退化成近似静止的 token，两条不同的 ego 轨迹
因此得到**逐位相同**的预测。当时差点据此得出"交通对 ego 无反应"的错误结论。测试里的 ego
必须沿自己的 heading 走。换成物理合理的轨迹后：192 个物体里 58 个不同，最大 0.917 m。

**`WaymoTargetBuilder` 是训练用的，推理不能用。** 它按 agent 的**未来**有效帧数决定预测谁，
还随机下采样。闭环里未来按定义全无效——用它等于一个都不预测，车全部静止。直接 `HeteroData(data)`。

**`TokenProcessor.preprocess` 原地修改地图字典。** 缓存的 `self._map` 被第一次 rollout 消费掉，
之后每次读到的都是自己的残渣。每次 rollout 要 `copy.deepcopy`。

**SMART 的历史窗口必须取过去，不能从 iteration 0 往后读。** 往后读等于开局就把 ego 两秒的
日志未来喂给模型。用 `get_ego_past_trajectory` / `get_past_tracked_objects`。

**性能**：瓶颈在 CPU（nuPlan 地图查询约 296 ms/步），不在 GPU。遮挡本身只加约 5%。

## 常用命令

```bash
PYTHONPATH=. ~/anaconda3/envs/flow_planner/bin/python -m pytest tests/ -q      # 48 个测试
PYTHONPATH=. ~/anaconda3/envs/flow_planner/bin/python scripts/run_benchmark.py --planner flow --scenarios 6
PYTHONPATH=. ~/anaconda3/envs/flow_planner/bin/python scripts/render_nuplan_gif.py --baseline <dir> --occluded <dir> --out-dir gifs
```

`scripts/run_benchmark.py` 是本地小规模用的手写循环，**用 `PerfectTrackingController`，复现不了已发表分数**。要对齐官方数字必须走 `run_simulation.py`（官方用 `two_stage_controller`），见 `scripts/server/run_val14.sh`。

## 风格

提交信息写清楚**为什么**，尤其是踩过的坑和被推翻的假设——这个项目里好几个结论是先错后改的，过程比结论有价值。数字要实测，不要估计；单场景的数字不能当结论。
