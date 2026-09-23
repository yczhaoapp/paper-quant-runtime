# Paper Quant Runtime

Paper Quant Runtime 是面向公开研究策略的可执行契约。它将策略声明、行情输入、训练制品、推理动作和回测报告连接到同一条可检查的运行流程。

当前仓库已实现规则、监督学习、强化学习三类各六个不同策略，覆盖字段与时间粒度校验、显式兼容转换、结构化失败、隔离执行和统一报告。项目自带逐项验收与严格容器验证；公开仓库和跨平台远端 CI 的交付状态以对应提交的实际收据为准，不由策略数量推断。

设计原则：

- 先根据声明和实际输入编译运行计划，再加载策略代码。
- 策略只使用公开的行情、账户、动作和制品接口；回测引擎保持独立。
- 转换默认关闭，启用时记录依据、受影响范围、关闭开关与输入输出哈希。
- 参考撮合默认拒绝现金不足的买入；需要融资的样例必须在政策中显式声明，报告保留该假设。
- 每个训练产物记录内容哈希，加载时重新校验。
- 失败返回机器可读的错误，不补造行情、模型或订单。
- 市价与限价订单均有明确的下一事件语义；未成交限价单显式过期。

接口与错误语义见 [契约说明](docs/contract.md)，可执行的引擎协议见[扩展边界](docs/engine-extension.md)，真实 PDF 与 PDF/HTML/TeX 接入边界见[论文接入说明](docs/paper-intake.md)；逐项能力与证据边界见 [验收矩阵](docs/development-plan.md)。

研究候选清单记录了 18 个拟实现的方法与来源核对状态；候选数量不是已完成数量。不同实现的效果比较使用中立观察格式，要求相同的输入哈希、方法标识和执行条件哈希，输出动作/成交路径与权益差。观察相同只说明该场景中的行为一致，不证明论文实验结果复现。格式和使用方式见 [比较协议](docs/comparison.md)。

当前开发样例使用 Python 3.12 或 3.13 和 `uv` 0.12.18；`uv.lock` 固定主机及 CI 依赖：

```bash
uv sync --locked --extra dev
uv run --no-sync python -m paperquant.cli schema --output schemas
uv run --no-sync python -m paperquant.cli run \
  --package examples/basic_rule \
  --dataset examples/data/dataset.json \
  --events examples/data/events.json \
  --output runs/demo
uv run --no-sync pytest -q
```

`runs/demo/report.json` 包含实际执行计划、兼容记录、逐事件决策、订单、成交和账户快照。失败时同一命令写入 `failure.json` 并返回非零退出码；同一输出目录的新尝试会使旧成功报告失效。严格隔离所需的镜像或 Docker 不可用时返回 `SANDBOX_UNAVAILABLE`。

严格模式将策略放进单独的只读、断网、非 root Docker worker；同一验收命令会无缓存构建镜像、对照三类策略的主机与容器轨迹，并生成与本轮尝试绑定的收据：

```bash
uv run --no-sync python scripts/verify_strict.py --require-clean --output runs/strict-verification
```

该命令还会逐项执行 `examples/catalog.json` 中的 18 个策略，要求每项生成真实决策、成交、论文主张绑定、独立 oracle 测试路径，以及主机和严格 worker 的一致轨迹；并验证一个真实 PDF 的页级接入和策略绑定。逐项报告与 `acceptance.json` 位于本轮 `attempts/<attempt_id>/acceptance/`，主收据保存该验收文件的 SHA-256。只需查看主机运行时，也可执行 `uv run --no-sync python scripts/acceptance.py --output runs/acceptance-host`。

隔离边界、实际容器配置核查及已知限制见 [沙箱说明](docs/sandbox.md)。

另一个研究样例是按 Zhu 等人论文中 VMA(2,20,0) 公式实现的日线规则；其附带行情为明确标记的合成测试夹具，不用于宣称复现论文的中国指数实验：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/rule.moving_average_crossover \
  --dataset examples/daily_vma/dataset.json \
  --events examples/daily_vma/events.json \
  --output runs/daily-vma
```

公开 AAPL 日线验证使用固定的 [Plotly 数据文件](https://github.com/plotly/datasets/blob/0c447c47b757ad74edecab31f0d72f849d2e67c2/finance-charts-apple.csv)，从本地原始 CSV 生成规范事件，并明确授权 `AAPL→DEMO` 映射：

```bash
uv run --no-sync python scripts/build_public_apple.py
uv run --no-sync python -m paperquant.cli run \
  --package strategies/rule.moving_average_crossover \
  --dataset examples/public_aapl/dataset.json \
  --events examples/public_aapl/events.json \
  --policy examples/public_aapl/policy.json \
  --output runs/public-aapl
```

这份数据用于真实公开行情的运行验证，不是论文原实验数据。原始文件与时间约定见 `data/public/SOURCE.json`；CSV 中预计算指标和复权列不会进入策略输入。

同一公开日线还验证了论文中的 TRB(50,0.01,C=10) 区间突破与固定持有期规则。它是另一种实质不同的策略，但仍使用同一份市场数据：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/rule.channel_breakout \
  --dataset examples/public_aapl/dataset.json \
  --events examples/public_aapl/events.json \
  --policy examples/public_aapl/policy.json \
  --output runs/public-channel
```

时间切片执行样例根据[最优执行框架论文](https://arxiv.org/pdf/2208.06244v1)中的 TWAP 基准调度思想，将 12 股母单拆为 12 个等量子订单，并在下一根公开 AAPL 日线开盘成交。它只验证预先确定的切片和订单契约；日线市价成交与原文的 BTC 分钟内订单簿限价及桶边界撮合并不相同：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/rule.time_sliced_execution \
  --dataset examples/public_aapl/dataset.json \
  --events examples/public_aapl/events.json \
  --policy examples/public_aapl/policy.json \
  --output runs/public-time-sliced
```

日线监督样例按 [Chipwanya 论文](https://arxiv.org/pdf/2310.16855)给出的次日方向标签、OHLCV 特征与 logistic 训练参数构建。训练标签截至公开样例第 321 根日线，回测从第 322 根开始；`lineage.json` 固定原始哈希、行号和可用时间。这是 AAPL 代理数据，不宣称复现论文中的日本股票准确率：

```bash
uv run --no-sync python scripts/build_public_direction.py
uv run --no-sync python -m paperquant.cli run \
  --package strategies/supervised.logistic_direction \
  --dataset examples/public_direction/dataset.json \
  --events examples/public_direction/events.json \
  --training examples/public_direction/training.json \
  --policy examples/public_direction/policy.json \
  --output runs/public-direction
```

另一个监督样例按[三日价格回归研究](https://arxiv.org/pdf/2310.09903v5)中的 Ridge 家族，用三根已完成日线的 OHLCV 预测三根交易日后的收盘价。训练仅用截至公开样例第 321 根日线可见的标签，回测自第 322 根开始。此处复现 L2 回归方法与三日窗口，未实现原文的 123 个技术指标和特征选择实验；按预测价决定单位目标持仓也是明确的运行时适配：

```bash
uv run --no-sync python scripts/build_public_ridge.py
uv run --no-sync python -m paperquant.cli run \
  --package strategies/supervised.ridge_three_day_price \
  --dataset examples/public_ridge/dataset.json \
  --events examples/public_ridge/events.json \
  --training examples/public_ridge/training.json \
  --policy examples/public_ridge/policy.json \
  --output runs/public-ridge
```

[分类器组合研究](https://arxiv.org/pdf/2107.13148v3)中的 Gaussian Naive Bayes 单模型组件也在同一公开 AAPL 留出集上运行。这里只训练一个分类器和四个 OHLCV 衍生特征；原文的四模型加权集成与动态特征筛选不在该样例中：

```bash
uv run --no-sync python scripts/build_public_gaussian.py
uv run --no-sync python -m paperquant.cli run \
  --package strategies/supervised.gaussian_nb_direction \
  --dataset examples/public_gaussian/dataset.json \
  --events examples/public_gaussian/events.json \
  --training examples/public_gaussian/training.json \
  --policy examples/public_gaussian/policy.json \
  --output runs/public-gaussian
```

另一项公开 AAPL 监督样例按 [Lauretto 等人的研究](https://arxiv.org/pdf/1301.4944v1)训练随机森林，预测买入卖出、卖出买回或无操作。Bootstrap 抽样、节点随机特征和多数投票均实际执行；三项衍生特征、浅树、固定止盈止损参数与代理行情不等于原文的巴西股票市场和调参实验：

```bash
uv run --no-sync python scripts/build_public_forest.py
uv run --no-sync python -m paperquant.cli run \
  --package strategies/supervised.random_forest_operations \
  --dataset examples/public_forest/dataset.json \
  --events examples/public_forest/events.json \
  --training examples/public_forest/training.json \
  --policy examples/public_forest/policy.json \
  --output runs/public-forest
```

三标的监督样例按 [Gu、Kelly、Xiu](https://dachxiu.chicagobooth.edu/download/ML.pdf) 的 Elastic Net 惩罚目标拟合次日收益，再按每日横截面的预测值选择一条多头和一条空头。样本为合成面板；两项特征、三标的、固定超参数及次日开盘成交均是缩减适配，不代表原文的大股票池和月度收益实验：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/supervised.cross_sectional_rank \
  --dataset examples/cross_sectional_rank/dataset.json \
  --events examples/cross_sectional_rank/events.json \
  --training examples/cross_sectional_rank/training.json \
  --output runs/cross-sectional-rank
```

订单簿队列不平衡样例通过同一入口执行监督学习的训练、制品保存与重载、逐 tick 预测和最小回测。训练标签与行情均为合成夹具，预测到目标持仓的映射另行披露：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/supervised.queue_imbalance \
  --dataset examples/l1_queues/dataset.json \
  --events examples/l1_queues/events.json \
  --training examples/l1_queues/training.json \
  --output runs/l1-queues
```

L1 微观价格规则只实现 [Stoikov 研究](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694)配套讲义中的布朗不平衡特例：此时微观价格等于按最优买卖队列量加权的中间价。它把该价格与最后成交价比较后映射为目标持仓，使用合成 L1 事件验证字段、推理和撮合；不声称完成论文的 Markov 校准与实证比较：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/rule.microprice_toy \
  --dataset examples/l1_queues/dataset.json \
  --events examples/l1_queues/events.json \
  --output runs/microprice-toy
```

两档订单簿做市样例按[Avellaneda–Stoikov 论文](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf)第 29、30 式计算库存相关保留价格和最优报价价差，并发出带限价的买卖子订单。32 条 L2 行情是合成夹具；参考撮合只让每张订单在下一事件有效，不能用它复现原论文的排队成交、泊松到达或收益分布：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/rule.reservation_quote \
  --dataset examples/l2_book/dataset.json \
  --events examples/l2_book/events.json \
  --output runs/reservation-quote
```

三标的配对规则按 [Gatev 等人的论文](https://repec.som.yale.edu/icfpub/publications/2573.pdf)用形成期标准化价格距离选一对，再用历史价差的两倍标准差开仓、价格交叉平仓。行情是合成的，且两条腿分别在各自的下一个事件成交；不能据此宣称复现原文的大规模市场收益或原子配对成交：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/rule.pairs_distance \
  --dataset examples/three_stock_pairs/dataset.json \
  --events examples/three_stock_pairs/events.json \
  --output runs/pairs-distance
```

有限时域执行样例根据 [Nevmyvaka、Feng、Kearns](https://www.cis.upenn.edu/~mkearns/papers/rlexec.pdf) 的倒序状态—动作价值更新，在合成压力、剩余时间和剩余买入量上学习子单大小。它实现强制到期完成，但把论文中的限价撤改单与真实订单簿撮合缩减为下一 tick 市价目标；因此不能用它声称复现原文的市场冲击或执行成本：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/reinforcement.execution_value_learning \
  --dataset examples/execution_lattice/dataset.json \
  --events examples/execution_lattice/events.json \
  --training examples/execution_lattice/training.json \
  --output runs/execution-value
```

同一份合成 tick 行情也可驱动 SARSA 方法适配样例；它训练的是状态—动作价值表，使用训练轨迹的实际下一动作更新，并把学习到的动作映射为单位目标持仓：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/reinforcement.sarsa_inventory \
  --dataset examples/l1_queues/dataset.json \
  --events examples/l1_queues/events.json \
  --training examples/l1_queues/sarsa_training.json \
  --output runs/sarsa
```

连续 Actor-Critic 样例按[组合管理研究](https://arxiv.org/pdf/1911.11880v2)的策略梯度、价值函数和组合对数收益思路，在训练期间自行抽样 AAPL 与现金的连续权重；随后只在独立公开行情区间推理。它用一维风险资产权重与线性函数逼近代替原文的多资产神经网络，参考引擎在下一开盘按目标股数成交：

```bash
uv run --no-sync python scripts/build_public_actor_critic.py
uv run --no-sync python -m paperquant.cli run \
  --package strategies/reinforcement.actor_critic_allocation \
  --dataset examples/public_actor_critic/dataset.json \
  --events examples/public_actor_critic/events.json \
  --training examples/public_actor_critic/training.json \
  --policy examples/public_actor_critic/policy.json \
  --output runs/public-actor-critic
```

离散 A2C 配对样例取自[Yang、Malik 的配对交易研究](https://arxiv.org/pdf/2407.16103)中的 RL1 分支：用前 20 个已完成配对收盘价拟合滚动回归，以当前残差、z-score、zone 和持仓构成观察，训练三动作 softmax actor 与 TD critic，分别在下一日开盘执行两条目标腿。样例使用独立合成日线配对和线性模型，不复现论文的 BTC-EUR/BTC-GBP 分钟数据、神经网络、原始收益或 RL2 连续仓位实验；两腿成交也不保证原子性：

```bash
uv run --no-sync python scripts/build_pairs_actor_critic.py
uv run --no-sync python -m paperquant.cli run \
  --package strategies/reinforcement.pairs_actor_critic \
  --dataset examples/pairs_actor_critic/dataset.json \
  --events examples/pairs_actor_critic/events.json \
  --training examples/pairs_actor_critic/training.json \
  --policy examples/pairs_actor_critic/policy.json \
  --output runs/pairs-actor-critic
```

风险厌恶 contextual Bandit 样例实现 [Lin、Wang、Zhou](https://arxiv.org/pdf/2206.12463v1) 的一维 disjoint normal-gamma 后验更新和 Thompson 抽样的均值—方差选臂。部署时仅用实际持仓区间的后续价格变化更新相应动作臂；L1 队列不平衡和单位持仓是合成缩减场景，不是论文的组合选择实证：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/reinforcement.risk_averse_bandit \
  --dataset examples/bandit_l1/dataset.json \
  --events examples/bandit_l1/events.json \
  --training examples/bandit_l1/training.json \
  --output runs/risk-averse-bandit
```

Double Q 样例使用同一运行契约，但训练时由两张价值表分别选择下一动作和评估其价值，并用训练 seed 决定每步更新哪张表。它复现的是双表更新算法；L1 状态、目标持仓、模拟成交仍是明确披露的交易环境适配：

```bash
uv run --no-sync python -m paperquant.cli run \
  --package strategies/reinforcement.double_q_market_making \
  --dataset examples/l1_queues/dataset.json \
  --events examples/l1_queues/events.json \
  --training examples/l1_queues/sarsa_training.json \
  --output runs/double-q
```
