# 六篇新论文定向接入补充版本

本补充版本沿用 `0.2.0` 的六篇接入和 `0.2.1` 两篇补强，并在独立分支 `codex/cnn-refinement`
形成软件版本 `0.2.2`。本轮完善 CNN 的配置验证、基线、分阶段训练和独立测试；
所有运行时源码及 Schema 与提交 `d009b62e725c53a3484021e4fef0be74d9dc2b23` 字节一致。
原封存提交 `747460544d5dabcafa9421031bdbc97a1bf28282` 及两轮历史收据保持原状。
本轮范围、完整负面结果、因果特征选模及协议偏差见 [CNN说明](cnn-refinement.txt)。
补充运行是本轮新产生的证据，不回写为先前六篇论文补验已经执行过训练或回测。

这里交付的是逐篇阅读后编写的上游策略包、方法说明，以及通过统一运行契约执行的验证。
它不提供对任意陌生 PDF 自动生成正确策略的保证，也不把局部实现称为整篇论文复现。
既有 18 个样例仍由原目录验收；新增 6 个样例单独验收，不替换既有案例。

## 来源、方法与差异

六篇主论文使用评测报告中的精确 arXiv 版本和 PDF SHA256。
`research/six-paper/source-lock.json` 另记录 SAC、PPO、Self-Imitation Learning 三篇算法来源。
`cases/*/recipe.json` 绑定全文摘要、页码文字锚点、策略源码摘要和 MethodSpec 摘要；
`claims.json` 与 `method.json` 分别记录方法主张、实际代码方法、独立 oracle 和排除范围。
PDF 页码从文件第一页计数，不使用印刷页码。锚点是定位辅助，不能代替人工理解和算法测试。

| 精确来源 | 实际实现 | 数据、训练与主要差异 |
| --- | --- | --- |
| [2403.07998v1](https://arxiv.org/abs/2403.07998v1) | 负单滞后 ADF 权重、精确全局最大权匹配、504 日滚动对数价差、月度选对、z-score 与 beta 美元对冲 | 四资产受控模拟；只实现 z-score 分支；子集动态规划替代 blossom；两腿非原子；不用原文收益实验。原文 Eq3 与后文方向矛盾时采用 §4.2 经济方向，z-score 用标准差。规则策略无训练入口执行需求。 |
| [2412.12458v1](https://arxiv.org/abs/2412.12458v1) | 训练段 AR(1) OLS、Eq8–10 OU 参数转换、历史 90 日分位信号 | 预选模拟价差与固定 hedge；形成期和运行期分离；不使用私有 WRDS、原股池或人为行业选对；Eq7 排印与 Eq9–10 不一致时遵循 Eq10。 |
| [2205.08382v1](https://arxiv.org/abs/2205.08382v1) | 两个卷积自编码器、三组 Conv/Pool/Conv 分支、many-to-one LSTM、经独立梯度验证的解析反向传播、概率阈值与下注量 | BTC 四小时数据；2024 Jan–Jun 训练、July 验证、固定 August–September 测试；完整步长一窗口、窗口相对价量、训练段 scaler、Adam 和验证损失检查点。最终 LSTM/FC 宽度4满足既有 worker 时限；省略 wavelet、XGBoost、dropout、大指标搜索及 SL/TP 引擎。原文质量代理与原实验复现状态单列。 |
| [2409.03762v1](https://arxiv.org/abs/2409.03762v1) | 59 分钟因果特征、训练内 4% 分位标签、BIC 选 GMM、簇内 KNN | BTC 一分钟数据；2024 Jan–Feb 训练、March 仅诊断验证、固定 April–May 测试。1,440 个训练块，训练段阈值/scaler/GMM/邻居库；对角协方差、ridge=.001、K=1..4、1NN。报告类别召回率与持有基线；不声称原三市场收益实验。原文末六分钟歧义按恰好六根处理。 |
| [2410.14504v2](https://arxiv.org/abs/2410.14504v2) | Gaussian/tanh actor、双 critic、熵项、目标软更新、库存约束报价、真实模拟账本奖励 | 四条行为轨迹与 128 更新；隐藏层 8；固定 alpha=0.2；采用现代 twin-Q SAC，省略参数抽取网络；明确量化连续潜变量为三种报价控制。保留 dt=.001、T=1、q=5、spread=.01、p=.2，使用简化扩散，不声称完整 non-Markov 订单簿。 |
| [2510.27334v1](https://arxiv.org/abs/2510.27334v1) | decision/action 双政策、绑定原始行为 logp 的 PPO clip、正优势 SIL、rho 与报价/撤单 | 五维观察、宽度 8；三条初始 on-policy 轨迹、4×32 样本更新；100 秒预热后开始代理，评估八笔间隔 40 秒 TWAP 子单；训练用晚启动 rPOV2 前缀而非原 rPOV1；Hawkes-like 模拟省略完整订单簿、排队、冲击与部分成交；每次成交 1bp 手续费单独披露。 |

支持算法原文：
[SAC](https://arxiv.org/abs/1812.05905v2)、
[PPO](https://arxiv.org/abs/1707.06347v2)、
[SIL](https://arxiv.org/abs/1806.05635v1)。
原文网络宽度、优化器、实验数据与完整训练日程不由上述核心公式测试证明。
Graph 与 CNN 输出持仓数量显式保留八位小数；这是策略的订单精度设置，
不在运行时隐式改写浮点输出，原始行情和训练载荷仍按各自摘要绑定。

## 0.2.0 已引入的契约 1.1 扩展

原 1.0 声明仍有效。新增 `CancelOrder(client_order_id, symbol, reason)`，
声明取消动作的策略必须显式使用 1.1。取消产生 `OrderEvent.status=cancelled`；
冷回放及证据验证同时核对订单 ID、标的和时间。参考引擎与 Backtrader 不宣称支持这一动作，
编译阶段会拒绝要求取消语义的策略。

新增 `PassiveQuoteEngine` / CLI `--engine passive`，范围为单标的 tick、
一侧最多一张当前最优价限价单、单位完整成交、持续挂单、明确撤单和流结束失效。
先处理前一事件挂单的成交，再向策略提供账户并接收新动作，禁止同事件新单成交。
逆向报价穿越或显式对手流量加独立抽样触发成交，当前最优价中点标记权益。
已实现和未实现盈亏均为扣费前价格损益，手续费从现金扣除并随 Fill 记录。
引擎不支持部分成交、排队优先级、多标的、任意深度报价、改单或内生市场冲击；
输入或动作超范围时产生结构化失败，不降级为另一种订单语义。
这是明确建模的参考模拟器，不是第三方实盘撮合能力证明。

`paperquant.strategy_api` 新增纯数值反向自动微分、DenseMLP、Conv1D 和 LSTMCell，
让训练在同一个隔离 worker 内执行；不开放额外文件、进程或网络权限。
补充 worker 镜像名为 `paperquant-worker:six-paper-0.2`，避免覆盖原封存镜像标记。

## 可重复执行

要求 Python 3.12/3.13、uv 0.12.18；严格执行还需要 Linux 容器兼容的 Docker。
先在宿主机明确准备依赖和论文，随后运行阶段无需网络。公开行情 CSV、转换脚本、数据切分、
模拟 seeds、完整行为收集账本和输入摘要已在 Git 内。
PDF 未断言再分发权，因此不放入补充源码归档；准备命令只获取固定 URL 并核对 SHA256。

```bash
uv sync --locked --extra dev
uv run --no-sync python -m scripts.prepare_six_sources --fetch --output runs/six-sources
uv run --no-sync python -m scripts.accept_six_papers --require-clean --output runs/six-host
uv run --no-sync python scripts/verify_strict.py --require-clean --output runs/strict-baseline
uv run --no-sync python -m scripts.accept_six_papers --strict --require-clean --output runs/six-final
```

`verify_strict` 无缓存构建新镜像，验证原 18 案例及所有安全测试。
`accept_six_papers --strict` 再执行六个新增案例的主机与断网隔离 worker 验证，比较完整
decision/order/fill/account 序列。此版本跨平台 workflow 已配置；本地证据不等于远端 CI 已执行。
输出目录使用仓库内 `runs/...`。单篇入口示例：

```bash
uv run --no-sync python -m paperquant.cli paper-run \
  --recipe research/six-paper/cases/ou_percentile/recipe.json \
  --dataset examples/six-paper/ou_percentile/dataset.json \
  --events examples/six-paper/ou_percentile/events.json \
  --training examples/six-paper/ou_percentile/training.json \
  --policy examples/six-paper/ou_percentile/policy.json \
  --output runs/ou-paper
```

SAC 与 PPO 使用 `--engine passive`，PPO 增加 `--fee-rate 0.0001`；
CNN 与 GMM 使用参考引擎和 `--fee-rate 0.0001`。其他例子按 catalog 的输入和费率运行。
纯规则 graph_matching 不传 `--training`。

方法或输入有意修改后需重新运行两份 builder，以更新 manifest、输入锁和方法/源码摘要；
修改后的证据必须标注新提交，不能沿用旧版本验收收据。

## 收据如何解释

`six-paper.json` 的 `passed` 范围仅为来源映射、契约运行、本地方法 oracle 和冷回放。
每篇分别记录真实训练状态、成交数量、零成交标志、收益账本、方法 JUnit 摘要；
`paper_experiment_reproduced=false`。`paper-run` 本身仍标记 `method_validation=not_run`，
方法通过只由补充门禁实际运行的独立 pytest/JUnit 证明。

零成交不是回测失败，但不证明策略交易表现有效。尤其 CNN 的原文质量门槛未达到时，
训练完成只证明生命周期和数值更新实际执行，不能宣称完成原文训练目标。
完整验收保存所有模型、训练输入、运行计划、日志、报告、Bundle 和 oracle JUnit。
失败会原子写入本轮 `failed` 收据并更新 attempt_id；成功从不借用历史尝试结果。

0.2.1 额外生成 `quality.json` 和 `prediction-audit.json`，重新加载实际模型并逐项核对
预测与运行决策；训练拟合、验证表现、测试混淆矩阵及成交数量分开报告。
`research/two-paper/contract-lock.json` 逐字节约束全部既有运行时及 Schema。

算法测试使用多参数、随机样本、独立公式、梯度性质及对照实现；
RL 还检验变更奖励确实改变模型，而不是返回固定模型或预存动作序列。
这些测试支持局部方法正确性，不能保证没有未知缺陷或推广到未验证的数据、依赖与语义。
