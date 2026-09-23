# 回测引擎扩展边界

运行时负责训练、保存、冷启动重载、输入编译和报告；回测引擎负责逐事件调用策略、订单状态、撮合、账户与成交。引擎不需要声明训练能力。`paperquant.engine.BacktestEngine` 是可信引擎端协议，要求实现 `capability(fields)`、`profile()` 与 `run(plan, strategy, events)`；调用方给出的 `EngineCapability` 必须与所选引擎实时返回的能力完全相同，否则在策略训练之前以 `ENGINE_UNSUPPORTED` 失败。`profile()` 记录实际初始现金、手续费和撮合语义，执行前后也核对其哈希。

`capability` 必须明确引擎 ID、接受的时间粒度、行情字段、动作，以及各粒度撮合与估值所需的 `execution_fields`。公共编译器只协商所选引擎的字段；参考引擎显式要求 tick 的 `price` 或 bar 的 `open/close`，quote-only 外部引擎可声明自身的报价字段。`run` 以已经编译的 `ExecutionPlan` 和有效事件为输入，返回顺序一致的 `Decision`、`OrderEvent`、`Fill`、`AccountSnapshot`。每条事件必须对应一条决策与账户快照；无交易也要有决策记录。运行时再次核对决策和账户轨迹、动作范围，以及成交对应的已接受和已完成订单，并拒绝先于接受时间的成交。引擎自身仍应校验计划中的策略标识、声明哈希、引擎 ID、有效事件哈希、能力与执行配置哈希，并把不支持的撮合或账户语义作为结构化错误返回，不能写零值冒充未知的持仓盈亏。`ReferenceEngine` 与限定能力的 `BacktraderEngine` 均实现此协议；测试驱动另验证无成交引擎也不被运行时硬编码排斥。

参考引擎也核对计划中的政策快照与哈希，按其中的资金模式处理购买力。外部引擎应把相同政策映射到自身执行配置；不能实现时应以结构化不支持错误退出。外部引擎意外抛出的普通异常由运行时转成 `BACKTEST_FAILED`；这并不证明该引擎的成交和账户数值正确，接入时仍需专门的差分测试。

接入第三方回测器时，适配器应在原生订单状态变化后生成统一订单事件和成交，使用原生交易流水计算持仓均价、已实现毛价格损益、按当前可用价格估值的未实现损益；手续费单列于 `Fill.fee` 并进入现金。完整账户字段是本版协议的固定义务，缺少任一字段的适配器不能声明实现该协议，应在编译期拒绝或扩展带有显式未知值的新版账户 Schema。订单在参考引擎中为下一事件一次性市场/可成交限价；若外部引擎采用部分成交、排队、撤单、不同滑点或同一时刻双腿原子成交，必须在独立执行配置中记录，并以同一输入和相同配置做逐笔差分。只对比最终权益不足以证明接口兼容。

原生 Backtrader 桥接固定版本 `1.9.78.123`，仅接受单标的 OHLCV 分钟/日线、`cash_only`、市价 `SubmitOrder` 和 `TargetPosition`。其 broker 在下一根开盘价成交；完成订单的价格、数量、手续费与实际持仓来自原生 broker，统一账户中的已实现价格盈亏据原生成交逐笔累计，浮动盈亏按原生持仓均价和当前已完成收盘价计算。多标的和无限融资由能力协商在训练前拒绝；限价与部分成交没有等价映射，也会明确失败。Backtrader 使用二进制浮点数，因此正式差分对数量、账户和盈亏使用 `0.000001` 绝对容差，对决策、方向、时刻及订单标识要求一致。规则、监督学习和强化学习三类公开 AAPL 案例均在主机与严格 worker 上走原生 broker，保存、重载、回放后与参考引擎逐笔差分；这只证明该限定能力集合，不证明 Nautilus、订单簿或实盘执行。

```bash
uv run --no-sync python -m paperquant.cli run --engine backtrader \
  --package strategies/rule.moving_average_crossover \
  --dataset examples/public_aapl/dataset.json \
  --events examples/public_aapl/events.json \
  --policy examples/public_aapl/policy.json \
  --output runs/native-sma
uv run --no-sync python -m paperquant.cli replay-bundle \
  --bundle runs/native-sma/bundle.json \
  --package strategies/rule.moving_average_crossover
```

策略隔离与引擎扩展相互独立。严格模式中的策略仍在无网络、只读 worker 中；引擎和账户账本留在主机可信侧。第三方适配器与参考引擎使用同一运行计划、训练证据哈希、worker 收据、失败输出与验收清单约束。
