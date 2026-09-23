# 黑箱运行对照

`Observation` 是两套独立实现之间的中立格式，包含场景、方法、源与有效行情哈希、执行条件哈希、初始现金、逐事件动作、成交和最终权益。对照器仅在这些前提字段完全一致时输出权益差与路径差；条件不同则输出 `incomparable`，不计算收益差。

`execution_profile_sha256` 必须取自共同审阅的机器可读撮合说明，包括时区、可用时间、撮合延迟、价格、手续费、滑点、现金和仓位限制。仅由操作者传入相同的哈希不能证明两个引擎真的遵守了相同撮合语义；还需对账户、订单和成交轨迹做差分验证。

本运行时的报告可以转换为观察记录：

```bash
uv run --no-sync python -m paperquant.cli observe \
  --report runs/demo/report.json \
  --scenario sample-minute-bars \
  --method moving-average-crossover \
  --profile-sha256 <64位十六进制执行条件摘要> \
  --initial-cash 100000 \
  --output runs/demo/observation.json
uv run --no-sync python -m paperquant.cli compare \
  --left path/to/first-observation.json \
  --right path/to/second-observation.json \
  --output path/to/comparison.json
```

另一实现应独立导出相同的 `Observation` Schema，不需要在本项目内导入其代码。动作 `reason`、订单标识及日志不参与路径一致性比较，以免实现特有的说明文字伪装成行为差异；这些字段仍应在各自原始报告中保留以供审计。结果对照只能说明运行行为或数值差异；论文方法忠实度必须另行核对原文、公式、实验数据与独立 oracle。
