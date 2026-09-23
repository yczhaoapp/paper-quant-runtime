# 公开论文接入与证据边界

论文接入有三个明确步骤：先取得并固定公开论文原文件，再人工审阅论文的方法并写入 `Recipe`、逐项 `MethodSpec` 和论文主张，最后让经审阅的策略包通过统一训练、推理和回测契约。每个方法步骤必须写明原文规则、运行时规则、适配差异、实现符号与独立 oracle 节点；验收检查原文、spec、策略源码的哈希，以及该节点在本轮执行并通过。文本提取器不会自行决定哪个公式适合交易，也不会自动生成策略代码；页级锚点只能证明指定文本确实出现在固定来源中，不能替代算法 oracle、数据时间切分或人工判断。

仓库内有三篇可离线重演的真实 CC BY 4.0 来源：[Yang 与 Malik 的配对 RL 论文](https://arxiv.org/pdf/2407.16103)、[Chipwanya 的方向分类论文](https://arxiv.org/pdf/2310.16855)和[de Meer Pardo 等人的执行框架论文](https://arxiv.org/pdf/2208.06244v1)。原 PDF、取得日期、许可证和 SHA-256 位于 `research/sources/`。对应 `research/recipes/*.json` 把页级文本锚点、人工审阅的策略边界、原文哈希和策略包源码哈希绑定；`tests/test_paper_intake.py` 核查三份原文并使对应策略完成运行。独立方法 oracle 分别见 `tests/test_pairs_actor_critic_paper.py`、`tests/test_logistic_direction_paper.py` 与 `tests/test_time_sliced_paper.py`。Yang 与 Malik 的案例使用合成日线、20 期窗口和线性 A2C；另外两个案例分别使用公开 AAPL 代理日线与简化的 TWAP 市价子单，均不宣称重现原论文市场或实验绩效。

```bash
uv run --no-sync python -m paperquant.cli paper-run \
  --recipe research/recipes/yang-malik-rl1.json \
  --dataset examples/pairs_actor_critic/dataset.json \
  --events examples/pairs_actor_critic/events.json \
  --training examples/pairs_actor_critic/training.json \
  --policy examples/pairs_actor_critic/policy.json \
  --output runs/paper-pair
```

`paper-run` 先核对原文、人工 recipe、结构化方法 spec、策略源码和主张，再调用通用运行时，最后仅凭制品冷启动回放；同一次尝试的 `paper-run.json` 记录论文、spec、主张、源码、报告和 bundle 的哈希。对单独的来源检查，仍可使用 `paper-check`。正式验收为三篇入库 PDF 各生成一条这种纵向运行收据，并与 18 项策略及本轮 oracle JUnit 合并发布。动态测试还把同一真实来源映射到两套不在正式 18 项目录中的不同订单计划，直接走 `paper-run`，检验目录外代码接入能力。

新论文无需修改运行时核心。对公开 PDF、HTML、TeX 输入，`paperquant.papers.extract_text` 提供显式文本提取；PDF 保留物理页号，HTML 和 TeX 只给一个文本块。扫描版 PDF 没有可提取文字时会失败，不会假装已 OCR；HTML 不执行脚本，TeX 不展开复杂宏或证明公式语义。可用 `scripts/fetch_paper.py --url HTTPS_URL --sha256 EXPECTED_HASH --output FILE` 显式下载，只有哈希匹配才发布文件。默认验收不联网；要纳入离线验收，须按再分发许可把固定来源、人工 recipe、主张与独立策略包一同提供。

验收把每项人工主张的 spec 步骤、实际实现符号、固定策略源码、独立 pytest 节点和本轮 JUnit 绑定；同一项运行另产生包含原始行情、训练请求、模型与完整轨迹的可离线校验 bundle，并在不训练的全新实例中回放。PDF 页级锚点和 oracle 一起构成当前人工复现链路；它不表示系统已能从全文自动生成代码，或证明与原论文全部实验一致。`MethodSpec` 是经过审阅的输入和核验约束，不是假装由全文机器抽取得到的算法真值。

本项目并不宣称对任意未知论文自动完成策略恢复，也不把三份 PDF 的文本锚点计作 18 份独立论文复现。其余 15 个策略有来源、实现边界与独立 oracle 绑定，但目前没有把其全文都作为离线原文件打包；目录中的 18 项计数是可运行且具有论文方法依据的策略计数。论文原实验数据和业绩没有在这里重现，具体差异以各 `research/claims/*.json` 为准。
