# 公开论文接入与证据边界

论文接入有三个明确步骤：先取得并固定公开论文原文件，再人工审阅论文的方法并写入 `Recipe`、逐项 `MethodSpec` 和论文主张，最后让经审阅的策略包通过统一训练、推理和回测契约。每个方法步骤必须写明原文规则、运行时规则、适配差异、实现符号与独立 oracle 节点；验收检查原文、spec、策略源码的哈希，以及该节点在本轮执行并通过。文本提取器不会自行决定哪个公式适合交易，也不会自动生成策略代码；页级锚点只能证明指定文本确实出现在固定来源中，不能替代算法 oracle、数据时间切分或人工判断。

仓库内有三篇可离线重演的真实 CC BY 4.0 来源：[Yang 与 Malik 的配对 RL 论文](https://arxiv.org/pdf/2407.16103)、[Chipwanya 的方向分类论文](https://arxiv.org/pdf/2310.16855)和[de Meer Pardo 等人的执行框架论文](https://arxiv.org/pdf/2208.06244v1)。原 PDF、取得日期、许可证和 SHA-256 位于 `research/sources/`。对应 `research/recipes/*.json` 把页级文本锚点、人工审阅的策略边界、原文哈希和策略包源码哈希绑定；`tests/test_paper_intake.py` 核查三份原文并使对应策略完成运行。独立方法 oracle 分别见 `tests/test_pairs_actor_critic_paper.py`、`tests/test_logistic_direction_paper.py` 与 `tests/test_time_sliced_paper.py`。Yang 与 Malik 的案例使用合成日线、20 期窗口和线性 A2C；另外两个案例分别使用公开 AAPL 代理日线与简化的 TWAP 市价子单，均不宣称重现原论文市场或实验绩效。

18 个策略对应 16 个不同的主要论文网址，Double Q 样例另需一篇 van Hasselt 算法原文。`research/source-lock.json` 分别固定 16 个主要来源和 1 个辅助算法来源的获取地址、标题、PDF 字节 SHA-256 与本地位置；对未确认再分发许可的 14 份 PDF，`research/source-cache/` 是明确的本地准备目录，不进入 Git。先在主机上显式取得，再离线验证所有字节和首页标题：

```bash
uv run --no-sync python scripts/verify_paper_sources.py \
  --fetch --output runs/paper-sources.json
uv run --no-sync python scripts/verify_paper_sources.py \
  --output runs/paper-sources-offline.json
```

来源锁定本身只证明取得了与研究声明匹配的可提取原文。`research/recipes/` 的三份入库 PDF recipe 与 `research/deep-recipes/` 的十五份本地来源 recipe 共同覆盖全部 18 个策略；每项主张同时具备页级锚点、方法步骤、实现符号、适配差异和绑定的独立 oracle。`reinforcement.double_q_market_making` 分别用 Spooner 等人的论文锚定做市环境、用 van Hasselt 原论文锚定双表算法。页级文字仍不能代替语义判断或完整原实验复现。

新增的[Busseti 与 Boyd VWAP 执行论文](https://arxiv.org/abs/1509.08503)不属于上述 16 个主要来源。`research/independent/busseti-static-vwap/` 固定 PDF 哈希、原文第 6、8 页锚点及人工审阅的静态常数价差公式输入。`scripts/build_busseti_case.py` 核验实际 PDF 后，根据审阅参数生成目录外策略包、方法 spec、recipe 和构建收据；随后 `paper-run` 核验论文到策略包绑定并执行回测。原论文的动态随机控制、原始 NYSE 数据和业绩没有复现。论文许可证未授权仓库再分发 PDF，因此该案例在主机上显式下载固定字节后执行：

```bash
uv run --no-sync python scripts/build_busseti_case.py \
  --fetch --output runs/busseti-build
uv run --no-sync python -m paperquant.cli paper-run \
  --recipe runs/busseti-build/recipe.json \
  --dataset examples/independent_busseti/dataset.json \
  --events examples/independent_busseti/events.json \
  --output runs/busseti-run
```

生成策略以原文式 (16) 的预期成交量份额拆单；五根明确标记的合成分钟线仅用于验证接口、订单和下一开盘成交。参数化独立 oracle 测试多种母单和成交量剖面，并检查总量守恒及完成后不再下单。这里的“从论文到代码”包括可执行的来源核验与结构化 spec 编译，但 spec 是人工审核的，不宣称机器已自主理解未知论文。

```bash
uv run --no-sync python -m paperquant.cli paper-run \
  --recipe research/recipes/yang-malik-rl1.json \
  --dataset examples/pairs_actor_critic/dataset.json \
  --events examples/pairs_actor_critic/events.json \
  --training examples/pairs_actor_critic/training.json \
  --policy examples/pairs_actor_critic/policy.json \
  --output runs/paper-pair
```

`paper-run` 先核对原文、人工 recipe、结构化方法 spec、策略源码和主张，再调用通用运行时，最后仅凭制品冷启动回放；同一次尝试的 `paper-run.json` 记录论文、spec、主张、源码、报告和 bundle 的哈希。对单独的来源检查，仍可使用 `paper-check`。基础严格容器验收保持三篇可直接离线分发的 PDF 深度案例；增强的论文深度验收在显式来源准备后逐项运行全部 18 个 catalog 策略，并另行生成独立新论文案例的构建、oracle、回测收据：

```bash
uv run --no-sync python -m scripts.accept_paper_depth \
  --fetch --output runs/paper-depth
```

`paper-depth.json` 启动时立即改为本轮 `running`，任何失败都会发布 `failed`，每轮产物隔离在新 attempt 目录。通过收据包含 18 条 `paper-run` 的 recipe、原文、spec 与 bundle 哈希，16 个主要来源、1 个算法辅助来源、1 个全新论文来源以及本轮 oracle JUnit 哈希。增强验收要求网络只用于主机上显式取得尚未缓存的公开原文；验收和策略执行阶段均可离线。基础验收仍另有两套基于同一入库论文的目录外动态策略测试，它们证明目录外接入，但不能充当新论文来源。

该收据还包含逐策略[四轴证据口径](evidence-axes.md)的矩阵哈希和汇总、八条公开 AAPL 的实际 `paper-run` 证据，以及本轮主机论文接入的 R1 范围。均线和突破的目录夹具为合成数据，因此另外运行固定公开行情补充案例；其余六条公开轨迹由目录论文运行直接验证。严格容器的 R2 只由 `verify_strict.py` 成功收据授予它实际执行的 18 个目录策略，不能沿用到尚未在容器内执行的全论文接入链路。

新论文无需修改运行时核心。对公开 PDF、HTML、TeX 输入，`paperquant.papers.extract_text` 提供显式文本提取；PDF 保留物理页号，HTML 和 TeX 只给一个文本块。扫描版 PDF 没有可提取文字时会失败，不会假装已 OCR；HTML 不执行脚本，TeX 不展开复杂宏或证明公式语义。可用 `scripts/fetch_paper.py --url HTTPS_URL --sha256 EXPECTED_HASH --output FILE` 显式下载，只有哈希匹配才发布文件。默认验收不联网；要纳入离线验收，须按再分发许可把固定来源、人工 recipe、主张与独立策略包一同提供。

验收把每项人工主张的 spec 步骤、实际实现符号、固定策略源码、独立 pytest 节点和本轮 JUnit 绑定；同一项运行另产生包含原始行情、训练请求、模型与完整轨迹的可离线校验 bundle，并在不训练的全新实例中回放。PDF 页级锚点和 oracle 一起构成当前人工复现链路；它不表示系统已能从全文自动生成代码，或证明与原论文全部实验一致。`MethodSpec` 是经过审阅的输入和核验约束，不是假装由全文机器抽取得到的算法真值。

本项目并不宣称对任意未知论文自动完成策略恢复，也不把 18 个策略误写成 18 篇互不相同的主要论文。当前有 16 篇主要来源、一篇 Double Q 算法辅助来源，以及一篇目录外独立新论文，共 18 篇不同来源；它们在验收中承担不同角色。目录中的 18 项计数是可运行、各具逐主张原文证据和方法 oracle 的策略计数。论文原实验数据和业绩没有在这里重现，具体差异以各 `research/claims/*.json` 为准。
