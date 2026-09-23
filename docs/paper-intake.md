# 公开论文接入与证据边界

论文接入有三个明确步骤：先取得并固定公开论文原文件，再人工审阅论文的方法并写入 `Recipe` 和论文主张，最后让经审阅的策略包通过统一训练、推理和回测契约。文本提取器不会自行决定哪个公式适合交易，也不会自动生成策略代码；页级锚点只能证明指定文本确实出现在固定来源中，不能替代算法 oracle、数据时间切分或人工判断。

本仓库内有一个可离线重演的真实来源：[Yang 与 Malik 的 CC BY 4.0 论文](https://arxiv.org/pdf/2407.16103)。原 PDF、取得日期、许可证和 SHA-256 见 `research/sources/SOURCE.json`；`research/recipes/yang-malik-rl1.json` 将第 7、8、9、12 页的四个文本锚点和 RL1 的离散配对动作、训练要求、策略包源码哈希绑定。`research/claims/pairs_actor_critic.json` 单独披露了合成日线、20 期窗口、线性 A2C、训练奖励与下一开盘双腿成交相对原实验的差异。`tests/test_paper_intake.py` 与 `tests/test_pairs_actor_critic_paper.py` 分别验证原文输入/绑定和数学更新/完整运行。

```bash
uv run --no-sync python -m paperquant.cli paper-check \
  --recipe research/recipes/yang-malik-rl1.json \
  --output runs/paper-check
uv run --no-sync python -m paperquant.cli run \
  --package strategies/reinforcement.pairs_actor_critic \
  --dataset examples/pairs_actor_critic/dataset.json \
  --events examples/pairs_actor_critic/events.json \
  --training examples/pairs_actor_critic/training.json \
  --output runs/paper-pair
```

新论文无需修改运行时核心。对公开 PDF、HTML、TeX 输入，`paperquant.papers.extract_text` 提供显式文本提取；PDF 保留物理页号，HTML 和 TeX 只给一个文本块。扫描版 PDF 没有可提取文字时会失败，不会假装已 OCR；HTML 不执行脚本，TeX 不展开复杂宏或证明公式语义。可用 `scripts/fetch_paper.py --url HTTPS_URL --sha256 EXPECTED_HASH --output FILE` 显式下载，只有哈希匹配才发布文件。默认验收不联网；要纳入离线验收，须按再分发许可把固定来源、人工 recipe、主张与独立策略包一同提供。

本项目并不宣称对任意未知论文自动完成策略恢复，也不把一个 PDF 的文本锚点计作 18 份独立论文复现。其他 17 个策略有来源、实现边界与独立 oracle 绑定，但目前没有把其全文都作为离线原文件打包；目录中的 18 项计数是可运行且具有论文方法依据的策略计数。论文原实验数据和业绩没有在这里重现，具体差异以各 `research/claims/*.json` 为准。
