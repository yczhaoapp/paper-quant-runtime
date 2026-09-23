# 策略隔离和严格验收

严格模式先在主机核对数据、策略清单和运行计划，再把单独的策略目录只读挂载给容器 worker。worker 只能通过 JSON 行协议接收训练请求或一条行情与账户快照，并返回模型字节或动作。行情编译、模型落盘、撮合、账户和报告由主机运行时完成；报告目录和主机模型目录不挂载到 worker。直接调用进程内 `run` 无法声明严格模式，严格入口是策略目录的 `run_package` 或 CLI。

运行时使用已检查的不可变镜像 ID 启动 worker，并在握手后读取 Docker 的实际容器配置。检查项包括断网、只读根目录、无 Linux capability、禁止提权、UID/GID 65532、只读策略目录、256 MiB 内存、64 个进程、0.5 CPU 和 16 MiB 的 noexec 临时目录。检查失败会返回 `SANDBOX_UNAVAILABLE`，不会转到进程内执行。静态源码扫描只是提前拒绝明显不符合策略接口的代码，不被当作安全边界。有关 Docker 只读根目录与 bind mount 的语义可参见 [Docker run 参考](https://docs.docker.com/reference/cli/docker/container/run) 和 [Docker bind mount 文档](https://docs.docker.com/engine/storage/bind-mounts/)。

容器允许策略读取其自身镜像内的非敏感系统文件，也可能在资源限制内启动子进程；本项目不声称 Python 进程内部完全禁用这些行为。隔离目标是阻断策略访问主机的私有数据、网络、报告和模型挂载，并让越界尝试失败。主机运行时、Docker daemon 和本地用户账户属于可信计算基。论文内容的正确复现须由单独的主张、公式和实验验证证明，沙箱收据不替代这些证据。

使用固定基础镜像摘要和固定 worker 依赖构建镜像。构建阶段仍可能访问包索引；运行阶段只使用本地镜像 ID，不需要下载依赖。完整本地门禁：

```bash
uv sync --locked --extra dev
uv run --no-sync python scripts/verify_strict.py --require-clean --output runs/strict-verification
```

脚本每次先原子发布新的 `running` 尝试并使旧成功结果失效，再无缓存构建、核查镜像、运行全部主机与严格容器测试，最后发布 `passed` 或 `failed`。`verification.json`、`image.json` 通过相同的 `attempt_id` 关联；成功收据还记录 Git commit、源码树、镜像与完整日志哈希。发布时增加 `--require-clean`，要求验证前后均为同一已提交的干净源码树。验收方应要求本轮 `verification.json.status == "passed"`，并核对 `attempt_id`、源码哈希与日志，而不是只看目录中是否存在旧报告。
