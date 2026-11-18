# 测试入口

从项目根目录执行。已有测试继续使用标准库 `unittest`，统一入口负责配置隔离、测试分组、超时与退出码。

```bash
uv sync --locked
uv run --locked python -m scripts.run_tests
```

无参数默认执行 `unit`。入口在新的 Python 进程中执行测试，避免上一次运行缓存的业务配置影响结果；同时显式设置当前检出目录为 `PROJECT_ROOT`，不依赖 `.env` 文件定位项目根目录。

## 离线测试

```bash
uv run --locked python -m scripts.run_tests unit
uv run --locked python -m scripts.run_tests unit --verbose
uv run --locked python -m scripts.run_tests unit --list
```

离线入口不读取项目 `.env`，覆盖继承的业务配置，使用虚拟凭证和 CPU 配置，并禁止下载模型。Python socket 及 Milvus gRPC 连接入口均被拦截；即使业务代码捕获了连接异常，整个入口仍返回失败。测试不需要本机 MongoDB、MinIO、Neo4j、Milvus 或模型权重。

测试通过自动发现加入。普通 `unittest.TestCase` 默认属于 `unit`；依赖真实数据库的测试类设置 `test_scope = "integration"`。因此新增普通测试文件会自动进入默认测试集，无需维护临时模块名单。

`test_import_main_graph.py`、`test_query_main_graph.py` 和 `test_neo4j.py` 是手动演示/诊断脚本。导入和测试收集不会执行其中的真实工作流。

## 数据库集成测试

```bash
cp .env.test.example .env.test.local
# 修改 .env.test.local，指向已启动的隔离测试实例。
uv run --locked python -m scripts.run_tests integration --env-file .env.test.local
```

也可以通过环境变量显式提供 `NEO4J_KG_TEST_URI`、`NEO4J_KG_TEST_PASSWORD` 和 `MILVUS_KG_TEST_URI`。入口不读取业务 `.env`，也不会把 `NEO4J_URI`、`NEO4J_PASSWORD` 或 `MILVUS_URL` 自动当成测试配置。显式文件仅导入上述三个测试字段。

目前有 14 项真实数据库测试，使用 `neo4j` 用户和 `neo4j` 数据库。Milvus 创建随机集合，Neo4j 使用随机商品名，结束时清理本次数据。测试实例需允许建集合、图谱约束和读写。缺少配置时直接返回配置错误；发生跳过时也不会把集成测试记为完成。

只查看用例不需要连接数据库：

```bash
uv run --locked python -m scripts.run_tests integration --list
```

## 在线冒烟

只有显式选择 `smoke` 才会读取真实服务配置并发起在线请求。默认读取项目根目录 `.env`，也可用 `--env-file` 指定其他文件；文件中的配置优先于继承的同名环境变量。

```bash
# 向模型服务发送两轮简短请求，验证工具调用。
uv run --locked python -m scripts.run_tests smoke --check model

# 将指定 PDF 发送到 MinerU，验证上传、解析和 Markdown 下载。
uv run --locked python -m scripts.run_tests smoke --check mineru --pdf /path/to/sample.pdf

# 两项都执行。
uv run --locked python -m scripts.run_tests smoke --check all --pdf /path/to/sample.pdf
```

MinerU 冒烟只在临时目录保存解析结果，不写入知识库。模型及 MinerU 检查均可能消耗已配置服务的额度。日志默认关闭，入口只打印检查结果，不输出凭证或完整上游错误体。

所有入口默认有 180 秒总超时，可以用 `--timeout 300` 调整；超时返回 124。正常通过返回 0，测试或在线检查失败返回非零，缺少配置/参数错误返回 2。

## 手动导入演示

完整业务导入会调用外部服务并写入知识库，必须显式提供文件：

```bash
uv run --locked python -m app.test.test_import_main_graph /path/to/document.pdf
```

## 后续 CI

CI 的默认命令与本地保持一致：

```bash
uv run --locked python -m scripts.run_tests unit
```

它不需要真实 `.env`、模型文件或服务密钥。数据库测试和在线冒烟应作为独立任务，使用显式配置，不混入默认离线检查。本次只整理测试入口，尚未添加 GitHub Actions 工作流。
