# RAG

[Neo4j 知识图谱接入说明](docs/neo4j_knowledge_graph.md)：导入与召回流程、数据结构、关键函数、事务行为及测试说明。

## 本地启动

```bash
uv sync --locked
uv run --locked python -m app.query_process.api.query_server
```

问答页面：http://127.0.0.1:8004/chat.html 。在另一个终端启动文档导入服务：

```bash
uv run --locked python -m app.import_process.api.import_server
```

导入页面：http://127.0.0.1:8001/import.html 。启动服务需要根目录 `.env` 中的实际模型、数据库及存储配置。

## 测试

```bash
uv run --locked python -m scripts.run_tests unit
```

默认离线测试不读取业务 `.env`，不访问模型或数据库。数据库集成、模型和 MinerU 冒烟等命令见[测试说明](docs/testing.md)。
