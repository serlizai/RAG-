"""手动运行的只读连接检查：验证服务连接和 NEO4J_DATABASE 指定的数据库。"""

import os

from dotenv import load_dotenv
from neo4j import READ_ACCESS

from app.clients.neo4j_utils import get_neo4j_driver


def main():
    load_dotenv()
    database = os.getenv("NEO4J_DATABASE") or "neo4j"
    with get_neo4j_driver() as driver:
        driver.verify_connectivity()
        # 能连接服务不代表配置的数据库存在；必须指定与图谱读写一致的数据库。
        with driver.session(database=database, default_access_mode=READ_ACCESS) as session:
            session.run("RETURN 1 AS ok").consume()
    print(f"Neo4j 数据库 {database!r} 连接检查通过（只读查询）")


if __name__ == "__main__":
    main()
