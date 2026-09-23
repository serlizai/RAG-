"""手动运行旧查询图的演示脚本。

只有显式运行此文件才会请求模型及数据库；测试发现或导入不会发起查询。
"""

import json


def main() -> None:
    from app.core.logger import logger
    from app.query_process.agent.main_graph import query_app
    from app.query_process.agent.state import create_query_default_state

    logger.info("===== 开始旧查询图演示 =====")
    initial_state = create_query_default_state(
        session_id="test_001", original_query="华为P60怎么样?"
    )
    final_state = None

    for event in query_app.stream(initial_state):
        for key, value in event.items():
            logger.info(f"节点: {key}, 输出结果: {value}")
            final_state = value

    logger.info(f"最终节点结果: {json.dumps(final_state, indent=4, ensure_ascii=False)}")
    logger.info("图结构:")
    query_app.get_graph().print_ascii()
    logger.info("===== 演示结束 =====")


if __name__ == "__main__":
    main()
