import os
from neo4j import GraphDatabase, Driver

_neo4j_driver = None    
def get_neo4j_driver() -> Driver:
    """
    获取 Neo4j 驱动实例
    """
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")))
    return _neo4j_driver