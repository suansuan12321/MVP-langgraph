#!/usr/bin/env python
# init.py - GraphRAG系统初始化脚本

import os
import sys
import time
import logging
import asyncio
import requests
from dotenv import load_dotenv
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/init.log')
    ]
)

# 加载环境变量
load_dotenv()

# 创建必要的目录
def create_directories():
    """创建必要的目录结构"""
    dirs = ['data', 'logs', 'milvus/data', 'milvus/conf', 'milvus/logs', 
            'neo4j/data', 'neo4j/logs', 'neo4j/import']
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logging.info("目录结构创建完成")

# 检查环境变量
def check_environment():
    """验证必要的环境变量是否已设置"""
    required_vars = {
        "DEEPSEEK_API_KEY": "DeepSeek API密钥",
        "DEEPSEEK_BASE_URL": "DeepSeek API基础URL",
        "DEEPSEEK_MODEL": "DeepSeek模型名称"
    }
    
    missing = []
    for var, desc in required_vars.items():
        if not os.environ.get(var):
            missing.append(f"{var} ({desc})")
    
    if missing:
        logging.error(f"缺少必要的环境变量: {', '.join(missing)}")
        print(f"请在.env文件中设置以下环境变量: {', '.join(missing)}")
        return False
    
    logging.info("环境变量检查通过")
    return True

# 等待服务就绪
def wait_for_service(service_name, url, max_retries=30, retry_interval=2):
    """等待服务就绪"""
    logging.info(f"等待{service_name}服务就绪...")
    
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logging.info(f"{service_name}服务已就绪")
                return True
        except requests.RequestException:
            pass
        
        logging.info(f"等待{service_name}服务... ({i+1}/{max_retries})")
        time.sleep(retry_interval)
    
    logging.error(f"{service_name}服务未就绪，超过最大重试次数")
    return False

# 初始化Milvus
async def initialize_milvus():
    """初始化Milvus向量数据库"""
    try:
        from pymilvus import connections, utility
        
        # 等待Milvus服务就绪
        milvus_host = os.environ.get("MILVUS_HOST", "milvus")
        milvus_port = os.environ.get("MILVUS_PORT", "19530")
        
        logging.info(f"连接Milvus数据库 ({milvus_host}:{milvus_port})...")
        
        # 重试连接
        max_retries = 10
        for i in range(max_retries):
            try:
                connections.connect("default", host=milvus_host, port=milvus_port)
                logging.info("Milvus连接成功")
                break
            except Exception as e:
                if i == max_retries - 1:
                    logging.error(f"Milvus连接失败: {e}")
                    return False
                logging.info(f"尝试连接Milvus... ({i+1}/{max_retries})")
                await asyncio.sleep(5)
        
        # 检查默认集合
        default_collections = ["default_collection", "demo_collection"]
        existing_collections = utility.list_collections()
        
        logging.info(f"现有集合: {existing_collections}")
        
        if not set(default_collections).intersection(set(existing_collections)):
            logging.info("默认集合不存在，请在启动后使用embedding.py创建")
        
        connections.disconnect("default")
        return True
    except Exception as e:
        logging.error(f"Milvus初始化失败: {e}")
        return False

# 初始化Neo4j
async def initialize_neo4j():
    """初始化Neo4j图数据库"""
    try:
        from neo4j import GraphDatabase
        
        uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
        user = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password")
        
        # 重试连接
        max_retries = 10
        for i in range(max_retries):
            try:
                driver = GraphDatabase.driver(uri, auth=(user, password))
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                    count = result.single()["count"]
                    logging.info(f"Neo4j连接成功，当前节点数: {count}")
                driver.close()
                return True
            except Exception as e:
                if i == max_retries - 1:
                    logging.error(f"Neo4j连接失败: {e}")
                    return False
                logging.info(f"尝试连接Neo4j... ({i+1}/{max_retries})")
                await asyncio.sleep(5)
    except Exception as e:
        logging.error(f"Neo4j初始化失败: {e}")
        return False

# 检查模型文件
def check_model_files():
    """检查并下载必要的模型文件"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        logging.info("检查嵌入模型...")
        
        # 检查BGE模型
        model_name = "BAAI/bge-large-zh-v1.5"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            logging.info(f"成功加载{model_name}模型")
            return True
        except Exception as e:
            logging.warning(f"自动下载{model_name}模型: {e}")
            return False
    except ImportError:
        logging.warning("缺少torch或transformers库，将使用在线API进行嵌入")
        return True

# 主函数
async def main():
    """主函数，执行初始化流程"""
    logging.info("开始GraphRAG初始化流程...")
    
    # 1. 创建目录
    create_directories()
    
    # 2. 检查环境变量
    if not check_environment():
        sys.exit(1)
    
    # 3. 检查模型文件
    check_model_files()
    
    # 4. 初始化服务
    milvus_ok = await initialize_milvus()
    neo4j_ok = await initialize_neo4j()
    
    # 5. 总结初始化状态
    if milvus_ok and neo4j_ok:
        logging.info("初始化完成，GraphRAG系统准备就绪")
        print("\n" + "="*50)
        print("GraphRAG系统初始化完成!")
        print("您现在可以运行以下命令开始使用:")
        print("1. 处理文档: python embedding.py")
        print("2. 启动问答系统: python graphRAG_query.py")
        print("="*50 + "\n")
    else:
        logging.warning("部分服务初始化失败，系统可能无法正常工作")
        if not milvus_ok:
            print("Milvus服务初始化失败，请检查日志")
        if not neo4j_ok:
            print("Neo4j服务初始化失败，请检查日志")

if __name__ == "__main__":
    asyncio.run(main())