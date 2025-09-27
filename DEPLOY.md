# GraphRAG 部署文档

GraphRAG是一个基于知识图谱的检索增强生成系统，结合了向量检索和知识图谱检索技术，提供智能问答服务。本文档提供完整的部署指南。


## 快速部署

### 1. 克隆代码库

```bash
git clone https://github.com/your-org/graphrag.git
cd graphrag
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

### 3. 启动服务

使用Docker Compose启动所有服务:

```bash
docker-compose up -d
```

首次启动时，系统将:
- 构建Docker镜像
- 启动Milvus向量数据库
- 启动Neo4j图数据库
- 初始化GraphRAG服务

### 4. 查看服务状态

```bash
docker-compose ps
```

所有服务状态应为`Up`:
```
Name Command State Ports
graphrag python init.py python gra ... Up 0.0.0.0:8501->8501/tcp

milvus /tini -- milvus run standalone Up 0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp

neo4j /sbin/tini -g -- /docker ... Up 0.0.0.0:7474->7474/tcp, 7473/tcp, 0.0.0.0:7687->7687/tcp
```

### 5. 初始化知识库

首次启动后，您需要处理文档并构建知识库:

```bash
# 进入GraphRAG容器
docker-compose exec graphrag bash

# 处理文档
python embedding.py

# 按照提示输入:
# - 文件路径 (例如: /app/data/document.pdf)
# - 集合名称 (例如: my_collection)
```

### 6. 启动问答系统

```bash
# 在GraphRAG容器内
python graphRAG_query.py

# 按照提示输入:
# - 集合名称 (与上一步相同)
# - 是否使用Neo4j (输入y)
# - 是否启用交互学习 (推荐输入y)
# - 是否启用性能监控 (推荐输入y)
```

## 组件详解

### GraphRAG服务

- **功能**: 核心系统，处理文档、创建知识图谱、提供问答功能
- **端口**: 8501
- **日志位置**: ./logs/
- **数据位置**: ./data/

### Milvus向量数据库

- **功能**: 存储和检索文档的向量表示，支持语义搜索
- **端口**: 19530 (服务)、9091 (管理界面)
- **数据位置**: ./milvus/data/
- **管理界面**: http://localhost:9091

### Neo4j图数据库

- **功能**: 存储知识图谱，支持结构化查询
- **端口**: 7474 (Web界面)