# 知识图谱驱动的RAG聊天机器人系统

## 项目简介

本项目是一个基于知识图谱的检索增强生成（GraphRAG）聊天机器人系统，结合了向量检索和知识图谱检索技术，提供更准确和相关的问答服务。

## 核心功能

### 1. 文档向量化处理 (`embedding.py`)
- 支持多种文档格式：PDF、DOCX、Markdown、TXT
- 异步并发处理，提高处理效率
- 支持OpenAI和HuggingFace两种嵌入模型
- 自动连接到Milvus向量数据库

### 2. 知识图谱构建与检索 (`graphRAG_query.py`)
- 智能实体提取：支持英文专有名词、中文实体、技术术语等
- 关系提取：基于上下文和句法结构分析实体间关系
- 混合检索策略：结合向量检索和知识图谱检索
- 支持Neo4j和NetworkX两种图谱存储方式
- 社区发现算法：使用Leiden算法进行层次聚类

## 技术架构

### 核心技术栈
- **LLM**: DeepSeek API
- **嵌入模型**: OpenAI text-embedding-3-large / BAAI/bge-large-zh-v1.5
- **向量数据库**: Milvus
- **图谱数据库**: Neo4j / NetworkX
- **框架**: LangGraph, LangChain

### 检索策略
1. **向量检索**: 基于语义相似度的文档检索
2. **知识图谱检索**: 基于实体关系的结构化检索
3. **混合检索**: 结合两种策略的并行检索

## 安装与配置

### 环境要求
- Python 3.8+
- Milvus向量数据库
- Neo4j数据库（可选）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 环境变量配置
创建 `.env` 文件并配置以下变量：

```env
# DeepSeek配置
DEEPSEEK_MODEL=your_model_name
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=your_api_base_url

# OpenAI配置（可选）
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=your_openai_api_base

# Neo4j配置（可选）
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

## 使用方法

### 1. 文档向量化处理
```bash
python embedding.py
```
按提示输入：
- 文件路径（支持多个文件，空格分隔）
- 数据库集合名称

### 2. 启动GraphRAG聊天机器人
```bash
python graphRAG_query.py
```
按提示输入：
- 数据库集合名称
- 是否使用Neo4j存储知识图谱
- 是否启用交互学习模式

### 3. 查询模式切换
在聊天过程中，可以使用以下命令切换检索模式：
- `vector`: 纯向量检索
- `kg`: 纯知识图谱检索
- `hybrid`: 混合检索（默认）

## 功能特性

### 知识图谱构建
- **实体提取**: 多层次实体识别算法
  - 英文专有名词识别
  - 中文实体提取（括号、引号内容）
  - 技术术语和产品名称识别
  - 基于句法分析的主语-宾语提取

- **关系提取**: 智能关系识别
  - 上下文关系分析
  - 关系指示词匹配
  - 句法结构分析
  - 关键词触发机制

### 检索增强
- **混合检索**: 结合向量和图谱检索
- **社区发现**: Leiden算法聚类
- **桥接节点识别**: 跨社区连接分析
- **动态学习**: 从用户交互中学习新知识

### 交互功能
- **知识图谱浏览**: 可视化实体和关系
- **人工审核**: 检索结果人工干预
- **学习确认**: 新知识添加确认机制
- **多轮对话**: 支持上下文记忆
- **历史回顾**: 查看和搜索历史对话记录
- **智能提问**: 自动生成相关补充问题

# 部署指南

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/suansuan12321/MVP-langgraph.git
cd knowledge-graph-rag-bot
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，填入API密钥
```

### 4. 启动Milvus服务
```bash
# 使用Docker启动Milvus
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

### 5. 启动Neo4j服务（可选）
```bash
# 使用Docker启动Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 6. 运行程序
```bash
# 第一步：处理文档
python embedding.py

# 第二步：启动聊天机器人
python graphRAG_query.py
```
```
