import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_deepseek import ChatDeepSeek
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
import asyncio
import time
import logging
from langchain_community.graphs import NetworkxEntityGraph
from langchain.schema import Document
import networkx as nx
import re
import pickle
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
import leidenalg as la
import igraph as ig
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

## 环境变量
load_dotenv()
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
deepseek_api_base = os.environ.get("DEEPSEEK_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_base = os.environ.get("OPENAI_API_BASE")
neo4juri = os.environ.get("NEO4J_URI")
neo4jusername = os.environ.get("NEO4J_USERNAME")
neo4jpassword = os.environ.get("NEO4J_PASSWORD")


class RAGState(TypedDict, total=False):
    query: str
    history: str
    docs: list
    enhanced_docs: list
    reranked_docs: list
    kg_docs: list
    found_docs: bool
    search_type: str  # 检索类型: "vector"-向量检索, "kg"-知识图谱检索, "hybrid"-混合检索
    answer: str


# 定义读写pickle函数
def read_gpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_gpickle(G, path):
    with open(path, 'wb') as f:
        pickle.dump(G, f)


def extract_entities(text):
    """实体提取函数"""
    import jieba
    import jieba.posseg as pseg

    start_time = time.time()

    if not text or len(text.strip()) == 0:
        return []
    entities = []

    # 1. 提取专有名词（英文）
    english_entities = re.findall(r'\b[A-Z][a-zA-Z0-9\-]*([ ][A-Z][a-zA-Z0-9\-]*)*\b', text)

    # 2. 提取中文关键实体
    # 2.1 括号内容
    cn_bracketed = re.findall(r'[《【「（](.*?)[》】」）]', text)
    # 2.2 引号内容
    cn_quoted = re.findall(r'[\"\'](.*?)[\"\']', text)

    # 3. 提取特定格式的实体
    patterns = [
        # 产品/系统名称
        r'(\w+[系统平台软件产品应用服务][V\d\.]*)',
        # 功能模块
        r'([\u4e00-\u9fa5]{2,6}[功能模块组件接口])',
        # 技术术语
        r'([A-Z][A-Za-z]+(?:\s[A-Z][A-Za-z]+)*)',
        # 数字编号
        r'(版本\s*\d+\.\d+)',
        # 角色名称
        r'([\u4e00-\u9fa5]{2,4}[员师长])',
        # 流程名称
        r'([\u4e00-\u9fa5]{2,8}流程)',
        # 接口名称
        r'([A-Za-z]+[Aa]pi|[A-Za-z]+[Ii]nterface)',
    ]

    pattern_entities = []
    for pattern in patterns:
        pattern_entities.extend(re.findall(pattern, text))

    # 4. 使用句法规则识别主语-谓语-宾语结构中的主语和宾语
    sentences = re.split(r'[。；！？.;!?\n]', text)
    svo_entities = []
    for sentence in sentences:
        if len(sentence) < 5:  # 忽略太短的句子
            continue

        # 尝试找到主语（句子开头的名词短语）
        subject_match = re.search(r'^([\u4e00-\u9fa5A-Za-z0-9]{2,10})', sentence.strip())
        if subject_match:
            svo_entities.append(subject_match.group(1))

        # 尝试找到宾语（动词后面的名词短语）
        verb_obj_matches = re.findall(r'[是为具有包含提供支持]([\u4e00-\u9fa5A-Za-z0-9]{2,15})', sentence)
        svo_entities.extend(verb_obj_matches)

    # 5. 使用关键词触发器查找特定领域实体
    domain_keywords = {
        '功能': r'([\u4e00-\u9fa5]{2,8}功能)',
        '模块': r'([\u4e00-\u9fa5]{2,8}模块)',
        '数据': r'([\u4e00-\u9fa5]{2,8}数据)',
        '界面': r'([\u4e00-\u9fa5]{2,8}界面)',
        '服务': r'([\u4e00-\u9fa5]{2,8}服务)',
        '流程': r'([\u4e00-\u9fa5]{2,8}流程)',
        '角色': r'([\u4e00-\u9fa5]{2,8}角色)',
    }

    domain_entities = []
    for keyword, pattern in domain_keywords.items():
        if keyword in text:
            domain_entities.extend(re.findall(pattern, text))

    # 6. 使用jieba进行分词，增强实体识别
    jieba_entities = []
    try:
        # 先添加已识别的实体到jieba词典，提高分词准确率
        initial_entities = english_entities + cn_bracketed + cn_quoted + pattern_entities
        for entity in initial_entities:
            if len(entity) >= 2:
                jieba.add_word(entity)

        # 对文本进行分词和词性标注
        words = pseg.cut(text)

        # 提取特定词性的词作为实体
        for word_pair in words:
            word = word_pair.word
            flag = word_pair.flag
            # 名词、机构名、地名、专有名词等可能是实体
            if (flag.startswith('n') or flag in ['nt', 'nz', 'ns', 'nr']) and len(word) >= 2:
                jieba_entities.append(word)

            # 专业术语可能是形容词+名词结构
            elif flag == 'a' and len(word) >= 2:
                # 找到后续词
                text_remainder = text[text.find(word) + len(word):]
                if text_remainder:
                    next_word_pairs = list(pseg.cut(text_remainder, HMM=False))
                    if next_word_pairs:
                        next_word = next_word_pairs[0].word
                        next_flag = next_word_pairs[0].flag
                        if next_flag.startswith('n') and len(next_word) >= 1:
                            compound = word + next_word
                            if 2 <= len(compound) <= 8:  # 限制长度
                                jieba_entities.append(compound)
    except Exception as e:
        logging.warning(f"jieba分词出错: {e}")

    # 合并所有实体
    all_entities = english_entities + cn_bracketed + cn_quoted + pattern_entities + svo_entities + domain_entities
    filtered_entities = []

    # 创建停用词集合以加速查找
    stop_words = set(
        ["若", "如", "的", "是", "在", "有", "和", "与", "或", "之", "用户", "如何", "进行", "如果", "可以", "怎么"])

    # 过滤和清洗实体-清洗空白字符、标点；太短或太长的实体；过滤停用词
    seen_entities = set()  # 用于去重
    for entity in all_entities:
        entity = entity.strip()
        # 1. 基本过滤
        if len(entity) < 2 or len(entity) > 30 or entity in stop_words or entity in seen_entities:
            continue

        # 2. 模式过滤 - 拒绝某些特定模式
        # 过滤仅包含数字和标点的实体
        if re.match(r'^[\d\s.,;:!?()[\]{}\"\'<>@#$%^&*_\-+=|\\~`]+$', entity):
            continue

        # 3. 过滤短词组合 (如"的是", "和在"等)
        if len(entity) <= 4 and all(word in stop_words for word in jieba.cut(entity)):
            continue

        # 4. 过滤嵌套实体 (如果较短实体完全包含在其他实体中)
        if any(entity != other_entity and entity in other_entity for other_entity in all_entities):
            # 如果这个实体完全包含在另一个实体中，且长度差异较大，则跳过
            if any(entity != other_entity and entity in other_entity and len(other_entity) >= len(entity) * 1.5 for
                   other_entity in all_entities):
                continue

        seen_entities.add(entity)
        filtered_entities.append(entity)

    # 对提取的实体进行评分，保留更有可能是真实实体的结果
    scored_entities = []
    for entity in filtered_entities:
        score = 0

        # 1. 频率评分 - 在文本中出现次数
        occurrences = text.count(entity)
        score += min(occurrences, 3)  # 最多加3分

        # 2. 位置评分 - 如果出现在括号或引号中更可能是实体
        if any(entity in item for item in cn_bracketed + cn_quoted):
            score += 2

        # 3. 词性评分 - 通过jieba确认的专有名词更可能是实体
        if entity in jieba_entities:
            score += 2

        # 4. 形式评分 - 特定格式（如首字母大写）更可能是实体
        if entity[0].isupper():
            score += 1

        # 5. 长度评分 - 适中长度更可能是有意义的实体
        if 2 <= len(entity) <= 6:
            score += 1

        # 保留得分较高的实体
        if score >= 2:  # 设置一个阈值，可以根据需要调整
            scored_entities.append(entity)

    end_time = time.time()
    logging.info(f"实体提取时间: {end_time - start_time:.4f}秒，提取了{len(scored_entities)}个实体")

    return scored_entities


def determine_relation_type(text, entity1, entity2):
    """确定关系类型"""
    text = text.strip()

    if len(text) > 50:
        # 寻找常见的关系指示词
        relation_indicators = [
            "是", "包含", "属于", "组成", "由", "使用", "提供",
            "管理", "控制", "负责", "依赖", "生成", "处理", "支持"
        ]

        # 找到第一个指示词及其上下文
        indicator_pos = -1
        found_indicator = ""

        for indicator in relation_indicators:
            pos = text.find(indicator)
            if pos != -1 and (indicator_pos == -1 or pos < indicator_pos):
                indicator_pos = pos
                found_indicator = indicator

        if indicator_pos != -1:
            # 提取指示词周围的文本
            start = max(0, indicator_pos - 10)
            end = min(len(text), indicator_pos + len(found_indicator) + 10)
            text = text[start:end]

    # 基于常见的关系模式进行匹配
    relation_mapping = {
        "是": "是",
        "为": "是",
        "即": "是",
        "指": "定义为",
        "定义为": "定义为",
        "包含": "包含",
        "拥有": "包含",
        "具有": "包含",
        "属于": "属于",
        "从属于": "属于",
        "组成": "组成",
        "构成": "组成",
        "由": "组成",
        "使用": "使用",
        "应用": "使用",
        "提供": "提供",
        "支持": "支持",
        "依赖": "依赖",
        "基于": "基于",
        "管理": "管理",
        "控制": "控制",
        "负责": "负责",
        "生成": "生成",
        "产生": "生成",
        "输出": "生成",
        "处理": "处理",
        "分析": "处理",
    }

    for key, relation in relation_mapping.items():
        if key in text:
            return relation

    # 设定关系默认值-“相关”
    return "相关"


def extract_relation_from_sentence(sentence, entity1, entity2):
    """提取句子中两个实体之间的关系"""
    # 找出实体在句子中的位置
    try:
        pos1 = sentence.find(entity1)
        pos2 = sentence.find(entity2)

        if pos1 == -1 or pos2 == -1:
            return None

        # 确定哪个实体在前面
        if pos1 < pos2:
            first_entity, second_entity = entity1, entity2
            middle_text = sentence[pos1 + len(entity1):pos2]
        else:
            first_entity, second_entity = entity2, entity1
            middle_text = sentence[pos2 + len(entity2):pos1]

        # 分析中间文本
        relation = determine_relation_type(middle_text, first_entity, second_entity)
        # 如果实体顺序与源目标不一致，可能需要调整关系方向
        if first_entity == entity2:  # 如果第二个实体是源
            # 某些关系需要反转
            inverse_relations = {
                "包含": "属于",
                "属于": "包含",
                "组成": "由组成",
                "由组成": "组成",
                "管理": "被管理",
                "被管理": "管理",
                "控制": "被控制",
                "被控制": "控制",
            }

            if relation in inverse_relations:
                relation = inverse_relations[relation]

        return relation

    except Exception:
        # 设定默认关系
        return "相关"


def extract_relationships(text, entities):
    """提取实体间关系总函数"""
    relationships = []

    # 为实体创建位置索引
    entity_positions = {}
    for entity in entities:
        positions = []
        start = 0
        while True:
            pos = text.find(entity, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(entity)))
            start = pos + 1
        if positions:
            entity_positions[entity] = positions

    # 1. 上下文关系提取
    for entity1 in entities:
        if entity1 not in entity_positions:
            continue

        for entity2 in entities:
            if entity1 == entity2 or entity2 not in entity_positions:
                continue

            # 检查两个实体是否在文本的相近位置
            for pos1_start, pos1_end in entity_positions[entity1]:
                for pos2_start, pos2_end in entity_positions[entity2]:
                    # 计算实体间的最小距离
                    if pos1_end < pos2_start:
                        distance = pos2_start - pos1_end
                        mid_text = text[pos1_end:pos2_start]
                    elif pos2_end < pos1_start:
                        distance = pos1_start - pos2_end
                        mid_text = text[pos2_end:pos1_start]
                    else:
                        # 实体有重叠，跳过
                        continue

                    # 只考虑距离较近的实体对
                    if distance < 100:
                        # 2. 分析中间文本以确定关系类型
                        relation = determine_relation_type(mid_text, entity1, entity2)

                        if relation:
                            relationships.append({
                                "source": entity1,
                                "target": entity2,
                                "relation": relation
                            })
                            # 一旦找到一个关系就停止查找
                            break

                # 如果已经找到关系，不再继续查找
                if any(r["source"] == entity1 and r["target"] == entity2 for r in relationships):
                    break

    # 3. 基于句法结构的关系提取
    sentences = re.split(r'[。；！？.;!?\n]', text)
    for sentence in sentences:
        # 找到句子中的所有实体
        sentence_entities = [e for e in entities if e in sentence]

        # 如果句子包含至少两个实体，尝试提取关系
        if len(sentence_entities) >= 2:
            for i, e1 in enumerate(sentence_entities):
                for j, e2 in enumerate(sentence_entities[i + 1:], i + 1):
                    # 检查这对实体是否已经有关系
                    if any(r["source"] == e1 and r["target"] == e2 for r in relationships) or \
                            any(r["source"] == e2 and r["target"] == e1 for r in relationships):
                        continue

                    # 提取关系
                    relation = extract_relation_from_sentence(sentence, e1, e2)
                    if relation:
                        relationships.append({
                            "source": e1,
                            "target": e2,
                            "relation": relation
                        })

    # 4. 基于关键词触发器的关系提取
    relation_patterns = [
        (r'(\w+)包含(\w+)', '包含'),
        (r'(\w+)属于(\w+)', '属于'),
        (r'(\w+)是(\w+)的一部分', '是组成部分'),
        (r'(\w+)由(\w+)组成', '组成'),
        (r'(\w+)使用(\w+)', '使用'),
        (r'(\w+)依赖于(\w+)', '依赖'),
        (r'(\w+)管理(\w+)', '管理'),
        (r'(\w+)负责(\w+)', '负责'),
    ]

    for pattern, rel_type in relation_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 2:
                source, target = match
                if source in entities and target in entities:
                    relationships.append({
                        "source": source,
                        "target": target,
                        "relation": rel_type
                    })

    return relationships


async def create_knowledge_graph(documents):
    """从文档创建知识图谱"""
    logging.info("创建知识图谱...")
    graph = NetworkxEntityGraph()

    # 记录文档来源，用于添加节点属性
    doc_sources = {}

    # 从文档中提取实体和关系
    total_entities = 0
    total_relationships = 0

    for doc in documents:
        try:
            content = doc.page_content
            source = doc.metadata.get("source", "unknown")

            # 使用增强的实体和关系提取
            entities = extract_entities(content)
            relationships = extract_relationships(content, entities)

            # 添加到图中
            for entity in entities:
                # 添加节点及其属性
                if entity not in graph._graph.nodes:
                    graph._graph.add_node(entity)
                    graph._graph.nodes[entity]["entity_type"] = "concept"  # 使用 entity_type 替代 type
                    graph._graph.nodes[entity]["text"] = content[:500]  # 存储前500个字符作为上下文
                    graph._graph.nodes[entity]["source"] = source
                    graph._graph.nodes[entity]["importance"] = 1  # 初始重要性分数
                else:
                    # 如果节点已存在，增加其重要性分数
                    if "importance" in graph._graph.nodes[entity]:
                        graph._graph.nodes[entity]["importance"] += 1
                    else:
                        graph._graph.nodes[entity]["importance"] = 1

                # 记录实体来源的文档
                if entity not in doc_sources:
                    doc_sources[entity] = []
                doc_sources[entity].append(source)

            # 添加关系
            for rel in relationships:
                source_entity = rel["source"]
                target_entity = rel["target"]
                relation = rel["relation"]

                # 检查两个实体是否已添加到图中
                if source_entity in graph._graph.nodes and target_entity in graph._graph.nodes:
                    # 直接使用 NetworkX 添加边和属性
                    if not graph._graph.has_edge(source_entity, target_entity):
                        graph._graph.add_edge(source_entity, target_entity,
                                              relation=relation,
                                              weight=1,
                                              context=content[:200])
                        total_relationships += 1
                    else:
                        # 如果边已存在，增加权重
                        if "weight" in graph._graph[source_entity][target_entity]:
                            graph._graph[source_entity][target_entity]["weight"] += 1
                        else:
                            graph._graph[source_entity][target_entity]["weight"] = 1

            total_entities += len(entities)

        except Exception as e:
            logging.error(f"处理文档时出错: {e}")
            continue

    try:
        # 计算度中心性，识别重要节点
        centrality = nx.degree_centrality(graph._graph)
        for node, score in centrality.items():
            graph._graph.nodes[node]["centrality"] = score

        # 标记核心节点（度中心性排名前20%的节点）
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        top_n = max(1, int(len(sorted_nodes) * 0.2))  # 至少1个节点
        for node, _ in sorted_nodes[:top_n]:
            graph._graph.nodes[node]["is_core"] = True

    except Exception as e:
        logging.warning(f"应用图分析算法失败: {e}")

    logging.info(f"提取了 {total_entities} 个实体和 {total_relationships} 个关系")

    # 对于大型图谱，可能需要过滤掉一些噪声
    if total_entities > 500:
        logging.info("图谱较大，进行节点过滤...")
        nodes_to_remove = []
        for node, data in graph._graph.nodes(data=True):
            # 移除重要性得分低且连接数少的节点
            if data.get("importance", 0) < 2 and graph._graph.degree(node) < 2:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            graph._graph.remove_node(node)

        logging.info(f"过滤后节点数量: {len(graph._graph.nodes)}")

    return graph

async def apply_hierarchical_clustering(graph):
    """应用Leiden算法进行层次聚类"""
    if not hasattr(graph, '_graph') or not graph._graph:
        logging.warning("知识图谱为空，无法进行聚类")
        return graph
    
    logging.info("开始应用Leiden算法进行层次聚类...")
    
    # 将NetworkX图转换为igraph格式
    nx_graph = graph._graph
    ig_graph = ig.Graph()
    
    # 添加节点
    node_mapping = {}
    for i, node in enumerate(nx_graph.nodes()):
        ig_graph.add_vertex(name=str(node))
        node_mapping[node] = i
    
    # 添加边
    for u, v, data in nx_graph.edges(data=True):
        weight = data.get('weight', 1.0)
        ig_graph.add_edge(node_mapping[u], node_mapping[v], weight=weight)
    
    # 应用Leiden算法
    partitions = la.find_partition(
        ig_graph, 
        la.ModularityVertexPartition, 
        weights='weight',
        n_iterations=10
    )
    
    # 记录每个节点的社区信息
    communities = {}
    for i, community in enumerate(partitions):
        for node_idx in community:
            node_name = list(node_mapping.keys())[list(node_mapping.values()).index(node_idx)]
            communities[node_name] = i
            # 将社区信息保存到节点属性
            nx_graph.nodes[node_name]['community_id'] = i
    
    # 计算社区内部连接紧密度
    community_cohesion = defaultdict(float)
    for community_id in set(communities.values()):
        community_nodes = [n for n, c in communities.items() if c == community_id]
        if len(community_nodes) > 1:
            internal_edges = nx_graph.subgraph(community_nodes).number_of_edges()
            max_possible_edges = len(community_nodes) * (len(community_nodes) - 1) / 2
            cohesion = internal_edges / max_possible_edges if max_possible_edges > 0 else 0
            community_cohesion[community_id] = cohesion
            
            # 为该社区的所有节点添加紧密度属性
            for node in community_nodes:
                nx_graph.nodes[node]['community_cohesion'] = cohesion
    
    # 识别社区之间的桥接节点
    bridge_nodes = []
    for node in nx_graph.nodes():
        neighbors = list(nx_graph.neighbors(node))
        neighbor_communities = [communities.get(neighbor, -1) for neighbor in neighbors]
        unique_communities = set(neighbor_communities)
        if len(unique_communities) > 1:  # 连接多个社区
            nx_graph.nodes[node]['is_bridge'] = True
            nx_graph.nodes[node]['connected_communities'] = list(unique_communities)
            bridge_nodes.append(node)
        else:
            nx_graph.nodes[node]['is_bridge'] = False
    
    logging.info(f"层次聚类完成，共识别出 {len(set(communities.values()))} 个社区")
    logging.info(f"识别出 {len(bridge_nodes)} 个桥接节点")
    
    return graph

async def browse_graph(graph):
    """图谱预览函数"""
    if not hasattr(graph, '_graph') or not graph._graph:
        print("知识图谱为空，无法浏览")
        return

    print("\n=== 知识图谱浏览 ===")
    print(f"图谱包含 {len(graph._graph.nodes)} 个实体和 {len(graph._graph.edges)} 个关系")

    while True:
        print("\n操作选项:")
        print("1. 查看所有实体")
        print("2. 搜索实体")
        print("3. 查看实体关系")
        print("4. 返回主菜单")

        choice = await async_input("请选择操作 (1-4): ")

        if choice == "1":
            # 显示所有实体，分页显示
            entities = list(graph._graph.nodes)
            page_size = 10
            total_pages = (len(entities) + page_size - 1) // page_size

            page = 1
            while page <= total_pages:
                start_idx = (page - 1) * page_size
                end_idx = min(start_idx + page_size, len(entities))

                print(f"\n--- 实体列表 (第 {page}/{total_pages} 页) ---")
                for i, entity in enumerate(entities[start_idx:end_idx], start=start_idx + 1):
                    print(f"{i}. {entity}")

                if page < total_pages:
                    next_page = await async_input("按Enter查看下一页，输入'q'返回: ")
                    if next_page.lower() == 'q':
                        break
                    page += 1
                else:
                    await async_input("已是最后一页，按Enter继续: ")
                    break

        elif choice == "2":
            # 搜索实体
            keyword = await async_input("请输入要搜索的实体关键词: ")
            if not keyword:
                continue

            matches = []
            for entity in graph._graph.nodes:
                if keyword.lower() in str(entity).lower():
                    matches.append(entity)

            if matches:
                print(f"\n找到 {len(matches)} 个匹配实体:")
                for i, entity in enumerate(matches, 1):
                    print(f"{i}. {entity}")

                # 查看详细信息
                entity_idx = await async_input("输入编号查看详情，或按Enter返回: ")
                if entity_idx.isdigit() and 1 <= int(entity_idx) <= len(matches):
                    entity = matches[int(entity_idx) - 1]
                    print(f"\n=== {entity} 详情 ===")

                    # 显示节点属性
                    node_data = graph._graph.nodes[entity]
                    if "text" in node_data:
                        print(f"内容: {node_data['text']}")
                    if "source" in node_data:
                        print(f"来源: {node_data['source']}")

                    # 显示相关关系
                    neighbors = list(graph._graph.neighbors(entity))
                    if neighbors:
                        print(f"\n相关实体 ({len(neighbors)}个):")
                        for i, neighbor in enumerate(neighbors[:10], 1):  # 显示前10个
                            edge_data = graph._graph.get_edge_data(entity, neighbor) or {}
                            relation = edge_data.get("relation", "相关")
                            print(f"{i}. {entity} --[{relation}]--> {neighbor}")
                    else:
                        print("\n没有相关实体")

                    await async_input("按Enter继续: ")
            else:
                print("未找到匹配实体")

        elif choice == "3":
            # 查看实体关系
            entity = await async_input("请输入要查看关系的实体名称: ")
            if not entity:
                continue

            # 寻找最匹配的实体
            best_match = None
            for node in graph._graph.nodes:
                if str(node).lower() == entity.lower():
                    best_match = node
                    break
                elif entity.lower() in str(node).lower():
                    best_match = node

            if best_match:
                print(f"\n=== {best_match} 的关系 ===")

                # 获取出边
                out_edges = []
                for neighbor in graph._graph.neighbors(best_match):
                    edge_data = graph._graph.get_edge_data(best_match, neighbor) or {}
                    relation = edge_data.get("relation", "相关")
                    out_edges.append((neighbor, relation, "out"))

                # 获取入边
                in_edges = []
                for node in graph._graph.nodes:
                    if node != best_match and graph._graph.has_edge(node, best_match):
                        edge_data = graph._graph.get_edge_data(node, best_match) or {}
                        relation = edge_data.get("relation", "相关")
                        in_edges.append((node, relation, "in"))

                # 显示所有关系
                all_edges = out_edges + in_edges
                if all_edges:
                    print(f"\n共有 {len(all_edges)} 个关系:")
                    for i, (related_entity, relation, direction) in enumerate(all_edges, 1):
                        if direction == "out":
                            print(f"{i}. {best_match} --[{relation}]--> {related_entity}")
                        else:
                            print(f"{i}. {related_entity} --[{relation}]--> {best_match}")

                    # 查看详细关系
                    edge_idx = await async_input("输入编号查看详情，或按Enter返回: ")
                    if edge_idx.isdigit() and 1 <= int(edge_idx) <= len(all_edges):
                        related_entity, relation, direction = all_edges[int(edge_idx) - 1]
                        print(f"\n=== 关系详情 ===")
                        if direction == "out":
                            print(f"主体: {best_match}")
                            print(f"关系: {relation}")
                            print(f"客体: {related_entity}")
                        else:
                            print(f"主体: {related_entity}")
                            print(f"关系: {relation}")
                            print(f"客体: {best_match}")

                        # 显示相关节点的文本内容
                        if "text" in graph._graph.nodes[related_entity]:
                            print(f"\n{related_entity}的内容:")
                            print(graph._graph.nodes[related_entity]["text"])

                        await async_input("按Enter继续: ")
                else:
                    print("该实体没有关系连接")
            else:
                print(f"未找到匹配实体: {entity}")

        elif choice == "4":
            # 返回主菜单
            break

        else:
            print("无效选择，请重试")

    return


class KGHybridRetriever:
    """知识图谱混合检索器，结合向量检索和关键词检索"""

    def __init__(self, graph, embeddings, similarity_top_k=3, include_text=True,
                 explore_global_knowledge=True, neo4j=None):
        self.graph = graph
        self.embeddings = embeddings
        self.similarity_top_k = similarity_top_k
        self.include_text = include_text
        self.explore_global_knowledge = explore_global_knowledge
        self.embedding_cache = {}
        self.keyword_cache = {}
        self.neo4j = neo4j

    async def get_cached_embedding(self, text):
        """获取缓存的嵌入向量，如无则计算并缓存"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        embedding = await asyncio.to_thread(self.embeddings.embed_query, text)
        self.embedding_cache[text] = embedding
        return embedding

    async def aretrieve(self, query: str) -> List[Document]:
        """混合检索方法，支持 Neo4j"""
        # 检查是否使用 Neo4j
        if self.neo4j and self.neo4j.connected:
            # 使用 Neo4j 进行检索
            neo4j_keyword_docs = self.neo4j.search_by_keyword(query, limit=self.similarity_top_k)

            # 如果支持向量搜索，尝试使用向量
            query_embedding = await self.get_cached_embedding(query)
            if isinstance(query_embedding, list):
                neo4j_vector_docs = self.neo4j.search_by_similarity(query, limit=self.similarity_top_k)
            else:
                neo4j_vector_docs = []

            # 合并 Neo4j 检索结果并去重
            all_neo4j_docs = neo4j_keyword_docs + neo4j_vector_docs
            unique_neo4j_docs = self._deduplicate(all_neo4j_docs)

            # 如果 Neo4j 检索结果足够，直接返回
            if len(unique_neo4j_docs) >= self.similarity_top_k:
                return unique_neo4j_docs[:self.similarity_top_k]

            # 如果结果不够，补充使用原始 NetworkX 检索
            keyword_docs = await asyncio.to_thread(self._keyword_search, query)
            vector_docs = await self._vector_search(query)
            all_docs = unique_neo4j_docs + keyword_docs + vector_docs
            return self._deduplicate(all_docs)[:self.similarity_top_k]
        else:
            # 使用原始 NetworkX 检索
            keyword_docs = await asyncio.to_thread(self._keyword_search, query)
            vector_docs = await self._vector_search(query)
            all_docs = keyword_docs + vector_docs
            return self._deduplicate(all_docs)[:self.similarity_top_k]

    def _keyword_search(self, query: str) -> List[Document]:
        """基于关键词的检索"""

        start_time = time.time()

        # 查询关键词缓存
        cache_key = query.lower()
        if cache_key in self.keyword_cache:
            logging.info("使用关键词检索缓存")
            return self.keyword_cache[cache_key]

        # 提取查询中的关键词
        keywords = cache_key.split()
        results = []

        # 搜索包含关键词的节点
        for node, data in self.graph._graph.nodes(data=True):
            node_text = str(node).lower()
            if any(keyword in node_text for keyword in keywords):
                # 如果探索全局知识，获取该节点的邻居节点
                if self.explore_global_knowledge:
                    neighbors = list(self.graph._graph.neighbors(node))
                    for neighbor in neighbors:
                        edge_data = self.graph._graph.get_edge_data(node, neighbor) or {}
                        relation = edge_data.get("relation", "related_to")

                        content = f"{node} {relation} {neighbor}"
                        if self.include_text and "text" in self.graph._graph.nodes[neighbor]:
                            content += f"\nDetails: {self.graph._graph.nodes[neighbor]['text']}"
                        results.append(Document(page_content=content, metadata={"source": "kg_keyword"}))

                # 添加当前节点信息
                content = f"Entity: {node}"
                if self.include_text and "text" in data:
                    content += f"\nDetails: {data['text']}"
                results.append(Document(page_content=content, metadata={"source": "kg_keyword"}))

        self.keyword_cache[cache_key] = results

        end_time = time.time()
        logging.info(f"keyword_search时间：{end_time - start_time}秒")
        return results

    # async def _vector_search(self, query: str) -> List[Document]:
    #     """基于向量的检索，利用社区结构"""
    #     start_time = time.time()

    #     # 获取查询的embedding
    #     query_embedding = await self.get_cached_embedding(query)

    #     results = []
    #     node_embeddings = {}

    #     nodes_to_embed = []
    #     nodes_list = []

    #     # 计算或获取所有节点的embedding并缓存，减少下轮对话检索时间
    #     for node, data in self.graph._graph.nodes(data=True):
    #         node_text = str(node)
    #         if "embedding" in data:
    #             node_embeddings[node] = data["embedding"]
    #         else:
    #             nodes_to_embed.append(node_text)
    #             nodes_list.append(node)

    #     # 批量处理嵌入
    #     if nodes_to_embed:
    #         batch_embeddings = await asyncio.to_thread(self.embeddings.embed_documents, nodes_to_embed)

    #         # 保存嵌入结果到节点和缓存
    #         for node, embedding in zip(nodes_list, batch_embeddings):
    #             node_embeddings[node] = embedding
    #             self.graph._graph.nodes[node]["embedding"] = embedding
    #             self.embedding_cache[str(node)] = embedding

    #     # 计算相似度并排序——余弦相似度
    #     similarities = {}
    #     for node, embedding in node_embeddings.items():
    #         similarity = self._cosine_similarity(query_embedding, embedding)
    #         similarities[node] = similarity

    #     sorted_nodes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    #     # 获取前K个最相似的节点及其邻居
    #     for node, sim in sorted_nodes[:self.similarity_top_k]:
    #         # 添加当前节点信息
    #         content = f"Entity: {node} (similarity: {sim:.4f})"
    #         if self.include_text and "text" in self.graph._graph.nodes[node]:
    #             content += f"\nDetails: {self.graph._graph.nodes[node]['text']}"

    #         results.append(Document(page_content=content, metadata={"source": "kg_vector", "similarity": sim}))

    #         # 如果探索全局知识，添加邻居节点
    #         if self.explore_global_knowledge:
    #             neighbors = list(self.graph._graph.neighbors(node))
    #             for neighbor in neighbors:
    #                 edge_data = self.graph._graph.get_edge_data(node, neighbor) or {}
    #                 relation = edge_data.get("relation", "related_to")

    #                 content = f"{node} {relation} {neighbor} (related to similarity: {sim:.4f})"
    #                 if self.include_text and "text" in self.graph._graph.nodes[neighbor]:
    #                     content += f"\nDetails: {self.graph._graph.nodes[neighbor]['text']}"
    #                 results.append(
    #                     Document(page_content=content, metadata={"source": "kg_vector", "similarity": sim * 0.8}))

    #     end_time = time.time()
    #     logging.info(f"vector_search时间：{end_time - start_time}秒")
    #     return results

    async def _vector_search(self, query: str) -> List[Document]:
        """基于向量的检索，利用社区结构"""
        start_time = time.time()

        # 获取查询的embedding
        query_embedding = await self.get_cached_embedding(query)
        
        # 计算所有节点与查询的相似度
        similarities = {}
        communities = defaultdict(list)
        
        for node, data in self.graph._graph.nodes(data=True):
            node_text = str(node)
            if "embedding" in data:
                embedding = data["embedding"]
            else:
                embedding = await self.get_cached_embedding(node_text)
                self.graph._graph.nodes[node]["embedding"] = embedding
            
            similarity = self._cosine_similarity(query_embedding, embedding)
            similarities[node] = similarity
            
            # 收集社区信息
            community_id = data.get('community_id', -1)
            communities[community_id].append((node, similarity))
        
        results = []
        
        # 先找出相似度最高的社区
        community_avg_sim = {}
        for community_id, nodes in communities.items():
            if community_id != -1:  # 跳过未分类节点
                avg_sim = sum(sim for _, sim in nodes) / len(nodes)
                community_avg_sim[community_id] = avg_sim
        
        # 按相似度排序社区
        sorted_communities = sorted(
            community_avg_sim.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]  # 取前3个最相关社区
        
        # 从相关社区中选择节点
        selected_nodes = set()
        for community_id, _ in sorted_communities:
            # 从每个社区选择最相关的节点
            community_nodes = communities[community_id]
            sorted_nodes = sorted(community_nodes, key=lambda x: x[1], reverse=True)
            
            # 取每个社区的前2个节点
            for node, sim in sorted_nodes[:2]:
                selected_nodes.add(node)
        
        # 添加桥接节点(如果相关)
        bridge_nodes = [
            (node, similarities[node]) 
            for node, data in self.graph._graph.nodes(data=True)
            if data.get('is_bridge', False) and similarities[node] > 0.5
        ]
        for node, sim in sorted(bridge_nodes, key=lambda x: x[1], reverse=True)[:2]:
            selected_nodes.add(node)
        
        # 构建结果文档
        for node in selected_nodes:
            sim = similarities[node]
            content = f"Entity: {node} (similarity: {sim:.4f})"
            if self.include_text and "text" in self.graph._graph.nodes[node]:
                content += f"\nDetails: {self.graph._graph.nodes[node]['text']}"
                
            # 添加社区信息
            community_id = self.graph._graph.nodes[node].get('community_id', -1)
            if community_id != -1:
                content += f"\nCommunity: {community_id}"
                
            results.append(Document(
                page_content=content, 
                metadata={
                    "source": "kg_vector", 
                    "similarity": sim,
                    "community_id": community_id
                }
            ))
            
            # 添加该节点的邻居
            if self.explore_global_knowledge:
                neighbors = list(self.graph._graph.neighbors(node))
                for neighbor in neighbors:
                    edge_data = self.graph._graph.get_edge_data(node, neighbor) or {}
                    relation = edge_data.get("relation", "related_to")
                    
                    content = f"{node} {relation} {neighbor} (related to similarity: {sim:.4f})"
                    if self.include_text and "text" in self.graph._graph.nodes[neighbor]:
                        content += f"\nDetails: {self.graph._graph.nodes[neighbor]['text']}"
                        
                    results.append(Document(
                        page_content=content, 
                        metadata={
                            "source": "kg_vector", 
                            "similarity": sim * 0.8,
                            "is_neighbor": True
                        }
                    ))

        end_time = time.time()
        logging.info(f"vector_search时间：{end_time - start_time}秒")
        return results

    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """去除重复的文档"""
        unique_contents = {}
        for doc in docs:
            # 使用内容的哈希作为唯一标识
            content_hash = hash(doc.page_content)
            if content_hash not in unique_contents or (
                    doc.metadata.get("similarity", 0) >
                    unique_contents[content_hash].metadata.get("similarity", 0)
            ):
                unique_contents[content_hash] = doc

        # 按相似度排序
        sorted_docs = sorted(
            unique_contents.values(),
            key=lambda x: x.metadata.get("similarity", 0),
            reverse=True
        )

        return sorted_docs

    async def _update_entity_embeddings(self, entities):
        """更新指定实体的嵌入向量"""
        nodes_to_embed = []
        nodes_list = []

        for entity in entities:
            if entity in self.graph._graph.nodes:
                nodes_to_embed.append(str(entity))
                nodes_list.append(entity)

        if nodes_to_embed:
            # 批量计算嵌入向量
            batch_embeddings = await asyncio.to_thread(self.embeddings.embed_documents, nodes_to_embed)

            # 保存嵌入结果到节点和缓存
            for node, embedding in zip(nodes_list, batch_embeddings):
                self.graph._graph.nodes[node]["embedding"] = embedding
                self.embedding_cache[str(node)] = embedding

            logging.info(f"已更新 {len(nodes_to_embed)} 个实体的嵌入向量")

    # async def learn_from_interaction(self, query, answer, context_docs=None):
    #     """从用户交互中学习新的实体和关系，并更新知识图谱"""
    #     logging.info("从用户交互中学习...")
    #     start_time = time.time()

    #     # 1. 构建学习材料 - 包含查询、回答和上下文
    #     learning_material = f"问题：{query}\n回答：{answer}"
    #     if context_docs:
    #         # 添加被引用的文档作为上下文
    #         for i, doc in enumerate(context_docs[:3]):  # 限制使用的文档数量
    #             learning_material += f"\n参考资料{i+1}：{doc.page_content[:500]}"

    #     # 2. 提取实体
    #     new_entities = extract_entities(learning_material)
    #     if not new_entities:
    #         logging.info("未从交互中提取到新实体")
    #         return False

    #     # 3. 提取关系
    #     new_relationships = extract_relationships(learning_material, new_entities)

    #     # 4. 更新知识图谱
    #     updated = False

    #     # 4.1 添加新实体
    #     for entity in new_entities:
    #         if entity not in self.graph._graph.nodes:
    #             self.graph._graph.add_node(entity)
    #             self.graph._graph.nodes[entity]["entity_type"] = "concept"
    #             self.graph._graph.nodes[entity]["text"] = learning_material[:500]
    #             self.graph._graph.nodes[entity]["source"] = "user_interaction"
    #             self.graph._graph.nodes[entity]["importance"] = 1
    #             self.graph._graph.nodes[entity]["learned"] = True  # 标记为学习所得
    #             updated = True

    #     # 4.2 添加新关系
    #     for rel in new_relationships:
    #         source_entity = rel["source"]
    #         target_entity = rel["target"]
    #         relation = rel["relation"]

    #         # 确保两个实体都在图中
    #         if source_entity not in self.graph._graph.nodes:
    #             continue
    #         if target_entity not in self.graph._graph.nodes:
    #             continue

    #         # 添加或更新边
    #         if not self.graph._graph.has_edge(source_entity, target_entity):
    #             self.graph._graph.add_edge(
    #                 source_entity, target_entity,
    #                 relation=relation,
    #                 weight=1,
    #                 context=learning_material[:200],
    #                 learned=True  # 标记为学习所得
    #             )
    #             updated = True
    #         else:
    #             # 如果边已存在，更新权重
    #             self.graph._graph[source_entity][target_entity]["weight"] += 1
    #             updated = True

    #     # 5. 如果有更新，重新计算实体嵌入
    #     if updated:
    #         await self._update_entity_embeddings(new_entities)
    #         logging.info(f"学习了 {len(new_entities)} 个实体和 {len(new_relationships)} 个关系")

    #     end_time = time.time()
    #     logging.info(f"learn_from_interaction时间：{end_time - start_time}秒")
    #     return updated

    async def learn_from_interaction(self, query, answer, context_docs=None):
        """从用户交互中学习并同时更新 Neo4j"""
        logging.info("从用户交互中学习...")
        start_time = time.time()
        
        # 1. 构建学习材料 - 包含查询、回答和上下文
        learning_material = f"问题：{query}\n回答：{answer}"
        if context_docs:
            # 添加被引用的文档作为上下文
            for i, doc in enumerate(context_docs[:3]):  # 限制使用的文档数量
                learning_material += f"\n参考资料{i+1}：{doc.page_content[:500]}"
        
        # 2. 提取实体
        new_entities = extract_entities(learning_material)
        if not new_entities:
            logging.info("未从交互中提取到新实体")
            return False
        
        # 3. 提取关系
        new_relationships = extract_relationships(learning_material, new_entities)
        
        # 4. 更新知识图谱
        updated = False
        
        # 4.1 添加新实体
        for entity in new_entities:
            if entity not in self.graph._graph.nodes:
                self.graph._graph.add_node(entity)
                self.graph._graph.nodes[entity]["entity_type"] = "concept"
                self.graph._graph.nodes[entity]["text"] = learning_material[:500]
                self.graph._graph.nodes[entity]["source"] = "user_interaction"
                self.graph._graph.nodes[entity]["importance"] = 1
                self.graph._graph.nodes[entity]["learned"] = True  # 标记为学习所得
                updated = True
        
        # 4.2 添加新关系
        for rel in new_relationships:
            source_entity = rel["source"]
            target_entity = rel["target"]
            relation = rel["relation"]
            
            # 确保两个实体都在图中
            if source_entity not in self.graph._graph.nodes:
                continue
            if target_entity not in self.graph._graph.nodes:
                continue
            
            # 添加或更新边
            if not self.graph._graph.has_edge(source_entity, target_entity):
                self.graph._graph.add_edge(
                    source_entity, target_entity,
                    relation=relation,
                    weight=1,
                    context=learning_material[:200],
                    learned=True  # 标记为学习所得
                )
                updated = True
            else:
                # 如果边已存在，更新权重
                self.graph._graph[source_entity][target_entity]["weight"] += 1
                updated = True
        
        # 5. 如果有更新，重新计算实体嵌入
        if updated:
            await self._update_entity_embeddings(new_entities)
            logging.info(f"学习了 {len(new_entities)} 个实体和 {len(new_relationships)} 个关系")
        
        # 更新 Neo4j
        if updated and self.neo4j and self.neo4j.connected:
            for entity in new_entities:
                self.neo4j.add_entity(entity, {
                    "text": learning_material[:500],
                    "source": "user_interaction",
                    "importance": 1,
                    "learned": True
                })
            
            for rel in new_relationships:
                source_entity = rel["source"]
                target_entity = rel["target"]
                relation = rel["relation"]
                
                valid_relation = re.sub(r'[^a-zA-Z0-9_]', '_', relation).upper()
                # 确保关系类型不以数字开头
                if valid_relation and valid_relation[0].isdigit():
                    valid_relation = f"REL_{valid_relation}"
                
                self.neo4j.add_relationship(
                    source_entity,
                    target_entity,
                    valid_relation,
                    {
                        "weight": 1,
                        "context": learning_material[:200],
                        "learned": True
                    }
                )
        
        end_time = time.time()
        logging.info(f"learn_from_interaction时间：{end_time - start_time}秒")
        return updated

class Neo4jKnowledgeGraph:
    """Neo4j 知识图谱管理类"""

    def __init__(self, uri=neo4juri, username=neo4jusername, password=neo4jpassword):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.connected = False

    def connect(self):
        """连接到 Neo4j 数据库"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # 测试连接
            with self.driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                count = result.single()["count"]
                logging.info(f"成功连接到 Neo4j 数据库，当前节点数: {count}")
            self.connected = True
            return True
        except Exception as e:
            logging.error(f"连接 Neo4j 失败: {e}")
            self.connected = False
            return False

    def close(self):
        """关闭 Neo4j 连接"""
        if self.driver:
            self.driver.close()
            self.connected = False

    def networkx_to_neo4j(self, nx_graph):
        """将 NetworkX 图转换到 Neo4j"""
        if not self.connected:
            if not self.connect():
                logging.error("无法连接到 Neo4j，转换失败")
                return False

        try:
            # 清空原有图数据（可选）
            self._clear_graph()

            # 添加节点
            nodes_added = 0
            for node, data in nx_graph.nodes(data=True):
                properties = {k: v for k, v in data.items() if
                              k != "embedding" and isinstance(v, (str, int, float, bool, list))}
                self._create_node(str(node), properties)
                nodes_added += 1

            # 添加边
            edges_added = 0
            for source, target, data in nx_graph.edges(data=True):
                raw_rel_type = data.get("relation", "RELATED_TO")
                rel_type = re.sub(r'[^a-zA-Z0-9_]', '_', raw_rel_type).upper()
                if rel_type and rel_type[0].isdigit():
                    rel_type = f"REL_{rel_type}"
                if not rel_type:
                    rel_type = "RELATED_TO"

                properties = {k: v for k, v in data.items() if
                              k != "relation" and isinstance(v, (str, int, float, bool, list))}
                self._create_relationship(str(source), str(target), rel_type, properties)
                edges_added += 1

            logging.info(f"成功将图谱转换到 Neo4j: {nodes_added} 个节点, {edges_added} 条关系")
            return True
        except Exception as e:
            logging.error(f"将图谱转换到 Neo4j 时出错: {e}")
            return False

    def _clear_graph(self):
        """清空图数据库"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def _create_node(self, node_id, properties=None):
        """创建节点"""
        if properties is None:
            properties = {}

        # 处理文本字段，防止过长
        if "text" in properties and isinstance(properties["text"], str) and len(properties["text"]) > 1000:
            properties["text"] = properties["text"][:1000] + "..."

        with self.driver.session() as session:
            cypher = """
            MERGE (n:Entity {id: $id}) 
            SET n += $properties
            """
            session.run(cypher, id=node_id, properties=properties)

    def _create_relationship(self, source_id, target_id, rel_type, properties=None):
        """创建关系"""
        if properties is None:
            properties = {}

        with self.driver.session() as session:
            cypher = f"""
            MATCH (s:Entity {{id: $source}})
            MATCH (t:Entity {{id: $target}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r += $properties
            """
            session.run(cypher, source=source_id, target=target_id, properties=properties)

    def search_by_keyword(self, keyword, limit=5):
        """基于关键词搜索实体和关系"""
        if not self.connected:
            if not self.connect():
                return []

        try:
            with self.driver.session() as session:
                # 使用全文搜索或模糊匹配
                cypher = """
                MATCH (n:Entity)
                WHERE n.id CONTAINS $keyword OR n.text CONTAINS $keyword
                RETURN n.id AS entity_id, n.text AS entity_text
                LIMIT $limit
                """
                result = session.run(cypher, keyword=keyword, limit=limit)
                entities = [{"entity": record["entity_id"], "text": record["entity_text"]} for record in result]

                # 如果找到实体，查找相关关系
                relations = []
                if entities:
                    for entity in entities[:3]:  # 限制处理前3个实体以提高性能
                        entity_id = entity["entity"]
                        cypher = """
                        MATCH (n:Entity {id: $entity_id})-[r]->(m:Entity)
                        RETURN n.id AS source, type(r) AS relation, m.id AS target, m.text AS target_text
                        LIMIT 5
                        UNION
                        MATCH (m:Entity)-[r]->(n:Entity {id: $entity_id})
                        RETURN m.id AS source, type(r) AS relation, n.id AS target, n.text AS target_text
                        LIMIT 5
                        """
                        rel_result = session.run(cypher, entity_id=entity_id)
                        for record in rel_result:
                            relations.append({
                                "source": record["source"],
                                "relation": record["relation"].lower().replace("_", " "),
                                "target": record["target"],
                                "text": record["target_text"]
                            })

                # 将结果转换为Document对象
                docs = []
                for entity in entities:
                    content = f"实体: {entity['entity']}\n详情: {entity.get('text', '')}"
                    docs.append(Document(page_content=content, metadata={"source": "kg_neo4j_keyword"}))

                for relation in relations:
                    content = f"{relation['source']} {relation['relation']} {relation['target']}\n详情: {relation.get('text', '')}"
                    docs.append(Document(page_content=content, metadata={"source": "kg_neo4j_relation"}))

                return docs

        except Exception as e:
            logging.error(f"Neo4j 搜索失败: {e}")
            return []

    def search_by_similarity(self, query_embedding, limit=5):
        """基于向量相似度搜索(需要Neo4j有向量索引支持)"""
        if not self.connected:
            if not self.connect():
                return []

        try:
            # 简化实现：返回查询内容中包含关键词的实体
            keywords = query_embedding.split()[:5] if isinstance(query_embedding, str) else []
            results = []

            for keyword in keywords:
                results.extend(self.search_by_keyword(keyword, limit=2))

            return results[:limit]
        except Exception as e:
            logging.error(f"Neo4j 向量搜索失败: {e}")
            return []

    def add_entity(self, entity_id, properties=None):
        """添加实体"""
        if not self.connected:
            if not self.connect():
                return False

        try:
            self._create_node(entity_id, properties)
            return True
        except Exception as e:
            logging.error(f"添加实体失败: {e}")
            return False

    def add_relationship(self, source_id, target_id, rel_type, properties=None):
        """添加关系"""
        if not self.connected:
            if not self.connect():
                return False

        try:
            self._create_relationship(source_id, target_id, rel_type, properties)
            return True
        except Exception as e:
            logging.error(f"添加关系失败: {e}")
            return False


async def save_knowledge_graph(graph, path):
    """保存更新后的知识图谱"""
    if hasattr(graph, '_graph') and graph._graph:
        logging.info(f"保存知识图谱到 {path}")
        write_gpickle(graph._graph, path)
        return True
    return False

async def visualize_communities(graph):
    """可视化社区聚类结果"""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    plt.rcParams['font.family'] = 'SimSun'
    plt.rcParams['axes.unicode_minus'] = False
    
    if not hasattr(graph, '_graph') or not graph._graph:
        print("图谱为空，无法可视化")
        return
    
    nx_graph = graph._graph
    
    # 检查是否已进行社区划分
    if not any('community_id' in data for _, data in nx_graph.nodes(data=True)):
        print("图谱尚未进行社区划分，请先执行层次聚类")
        return
    
    # 获取所有社区ID
    communities = set()
    for _, data in nx_graph.nodes(data=True):
        comm_id = data.get('community_id', -1)
        if comm_id != -1:
            communities.add(comm_id)
    
    # 创建颜色映射
    colors = list(mcolors.TABLEAU_COLORS.values())
    if len(communities) > len(colors):
        colors = list(mcolors.CSS4_COLORS.values())
    
    color_map = {}
    for i, comm_id in enumerate(communities):
        color_map[comm_id] = colors[i % len(colors)]
    
    # 为每个节点分配颜色
    node_colors = []
    for node, data in nx_graph.nodes(data=True):
        comm_id = data.get('community_id', -1)
        if comm_id in color_map:
            node_colors.append(color_map[comm_id])
        else:
            node_colors.append('gray')  # 未分类节点
    
    # 为桥接节点使用特殊形状
    node_shapes = []
    for _, data in nx_graph.nodes(data=True):
        if data.get('is_bridge', False):
            node_shapes.append('s')  # 正方形
        else:
            node_shapes.append('o')  # 圆形
    
    pos = nx.spring_layout(nx_graph, seed=42)
    
    plt.figure(figsize=(12, 10))
    
    # 绘制边
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.3)
    
    # 分别绘制不同形状的节点
    node_list = list(nx_graph.nodes())
    for shape in set(node_shapes):
        indices = [i for i, s in enumerate(node_shapes) if s == shape]
        nodelist = [node_list[i] for i in indices]
        nodecolor = [node_colors[i] for i in indices]
        
        if shape == 'o':
            nx.draw_networkx_nodes(nx_graph, pos, nodelist=nodelist, node_color=nodecolor, alpha=0.8)
        else:  # 正方形
            nx.draw_networkx_nodes(nx_graph, pos, nodelist=nodelist, node_color=nodecolor, node_shape=shape, alpha=0.8)
    
    # 添加标签
    nx.draw_networkx_labels(nx_graph, pos, font_size=8)
    
    # 添加图例
    patches = []
    for comm_id, color in color_map.items():
        patches.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, 
                                 label=f'社区 {comm_id}'))
    
    # 添加形状图例
    patches.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='普通节点'))
    patches.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='桥接节点'))
    
    plt.legend(handles=patches, loc='best')
    plt.title(f'知识图谱社区结构 ({len(communities)}个社区)')
    plt.axis('off')
    
    # 保存图片
    plt.savefig(f'knowledge_graph_communities.png', dpi=300)
    print(f"社区可视化图已保存为 knowledge_graph_communities.png")
    plt.show()

# 1. 定义检索节点
async def retrieve(state):
    query = state["query"]
    search_type = state.get("search_type", "hybrid")  # 默认使用混合检索

    start_time = time.time()

    if search_type == "vector":
        # 向量检索
        reranked_docs = await vectorstore.asimilarity_search(
            query, k=5, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]}
        )
        kg_docs = []
    elif search_type == "kg":
        # 仅知识图谱检索
        kg_docs = await kg_retriever.aretrieve(query)
        reranked_docs = []
    else:  # hybrid——默认使用
        # 混合检索: 向量检索 + 知识图谱检索
        vector_docs_future = vectorstore.asimilarity_search(
            query, k=3, ranker_type="weighted", ranker_params={"weights": [0.6, 0.4]}
        )
        kg_docs_future = kg_retriever.aretrieve(query)

        # 并行执行两种检索
        reranked_docs, kg_docs = await asyncio.gather(vector_docs_future, kg_docs_future)

    end_time = time.time()
    logging.info(f"retrieve时间：{end_time - start_time}秒")

    found_docs = bool(reranked_docs) or bool(kg_docs)
    return {
        "query": query,
        "reranked_docs": reranked_docs,
        "kg_docs": kg_docs,
        "found_docs": found_docs
    }

def route_after_retrieve(state):
    if state["found_docs"]:
        return "enhance"
    else:
        return "generate"

async def human_review_retrieval(state):
    """人工审核检索结果"""
    print("\n=== 检索结果人工审核 ===")
    vector_docs = state.get("reranked_docs", [])
    kg_docs = state.get("kg_docs", [])
    
    # 展示检索结果
    print(f"向量检索结果: {len(vector_docs)} 个文档")
    for i, doc in enumerate(vector_docs, 1):
        print(f"{i}. {doc.page_content[:100]}...")
    
    print(f"\n知识图谱检索结果: {len(kg_docs)} 个文档")
    for i, doc in enumerate(kg_docs, 1):
        print(f"{i}. {doc.page_content[:100]}...")
    
    # 人工干预
    action = await async_input("\n请选择操作: \n1. 继续处理 \n2. 删除某个结果 \n3. 添加额外信息 \n选择: ")
    
    if action == "2":
        # 删除操作
        doc_type = await async_input("删除哪类文档? (vector/kg): ")
        doc_id = await async_input("输入文档编号: ")
        if doc_id.isdigit():
            doc_id = int(doc_id)
            if doc_type.lower() == "vector" and 1 <= doc_id <= len(vector_docs):
                state["reranked_docs"].pop(doc_id-1)
                print(f"已删除向量文档 {doc_id}")
            elif doc_type.lower() == "kg" and 1 <= doc_id <= len(kg_docs):
                state["kg_docs"].pop(doc_id-1)
                print(f"已删除知识图谱文档 {doc_id}")
    
    elif action == "3":
        # 添加额外信息
        extra_info = await async_input("请输入要添加的额外信息: ")
        if extra_info:
            new_doc = Document(page_content=extra_info, metadata={"source": "human_input"})
            state["kg_docs"].append(new_doc)
            print("已添加人工提供的信息")
    
    return state


# 2. 定义增强节点，对检索到的文档进行"增强"
async def enhance(state):
    start_time = time.time()
    vector_docs = state["reranked_docs"]
    kg_docs = state.get("kg_docs", [])

    # 显示检索结果
    for i, doc in enumerate(vector_docs, 1):
        print(f"【向量检索结果{i}】")
        print(doc.page_content)
        print("-" * 50)

    for i, doc in enumerate(kg_docs, 1):
        print(f"【知识图谱检索结果{i}】")
        print(doc.page_content)
        print("-" * 50)

    # 合并文档进行增强
    all_docs = vector_docs + kg_docs

    async def enhance_one(d):
        prompt = f"请对下面这段内容进行美化润色，使其表达更通顺正式：\n\n{d.page_content}"
        enhanced = await llm.ainvoke(prompt)
        return enhanced.content if hasattr(enhanced, "content") else str(enhanced)

    # 并发执行所有润色任务
    enhanced_docs = await asyncio.gather(*(enhance_one(d) for d in all_docs))

    end_time = time.time()
    logging.info(f"enhance时间：{end_time - start_time}秒")
    return {
        **state,
        "enhanced_docs": enhanced_docs,
        "docs": all_docs
    }


# 3. 定义生成节点
async def generate(state):
    start_time = time.time()
    query = state["query"]
    history = state.get("history", "")
    found_docs = state.get("found_docs", False)

    if "enhanced_docs" in state and state["enhanced_docs"]:
        context = "\n".join(state["enhanced_docs"])
    elif "docs" in state and state["docs"]:
        context = "\n".join([d.page_content for d in state["docs"]])
    else:
        context = ""

    # 组合知识图谱和向量检索结果
    prompt = f"""对话历史摘要：{history}
                已知信息：{context}

                请根据上述内容回答：{query}

                回答时，请综合考虑知识图谱和向量检索提供的信息，注意实体之间的关系。
                如果知识图谱提供了额外的实体关系，请在回答中合理利用这些关系信息。
            """

    answer_text = ""
    async for chunk in llm.astream(prompt, stream=True):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
        answer_text += str(content)

    if found_docs:
        pass
    else:
        answer_text = f"未从您的文档中检索到内容，直接生成：\n{answer_text}"
    print('\n')
    end_time = time.time()
    logging.info(f"generate时间：{end_time - start_time}秒")
    return {
        "query": query,
        "docs": state.get("docs", []),
        "enhanced_docs": state.get("enhanced_docs", []),
        "found_docs": found_docs,
        "answer": answer_text
    }


# 4. 学习节点
async def learn(state):
    """经人工确认后，可从用户交互中学习"""
    query = state["query"]
    answer = state["answer"]
    docs = state.get("docs", [])
    
    # 仅当检索到文档时尝试学习
    if state.get("found_docs", False):
        # 先提取潜在实体和关系
        learning_material = f"问题：{query}\n回答：{answer}"
        new_entities = extract_entities(learning_material)
        
        if new_entities:
            print("\n=== 知识图谱学习确认 ===")
            print(f"从交互中识别出的实体: {', '.join(new_entities)}")
            
            confirm = await async_input("是否将这些知识添加到知识图谱? (y/n): ")
            if confirm.lower() == 'y':
                # 使用检索到的文档作为上下文
                updated = await kg_retriever.learn_from_interaction(query, answer, docs)
                if updated:
                    # 如果图谱有更新，则保存
                    graph_path = f"{collection_name}_graph.gpickle"
                    await save_knowledge_graph(kg_retriever.graph, graph_path)
                    print("知识已成功添加到图谱")
            else:
                print("已取消知识添加")
    
    return state



async def async_input(prompt):
    return await asyncio.to_thread(input, prompt)


async def main():
    global vectorstore, llm, memory, kg_retriever, collection_name

    collection_name = await async_input("请输入数据库集合名称：")
    start_time = time.time()

    # Neo4j 配置
    use_neo4j = await async_input("是否使用 Neo4j 存储知识图谱? (y/n): ")
    neo4j_instance = None

    if use_neo4j.lower() == 'y':
        neo4j_uri = await async_input("输入 Neo4j URI: ") or neo4juri
        neo4j_username = await async_input("输入 Neo4j 用户名: ") or neo4jusername
        neo4j_password = await async_input("输入 Neo4j 密码: ") or neo4jpassword

        # 创建 Neo4j 连接
        neo4j_instance = Neo4jKnowledgeGraph(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )

        # 连接测试
        if not neo4j_instance.connect():
            logging.warning("Neo4j 连接失败，将退回到使用 NetworkX 进行图谱存储")
            neo4j_instance = None

    # 连接 Milvus
    if openai_api_key:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_api_key,
                                      openai_api_base=openai_api_base)
    else:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

    try:
        from pymilvus import connections, Collection, utility

        # 建立连接
        logging.info("连接 Milvus 以检查集合结构...")
        connections.connect("default", host="localhost", port="19530")

        # 检查集合是否存在
        if utility.has_collection(collection_name):
            collection = Collection(name=collection_name)
            schema = collection.schema
            logging.info(f"集合 {collection_name} 的字段：")
            field_names = [field.name for field in schema.fields]
            logging.info(f"字段名称: {field_names}")

            # 尝试找到可能的文本字段
            possible_text_fields = [name for name in field_names if
                                    any(keyword in name.lower() for keyword in
                                        ["text", "content", "document", "page", "string", "txt"])]

            if possible_text_fields:
                logging.info(f"可能的文本字段: {possible_text_fields}")
                text_field = possible_text_fields[0]  #
            else:
                logging.warning(f"未找到可能的文本字段，尝试几个常见名称")
                for field_name in ["page_content", "content", "document"]:
                    if field_name in field_names:
                        text_field = field_name
                        logging.info(f"使用 {text_field} 作为文本字段")
                        break
                else:
                    logging.warning("未找到任何已知的文本字段，将使用默认的 'text'")
                    text_field = "text"
        else:
            logging.warning(f"集合 {collection_name} 不存在")
            text_field = "text"

        connections.disconnect("default")
    except Exception as e:
        logging.error(f"检查集合结构失败: {e}")
        text_field = "text"

    logging.info(f"使用 '{text_field}' 作为文本字段初始化 Milvus")

    # 创建 vectorstore
    vectorstore = await asyncio.to_thread(
        Milvus,
        embedding_function=embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name=collection_name,
        text_field=text_field,
    )

    end_time = time.time()
    logging.info(f"连接Milvus数据库时间：{end_time - start_time}秒")

    # 加载知识图谱
    graph = NetworkxEntityGraph()
    graph_path = f"{collection_name}_graph.gpickle"

    try:
        if os.path.exists(graph_path):
            logging.info(f"从文件加载知识图谱: {graph_path}")
            g = read_gpickle(graph_path)
            graph._graph = g
            logging.info(f"知识图谱加载成功，包含 {len(graph._graph.nodes)} 个节点和 {len(graph._graph.edges)} 条边")

            # 如果使用Neo4j且已有知识图谱，询问是否迁移到Neo4j
            if neo4j_instance and neo4j_instance.connected:
                migrate_to_neo4j = await async_input("是否将已有知识图谱迁移到 Neo4j? (y/n): ")
                if migrate_to_neo4j.lower() == 'y':
                    logging.info("开始将知识图谱迁移到Neo4j...")
                    if neo4j_instance.networkx_to_neo4j(graph._graph):
                        logging.info("成功将知识图谱迁移到Neo4j")
                    else:
                        logging.error("知识图谱迁移到Neo4j失败")
        else:
            logging.info(f"知识图谱文件不存在: {graph_path}，将从向量存储创建新图谱")
            create_graph = await async_input("是否从向量存储创建知识图谱？(y/n): ")
            if create_graph.lower() == 'y':
                # 从Milvus获取所有文档
                logging.info("从Milvus检索文档...")
                try:
                    # 方法1: 使用列表查询
                    docs = await vectorstore.asimilarity_search("", k=5000)  # 空查询以获取所有文档

                    if not docs:
                        # 方法2: 如果方法1失败，尝试使用多个常见查询词
                        logging.info("尝试使用通用查询词获取文档...")
                        common_queries = ["介绍", "什么", "如何", "为什么", "在哪里", "什么时候", "怎么样"]
                        all_docs = []
                        for query in common_queries:
                            batch_docs = await vectorstore.asimilarity_search(query, k=200)
                            all_docs.extend(batch_docs)

                        # 去重
                        seen_contents = set()
                        docs = []
                        for doc in all_docs:
                            if doc.page_content not in seen_contents:
                                seen_contents.add(doc.page_content)
                                docs.append(doc)

                    logging.info(f"从向量存储获取了 {len(docs)} 个文档")

                    # 创建知识图谱
                    if docs:
                        logging.info("开始创建知识图谱...")
                        graph = await create_knowledge_graph(docs)

                        # 创建或加载知识图谱后
                        if hasattr(graph, '_graph') and graph._graph:
                            # 应用层次聚类
                            clustering_choice = await async_input("是否应用Leiden算法进行层次聚类? (y/n): ")
                            if clustering_choice.lower() == 'y':
                                graph = await apply_hierarchical_clustering(graph)
                                viz_choice = await async_input("是否可视化社区聚类结果? (y/n): ")
                                if viz_choice.lower() == 'y':
                                    await visualize_communities(graph)

                        # 保存知识图谱到文件
                        logging.info(f"保存知识图谱到文件: {graph_path}")
                        write_gpickle(graph._graph, graph_path)
                        logging.info(
                            f"知识图谱创建成功，包含 {len(graph._graph.nodes)} 个节点和 {len(graph._graph.edges)} 条边")



                        # 如果使用Neo4j，将新创建的知识图谱保存到Neo4j
                        if neo4j_instance and neo4j_instance.connected:
                            logging.info("开始将新创建的知识图谱保存到Neo4j...")
                            if neo4j_instance.networkx_to_neo4j(graph._graph):
                                logging.info("成功将知识图谱保存到Neo4j")
                            else:
                                logging.error("知识图谱保存到Neo4j失败")
                    else:
                        logging.warning("未能获取文档，将使用空图谱")
                except Exception as e:
                    logging.error(f"从向量存储获取文档失败: {e}")
                    logging.info("将使用空图继续")
            else:
                logging.info("用户选择不创建知识图谱，将使用空图继续")

    except Exception as e:
        logging.error(f"加载或创建知识图谱失败: {e}")
        logging.info("将使用空图继续")

    # 显示图谱信息
    if hasattr(graph, '_graph') and graph._graph:
        logging.info(f"知识图谱信息:")
        logging.info(f"- 节点数量: {len(graph._graph.nodes)}")
        logging.info(f"- 边数量: {len(graph._graph.edges)}")

        # 显示部分节点示例
        node_sample = list(graph._graph.nodes)[:min(5, len(graph._graph.nodes))]
        if node_sample:
            logging.info(f"- 节点示例: {node_sample}")

        # 显示部分边的示例
        edge_sample = list(graph._graph.edges)[:min(5, len(graph._graph.edges))]
        if edge_sample:
            logging.info(f"- 边示例: {edge_sample}")

        # 询问用户是否要浏览知识图谱
        browse = await async_input("是否要浏览知识图谱? (y/n): ")
        if browse.lower() == 'y':
            await browse_graph(graph)

    else:
        logging.warning("知识图谱为空")

    # 初始化知识图谱检索器global
    kg_retriever = KGHybridRetriever(
        graph=graph,
        embeddings=embeddings,
        similarity_top_k=3,
        include_text=True,
        explore_global_knowledge=True,
        neo4j=neo4j_instance
    )

    llm = ChatDeepSeek(model="ep-20250212134213-cwsj8", api_key=deepseek_api_key, api_base=deepseek_api_base)
    memory = ConversationSummaryMemory(llm=llm)

    # 构建 langgraph 流程
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("human_review_retrieval", human_review_retrieval)
    graph.add_node("enhance", enhance)
    graph.add_node("generate", generate)
    graph.add_node("learn", learn)
    graph.add_conditional_edges("retrieve", route_after_retrieve, {'enhance': 'human_review_retrieval', 'generate': 'generate'})
    graph.add_edge("human_review_retrieval", "enhance")
    graph.add_edge("enhance", "generate")
    graph.add_edge("generate", "learn")
    graph.add_edge("learn", END)
    graph.set_entry_point("retrieve")
    app = graph.compile()

    # 添加学习配置
    learning_config = {
        "enabled": True,  # 是否启用学习
        "min_confidence": 0.7,  # 学习的最低置信度
        "auto_save_interval": 10,  # 每处理10次查询自动保存图谱
        "learned_entities_limit": 50,  # 每次学习的最大实体数
    }
    learning_enabled = await async_input("是否启用交互学习模式? (y/n): ")
    learning_config["enabled"] = learning_enabled.lower() == 'y'
    query_count = 0

    while True:
        query = await async_input("请输入你的问题（输入exit退出，输入'vector'/'kg'/'hybrid'可切换检索模式）：")
        if query.strip().lower() == "exit":
            if neo4j_instance:
                neo4j_instance.close()
                logging.info("已关闭 Neo4j 连接")
            break

        # 检查是否是切换检索模式的命令
        if query.strip().lower() in ["vector", "kg", "hybrid"]:
            search_type = query.strip().lower()
            print(f"检索模式已切换为: {search_type}")
            continue

        history_vars = await asyncio.to_thread(memory.load_memory_variables, {})
        history = history_vars.get("history", "")

        # 确定搜索类型
        if query.startswith("vector:"):
            search_type = "vector"
            query = query[7:].strip()
        elif query.startswith("kg:"):
            search_type = "kg"
            query = query[3:].strip()
        elif query.startswith("hybrid:"):
            search_type = "hybrid"
            query = query[7:].strip()
        else:
            search_type = "hybrid"  # 默认使用混合检索

        result = await app.ainvoke({"query": query, "history": history, "search_type": search_type})
        await asyncio.to_thread(memory.save_context, {"input": query}, {"output": result["answer"]})
        print("\n")
        query_count += 1
        if learning_config["enabled"] and query_count % learning_config["auto_save_interval"] == 0:
            await save_knowledge_graph(kg_retriever.graph, graph_path)
            logging.info(f"自动保存知识图谱，已处理 {query_count} 次查询")

            # 同时更新 Neo4j 数据库
            if neo4j_instance and neo4j_instance.connected:
                logging.info("自动更新 Neo4j 知识图谱...")
                if neo4j_instance.networkx_to_neo4j(kg_retriever.graph._graph):
                    logging.info("成功更新 Neo4j 知识图谱")
                else:
                    logging.error("更新 Neo4j 知识图谱失败")


if __name__ == "__main__":
    asyncio.run(main())