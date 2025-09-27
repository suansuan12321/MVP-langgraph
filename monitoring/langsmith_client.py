import os
import time
import asyncio
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    from langsmith import Client
    from langsmith.evaluation import evaluate
    from langsmith.schemas import Run, Example
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    logging.warning("LangSmith not available. Install with: pip install langsmith")

from dotenv import load_dotenv

load_dotenv()

@dataclass
class QueryMetrics:
    """查询指标数据类"""
    query_id: str
    query: str
    answer: str
    response_time: float
    similarity_scores: List[float]
    token_usage: Dict[str, int]
    search_type: str
    found_docs: bool
    docs_count: int
    error: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class LangSmithClient:
    """LangSmith 客户端封装"""
    
    def __init__(self, project_name: str = "mvp-langgraph-rag"):
        self.project_name = project_name
        self.client = None
        self.enabled = False
        
        if LANGSMITH_AVAILABLE:
            api_key = os.getenv("LANGSMITH_API_KEY")
            if api_key:
                self.client = Client(api_key=api_key)
                self.enabled = True
                logging.info(f"LangSmith client initialized for project: {project_name}")
            else:
                logging.warning("LANGSMITH_API_KEY not found. LangSmith monitoring disabled.")
        else:
            logging.warning("LangSmith not installed. Monitoring disabled.")
    
    async def log_query_execution(
        self,
        query: str,
        answer: str,
        response_time: float,
        similarity_scores: List[float],
        search_type: str,
        found_docs: bool,
        docs_count: int,
        token_usage: Dict[str, int] = None,
        error: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """记录查询执行信息"""
        if not self.enabled:
            return "monitoring_disabled"
        
        query_id = f"query_{int(time.time() * 1000)}"
        
        try:
            # 创建运行记录
            run_data = {
                "name": "graphrag_query",
                "run_type": "chain",
                "inputs": {"query": query},
                "outputs": {"answer": answer},
                "start_time": time.time() - response_time,
                "end_time": time.time(),
                "run_id": query_id,
                "project_name": self.project_name,
                "metadata": {
                    "response_time": response_time,
                    "similarity_scores": similarity_scores,
                    "search_type": search_type,
                    "found_docs": found_docs,
                    "docs_count": docs_count,
                    "token_usage": token_usage or {},
                    "error": error,
                    **(metadata or {})
                }
            }
            
            # 发送到 LangSmith
            await asyncio.to_thread(self.client.create_run, **run_data)
            
            logging.info(f"Logged query execution to LangSmith: {query_id}")
            return query_id
            
        except Exception as e:
            logging.error(f"Failed to log to LangSmith: {e}")
            return "logging_failed"
    
    async def log_retrieval_metrics(
        self,
        query_id: str,
        vector_docs: List[Any],
        kg_docs: List[Any],
        similarity_scores: List[float],
        retrieval_time: float
    ):
        """记录检索指标"""
        if not self.enabled:
            return
        
        try:
            retrieval_data = {
                "name": "retrieval_step",
                "run_type": "retriever",
                "inputs": {"query_id": query_id},
                "outputs": {
                    "vector_docs_count": len(vector_docs),
                    "kg_docs_count": len(kg_docs),
                    "similarity_scores": similarity_scores
                },
                "metadata": {
                    "retrieval_time": retrieval_time,
                    "avg_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
                    "max_similarity": max(similarity_scores) if similarity_scores else 0,
                    "min_similarity": min(similarity_scores) if similarity_scores else 0
                }
            }
            
            await asyncio.to_thread(self.client.create_run, **retrieval_data)
            
        except Exception as e:
            logging.error(f"Failed to log retrieval metrics: {e}")
    
    async def log_generation_metrics(
        self,
        query_id: str,
        generation_time: float,
        token_usage: Dict[str, int],
        model_info: Dict[str, Any]
    ):
        """记录生成指标"""
        if not self.enabled:
            return
        
        try:
            generation_data = {
                "name": "generation_step",
                "run_type": "llm",
                "inputs": {"query_id": query_id},
                "outputs": {"token_usage": token_usage},
                "metadata": {
                    "generation_time": generation_time,
                    "model_info": model_info,
                    "tokens_per_second": token_usage.get("total_tokens", 0) / generation_time if generation_time > 0 else 0
                }
            }
            
            await asyncio.to_thread(self.client.create_run, **generation_data)
            
        except Exception as e:
            logging.error(f"Failed to log generation metrics: {e}")
    
    def create_dataset(self, name: str, description: str = "") -> str:
        """创建数据集"""
        if not self.enabled:
            return "monitoring_disabled"
        
        try:
            dataset = self.client.create_dataset(
                dataset_name=name,
                description=description
            )
            logging.info(f"Created dataset: {dataset.name}")
            return dataset.id
        except Exception as e:
            logging.error(f"Failed to create dataset: {e}")
            return "creation_failed"
    
    def add_examples(self, dataset_id: str, examples: List[Dict[str, Any]]):
        """添加示例到数据集"""
        if not self.enabled:
            return
        
        try:
            for example in examples:
                self.client.create_example(
                    inputs=example["inputs"],
                    outputs=example["outputs"],
                    dataset_id=dataset_id
                )
            logging.info(f"Added {len(examples)} examples to dataset")
        except Exception as e:
            logging.error(f"Failed to add examples: {e}")