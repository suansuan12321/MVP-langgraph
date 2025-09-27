import asyncio
import logging
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

from .langsmith_client import LangSmithClient
from .metrics_collector import MetricsCollector

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config_path: str = "config/monitoring.yaml"):
        self.config = self._load_config(config_path)
        self.metrics_collector = MetricsCollector()
        self.langsmith_client = None
        
        if self.config.get("monitoring", {}).get("enabled", False):
            self.langsmith_client = LangSmithClient(
                project_name=self.config.get("langsmith", {}).get("project", "mvp-langgraph-rag")
            )
        
        # 性能阈值
        self.thresholds = self.config.get("monitoring", {}).get("thresholds", {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logging.warning(f"Config file not found: {config_path}")
                return {}
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            return {}
    
    async def monitor_query(
        self,
        query: str,
        answer: str,
        response_time: float,
        similarity_scores: list[float],
        search_type: str,
        found_docs: bool,
        docs_count: int,
        token_usage: Dict[str, int] = None,
        error: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """监控查询执行"""
        query_id = self.metrics_collector.start_query(query, query)
        
        try:
            # 记录相似度分数
            self.metrics_collector.record_similarity_scores(query_id, similarity_scores)
            
            # 检查性能阈值
            self._check_thresholds(response_time, similarity_scores, error)
            
            # 上报到 LangSmith
            if self.langsmith_client:
                await self.langsmith_client.log_query_execution(
                    query=query,
                    answer=answer,
                    response_time=response_time,
                    similarity_scores=similarity_scores,
                    search_type=search_type,
                    found_docs=found_docs,
                    docs_count=docs_count,
                    token_usage=token_usage,
                    error=error,
                    metadata=metadata
                )
            
            # 结束查询计时
            self.metrics_collector.end_query(query_id, success=(error is None), error=error)
            
            return query_id
            
        except Exception as e:
            logging.error(f"Failed to monitor query: {e}")
            self.metrics_collector.end_query(query_id, success=False, error=str(e))
            return "monitoring_failed"
    
    def _check_thresholds(self, response_time: float, similarity_scores: list[float], error: Optional[str]):
        """检查性能阈值"""
        # 检查响应时间阈值
        max_response_time = self.thresholds.get("max_response_time", 30.0)
        if response_time > max_response_time:
            logging.warning(f"Response time {response_time:.2f}s exceeds threshold {max_response_time}s")
        
        # 检查相似度分数阈值
        min_similarity = self.thresholds.get("min_similarity_score", 0.7)
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            if avg_similarity < min_similarity:
                logging.warning(f"Average similarity {avg_similarity:.3f} below threshold {min_similarity}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        stats = self.metrics_collector.get_overall_stats()
        
        report = {
            "overall_stats": stats,
            "thresholds": self.thresholds,
            "config": {
                "monitoring_enabled": self.config.get("monitoring", {}).get("enabled", False),
                "langsmith_enabled": self.langsmith_client is not None
            }
        }
        
        return report
    
    async def export_metrics(self, filepath: str):
        """导出指标到文件"""
        try:
            import json
            report = self.get_performance_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logging.info(f"Metrics exported to: {filepath}")
        except Exception as e:
            logging.error(f"Failed to export metrics: {e}")