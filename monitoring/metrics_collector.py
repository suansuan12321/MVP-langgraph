import time
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import statistics

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    similarity_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    success_count: int = 0
    total_queries: int = 0
    
    def add_response_time(self, response_time: float):
        self.response_times.append(response_time)
        self.total_queries += 1
    
    def add_similarity_score(self, score: float):
        self.similarity_scores.append(score)
    
    def add_error(self):
        self.error_count += 1
    
    def add_success(self):
        self.success_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "total_queries": self.total_queries,
            "error_rate": self.error_count / max(self.total_queries, 1),
            "success_rate": self.success_count / max(self.total_queries, 1)
        }
        
        if self.response_times:
            stats.update({
                "avg_response_time": statistics.mean(self.response_times),
                "median_response_time": statistics.median(self.response_times),
                "max_response_time": max(self.response_times),
                "min_response_time": min(self.response_times),
                "p95_response_time": self._percentile(self.response_times, 95),
                "p99_response_time": self._percentile(self.response_times, 99)
            })
        
        if self.similarity_scores:
            stats.update({
                "avg_similarity": statistics.mean(self.similarity_scores),
                "median_similarity": statistics.median(self.similarity_scores),
                "max_similarity": max(self.similarity_scores),
                "min_similarity": min(self.similarity_scores)
            })
        
        return stats
    
    def _percentile(self, data: deque, percentile: int) -> float:
        """计算百分位数"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.query_metrics: Dict[str, Dict[str, Any]] = {}
        self.start_times: Dict[str, float] = {}
        
    def start_query(self, query_id: str, query: str) -> str:
        """开始查询计时"""
        self.start_times[query_id] = time.time()
        self.query_metrics[query_id] = {
            "query": query,
            "start_time": self.start_times[query_id],
            "stages": {}
        }
        return query_id
    
    def end_query(self, query_id: str, success: bool = True, error: Optional[str] = None):
        """结束查询计时"""
        if query_id not in self.start_times:
            return
        
        end_time = time.time()
        response_time = end_time - self.start_times[query_id]
        
        self.metrics.add_response_time(response_time)
        if success:
            self.metrics.add_success()
        else:
            self.metrics.add_error()
        
        # 更新查询指标
        if query_id in self.query_metrics:
            self.query_metrics[query_id].update({
                "end_time": end_time,
                "response_time": response_time,
                "success": success,
                "error": error
            })
        
        # 清理
        del self.start_times[query_id]
    
    def record_stage(self, query_id: str, stage_name: str, duration: float, metadata: Dict[str, Any] = None):
        """记录阶段性能"""
        if query_id in self.query_metrics:
            self.query_metrics[query_id]["stages"][stage_name] = {
                "duration": duration,
                "metadata": metadata or {}
            }
    
    def record_similarity_scores(self, query_id: str, scores: List[float]):
        """记录相似度分数"""
        if scores:
            avg_score = sum(scores) / len(scores)
            self.metrics.add_similarity_score(avg_score)
            
            if query_id in self.query_metrics:
                self.query_metrics[query_id]["similarity_scores"] = scores
                self.query_metrics[query_id]["avg_similarity"] = avg_score
    
    def get_query_metrics(self, query_id: str) -> Optional[Dict[str, Any]]:
        """获取特定查询的指标"""
        return self.query_metrics.get(query_id)
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """获取整体统计信息"""
        return self.metrics.get_stats()
    
    def reset_metrics(self):
        """重置指标"""
        self.metrics = PerformanceMetrics()
        self.query_metrics.clear()
        self.start_times.clear()