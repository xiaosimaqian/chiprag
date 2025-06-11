# chiprag/modules/core/granularity_retriever.py

from typing import Dict, Any, List, Optional
from .base_retriever import BaseRetriever
from .basic_retriever import BasicRetriever
from ..utils.embedding_manager import EmbeddingManager
from ..utils.llm_manager import LLMManager
from ..knowledge.transfer import KnowledgeTransfer
from ..knowledge.fusion import KnowledgeFusion
import torch
import logging

logger = logging.getLogger(__name__)

class GranularityRetriever(BaseRetriever):
    """多粒度检索器，支持不同层次的知识检索"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.levels = config.get('levels', ['system', 'module', 'component'])
        super().__init__(config)
        self.embedding_manager = EmbeddingManager(config.get('embedding', {}))
        self.llm_manager = LLMManager(config.get('llm', {}))
        
        # 初始化知识迁移和融合
        self.knowledge_transfer = KnowledgeTransfer(config)
        self.knowledge_fusion = KnowledgeFusion(config)
        
        # 初始化各粒度检索器
        self.retrievers = {
            'system': self._init_system_retriever(),
            'function': self._init_function_retriever(),
            'module': self._init_module_retriever(),
            'cell': self._init_cell_retriever()
        }
        
    def retrieve(self, query: Dict[str, Any], context: Optional[Dict] = None) -> List[Dict]:
        """执行多粒度检索
        
        Args:
            query: 查询字典
            context: 上下文信息
            
        Returns:
            List[Dict]: 检索结果
        """
        results = []
        
        # 在各粒度执行检索
        for level, retriever in self.retrievers.items():
            level_results = retriever.retrieve(query, context)
            
            # 知识迁移
            transferred_results = self.knowledge_transfer.transfer(
                results=level_results,
                target_level=level
            )
            
            # 知识融合
            fused_results = self.knowledge_fusion.fuse(
                results=transferred_results,
                level=level
            )
            
            results.extend(fused_results)
            
        return results
        
    def compute_similarity(self, query: Dict[str, Any], item: Dict[str, Any]) -> float:
        """计算多粒度相似度
        
        Args:
            query: 查询字典
            item: 待比较项
            
        Returns:
            float: 相似度分数
        """
        # 计算各粒度的相似度
        similarities = {}
        for level, retriever in self.retrievers.items():
            similarities[level] = retriever.compute_similarity(query, item)
            
        # 加权融合
        return self._weighted_fusion(similarities)
        
    def _init_system_retriever(self) -> BaseRetriever:
        """初始化系统级检索器"""
        return BasicRetriever(self.config.get('system_retriever', {}))
        
    def _init_function_retriever(self) -> BaseRetriever:
        """初始化功能级检索器"""
        return BasicRetriever(self.config.get('function_retriever', {}))
        
    def _init_module_retriever(self) -> BaseRetriever:
        """初始化模块级检索器"""
        return BasicRetriever(self.config.get('module_retriever', {}))
        
    def _init_cell_retriever(self) -> BaseRetriever:
        """初始化单元级检索器"""
        return BasicRetriever(self.config.get('cell_retriever', {}))
        
    def _add_level_info(self, results: List[Dict], level: str) -> List[Dict]:
        """添加粒度信息
        
        Args:
            results: 检索结果
            level: 粒度级别
            
        Returns:
            List[Dict]: 添加粒度信息的结果
        """
        for result in results:
            result['level'] = level
        return results
        
    def _fuse_results(self, results: List[Dict]) -> List[Dict]:
        """融合多粒度结果
        
        Args:
            results: 检索结果
            
        Returns:
            List[Dict]: 融合后的结果
        """
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 去重
        seen = set()
        unique_results = []
        for result in results:
            key = result.get('id', '')
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
                
        return unique_results
        
    def _weighted_fusion(self, similarities: Dict[str, float]) -> float:
        """加权融合相似度
        
        Args:
            similarities: 各粒度相似度
            
        Returns:
            float: 融合后的相似度
        """
        weights = self.config.get('granularity_weights', {
            'system': 1.0,
            'function': 0.8,
            'module': 0.6,
            'cell': 0.4
        })
        
        return sum(
            similarities[level] * weights[level]
            for level in similarities
        ) / sum(weights.values())

class MultiGranularityRetrieval:
    """多粒度检索器"""
    
    def __init__(self, config):
        self.config = config
        self._init_components()
        self.current_granularity = None

    def _init_components(self):
        """初始化组件"""
        from modules.knowledge.knowledge_base import KnowledgeBase
        self.knowledge_base = KnowledgeBase(self.config.get('knowledge_base', {}))
        self.llm_manager = LLMManager(self.config.get('llm_config', {}))

    def set_granularity(self, granularity):
        """设置检索粒度"""
        self.current_granularity = granularity
        return self

    def retrieve(self, query, top_k=3):
        """检索知识"""
        return self.knowledge_base.get_similar_cases(query, top_k=top_k)