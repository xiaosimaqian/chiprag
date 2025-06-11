# chiprag/modules/core/basic_retriever.py

from typing import Dict, Any, List, Optional
from .base_retriever import BaseRetriever
from ..utils.llm_manager import LLMManager
import torch
import logging

logger = logging.getLogger(__name__)

class BasicRetriever(BaseRetriever):
    """基础检索器，使用 Ollama 进行特征提取和解释生成"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm_manager = LLMManager(config.get('llm', {}))
        
    def retrieve(self, query: str, context: Optional[Dict] = None, knowledge_base: Optional[Any] = None) -> List[Dict]:
        """执行检索
        
        Args:
            query: 查询文本
            context: 上下文信息
            knowledge_base: 知识库实例
            
        Returns:
            List[Dict]: 检索结果列表
        """
        # 1. 特征提取
        features = self._extract_features(query, context)
        
        # 2. 相似度计算
        if knowledge_base is not None:
            results = self._compute_similarities_with_kb(features, knowledge_base)
        else:
            results = self._compute_similarities(features)
        
        # 3. 生成解释
        results = self._generate_explanations(query, results)
        
        return results
        
    def compute_similarity(self, query: str, item: str) -> float:
        """计算文本相似度
        
        Args:
            query: 查询文本
            item: 待比较文本
            
        Returns:
            float: 相似度分数
        """
        # 使用 LLM 计算相似度
        return self.llm_manager.compute_similarity(query, item)
        
    def _extract_features(self, query: str, context: Optional[Dict] = None) -> Dict:
        """提取特征
        
        Args:
            query: 查询文本
            context: 上下文信息
            
        Returns:
            Dict: 特征字典
        """
        return self.llm_manager.extract_features(query, context)
        
    def _compute_similarities(self, features: Dict) -> List[Dict]:
        """计算相似度
        
        Args:
            features: 特征字典
            
        Returns:
            List[Dict]: 相似度结果
        """
        return self.llm_manager.compute_similarities(features)
        
    def _compute_similarities_with_kb(self, features: Dict, knowledge_base: Any) -> List[Dict]:
        """使用知识库计算相似度
        
        Args:
            features: 特征字典
            knowledge_base: 知识库实例
            
        Returns:
            List[Dict]: 相似度结果
        """
        # 从知识库中检索相似案例
        similar_cases = knowledge_base.get_similar_cases(features)
        
        # 转换为标准格式
        results = []
        for case in similar_cases:
            results.append({
                'id': case.get('id', ''),
                'content': case.get('content', ''),
                'similarity': case.get('similarity', 0.0),
                'metadata': case.get('metadata', {})
            })
            
        return results
        
    def _generate_explanations(self, query: str, results: List[Dict]) -> List[Dict]:
        """生成解释
        
        Args:
            query: 查询文本
            results: 检索结果
            
        Returns:
            List[Dict]: 带解释的结果
        """
        return self.llm_manager.generate_explanations(query, results)