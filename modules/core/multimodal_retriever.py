# chiprag/modules/core/multimodal_retriever.py

import torch
from typing import Dict, List, Any
import logging
from ..encoders import ResNetImageEncoder, BertTextEncoder, KGEncoder

logger = logging.getLogger(__name__)

class MultimodalRetriever:
    """多模态检索器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化多模态检索器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化编码器
        self.image_encoder = ResNetImageEncoder(config.get('image_encoder', {}))
        self.text_encoder = BertTextEncoder(config.get('text_encoder', {}))
        self.graph_encoder = KGEncoder(config.get('graph_encoder', {}))
        
        # 设置权重
        self.weights = {
            'image': config.get('image_weight', 0.3),
            'text': config.get('text_weight', 0.4),
            'graph': config.get('graph_weight', 0.3)
        }
        
    def encode_query(self, query: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """编码查询
        
        Args:
            query: 查询字典，包含文本、图像和知识图谱
            
        Returns:
            Dict[str, torch.Tensor]: 各模态的编码向量
        """
        encodings = {}
        
        # 编码文本
        if 'text' in query:
            text_encoding = self.text_encoder.encode(query['text'])
            encodings['text'] = text_encoding
            
        # 编码图像
        if 'image' in query:
            image_encoding = self.image_encoder.encode(query['image'])
            encodings['image'] = image_encoding
            
        # 编码知识图谱
        if 'graph' in query:
            graph_encoding = self.graph_encoder.encode(query['graph'])
            encodings['graph'] = graph_encoding
            
        return encodings
        
    def retrieve(self, query: Dict[str, Any], knowledge_base: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """检索相关知识
        
        Args:
            query: 查询字典
            knowledge_base: 知识库
            top_k: 返回结果数量
            
        Returns:
            List[Dict[str, Any]]: 检索结果
        """
        # 编码查询
        query_encodings = self.encode_query(query)
        
        # 计算相似度
        results = []
        for item in knowledge_base:
            item_encodings = self.encode_query(item)
            
            # 计算各模态的相似度
            similarities = {}
            for modality in query_encodings:
                if modality in item_encodings:
                    sim = self.image_encoder.compute_similarity(
                        query_encodings[modality],
                        item_encodings[modality]
                    )
                    similarities[modality] = sim
                    
            # 计算加权平均相似度
            weighted_sim = sum(
                similarities.get(modality, 0) * self.weights[modality]
                for modality in self.weights
            )
            
            results.append({
                'item': item,
                'similarity': weighted_sim,
                'modality_similarities': similarities
            })
            
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]