import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class KGEncoder(nn.Module):
    """知识图谱编码器"""
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 100, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        """
        初始化知识图谱编码器
        
        Args:
            num_entities: 实体数量
            num_relations: 关系数量
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            dropout: Dropout比率
        """
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 实体嵌入
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        # 关系嵌入
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化嵌入
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
        # 添加多层感知机
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            triples: 三元组张量 [batch_size, 3]
            
        Returns:
            三元组得分
        """
        # 获取实体和关系的嵌入
        head_emb = self.entity_embeddings(triples[:, 0])
        relation_emb = self.relation_embeddings(triples[:, 1])
        tail_emb = self.entity_embeddings(triples[:, 2])
        
        # 计算三元组得分
        score = self._compute_score(head_emb, relation_emb, tail_emb)
        
        return score
        
    def _compute_score(self, head_emb: torch.Tensor, relation_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        """
        计算三元组得分
        
        Args:
            head_emb: 头实体嵌入
            relation_emb: 关系嵌入
            tail_emb: 尾实体嵌入
            
        Returns:
            三元组得分
        """
        # 使用TransE评分函数
        score = head_emb + relation_emb - tail_emb
        return torch.norm(score, p=2, dim=1)
        
    def get_entity_embedding(self, entity_id: int) -> torch.Tensor:
        """
        获取实体嵌入
        
        Args:
            entity_id: 实体ID
            
        Returns:
            实体嵌入向量
        """
        return self.entity_embeddings(torch.tensor(entity_id))
        
    def get_relation_embedding(self, relation_id: int) -> torch.Tensor:
        """
        获取关系嵌入
        
        Args:
            relation_id: 关系ID
            
        Returns:
            关系嵌入向量
        """
        return self.relation_embeddings(torch.tensor(relation_id))
        
    def compute_entity_similarity(self, entity1_id: int, entity2_id: int) -> float:
        """
        计算两个实体的相似度
        
        Args:
            entity1_id: 第一个实体ID
            entity2_id: 第二个实体ID
            
        Returns:
            相似度分数
        """
        emb1 = self.get_entity_embedding(entity1_id)
        emb2 = self.get_entity_embedding(entity2_id)
        
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()
        
    def compute_relation_similarity(self, relation1_id: int, relation2_id: int) -> float:
        """
        计算两个关系的相似度
        
        Args:
            relation1_id: 第一个关系ID
            relation2_id: 第二个关系ID
            
        Returns:
            相似度分数
        """
        emb1 = self.get_relation_embedding(relation1_id)
        emb2 = self.get_relation_embedding(relation2_id)
        
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()
        
    def predict_tail(self, head_id: int, relation_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """
        预测尾实体
        
        Args:
            head_id: 头实体ID
            relation_id: 关系ID
            k: 返回的候选数量
            
        Returns:
            候选尾实体列表及其得分
        """
        head_emb = self.get_entity_embedding(head_id)
        relation_emb = self.get_relation_embedding(relation_id)
        
        # 计算所有可能的尾实体得分
        all_tails = torch.arange(self.num_entities)
        tail_embs = self.entity_embeddings(all_tails)
        
        scores = torch.norm(head_emb + relation_emb - tail_embs, p=2, dim=1)
        
        # 获取得分最高的k个尾实体
        topk_scores, topk_indices = torch.topk(scores, k)
        
        return [(idx.item(), score.item()) for idx, score in zip(topk_indices, topk_scores)]
        
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'embedding_dim': self.embedding_dim
        }, path)
        logger.info(f"模型已保存到: {path}")
        
    @classmethod
    def load_model(cls, path: str) -> 'KGEncoder':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path)
        model = cls(
            num_entities=checkpoint['num_entities'],
            num_relations=checkpoint['num_relations'],
            embedding_dim=checkpoint['embedding_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已从 {path} 加载")
        return model 