# chiprag/modules/encoders/base_encoder.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import torch
import logging

logger = logging.getLogger(__name__)

class BaseEncoder(ABC):
    """编码器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化编码器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._init_model()
        
    @abstractmethod
    def _init_model(self):
        """初始化模型"""
        pass
        
    @abstractmethod
    def encode(self, data: Any) -> torch.Tensor:
        """编码数据
        
        Args:
            data: 输入数据
            
        Returns:
            torch.Tensor: 编码后的向量
        """
        pass
        
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """预处理数据
        
        Args:
            data: 输入数据
            
        Returns:
            Any: 预处理后的数据
        """
        pass
        
    def compute_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """计算两个向量的相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            float: 相似度分数
        """
        return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()