import logging
from typing import Dict, List, Any
import numpy as np

logger = logging.getLogger(__name__)

class MultimodalFusion:
    """多模态融合类"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化多模态融合器
        
        Args:
            config: 配置信息，包含各模态的权重
        """
        self.text_weight = config.get('text_weight', 0.4)
        self.image_weight = config.get('image_weight', 0.3)
        self.graph_weight = config.get('graph_weight', 0.3)
        
        # 验证权重和为1
        total_weight = self.text_weight + self.image_weight + self.graph_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"权重和不为1: {total_weight}，将进行归一化")
            self.text_weight /= total_weight
            self.image_weight /= total_weight
            self.graph_weight /= total_weight
            
        logger.info("多模态融合器初始化完成")
        
    def fuse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """融合多模态数据
        
        Args:
            data: 包含各模态特征和元数据的字典
            
        Returns:
            融合后的特征和元数据
        """
        # 提取各模态特征
        text_features = np.array(data['text']['features'])
        image_features = np.array(data['image']['features'])
        graph_features = np.array(data['graph']['features'])
        
        # 确保特征维度一致
        if not (len(text_features) == len(image_features) == len(graph_features)):
            raise ValueError("各模态特征维度不一致")
            
        # 加权融合
        fused_features = (
            self.text_weight * text_features +
            self.image_weight * image_features +
            self.graph_weight * graph_features
        )
        
        # 合并元数据
        fused_metadata = {
            'text': data['text']['metadata'],
            'image': data['image']['metadata'],
            'graph': data['graph']['metadata']
        }
        
        return {
            'features': fused_features.tolist(),
            'metadata': fused_metadata
        } 