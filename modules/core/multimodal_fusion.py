import logging
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.preprocessing import normalize

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
        
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """归一化特征向量
        
        Args:
            features: 特征向量
            
        Returns:
            归一化后的特征向量
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return normalize(features, norm='l2', axis=1).squeeze()
        
    def _validate_features(self, features: np.ndarray, modality: str) -> None:
        """验证特征向量的有效性
        
        Args:
            features: 特征向量
            modality: 模态名称
            
        Raises:
            ValueError: 特征向量无效
        """
        if not isinstance(features, np.ndarray):
            raise ValueError(f"{modality}特征必须是numpy数组")
        if features.size == 0:
            raise ValueError(f"{modality}特征不能为空")
        if np.isnan(features).any():
            raise ValueError(f"{modality}特征包含NaN值")
            
    def fuse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """融合多模态数据
        
        Args:
            data: 包含各模态特征和元数据的字典
            
        Returns:
            融合后的特征和元数据
            
        Raises:
            ValueError: 数据格式错误或特征无效
        """
        try:
            # 提取各模态特征
            text_features = np.array(data.get('text', {}).get('features', []))
            image_features = np.array(data.get('image', {}).get('features', []))
            graph_features = np.array(data.get('graph', {}).get('features', []))
            
            # 验证特征
            if text_features.size > 0:
                self._validate_features(text_features, 'text')
            if image_features.size > 0:
                self._validate_features(image_features, 'image')
            if graph_features.size > 0:
                self._validate_features(graph_features, 'graph')
                
            # 计算有效模态的权重
            valid_modalities = []
            if text_features.size > 0:
                valid_modalities.append(('text', text_features, self.text_weight))
            if image_features.size > 0:
                valid_modalities.append(('image', image_features, self.image_weight))
            if graph_features.size > 0:
                valid_modalities.append(('graph', graph_features, self.graph_weight))
                
            if not valid_modalities:
                raise ValueError("没有有效的模态特征")
                
            # 归一化权重
            total_weight = sum(weight for _, _, weight in valid_modalities)
            normalized_weights = [weight/total_weight for _, _, weight in valid_modalities]
            
            # 归一化特征
            normalized_features = []
            for _, features, _ in valid_modalities:
                normalized_features.append(self._normalize_features(features))
                
            # 确保特征维度一致
            feature_dims = [f.shape for f in normalized_features]
            if len(set(feature_dims)) > 1:
                raise ValueError(f"各模态特征维度不一致: {feature_dims}")
                
            # 加权融合
            fused_features = np.zeros_like(normalized_features[0])
            for features, weight in zip(normalized_features, normalized_weights):
                fused_features += weight * features
                
            # 合并元数据
            fused_metadata = {}
            for modality, _, _ in valid_modalities:
                if modality in data and 'metadata' in data[modality]:
                    fused_metadata[modality] = data[modality]['metadata']
                    
            return {
                'features': fused_features.tolist(),
                'metadata': fused_metadata
            }
            
        except Exception as e:
            logger.error(f"多模态融合失败: {str(e)}")
            raise 