from typing import Dict, List, Any, Tuple
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import psutil
import os
import gc
from collections import defaultdict

logger = logging.getLogger(__name__)

class LayoutQualityEvaluator:
    """布局质量评估器，实现多目标评估和反馈分析"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nets = []
        
    def evaluate(self, layout: Dict[str, Any]) -> float:
        # 返回单个分数而不是字典
        metrics = self.calculate_metrics(layout)
        return metrics['overall_score']
        
    def calculate_metrics(self, layout: Dict[str, Any]) -> Dict[str, float]:
        return {
            "wirelength": 0.0,
            "congestion": 0.0,
            "timing": 0.0,
            "power": 0.0,
            "overall_score": 0.0
        }

class SceneAdaptiveRetrieval:
    def __init__(self, knowledge_base: Dict):
        """初始化场景自适应检索
        
        Args:
            knowledge_base: 知识库
        """
        self.knowledge_base = knowledge_base
        self.scene_features = {
            'timing_critical': False,
            'power_critical': False,
            'area_critical': False,
            'congestion_critical': False
        }
        
    def analyze_scene(self, scene_data: Dict) -> Dict[str, Any]:
        """分析场景特征
        
        Args:
            scene_data: 场景数据
            
        Returns:
            场景特征
        """
        features = self.scene_features.copy()
        
        # 分析时序特征
        if 'timing' in scene_data:
            timing_data = scene_data['timing']
            if 'critical_path_delay' in timing_data:
                delay = timing_data['critical_path_delay']
                target_delay = timing_data.get('target_delay', 0.0)
                if target_delay > 0 and delay / target_delay > 0.8:
                    features['timing_critical'] = True
                    
        # 分析功耗特征
        if 'power' in scene_data:
            power_data = scene_data['power']
            if 'total_power' in power_data:
                power = power_data['total_power']
                target_power = power_data.get('target_power', 0.0)
                if target_power > 0 and power / target_power > 0.8:
                    features['power_critical'] = True
                    
        # 分析面积特征
        if 'area' in scene_data:
            area_data = scene_data['area']
            if 'total_area' in area_data:
                area = area_data['total_area']
                target_area = area_data.get('target_area', 0.0)
                if target_area > 0 and area / target_area > 0.8:
                    features['area_critical'] = True
                    
        # 分析拥塞特征
        if 'congestion' in scene_data:
            congestion_data = scene_data['congestion']
            if 'average_congestion' in congestion_data:
                congestion = congestion_data['average_congestion']
                target_congestion = congestion_data.get('target_congestion', 0.0)
                if target_congestion > 0 and congestion / target_congestion > 0.8:
                    features['congestion_critical'] = True
                    
        return features
        
    def adjust_search_range(self, search_range: Dict, 
                           scene_features: Dict) -> Dict:
        """调整检索范围
        
        Args:
            search_range: 初始检索范围
            scene_features: 场景特征
            
        Returns:
            调整后的检索范围
        """
        adjusted_range = search_range.copy()
        
        # 根据场景特征调整检索范围
        if scene_features['timing_critical']:
            adjusted_range['timing_weight'] = 0.4
            adjusted_range['power_weight'] = 0.2
            adjusted_range['area_weight'] = 0.2
            adjusted_range['congestion_weight'] = 0.2
        elif scene_features['power_critical']:
            adjusted_range['timing_weight'] = 0.2
            adjusted_range['power_weight'] = 0.4
            adjusted_range['area_weight'] = 0.2
            adjusted_range['congestion_weight'] = 0.2
        elif scene_features['area_critical']:
            adjusted_range['timing_weight'] = 0.2
            adjusted_range['power_weight'] = 0.2
            adjusted_range['area_weight'] = 0.4
            adjusted_range['congestion_weight'] = 0.2
        elif scene_features['congestion_critical']:
            adjusted_range['timing_weight'] = 0.2
            adjusted_range['power_weight'] = 0.2
            adjusted_range['area_weight'] = 0.2
            adjusted_range['congestion_weight'] = 0.4
            
        return adjusted_range
        
    def optimize_retrieval(self, search_range: Dict) -> Dict:
        """优化检索结果
        
        Args:
            search_range: 检索范围
            
        Returns:
            优化后的检索范围
        """
        optimized_range = search_range.copy()
        
        # 根据知识库特征优化检索范围
        if 'knowledge_stats' in self.knowledge_base:
            stats = self.knowledge_base['knowledge_stats']
            
            # 调整权重以平衡知识分布
            total_weight = sum(optimized_range.values())
            for key in optimized_range:
                if key in stats:
                    optimized_range[key] = (optimized_range[key] + 
                                          stats[key]) / 2
                    
            # 归一化权重
            total = sum(optimized_range.values())
            for key in optimized_range:
                optimized_range[key] /= total
                
        return optimized_range

class LightweightDeployment:
    def __init__(self, model: nn.Module):
        """初始化轻量级部署
        
        Args:
            model: 待部署模型
        """
        self.model = model
        self.quantized_model = None
        self.pruned_model = None
        self.compressed_model = None
        
    def quantize(self, bits: int = 8) -> None:
        """量化模型
        
        Args:
            bits: 量化位数
        """
        # 使用PyTorch的量化功能
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
    def prune(self, sparsity: float = 0.5) -> None:
        """剪枝模型
        
        Args:
            sparsity: 稀疏度
        """
        if self.quantized_model is None:
            model = self.model
        else:
            model = self.quantized_model
            
        # 使用PyTorch的剪枝功能
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.utils.prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=sparsity
                )
                
        self.pruned_model = model
        
    def compress(self) -> None:
        """压缩模型"""
        if self.pruned_model is None:
            model = self.model
        else:
            model = self.pruned_model
            
        # 使用ONNX进行模型压缩
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            "compressed_model.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                         'output': {0: 'batch_size'}}
        )
        
        self.compressed_model = model
        
    def get_deployed_model(self) -> nn.Module:
        """获取部署后的模型
        
        Returns:
            部署后的模型
        """
        if self.compressed_model is not None:
            return self.compressed_model
        elif self.pruned_model is not None:
            return self.pruned_model
        elif self.quantized_model is not None:
            return self.quantized_model
        else:
            return self.model 