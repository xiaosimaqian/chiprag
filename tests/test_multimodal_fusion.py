# chiprag/tests/test_multimodal_fusion.py

import unittest
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
import shutil

from modules.core.chip_retriever import ChipRetriever
from modules.utils.config_loader import ConfigLoader
from modules.core.multimodal_fusion import MultimodalFusion
from modules.knowledge.knowledge_base import KnowledgeBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMultimodalFusion(unittest.TestCase):
    """多模态融合检索测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        # 创建测试目录
        cls.test_dir = Path('/tmp/chiprag_test')
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建模型目录
        cls.model_dir = cls.test_dir / 'models'
        cls.model_dir.mkdir(exist_ok=True)
        
        # 创建缓存目录
        cls.cache_dir = cls.test_dir / 'cache'
        cls.cache_dir.mkdir(exist_ok=True)
        
        # 设置测试配置
        cls.config = {
            'knowledge_base': {
                'path': str(cls.test_dir),
                'format': 'json',
                'text_path': str(cls.test_dir / 'text'),
                'image_path': str(cls.test_dir / 'images'),
                'structured_data_path': str(cls.test_dir / 'structured'),
                'graph_path': str(cls.test_dir / 'graph'),
                'layout_experience_path': str(cls.test_dir / 'layout'),
                'cache_dir': str(cls.cache_dir)
            },
            'fusion': {
                'text_weight': 0.4,
                'image_weight': 0.3,
                'graph_weight': 0.3
            },
            'text_encoder': {
                'use_ollama': True,
                'ollama_model': 'llama2',
                'ollama_url': 'http://localhost:11434',
                'model_path': str(cls.model_dir / 'bert')
            },
            'image_encoder': {
                'model_path': str(cls.model_dir / 'resnet50')
            },
            'graph_encoder': {
                'model_path': str(cls.model_dir / 'graphsage')
            }
        }
        
        # 创建测试数据
        cls.test_data = {
            'text': {
                'features': np.random.rand(768).astype(np.float32),
                'metadata': {'name': 'Test Text'}
            },
            'image': {
                'features': np.random.rand(2048).astype(np.float32),
                'metadata': {'name': 'Test Image'}
            },
            'graph': {
                'features': np.random.rand(512).astype(np.float32),
                'metadata': {'name': 'Test Graph'}
            }
        }
        
        # 初始化融合器
        cls.fusion = MultimodalFusion(cls.config['fusion'])
        
        # 创建测试用的知识库
        cls.knowledge_base = KnowledgeBase(cls.config['knowledge_base'])
        
        # 加载测试数据
        test_data_dir = Path(__file__).parent / 'test_data'
        with open(test_data_dir / 'test_kb.json', 'r', encoding='utf-8') as f:
            cls.knowledge_base = json.load(f)
            
        with open(test_data_dir / 'test_queries.json', 'r', encoding='utf-8') as f:
            cls.queries = json.load(f)
            
        # 初始化检索器
        cls.retriever = ChipRetriever(cls.config)
        
    def test_basic_retrieval(self):
        """测试基础检索功能"""
        query = self.queries['basic']
        results = self.retriever.retrieve(query)
        
        # 检查结果
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        # 检查结果格式
        for result in results:
            self.assertIn('content', result)
            self.assertIn('metadata', result)
            self.assertIn('features', result)
            self.assertIn('score', result)
            
    def test_multimodal_fusion(self):
        """测试多模态融合"""
        query = self.queries['multimodal']
        results = self.retriever.retrieve(query)
        
        # 检查多模态特征
        for result in results:
            features = result['features']
            self.assertIn('text', features)
            self.assertIn('image', features)
            self.assertIn('graph', features)
            
            # 检查模态权重
            metadata = result['metadata']
            self.assertIn('modal_weights', metadata)
            self.assertAlmostEqual(
                sum(metadata['modal_weights'].values()),
                1.0,
                places=5
            )
            
    def test_granularity_fusion(self):
        """测试多粒度融合"""
        query = self.queries['granularity']
        results = self.retriever.retrieve(query)
        
        # 检查粒度信息
        for result in results:
            metadata = result['metadata']
            self.assertIn('level', metadata)
            self.assertIn('granularity_score', metadata)
            
            # 检查粒度权重
            self.assertIn('granularity_weights', metadata)
            self.assertAlmostEqual(
                sum(metadata['granularity_weights'].values()),
                1.0,
                places=5
            )
            
    def test_hybrid_fusion(self):
        """测试混合融合"""
        query = self.queries['hybrid']
        results = self.retriever.retrieve(query)
        
        # 检查混合特征
        for result in results:
            # 检查多模态特征
            features = result['features']
            self.assertIn('text', features)
            self.assertIn('image', features)
            self.assertIn('graph', features)
            
            # 检查粒度信息
            metadata = result['metadata']
            self.assertIn('level', metadata)
            self.assertIn('granularity_score', metadata)
            
            # 检查融合分数
            self.assertIn('fusion_score', metadata)
            self.assertGreaterEqual(metadata['fusion_score'], 0)
            self.assertLessEqual(metadata['fusion_score'], 1)
            
    def test_ranking(self):
        """测试结果排序"""
        query = self.queries['ranking']
        results = self.retriever.retrieve(query)
        
        # 检查排序
        scores = [result['score'] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
    def test_explanation_generation(self):
        """测试解释生成"""
        query = self.queries['explanation']
        results = self.retriever.retrieve(query)
        
        # 检查解释
        for result in results:
            self.assertIn('explanation', result)
            self.assertIsInstance(result['explanation'], str)
            self.assertGreater(len(result['explanation']), 0)
            
    def test_suggestion_generation(self):
        """测试建议生成"""
        query = self.queries['suggestion']
        results = self.retriever.retrieve(query)
        
        # 检查建议
        for result in results:
            self.assertIn('suggestions', result)
            self.assertIsInstance(result['suggestions'], list)
            self.assertGreater(len(result['suggestions']), 0)
            
    def test_confidence_calculation(self):
        """测试置信度计算"""
        query = self.queries['confidence']
        results = self.retriever.retrieve(query)
        
        # 检查置信度
        for result in results:
            self.assertIn('confidence', result)
            self.assertGreaterEqual(result['confidence'], 0)
            self.assertLessEqual(result['confidence'], 1)
            
    def test_performance(self):
        """测试性能"""
        import time
        
        # 测试检索时间
        start_time = time.time()
        query = self.queries['performance']
        results = self.retriever.retrieve(query)
        end_time = time.time()
        
        # 检查性能
        retrieval_time = end_time - start_time
        self.assertLess(retrieval_time, 5.0)  # 假设5秒为性能阈值
        
        # 检查结果数量
        self.assertGreater(len(results), 0)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空查询
        results = self.retriever.retrieve({})
        self.assertEqual(len(results), 0)
        
        # 测试无效查询
        results = self.retriever.retrieve({'invalid': 'query'})
        self.assertEqual(len(results), 0)
        
        # 测试空知识库
        results = self.retriever.retrieve(
            self.queries['basic'],
            knowledge_base=[]
        )
        self.assertEqual(len(results), 0)

    def test_fusion(self):
        """测试多模态融合功能"""
        # 执行融合
        fused_features = self.fusion.fuse(self.test_data)
        
        # 验证结果
        self.assertIsInstance(fused_features, dict)
        self.assertIn('features', fused_features)
        self.assertIn('metadata', fused_features)
        self.assertIsInstance(fused_features['features'], list)
        self.assertEqual(len(fused_features['features']), 3)

    def test_fusion_weights(self):
        """测试融合权重"""
        weights = self.fusion.weights
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights['text'], 0)
        self.assertGreater(weights['image'], 0)
        self.assertGreater(weights['graph'], 0)
        
    def test_feature_fusion(self):
        """测试特征融合"""
        # 融合特征
        fused_data = self.fusion.fuse(self.test_data)
        
        # 验证结果
        self.assertIn('features', fused_data)
        self.assertIn('metadata', fused_data)
        self.assertIsInstance(fused_data['features'], list)
        self.assertIsInstance(fused_data['metadata'], dict)
        
    def test_retrieval(self):
        """测试检索功能"""
        # 准备查询
        query = {
            'text': 'test query',
            'image': np.random.rand(224, 224, 3).astype(np.float32),
            'graph': {'nodes': [], 'edges': []}
        }
        
        # 执行检索
        results = self.retriever.modal_retriever.retrieve(
            query,
            context={'knowledge_base': [self.test_data]}
        )
        
        # 验证结果
        self.assertIsInstance(results, list)
        if results:
            self.assertIn('similarity', results[0])
            self.assertIn('modality_similarities', results[0])
            
    def test_similarity_computation(self):
        """测试相似度计算"""
        # 准备测试数据
        query = {
            'text': 'test query',
            'image': np.random.rand(224, 224, 3).astype(np.float32),
            'graph': {'nodes': [], 'edges': []}
        }
        
        # 计算相似度
        similarity = self.retriever.modal_retriever.compute_similarity(
            query,
            self.test_data
        )
        
        # 验证结果
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0)
        self.assertLessEqual(similarity, 1)
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试空数据
        with self.assertRaises(ValueError):
            self.fusion.fuse({})
            
        # 测试无效特征
        invalid_data = {
            'text': {
                'features': np.array([np.nan]),
                'metadata': {}
            }
        }
        with self.assertRaises(ValueError):
            self.fusion.fuse(invalid_data)
            
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        # 清理临时文件
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

def main():
    """运行测试"""
    unittest.main()

if __name__ == '__main__':
    main()