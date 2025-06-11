"""
检索系统测试模块
"""

import unittest
import json
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.core.granularity_retriever import MultiGranularityRetrieval
from modules.knowledge.knowledge_base import KnowledgeBase
from modules.utils.llm_manager import LLMManager

class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.config = {
            'knowledge_base': {
                'path': '/tmp/chiprag_test',
                'format': 'json',
                'text_path': '/tmp/chiprag_test/text',
                'image_path': '/tmp/chiprag_test/images',
                'structured_data_path': '/tmp/chiprag_test/structured',
                'graph_path': '/tmp/chiprag_test/graph',
                'layout_experience_path': '/tmp/chiprag_test/layout',
                'cache_dir': '/tmp/chiprag_test/cache'
            },
            'hierarchy_config': {
                'max_depth': 3,
                'min_components': 2,
                'max_components': 10,
                'similarity_threshold': 0.8
            },
            'llm_config': {
                'base_url': 'http://localhost:8000',
                'model_name': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'max_tokens': 1000,
                'top_p': 0.9,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            }
        }
        
        # 初始化知识库
        cls.knowledge_base = KnowledgeBase(cls.config['knowledge_base'])
        
        # 添加测试数据
        test_data = {
            'name': 'test_module',
            'type': 'module',
            'features': {
                'wirelength': 500,
                'congestion': 0.5,
                'timing': 5.0
            },
            'metadata': {
                'description': 'Test module for retrieval',
                'author': 'Test Author',
                'date': '2024-03-20'
            },
            'components': [
                {
                    'name': 'comp1',
                    'type': 'component',
                    'x': 0,
                    'y': 0,
                    'width': 100,
                    'height': 100
                },
                {
                    'name': 'comp2',
                    'type': 'component',
                    'x': 200,
                    'y': 0,
                    'width': 100,
                    'height': 100
                }
            ],
            'nets': [
                {
                    'name': 'net1',
                    'source': 'comp1',
                    'target': 'comp2',
                    'type': 'signal',
                    'connections': [
                        {'from': 'comp1', 'to': 'comp2'}
                    ]
                }
            ]
        }
        optimization_result = {
            'wirelength': 500,
            'congestion': 0.5,
            'timing': 5.0,
            'score': 0.8
        }
        cls.knowledge_base.add_case(test_data, optimization_result)
        
        # 初始化检索器
        cls.retriever = MultiGranularityRetrieval(cls.config)
        
    def test_multi_granularity_initialization(self):
        """测试多粒度检索器初始化"""
        self.assertIsNotNone(self.retriever)
        self.assertTrue(hasattr(self.retriever, 'retrieve'))
        self.assertTrue(hasattr(self.retriever, 'set_granularity'))
        
    def test_multi_granularity_retrieval(self):
        """测试多粒度检索"""
        # 准备测试数据
        test_layout = {
            "name": "test_layout",
            "components": [
                {
                    "name": "comp1",
                    "type": "memory",
                    "position": {"x": 0, "y": 0},
                    "size": {"width": 100, "height": 100}
                }
            ],
            "hierarchy": {
                "levels": ["top", "module"],
                "modules": ["mem"],
                "max_depth": 2
            }
        }
        
        # 设置粒度
        self.retriever.set_granularity("module")
        
        # 执行检索
        results = self.retriever.retrieve(test_layout)
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertGreater(len(results), 0)
        self.assertIn("module", str(results))
        
    def test_combined_retrieval(self):
        """测试组合检索"""
        # 准备测试数据
        test_layout = {
            "name": "test_layout",
            "components": [
                {
                    "name": "comp1",
                    "type": "memory",
                    "position": {"x": 0, "y": 0},
                    "size": {"width": 100, "height": 100}
                }
            ],
            "constraints": {
                "power": {"max_power": 1.0},
                "timing": {"max_delay": 1.0}
            }
        }
        
        # 设置检索粒度
        self.retriever.set_granularity("module")
        
        # 执行多粒度检索
        layout_results = self.retriever.retrieve(test_layout)
        
        # 验证结果
        self.assertIsNotNone(layout_results)
        self.assertIn("module", str(layout_results))
        
if __name__ == '__main__':
    unittest.main()
