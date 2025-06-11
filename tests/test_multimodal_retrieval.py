# chiprag/scripts/test_multimodal_retrieval.py

import unittest
from typing import Dict, Any
import logging
from modules.core.modal_retriever import ModalRetriever
from modules.knowledge.knowledge_base import KnowledgeBase

class TestMultimodalRetrieval(unittest.TestCase):
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
            'image_encoder': {
                'model_name': 'resnet50',
                'pretrained': True,
                'feature_dim': 2048,
                'device': 'cpu'
            },
            'text_encoder': {
                'model_name': 'bert-base-uncased',
                'max_length': 512,
                'device': 'cpu'
            },
            'graph_encoder': {
                'embedding_dim': 256,
                'num_layers': 2,
                'dropout': 0.1
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
        
        # 创建测试数据
        cls.test_data = {
            'circuit': {
                'features': [0.1, 0.2, 0.3],
                'metadata': {'name': 'Test Circuit'}
            },
            'module': {
                'features': [0.4, 0.5, 0.6],
                'metadata': {'name': 'Test Module'}
            }
        }
        
        # 初始化检索器
        cls.retriever = ModalRetriever(cls.config)
        cls.knowledge_base = KnowledgeBase(cls.config['knowledge_base'])

    def test_modal_retrieval(self):
        """测试多模态检索"""
        query = {
            'text': 'Design a high-performance ALU',
            'image': 'path/to/layout.png',
            'graph': {
                'nodes': ['ALU', 'Area', 'Performance'],
                'edges': [
                    ['ALU', 'Area', 'constraint'],
                    ['ALU', 'Performance', 'optimize']
                ]
            }
        }
        results = self.retriever.retrieve(query)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        
        for result in results:
            self.assertIn('content', result)
            self.assertIn('metadata', result)
            self.assertIn('features', result)
            self.assertIn('score', result)

def main():
    unittest.main()

if __name__ == '__main__':
    main()