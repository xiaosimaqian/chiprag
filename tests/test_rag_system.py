import unittest
import json
from pathlib import Path
from run_rag import RAGController
import logging

class TestRAGSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """设置测试环境"""
        cls.logger = logging.getLogger('CHIPRAG_System_Test')
        cls.logger.info("开始步骤: 初始化测试环境")
        
        # 加载配置
        with open('configs/system.json', 'r') as f:
            cls.config = json.load(f)
        
        # 初始化RAG系统
        cls.controller = RAGController(cls.config)
        
        cls.logger.info("测试环境初始化完成")
        
    def test_system_initialization(self):
        """测试系统初始化"""
        self.assertIsNotNone(self.controller.knowledge_base)
        self.assertIsNotNone(self.controller.llm_manager)
        self.assertIsNotNone(self.controller.embedding_manager)
        self.assertIsNotNone(self.controller.layout_generator)
        self.assertIsNotNone(self.controller.evaluator)
        self.assertIsNotNone(self.controller.rag_system)
        
    def test_knowledge_retrieval(self):
        """测试知识检索和增强"""
        design_spec = {
            "name": "test_design",
            "area": {
                "width": 1000,
                "height": 1000
            },
            "components": [
                {
                    "name": "comp1",
                    "type": "macro",
                    "width": 100,
                    "height": 100
                }
            ],
            "nets": [
                {
                    "name": "net1",
                    "pins": [
                        {"component": "comp1", "x": 50, "y": 50}
                    ]
                }
            ],
            "constraints": {
                "max_wirelength": 1000,
                "max_congestion": 0.8
            }
        }
        
        # 使用完整的检索和增强功能
        enhanced_knowledge = self.controller.rag_system.retrieve_and_enhance(
            design_input=design_spec,
            constraints=design_spec["constraints"]
        )
        
        # 验证返回的知识结构
        self.assertIsNotNone(enhanced_knowledge)
        self.assertIsInstance(enhanced_knowledge, dict)
        
        # 验证基本知识结构
        self.assertIn('area_utilization', enhanced_knowledge)
        self.assertIn('routing_quality', enhanced_knowledge)
        self.assertIn('timing_performance', enhanced_knowledge)
        self.assertIn('power_distribution', enhanced_knowledge)
        
        # 验证增强信息
        self.assertIn('enhanced_suggestions', enhanced_knowledge)
        self.assertIn('optimization_metrics', enhanced_knowledge)
        
        # 验证元信息
        self.assertIn('metadata', enhanced_knowledge)
        self.assertIn('source', enhanced_knowledge['metadata'])
        self.assertIn('timestamp', enhanced_knowledge['metadata'])
        self.assertIn('version', enhanced_knowledge['metadata'])
        
    def test_layout_generation(self):
        """测试布局生成"""
        design_spec = {
            "name": "test_design",
            "area": {
                "width": 1000,
                "height": 1000
            },
            "die_area": {
                "width": 1000,
                "height": 1000
            },
            "components": [
                {
                    "name": "comp1",
                    "type": "macro",
                    "width": 100,
                    "height": 100
                }
            ],
            "nets": [
                {
                    "name": "net1",
                    "pins": [
                        {"component": "comp1", "x": 50, "y": 50}
                    ]
                }
            ],
            "constraints": {
                "max_wirelength": 1000,
                "max_congestion": 0.8
            }
        }
        
        # 1. 检索和增强知识
        enhanced_knowledge = self.controller.rag_system.retrieve_and_enhance(
            design_input=design_spec,
            constraints=design_spec["constraints"]
        )
        
        # 2. 使用完整的布局生成功能
        layout = self.controller.layout_generator.generate_layout(
            design_info=design_spec,
            hierarchy_info={},  # 空层次信息，表示顶层设计
            knowledge_base=enhanced_knowledge  # 使用增强后的知识
        )
        
        # 验证布局结构
        self.assertIsNotNone(layout)
        self.assertIsInstance(layout, dict)
        self.assertIn("components", layout)
        self.assertIn("nets", layout)
        
        # 验证组件位置
        self.assertEqual(len(layout["components"]), 1)
        comp = layout["components"][0]
        self.assertEqual(comp["name"], "comp1")
        self.assertEqual(comp["type"], "macro")
        self.assertEqual(comp["width"], 100)
        self.assertEqual(comp["height"], 100)
        self.assertIn("x", comp)
        self.assertIn("y", comp)
        
    def test_layout_evaluation(self):
        """测试布局评估"""
        layout = {
            "components": [
                {
                    "name": "comp1",
                    "type": "macro",
                    "x": 0,
                    "y": 0,
                    "width": 100,
                    "height": 100
                }
            ],
            "nets": [
                {
                    "name": "net1",
                    "pins": [
                        {"component": "comp1", "x": 50, "y": 50}
                    ]
                }
            ]
        }
        
        evaluation = self.controller.evaluator.evaluate(layout)
        self.assertIsNotNone(evaluation)
        self.assertIsInstance(evaluation, dict)
        self.assertIn("wirelength", evaluation)
        self.assertIn("congestion", evaluation)
        self.assertIn("timing", evaluation)
        
    def test_full_workflow(self):
        """测试完整工作流程"""
        design_spec = {
            "name": "test_design",
            "area": {
                "width": 1000,
                "height": 1000
            },
            "die_area": {
                "width": 1000,
                "height": 1000
            },
            "components": [
                {
                    "name": "comp1",
                    "type": "macro",
                    "width": 100,
                    "height": 100
                }
            ],
            "nets": [
                {
                    "name": "net1",
                    "pins": [
                        {"component": "comp1", "x": 50, "y": 50}
                    ]
                }
            ],
            "constraints": {
                "max_wirelength": 1000,
                "max_congestion": 0.8
            }
        }
        
        # 1. 检索和增强知识
        enhanced_knowledge = self.controller.rag_system.retrieve_and_enhance(
            design_input=design_spec,
            constraints=design_spec["constraints"]
        )
        
        # 2. 生成布局
        layout = self.controller.layout_generator.generate_layout(
            design_info=design_spec,
            hierarchy_info={},
            knowledge_base=enhanced_knowledge  # 使用增强后的知识
        )
        
        # 3. 评估布局
        evaluation = self.controller.evaluator.evaluate(layout)
        
        # 验证结果
        self.assertIsNotNone(enhanced_knowledge)
        self.assertIsNotNone(layout)
        self.assertIsNotNone(evaluation)
        
        # 验证知识结构
        self.assertIn('area_utilization', enhanced_knowledge)
        self.assertIn('routing_quality', enhanced_knowledge)
        self.assertIn('timing_performance', enhanced_knowledge)
        self.assertIn('power_distribution', enhanced_knowledge)
        self.assertIn('enhanced_suggestions', enhanced_knowledge)
        self.assertIn('optimization_metrics', enhanced_knowledge)
        
        # 验证布局结构
        self.assertIn("components", layout)
        self.assertIn("nets", layout)
        
        # 验证评估结果
        self.assertIn("wirelength_score", evaluation)
        self.assertIn("congestion_score", evaluation)
        self.assertIn("timing_score", evaluation)
        self.assertIn("overall_score", evaluation)
        
if __name__ == '__main__':
    unittest.main() 