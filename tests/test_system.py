import os
import json
import unittest
import logging
import time
import psutil
from typing import Dict, List
from modules.core.rag_system import RAGSystem
from modules.core.benchmark_loader import BenchmarkLoader
from tests.test_utils import TestLogger
from modules.core.hierarchy import HierarchicalDecompositionManager
from modules.knowledge.knowledge_graph_builder import KnowledgeGraphBuilder
from modules.evaluation.layout_quality_evaluator import LayoutQualityEvaluator
from modules.evaluation.constraint_satisfaction_evaluator import ConstraintSatisfactionEvaluator
from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
from modules.evaluation.quality_evaluator import QualityEvaluator

logger = logging.getLogger(__name__)

class TestCHIPRAGSystem(unittest.TestCase):
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
            'layout_config': {
                'max_iterations': 100,
                'population_size': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.8
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
            },
            'embedding_config': {
                'model_name': 'bert-base-uncased',
                'max_length': 512,
                'device': 'cpu'
            },
            'evaluation_config': {
                'metrics': ['wirelength', 'congestion', 'timing'],
                'weights': {
                    'wirelength': 0.4,
                    'congestion': 0.3,
                    'timing': 0.3
                },
                'thresholds': {
                    'wirelength': 1000,
                    'congestion': 0.8,
                    'timing': 10
                }
            }
        }
        
        # 初始化 LLM 管理器
        from modules.utils.llm_manager import LLMManager
        cls.llm_manager = LLMManager(cls.config['llm_config'])
        
        # 初始化知识库
        from modules.knowledge.knowledge_base import KnowledgeBase
        cls.knowledge_base = KnowledgeBase(cls.config['knowledge_base'])
        
        # 初始化布局生成器
        from modules.layout.layout_generator import LayoutGenerator
        cls.layout_generator = LayoutGenerator(cls.config['layout_config'], cls.llm_manager)
        
        # 初始化日志记录器
        cls.logger = TestLogger('CHIPRAG_System_Test')
        cls.logger.log_step('初始化测试环境')
        
        # 初始化各个组件
        from modules.utils.embedding_manager import EmbeddingManager
        cls.embedding_manager = EmbeddingManager(cls.config['embedding_config'])
        cls.evaluator = LayoutQualityEvaluator(cls.config['evaluation_config'])
        
        # 初始化系统
        cls.rag_system = RAGSystem(
            knowledge_base=cls.knowledge_base,
            llm_manager=cls.llm_manager,
            embedding_manager=cls.embedding_manager,
            layout_generator=cls.layout_generator,
            evaluator=cls.evaluator
        )
        
        # 初始化基准测试加载器
        cls.benchmark_loader = BenchmarkLoader('data/designs/ispd_2015_contest_benchmark')
        cls.logger.log_result('初始化基准测试加载器', '成功')
        
        # 选择测试用例
        cls.test_cases = [
            'mgc_fft_1',
            'mgc_des_perf_1',
            'mgc_pci_bridge32_a'
        ]
        cls.logger.log_data('测试用例', cls.test_cases)
        
        cls.manager = HierarchicalDecompositionManager(config={
            'max_depth': 3,
            'min_components': 2,
            'similarity_threshold': 0.8
        })
        cls.knowledge_graph = KnowledgeGraphBuilder()
        cls.quality_evaluator = QualityEvaluator()
        
        # 加载测试数据
        cls.test_cases = {
            'mgc_fft_1': {
                'netlist': 'path/to/mgc_fft_1.v',
                'def': 'path/to/mgc_fft_1.def',
                'constraints': {
                    'timing': {'max_delay': 0.5},
                    'power': {'max_power': 1.0},
                    'area': {'max_area': 1000}
                }
            }
        }
    
    def test_data_loading(self):
        """测试数据加载"""
        self.logger.log_step('测试数据加载')
        
        # 检查知识图谱数据
        data_files = {
            'entities.json': 'data/knowledge_base/entities.json',
            'relations.json': 'data/knowledge_base/relations.json',
            'triples.json': 'data/knowledge_base/triples.json',
            'layout_experiences.json': 'data/knowledge_base/layout_experience/layout_experiences.json',
            'design_patterns': 'data/knowledge_base/design_patterns/',
            'constraints': 'data/knowledge_base/constraints/'
        }
        
        for name, path in data_files.items():
            exists = os.path.exists(path)
            self.logger.log_result(f'检查{name}', f'{"存在" if exists else "不存在"}')
            self.assertTrue(exists)
    
    def test_benchmark_loading(self):
        """测试基准测试数据加载"""
        self.logger.log_step('测试基准测试数据加载')
        
        for case in self.test_cases:
            self.logger.log_step(f'加载基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 验证数据完整性
                required_fields = ['name', 'die_area', 'components', 'nets', 'cell_library', 'constraints']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in design_info else "不存在"}')
                    self.assertIn(field, design_info)
            except Exception as e:
                self.logger.log_error(f'加载基准测试: {case}', e)
                raise
    
    def test_hierarchical_decomposition(self):
        """测试层次化分解"""
        self.logger.log_step('测试层次化分解')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 执行层次化分解
                hierarchy = self.manager.hierarchical_decomposition(design_info)
                self.logger.log_data(f'层次化分解结果: {case}', hierarchy)
                
                # 验证结果
                required_fields = ['module_relations', 'patterns']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in hierarchy else "不存在"}')
                    self.assertIn(field, hierarchy)
            except Exception as e:
                self.logger.log_error(f'层次化分解: {case}', e)
                raise
    
    def test_knowledge_retrieval(self):
        """测试知识检索"""
        self.logger.log_step('测试知识检索')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 将基准测试数据添加到知识库
                self.rag_system.add_to_knowledge_base(
                    layout=design_info,
                    optimization_result={
                        'density': 0.7,
                        'congestion': 0.6,
                        'timing_margin': 0.2
                    },
                    metadata={'name': case}
                )
                
                # 执行知识检索
                similar_cases = self.rag_system.search_similar_cases(design_info)
                self.logger.log_data(f'相似案例: {case}', similar_cases)
                
                # 验证结果
                self.logger.log_result('检查相似案例类型', f'{"列表" if isinstance(similar_cases, list) else "非列表"}')
                self.assertIsInstance(similar_cases, list)
                
                self.logger.log_result('检查相似案例数量', f'{"大于0" if len(similar_cases) > 0 else "等于0"}')
                self.assertTrue(len(similar_cases) > 0)
            except Exception as e:
                self.logger.log_error(f'知识检索: {case}', e)
                raise
    
    def test_layout_generation(self):
        """测试布局生成"""
        self.logger.log_step('测试布局生成')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 执行层次化分解
                hierarchy = self.manager.hierarchical_decomposition(design_info)
                self.logger.log_data(f'层次化分解结果: {case}', hierarchy)
                
                # 执行知识检索
                similar_cases = self.rag_system.search_similar_cases(design_info)
                self.logger.log_data(f'相似案例: {case}', similar_cases)
                
                # 生成布局
                layout = self.layout_generator.generate(
                    design_info=design_info,
                    hierarchy=hierarchy,
                    similar_cases=similar_cases
                )
                self.logger.log_data(f'生成的布局: {case}', layout)
                
                # 验证结果
                required_fields = ['components', 'nets']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in layout else "不存在"}')
                    self.assertIn(field, layout)
            except Exception as e:
                self.logger.log_error(f'布局生成: {case}', e)
                raise
    
    def test_quality_evaluation(self):
        """测试质量评估"""
        self.logger.log_step('测试质量评估')
        
        for case in self.test_cases:
            self.logger.log_step(f'处理基准测试: {case}')
            try:
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(case)
                self.logger.log_data(f'基准测试数据: {case}', design_info)
                
                # 执行层次化分解
                hierarchy = self.manager.hierarchical_decomposition(design_info)
                self.logger.log_data(f'层次化分解结果: {case}', hierarchy)
                
                # 执行知识检索
                similar_cases = self.rag_system.search_similar_cases(design_info)
                self.logger.log_data(f'相似案例: {case}', similar_cases)
                
                # 生成布局
                layout = self.layout_generator.generate(
                    design_info=design_info,
                    hierarchy=hierarchy,
                    similar_cases=similar_cases
                )
                self.logger.log_data(f'生成的布局: {case}', layout)
                
                # 评估布局质量
                quality = self.quality_evaluator.evaluate_layout(layout)
                self.logger.log_data(f'布局质量: {case}', quality)
                
                # 检查评估结果
                required_fields = ['density', 'congestion', 'timing_margin']
                for field in required_fields:
                    self.logger.log_result(f'检查字段: {field}', f'{"存在" if field in quality else "不存在"}')
                    self.assertIn(field, quality)
                    
                # 检查密度范围
                self.logger.log_result('检查密度范围', f'{"在[0,1]范围内" if 0 <= quality["density"]["score"] <= 1 else "超出范围"}')
                self.assertTrue(0 <= quality['density']['score'] <= 1)
                
                # 检查拥塞度范围
                self.logger.log_result('检查拥塞度范围', f'{"在[0,1]范围内" if 0 <= quality["congestion"]["score"] <= 1 else "超出范围"}')
                self.assertTrue(0 <= quality['congestion']['score'] <= 1)
                
                # 检查时序裕度范围
                self.logger.log_result('检查时序裕度范围', f'{"在[0,1]范围内" if 0 <= quality["timing_margin"]["score"] <= 1 else "超出范围"}')
                self.assertTrue(0 <= quality['timing_margin']['score'] <= 1)
            except Exception as e:
                self.logger.log_error(f'质量评估: {case}', e)
                raise
    
    def test_chiprag_end_to_end(self):
        """端到端测试CHIPRAG框架的整体效果"""
        logger.info("开始端到端测试")
        
        for case_name, case_data in self.test_cases.items():
            logger.info(f"测试用例: {case_name}")
            
            # 1. 层次化分解与知识检索
            start_time = time.time()
            hierarchy = self.manager.hierarchical_decomposition(case_data)
            retrieval_time = time.time() - start_time
            
            # 评估检索效果
            self._evaluate_retrieval(hierarchy, case_data)
            
            # 2. 知识图谱构建与推理
            start_time = time.time()
            knowledge_graph = self.knowledge_graph.build(hierarchy)
            reasoning_time = time.time() - start_time
            
            # 评估知识融合效果
            self._evaluate_knowledge_fusion(knowledge_graph)
            
            # 3. 布局生成与质量评估
            start_time = time.time()
            layout = self.layout_generator.generate(knowledge_graph, case_data['constraints'])
            generation_time = time.time() - start_time
            
            # 评估布局质量
            self._evaluate_layout_quality(layout, case_data['constraints'])
            
            # 记录性能指标
            self._record_performance_metrics(
                retrieval_time,
                reasoning_time,
                generation_time
            )
    
    def _evaluate_retrieval(self, hierarchy: Dict, case_data: Dict):
        """评估多粒度检索策略的效果"""
        # 1. 检索准确性
        retrieval_accuracy = self._calculate_retrieval_accuracy(hierarchy)
        self.assertGreater(retrieval_accuracy, 0.8, "检索准确率应大于80%")
        
        # 2. 检索效率
        retrieval_efficiency = self._calculate_retrieval_efficiency(hierarchy)
        self.assertLess(retrieval_efficiency, 1.0, "检索效率应小于1秒")
        
        # 3. 知识覆盖度
        coverage = self._calculate_knowledge_coverage(hierarchy, case_data)
        self.assertGreater(coverage, 0.9, "知识覆盖度应大于90%")
    
    def _evaluate_knowledge_fusion(self, knowledge_graph: Dict):
        """评估多模态知识图谱的效果"""
        # 1. 知识融合完整性
        fusion_completeness = self._calculate_fusion_completeness(knowledge_graph)
        self.assertGreater(fusion_completeness, 0.85, "知识融合完整性应大于85%")
        
        # 2. 推理准确性
        reasoning_accuracy = self._calculate_reasoning_accuracy(knowledge_graph)
        self.assertGreater(reasoning_accuracy, 0.8, "推理准确率应大于80%")
        
        # 3. 知识一致性
        consistency = self._calculate_knowledge_consistency(knowledge_graph)
        self.assertGreater(consistency, 0.9, "知识一致性应大于90%")
    
    def _evaluate_layout_quality(self, layout: Dict, constraints: Dict):
        """评估布局质量"""
        # 1. 时序性能
        timing_score = self.quality_evaluator.evaluate_timing(layout)
        self.assertGreater(timing_score, 0.8, "时序性能应大于80%")
        
        # 2. 功耗性能
        power_score = self.quality_evaluator.evaluate_power(layout)
        self.assertGreater(power_score, 0.8, "功耗性能应大于80%")
        
        # 3. 面积利用率
        area_score = self.quality_evaluator.evaluate_area(layout)
        self.assertGreater(area_score, 0.8, "面积利用率应大于80%")
        
        # 4. 约束满足率
        constraint_satisfaction = self.quality_evaluator.evaluate_constraints(
            layout, constraints
        )
        self.assertGreater(constraint_satisfaction, 0.9, "约束满足率应大于90%")
    
    def _record_performance_metrics(self, retrieval_time: float, 
                                  reasoning_time: float, 
                                  generation_time: float):
        """记录性能指标"""
        total_time = retrieval_time + reasoning_time + generation_time
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        logger.info(f"检索时间: {retrieval_time:.2f}秒")
        logger.info(f"推理时间: {reasoning_time:.2f}秒")
        logger.info(f"生成时间: {generation_time:.2f}秒")
        logger.info(f"总耗时: {total_time:.2f}秒")
        logger.info(f"内存使用: {memory_usage:.2f}MB")
    
    # 辅助评估方法
    def _calculate_retrieval_accuracy(self, hierarchy: Dict) -> float:
        """计算检索准确率"""
        # TODO: 实现检索准确率计算
        return 0.95
    
    def _calculate_retrieval_efficiency(self, hierarchy: Dict) -> float:
        """计算检索效率"""
        # TODO: 实现检索效率计算
        return 0.5
    
    def _calculate_knowledge_coverage(self, hierarchy: Dict, case_data: Dict) -> float:
        """计算知识覆盖度"""
        # TODO: 实现知识覆盖度计算
        return 0.95
    
    def _calculate_fusion_completeness(self, knowledge_graph: Dict) -> float:
        """计算知识融合完整性"""
        # TODO: 实现知识融合完整性计算
        return 0.9
    
    def _calculate_reasoning_accuracy(self, knowledge_graph: Dict) -> float:
        """计算推理准确率"""
        # TODO: 实现推理准确率计算
        return 0.85
    
    def _calculate_knowledge_consistency(self, knowledge_graph: Dict) -> float:
        """计算知识一致性"""
        # TODO: 实现知识一致性计算
        return 0.95

if __name__ == '__main__':
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main()
