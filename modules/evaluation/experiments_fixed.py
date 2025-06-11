import os
import time
import json
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
from modules.utils.benchmark_loader import BenchmarkLoader
from modules.core.rag_system import RAGSystem
from .layout_generators import ORAssistant, GLayout, ORAssistantSim, GLayoutSim
from chiprag.config.experiment_config import ExperimentConfig

class ExperimentEvaluator:
    def __init__(self):
        # 加载实验配置
        self.config = ExperimentConfig()
        
        # 初始化基准测试加载器
        self.benchmark_loader = BenchmarkLoader(self.config.benchmark_path)
        
        # 初始化布局生成器
        self.layout_generators = {
            'ORAssistant': ORAssistant(self.config.layout_generators['ORAssistant']['config']),
            'GLayout': GLayout(self.config.layout_generators['GLayout']['config']),
            'ORAssistant-Sim': ORAssistantSim(self.config.layout_generators['ORAssistant-Sim']['config']),
            'GLayout-Sim': GLayoutSim(self.config.layout_generators['GLayout-Sim']['config'])
        }
        
        # 初始化RAG系统
        self.rag_system = None  # 暂时不初始化，等待需要时再初始化
        
        # 设置随机种子
        np.random.seed(self.config.experiment_params['seed'])
        
    def run_all_experiments(self) -> Dict[str, Any]:
        """运行所有实验"""
        results = {
            'layout_quality': {},
            'constraint_satisfaction': {},
            'optimization_efficiency': {}
        }
        
        # 对每个布局生成器运行实验
        for method_name, generator in self.layout_generators.items():
            print(f"\n运行 {method_name} 实验...")
            
            # 初始化结果
            results['layout_quality'][method_name] = {
                'density': [],
                'congestion': [],
                'timing_margin': []
            }
            results['constraint_satisfaction'][method_name] = {
                'mgc_fft_1': [],
                'mgc_des_perf_1': [],
                'mgc_pci_bridge32_a': []
            }
            results['optimization_efficiency'][method_name] = {
                'generation_time': [],
                'iterations': [],
                'total_time': []
            }
            
            # 对每个基准测试运行多次实验
            for benchmark in self.config.benchmarks:
                print(f"\n处理基准测试: {benchmark}")
                
                # 加载基准测试数据
                design_info = self.benchmark_loader.load_benchmark(benchmark)
                
                for run in range(self.config.experiment_params['num_runs']):
                    print(f"运行 {run + 1}/{self.config.experiment_params['num_runs']}")
                    
                    # 生成布局
                    start_time = time.time()
                    layout = generator.generate_layout(design_info)
                    generation_time = time.time() - start_time
                    
                    # 评估布局质量
                    quality = self._evaluate_layout_quality(layout, design_info)
                    results['layout_quality'][method_name]['density'].append(quality['density'])
                    results['layout_quality'][method_name]['congestion'].append(quality['congestion'])
                    results['layout_quality'][method_name]['timing_margin'].append(quality['timing_margin'])
                    
                    # 评估约束满足率
                    satisfaction = self._evaluate_constraint_satisfaction(layout, design_info)
                    results['constraint_satisfaction'][method_name][benchmark].append(satisfaction)
                    
                    # 评估优化效率
                    results['optimization_efficiency'][method_name]['generation_time'].append(generation_time)
                    results['optimization_efficiency'][method_name]['iterations'].append(generator.iterations)
                    results['optimization_efficiency'][method_name]['total_time'].append(time.time() - start_time)
        
        # 计算平均值
        self._calculate_averages(results)
        
        # 保存结果
        self._save_results(results)
        
        return results 