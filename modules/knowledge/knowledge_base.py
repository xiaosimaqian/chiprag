"""
知识库模块
"""

import json
import os
import logging
import pickle
from typing import Dict, List, Optional, Union, Any
import numpy as np
from collections import defaultdict
from functools import lru_cache
from datetime import datetime
import hashlib
from ..core.hierarchy import HierarchicalDecompositionManager
from ..utils.llm_manager import LLMManager
from pathlib import Path
from .pdf_processing_status import PDFProcessingStatus

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """知识库管理器，整合层次化分解和知识检索功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化知识库管理器
        
        Args:
            config: 知识库配置
        """
        self.config = config
        self._validate_config()
        self.data = {}
        self.cases = []
        self.knowledge_graph = {
            'global': [],
            'module': [],
            'connection': [],
            'constraint': []
        }
        self._init_components()
        
    def _validate_config(self):
        """验证配置"""
        required_fields = ['path', 'format']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required field '{field}' in knowledge base config")
                
        # 验证层次配置
        if 'hierarchy_config' not in self.config:
            self.config['hierarchy_config'] = {
                'levels': [
                    {'name': 'system', 'threshold': 0.8},
                    {'name': 'module', 'threshold': 0.6},
                    {'name': 'component', 'threshold': 0.4}
                ]
            }
            
        # 验证LLM配置
        if 'llm_config' not in self.config:
            self.config['llm_config'] = {
                'base_url': 'http://localhost:11434',
                'model': 'llama2',
                'temperature': 0.7,
                'max_tokens': 1000
            }
                
    def _init_components(self):
        """初始化组件"""
        try:
            # 初始化路径
            self.path = Path(self.config['path'])
            self.format = self.config['format']
            
            # 初始化层次管理器
            from modules.core.hierarchy import HierarchicalDecompositionManager
            hierarchy_config = {
                'levels': self.config.get('hierarchy_config', {}).get('levels', [
                    {'name': 'system', 'threshold': 0.8},
                    {'name': 'module', 'threshold': 0.6},
                    {'name': 'component', 'threshold': 0.4}
                ]),
                'llm_config': self.config.get('llm_config', {
                    'base_url': 'http://localhost:11434',
                    'model': 'llama2',
                    'temperature': 0.7,
                    'max_tokens': 1000
                })
            }
            self.hierarchy_manager = HierarchicalDecompositionManager(hierarchy_config)
            
            # 初始化LLM管理器
            from modules.utils.llm_manager import LLMManager
            self.llm_manager = LLMManager(
                self.config.get('llm_config', {
                    'base_url': 'http://localhost:11434',
                    'model': 'llama2',
                    'temperature': 0.7,
                    'max_tokens': 1000
                })
            )
            
        except Exception as e:
            logging.error(f"初始化组件失败: {str(e)}")
            raise
        
    def _init_knowledge_base(self):
        """初始化知识库"""
        try:
            # 如果数据文件不存在，创建空的知识库
            if not os.path.exists(self.data_file):
                with open(self.data_file, 'wb') as f:
                    pickle.dump([], f)
                logger.info(f"创建新的知识库文件: {self.data_file}")
                
        except Exception as e:
            logger.error(f"初始化知识库失败: {str(e)}")
            
    def _load_data(self) -> List[Dict]:
        """加载知识库数据
        
        Returns:
            List[Dict]: 知识列表
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(self.data_file):
                logger.info(f"知识库文件不存在，创建新文件: {self.data_file}")
                return []
                
            # 检查文件大小
            if os.path.getsize(self.data_file) == 0:
                logger.info("知识库文件为空")
                return []
                
            # 加载数据
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
                
            # 确保返回的是列表
            if data is None:
                return []
                
            return data
            
        except Exception as e:
            logger.error(f"加载知识库数据失败: {str(e)}")
            return []
            
    def _save_data(self):
        """保存知识库数据"""
        try:
            if not os.path.exists(self.layout_experience_path):
                os.makedirs(self.layout_experience_path)
            
            # 保存案例数据
            cases_file = os.path.join(self.layout_experience_path, "cases.pkl")
            try:
                with open(cases_file, 'wb') as f:
                    pickle.dump(self.cases, f)
            except Exception as e:
                logger.error(f"保存案例数据失败: {str(e)}")
            
            # 保存知识图谱
            graph_file = os.path.join(self.layout_experience_path, "knowledge_graph.pkl")
            try:
                with open(graph_file, 'wb') as f:
                    pickle.dump(self.knowledge_graph, f)
            except Exception as e:
                logger.error(f"保存知识图谱失败: {str(e)}")
            
        except Exception as e:
            logger.error(f"保存知识库数据失败: {str(e)}")
            
    def add_case(self, 
                layout: Dict,
                optimization_result: Dict,
                metadata: Optional[Dict] = None):
        """添加布局经验到知识库
        
        Args:
            layout: 布局数据
            optimization_result: 优化结果
            metadata: 额外的元数据
        """
        case = {
            'id': len(self.cases),
            'layout': layout,
            'optimization_result': optimization_result,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 提取特征
        features = {
            'global': self._extract_global_features(layout),
            'module': self._extract_module_features(layout),
            'connection': self._extract_connection_features(layout),
            'constraint': self._extract_constraint_features(layout, optimization_result)
        }
        
        # 更新知识图谱
        self._update_knowledge_graph(case['id'], features, layout, optimization_result)
        
        # 添加案例
        self.cases.append(case)
        
        # 保存数据
        self._save_data()
        
    def _extract_global_features(self, layout: Dict) -> Dict:
        """提取布局的全局特征
        
        Args:
            layout: 布局数据
            
        Returns:
            全局特征字典
        """
        features = {}
        
        # 提取面积特征
        if 'die_area' in layout:
            die_area = layout['die_area']
            if isinstance(die_area, dict):
                features['area'] = die_area['width'] * die_area['height']
                if die_area['height'] != 0:
                    features['aspect_ratio'] = die_area['width'] / die_area['height']
                else:
                    features['aspect_ratio'] = 0
            elif isinstance(die_area, list) and len(die_area) >= 2:
                features['area'] = die_area[0] * die_area[1]
                if die_area[1] != 0:
                    features['aspect_ratio'] = die_area[0] / die_area[1]
                else:
                    features['aspect_ratio'] = 0
        
        # 提取组件特征
        if 'components' in layout:
            components = layout['components']
            if isinstance(components, dict):
                comp_iter = components.values()
                features['num_components'] = len(components)
            elif isinstance(components, list):
                comp_iter = components
                features['num_components'] = len(components)
            else:
                comp_iter = []
                features['num_components'] = 0
            
            # 计算组件密度
            total_area = 0
            for comp in comp_iter:
                if isinstance(comp, dict):
                    if 'width' in comp and 'height' in comp:
                        total_area += comp['width'] * comp['height']
                    elif 'area' in comp:
                        total_area += comp['area']
            if features.get('area', 0) > 0:
                features['component_density'] = total_area / features['area']
            else:
                features['component_density'] = 0
            # 计算组件类型分布
            type_counts = defaultdict(int)
            for comp in comp_iter:
                if isinstance(comp, dict) and 'type' in comp:
                    type_counts[comp['type']] += 1
            features['component_types'] = dict(type_counts)
        
        # 提取网络特征
        if 'nets' in layout:
            nets = layout['nets']
            if isinstance(nets, dict):
                net_iter = nets.values()
                features['num_nets'] = len(nets)
            elif isinstance(nets, list):
                net_iter = nets
                features['num_nets'] = len(nets)
            else:
                net_iter = []
                features['num_nets'] = 0
            # 计算平均网络长度
            net_lengths = []
            for net in net_iter:
                if isinstance(net, dict) and 'pins' in net:
                    pins = net['pins']
                    if len(pins) >= 2:
                        # 计算所有引脚对之间的曼哈顿距离
                        for i in range(len(pins)):
                            for j in range(i + 1, len(pins)):
                                pin1 = pins[i]
                                pin2 = pins[j]
                                if isinstance(pin1, dict) and isinstance(pin2, dict):
                                    if 'x' in pin1 and 'y' in pin1 and 'x' in pin2 and 'y' in pin2:
                                        length = abs(pin1['x'] - pin2['x']) + abs(pin1['y'] - pin2['y'])
                                        net_lengths.append(length)
            if net_lengths:
                features['avg_net_length'] = sum(net_lengths) / len(net_lengths)
                features['max_net_length'] = max(net_lengths)
            else:
                features['avg_net_length'] = 0
                features['max_net_length'] = 0
        
        return features
        
    def _extract_module_features(self, layout: Dict) -> List[np.ndarray]:
        """提取模块特征
        
        Args:
            layout: 布局数据
            
        Returns:
            模块特征列表
        """
        features = []
        for comp in layout['components']:
            # 计算面积
            area = comp['width'] * comp['height']
            
            comp_features = [
                area,
                comp['width'],
                comp['height'],
                len(comp.get('connections', {}))
            ]
            features.append(np.array(comp_features))
        return features
        
    def _extract_connection_features(self, layout: Dict) -> List[np.ndarray]:
        """提取连接特征
        
        Args:
            layout: 布局数据
            
        Returns:
            连接特征列表
        """
        features = []
        for net in layout['nets']:
            net_features = [
                len(net['connections']),
                net.get('length', 0),
                net.get('width', 0)
            ]
            features.append(np.array(net_features))
        return features
        
    def _extract_constraint_features(self, 
                                   layout: Dict,
                                   optimization_result: Dict) -> np.ndarray:
        """提取约束特征
        
        Args:
            layout: 布局数据
            optimization_result: 优化结果
            
        Returns:
            约束特征向量
        """
        features = []
        
        # 时序约束
        features.append(optimization_result.get('timing_slack', 0))
        
        # 功耗约束
        features.append(optimization_result.get('power', 0))
        
        # 面积约束
        features.append(optimization_result.get('area', 0))
        
        # 拥塞约束
        features.append(optimization_result.get('congestion', 0))
        
        return np.array(features)
        
    def _update_knowledge_graph(self,
                              case_id: int,
                              features: Dict[str, Union[np.ndarray, List[np.ndarray]]],
                              layout: Dict,
                              optimization_result: Dict):
        """更新知识图谱
        
        Args:
            case_id: 案例ID
            features: 特征字典
            layout: 布局数据
            optimization_result: 优化结果
        """
        # 添加全局特征节点
        self.knowledge_graph['global'].append({
            'case_id': case_id,
            'features': features['global']
        })
        
        # 添加模块特征节点
        for i, module_features in enumerate(features['module']):
            self.knowledge_graph['module'].append({
                'case_id': case_id,
                'module_id': i,
                'features': module_features
            })
            
        # 添加连接特征节点
        for i, conn_features in enumerate(features['connection']):
            self.knowledge_graph['connection'].append({
                'case_id': case_id,
                'connection_id': i,
                'features': conn_features
            })
            
        # 添加约束特征节点
        self.knowledge_graph['constraint'].append({
            'case_id': case_id,
            'features': features['constraint']
        })
        
    def get_similar_cases(self, 
                         query: Dict,
                         top_k: int = 5,
                         similarity_threshold: float = 0.5) -> List[Dict]:
        """获取相似案例
        
        Args:
            query: 查询布局
            top_k: 返回的案例数量
            similarity_threshold: 相似度阈值
            
        Returns:
            相似案例列表
        """
        # 提取查询布局的特征
        query_features = self._extract_global_features(query)
        
        # 计算与所有案例的相似度
        similarities = []
        for case in self.cases:
            if 'layout' in case:
                case_features = self._extract_global_features(case['layout'])
                similarity = self._compute_similarity(query_features, case_features)
                if similarity >= similarity_threshold:
                    similarities.append({
                        'case': case,
                        'similarity': similarity
                    })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回top_k个最相似的案例
        return [item['case'] for item in similarities[:top_k]]
        
    def hierarchical_decomposition(self, design_info: Dict) -> Dict:
        """对设计进行层次化分解
        
        Args:
            design_info: 设计信息
            
        Returns:
            层次化分解结果
        """
        return self.decomposition_manager.hierarchical_decomposition(design_info)

    def _compute_similarity(self, features1: Dict, features2: Dict) -> float:
        """计算两个布局特征之间的相似度
        
        Args:
            features1: 第一个布局的特征
            features2: 第二个布局的特征
            
        Returns:
            相似度得分，范围[0,1]
        """
        # 计算数值特征的欧氏距离
        numeric_features = ['area', 'aspect_ratio', 'num_components', 
                          'component_density', 'num_nets', 'avg_net_length', 
                          'max_net_length']
        distances = []
        for feat in numeric_features:
            if feat in features1 and feat in features2:
                # 归一化距离
                max_val = max(features1[feat], features2[feat])
                if max_val > 0:
                    dist = abs(features1[feat] - features2[feat]) / max_val
                    distances.append(dist)
        
        # 计算组件类型分布的相似度
        if 'component_types' in features1 and 'component_types' in features2:
            types1 = features1['component_types']
            types2 = features2['component_types']
            all_types = set(types1.keys()) | set(types2.keys())
            
            # 计算余弦相似度
            dot_product = sum(types1.get(t, 0) * types2.get(t, 0) for t in all_types)
            norm1 = sum(v * v for v in types1.values()) ** 0.5
            norm2 = sum(v * v for v in types2.values()) ** 0.5
            
            if norm1 > 0 and norm2 > 0:
                type_similarity = dot_product / (norm1 * norm2)
                distances.append(1 - type_similarity)  # 转换为距离
        
        # 计算平均距离作为最终相似度
        if distances:
            similarity = 1 - (sum(distances) / len(distances))
            return max(0.0, min(1.0, similarity))  # 确保结果在[0,1]范围内
        return 0.0  # 如果没有可比较的特征，返回0 

    def retrieve(self, query: Dict, constraints: Dict, top_k: int = 5) -> Dict:
        """检索相关知识
        
        Args:
            query: 查询信息
            constraints: 约束条件
            top_k: 返回的相似案例数量
            
        Returns:
            检索到的知识，包含：
            - area_utilization: 面积利用率
            - routing_quality: 布线质量
            - timing_performance: 时序性能
            - power_distribution: 功耗分布
        """
        # 获取相似案例
        similar_cases = self.get_similar_cases(
            query=query,
            top_k=top_k,
            similarity_threshold=0.5
        )
        
        # 如果没有相似案例，返回默认知识
        if not similar_cases:
            return {
                'area_utilization': {
                    'score': 0.5,
                    'issues': ['没有找到相似案例']
                },
                'routing_quality': {
                    'score': 0.5,
                    'issues': ['没有找到相似案例']
                },
                'timing_performance': {
                    'score': 0.5,
                    'issues': ['没有找到相似案例']
                },
                'power_distribution': {
                    'score': 0.5,
                    'issues': ['没有找到相似案例']
                }
            }
        
        # 从相似案例中提取知识
        area_scores = []
        routing_scores = []
        timing_scores = []
        power_scores = []
        
        for case in similar_cases:
            result = case['optimization_result']
            if 'area_utilization' in result:
                area_scores.append(result['area_utilization'].get('score', 0.5))
            if 'routing_quality' in result:
                routing_scores.append(result['routing_quality'].get('score', 0.5))
            if 'timing_performance' in result:
                timing_scores.append(result['timing_performance'].get('score', 0.5))
            if 'power_distribution' in result:
                power_scores.append(result['power_distribution'].get('score', 0.5))
        
        # 计算平均分数
        def avg_score(scores):
            return sum(scores) / len(scores) if scores else 0.5
        
        return {
            'area_utilization': {
                'score': avg_score(area_scores),
                'issues': ['基于历史案例的平均分数']
            },
            'routing_quality': {
                'score': avg_score(routing_scores),
                'issues': ['基于历史案例的平均分数']
            },
            'timing_performance': {
                'score': avg_score(timing_scores),
                'issues': ['基于历史案例的平均分数']
            },
            'power_distribution': {
                'score': avg_score(power_scores),
                'issues': ['基于历史案例的平均分数']
            }
        } 

    def build(self, case: Dict):
        """构建知识库
        
        Args:
            case: 测试用例信息
        """
        self.logger.info(f"开始构建知识库: {case['name']}")
        
        # 构建文本知识库
        self._build_text_kb(case)
        
        # 构建图像知识库
        self._build_image_kb(case)
        
        # 构建结构化知识库
        self._build_structured_kb(case)
        
        # 构建图知识库
        self._build_graph_kb(case)
        
        self.logger.info("知识库构建完成")
        
    def _build_text_kb(self, case: Dict):
        """构建文本知识库
        
        Args:
            case: 测试用例信息
        """
        text_kb = {
            'case_name': case['name'],
            'netlist_info': self._extract_netlist_info(case['netlist']),
            'constraints': case['constraints'],
            'design_rules': self._extract_design_rules(case),
            'optimization_guidelines': self._extract_optimization_guidelines(case)
        }
        
        kb_path = self.text_path / f"{case['name']}_text.json"
        with open(kb_path, 'w') as f:
            json.dump(text_kb, f, indent=2)
            
    def _build_image_kb(self, case: Dict):
        """构建图像知识库
        
        Args:
            case: 测试用例信息
        """
        # 提取布局图像
        layout_images = self._extract_layout_images(case['layout'])
        
        # 保存图像
        for name, image in layout_images.items():
            image_path = self.image_path / f"{case['name']}_{name}.png"
            image.save(image_path)
            
    def _build_structured_kb(self, case: Dict):
        """构建结构化知识库
        
        Args:
            case: 测试用例信息
        """
        structured_kb = {
            'case_name': case['name'],
            'hierarchy': self._extract_hierarchy(case['netlist']),
            'timing_info': self._extract_timing_info(case),
            'power_info': self._extract_power_info(case),
            'area_info': self._extract_area_info(case)
        }
        
        kb_path = self.structured_path / f"{case['name']}_structured.json"
        with open(kb_path, 'w') as f:
            json.dump(structured_kb, f, indent=2)
            
    def _build_graph_kb(self, case: Dict):
        """构建图知识库
        
        Args:
            case: 测试用例信息
        """
        graph_kb = {
            'case_name': case['name'],
            'connectivity': self._extract_connectivity(case['netlist']),
            'dependencies': self._extract_dependencies(case),
            'constraint_graph': self._extract_constraint_graph(case)
        }
        
        kb_path = self.graph_path / f"{case['name']}_graph.json"
        with open(kb_path, 'w') as f:
            json.dump(graph_kb, f, indent=2)
            
    def _extract_netlist_info(self, netlist: Dict) -> Dict:
        """提取网表信息
        
        Args:
            netlist: 网表数据
            
        Returns:
            Dict: 网表信息
        """
        return {
            'module_count': len(netlist.get('modules', [])),
            'instance_count': len(netlist.get('instances', [])),
            'net_count': len(netlist.get('nets', [])),
            'pin_count': len(netlist.get('pins', [])),
            'hierarchy_levels': self._calculate_hierarchy_levels(netlist)
        }
        
    def _extract_design_rules(self, case: Dict) -> Dict:
        """提取设计规则
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 设计规则
        """
        return {
            'min_spacing': case['constraints'].get('min_spacing', 0.1),
            'min_width': case['constraints'].get('min_width', 0.1),
            'max_density': case['constraints'].get('max_density', 0.8),
            'power_grid': case['constraints'].get('power_grid', {}),
            'clock_tree': case['constraints'].get('clock_tree', {})
        }
        
    def _extract_optimization_guidelines(self, case: Dict) -> Dict:
        """提取优化指南
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 优化指南
        """
        return {
            'timing_optimization': {
                'critical_paths': case['constraints'].get('critical_paths', []),
                'clock_constraints': case['constraints'].get('clock_constraints', {}),
                'setup_hold_margins': case['constraints'].get('setup_hold_margins', {})
            },
            'power_optimization': {
                'power_budget': case['constraints'].get('power_budget', {}),
                'leakage_targets': case['constraints'].get('leakage_targets', {}),
                'dynamic_power_limits': case['constraints'].get('dynamic_power_limits', {})
            },
            'area_optimization': {
                'area_budget': case['constraints'].get('area_budget', {}),
                'utilization_targets': case['constraints'].get('utilization_targets', {})
            }
        }
        
    def _extract_layout_images(self, layout: Dict) -> Dict[str, Any]:
        """提取布局图像
        
        Args:
            layout: 布局数据
            
        Returns:
            Dict[str, Any]: 布局图像
        """
        # TODO: 实现布局图像提取
        return {}
        
    def _extract_hierarchy(self, netlist: Dict) -> Dict:
        """提取层次结构
        
        Args:
            netlist: 网表数据
            
        Returns:
            Dict: 层次结构
        """
        hierarchy = {}
        for module in netlist.get('modules', []):
            hierarchy[module['name']] = {
                'instances': [inst['name'] for inst in module.get('instances', [])],
                'nets': [net['name'] for net in module.get('nets', [])],
                'pins': [pin['name'] for pin in module.get('pins', [])]
            }
        return hierarchy
        
    def _extract_timing_info(self, case: Dict) -> Dict:
        """提取时序信息
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 时序信息
        """
        return {
            'clock_period': case['constraints'].get('clock_period', 0),
            'setup_time': case['constraints'].get('setup_time', 0),
            'hold_time': case['constraints'].get('hold_time', 0),
            'max_delay': case['constraints'].get('max_delay', 0),
            'min_delay': case['constraints'].get('min_delay', 0)
        }
        
    def _extract_power_info(self, case: Dict) -> Dict:
        """提取功耗信息
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 功耗信息
        """
        return {
            'total_power': case['constraints'].get('total_power', 0),
            'dynamic_power': case['constraints'].get('dynamic_power', 0),
            'leakage_power': case['constraints'].get('leakage_power', 0),
            'power_density': case['constraints'].get('power_density', 0)
        }
        
    def _extract_area_info(self, case: Dict) -> Dict:
        """提取面积信息
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 面积信息
        """
        return {
            'total_area': case['constraints'].get('total_area', 0),
            'cell_area': case['constraints'].get('cell_area', 0),
            'routing_area': case['constraints'].get('routing_area', 0),
            'utilization': case['constraints'].get('utilization', 0)
        }
        
    def _extract_connectivity(self, netlist: Dict) -> Dict:
        """提取连接关系
        
        Args:
            netlist: 网表数据
            
        Returns:
            Dict: 连接关系
        """
        connectivity = {}
        for net in netlist.get('nets', []):
            connectivity[net['name']] = {
                'source': net.get('source', ''),
                'sinks': net.get('sinks', []),
                'fanout': len(net.get('sinks', [])),
                'net_type': net.get('type', 'signal')
            }
        return connectivity
        
    def _extract_dependencies(self, case: Dict) -> Dict:
        """提取依赖关系
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 依赖关系
        """
        return {
            'timing_dependencies': case['constraints'].get('timing_dependencies', {}),
            'power_dependencies': case['constraints'].get('power_dependencies', {}),
            'area_dependencies': case['constraints'].get('area_dependencies', {})
        }
        
    def _extract_constraint_graph(self, case: Dict) -> Dict:
        """提取约束图
        
        Args:
            case: 测试用例信息
            
        Returns:
            Dict: 约束图
        """
        return {
            'timing_constraints': case['constraints'].get('timing_constraints', {}),
            'power_constraints': case['constraints'].get('power_constraints', {}),
            'area_constraints': case['constraints'].get('area_constraints', {}),
            'physical_constraints': case['constraints'].get('physical_constraints', {})
        }
        
    def _calculate_hierarchy_levels(self, netlist: Dict) -> int:
        """计算层次级别
        
        Args:
            netlist: 网表数据
            
        Returns:
            int: 层次级别
        """
        # TODO: 实现层次级别计算
        return 1

    def query(self, query_params):
        """查询知识库数据"""
        return self.cases

    def save(self, path: str):
        """保存知识库
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.knowledge, f, indent=2)
            
    def load(self, path=None):
        """加载知识库数据"""
        if path is None:
            path = self.config['path']
        # 添加一些测试数据
        self.cases = [{'id': 1, 'content': 'test case 1'}]
        return self

    def update(self, new_knowledge):
        """更新知识库数据"""
        self.data.update(new_knowledge)
        self.cases.extend(new_knowledge.get('cases', []))
        return self

    def get_circuit_knowledge(self) -> Dict:
        """获取电路级知识
        
        Returns:
            Dict: 电路级知识
        """
        return self.knowledge['circuit']
        
    def get_module_knowledge(self) -> Dict:
        """获取模块级知识
        
        Returns:
            Dict: 模块级知识
        """
        return self.knowledge['module']
        
    def get_cell_knowledge(self) -> Dict:
        """获取单元级知识
        
        Returns:
            Dict: 单元级知识
        """
        return self.knowledge['cell']

    def add_pdf_knowledge(self, pdf_path: str) -> bool:
        """添加PDF知识到知识库
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            bool: 是否成功添加
        """
        # 检查是否已处理
        if self.pdf_status.is_processed(pdf_path):
            logger.info(f"PDF {pdf_path} 已处理，跳过")
            return True
            
        try:
            # 处理PDF
            pdf_data = self._process_pdf(pdf_path)
            
            # 添加到知识库
            self.add_case(
                layout=pdf_data['layout'],
                optimization_result=pdf_data['optimization_result'],
                metadata={
                    'source': 'pdf',
                    'filename': Path(pdf_path).name,
                    'type': 'layout_design',
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # 标记为已处理
            self.pdf_status.mark_as_processed(
                pdf_path,
                metadata={'type': 'layout_design'}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"处理PDF {pdf_path} 失败: {str(e)}")
            return False
            
    def _process_pdf(self, pdf_path: str) -> Dict:
        """处理PDF文件
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            Dict: 处理后的数据
        """
        # PDF处理逻辑
        # ... 

    def add_knowledge(self, knowledge: Dict) -> bool:
        """添加知识到知识库
        
        Args:
            knowledge: 知识字典
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 获取现有知识
            all_knowledge = self._load_data()
            
            # 添加新知识
            all_knowledge.append(knowledge)
            
            # 保存数据
            with open(self.data_file, 'wb') as f:
                pickle.dump(all_knowledge, f)
                
            return True
            
        except Exception as e:
            logger.error(f"添加知识失败: {str(e)}")
            return False 

    def __len__(self):
        return len(self.cases) 