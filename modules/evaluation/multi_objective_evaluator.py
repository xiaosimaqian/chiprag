from typing import Dict, List, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MultiObjectiveEvaluator:
    def __init__(self, config: Dict):
        """初始化多目标评估器
        
        Args:
            config: 评估器配置
        """
        self.config = config
        self.weights = config.get('weights', {
            'wirelength': 0.3,
            'congestion': 0.3,
            'timing': 0.4
        })
        self.thresholds = config.get('thresholds', {
            'wirelength': 1.0,
            'congestion': 1.0,
            'timing': 1.0
        })
        self.metrics = config.get('metrics', {
            'wirelength': {
                'type': 'minimize',
                'weight': 0.3,
                'threshold': 1.0
            },
            'congestion': {
                'type': 'minimize',
                'weight': 0.3,
                'threshold': 1.0
            },
            'timing': {
                'type': 'minimize',
                'weight': 0.4,
                'threshold': 1.0
            }
        })
        
    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """评估布局质量
        
        Args:
            data: 包含各项指标的字典
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 1. 计算各项指标的得分
            wirelength_score = self._normalize_score(
                self._calculate_wirelength(data['layout']),
                'wirelength'
            )
            congestion_score = self._normalize_score(
                self._calculate_congestion(data['layout']),
                'congestion'
            )
            timing_score = self._normalize_score(
                self._calculate_timing(data['layout']),
                'timing'
            )
            
            # 2. 计算加权总分
            overall_score = (
                self.weights['wirelength'] * wirelength_score +
                self.weights['congestion'] * congestion_score +
                self.weights['timing'] * timing_score
            )
            
            # 3. 计算约束满足度
            constraint_satisfaction = self._evaluate_constraints(data)
            
            # 4. 生成评估报告
            evaluation_report = self._generate_report(
                wirelength_score,
                congestion_score,
                timing_score,
                overall_score,
                constraint_satisfaction
            )
            
            # 5. 返回评估结果
            return {
                'wirelength_score': wirelength_score,
                'congestion_score': congestion_score,
                'timing_score': timing_score,
                'overall_score': overall_score,
                'constraint_satisfaction': constraint_satisfaction,
                'evaluation_report': evaluation_report,
                'details': {
                    'wirelength': {
                        'value': self._calculate_wirelength(data['layout']),
                        'threshold': self.thresholds['wirelength'],
                        'score': wirelength_score,
                        'weight': self.weights['wirelength']
                    },
                    'congestion': {
                        'value': self._calculate_congestion(data['layout']),
                        'threshold': self.thresholds['congestion'],
                        'score': congestion_score,
                        'weight': self.weights['congestion']
                    },
                    'timing': {
                        'value': self._calculate_timing(data['layout']),
                        'threshold': self.thresholds['timing'],
                        'score': timing_score,
                        'weight': self.weights['timing']
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"评估失败: {str(e)}")
            return {
                'wirelength_score': 0.0,
                'congestion_score': 0.0,
                'timing_score': 0.0,
                'overall_score': 0.0,
                'constraint_satisfaction': 0.0,
                'evaluation_report': f"评估失败: {str(e)}",
                'error': str(e)
            }

    def _normalize_score(self, value: float, metric: str) -> float:
        """归一化分数
        
        Args:
            value: 原始值
            metric: 指标名称
            
        Returns:
            归一化后的分数
        """
        if metric in self.metrics:
            metric_config = self.metrics[metric]
            threshold = metric_config['threshold']
            
            if metric_config['type'] == 'minimize':
                return min(1.0, threshold / (value + 1e-6))
            else:  # maximize
                return min(1.0, value / (threshold + 1e-6))
        return value
        
    def _calculate_wirelength(self, layout: Dict) -> float:
        """计算线长
        
        Args:
            layout: 布局数据
            
        Returns:
            线长得分
        """
        try:
            # 1. 获取布线信息
            routing = layout.get('routing', [])
            if not routing:
                return 0.0
                
            # 2. 计算每条线的长度
            total_length = 0.0
            for wire in routing:
                start_point = wire[0:2]  # 起点坐标
                end_point = wire[2:4]    # 终点坐标
                
                # 计算曼哈顿距离
                length = abs(end_point[0] - start_point[0]) + \
                        abs(end_point[1] - start_point[1])
                total_length += length
                
            return total_length
            
        except Exception as e:
            logger.error(f"计算线长失败: {str(e)}")
            return 0.0
        
    def _calculate_congestion(self, layout: Dict) -> float:
        """计算拥塞度
        
        Args:
            layout: 布局数据
            
        Returns:
            拥塞度得分
        """
        try:
            # 1. 获取布局信息
            placement = layout.get('placement', [])
            if not placement:
                return 0.0
                
            # 2. 创建网格
            grid_size = 10  # 10x10网格
            grid = np.zeros((grid_size, grid_size))
            
            # 3. 统计每个网格中的组件数量
            for component in placement:
                x, y = component[0:2]  # 组件位置
                grid_x = int(x * grid_size)
                grid_y = int(y * grid_size)
                
                if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                    grid[grid_x, grid_y] += 1
                    
            # 4. 计算拥塞度
            max_congestion = np.max(grid)
            avg_congestion = np.mean(grid)
            
            return max_congestion / (avg_congestion + 1e-6)
            
        except Exception as e:
            logger.error(f"计算拥塞度失败: {str(e)}")
            return 0.0
        
    def _calculate_timing(self, layout: Dict) -> float:
        """计算时序
        
        Args:
            layout: 布局数据
            
        Returns:
            时序得分
        """
        try:
            # 1. 获取时序信息
            timing = layout.get('timing', {})
            if not timing:
                return 0.0
                
            # 2. 计算关键路径延迟
            max_delay = timing.get('max_delay', 0.0)
            setup_time = timing.get('setup_time', 0.0)
            hold_time = timing.get('hold_time', 0.0)
            
            # 3. 计算时序得分
            timing_score = max_delay / (setup_time + hold_time + 1e-6)
            
            return timing_score
            
        except Exception as e:
            logger.error(f"计算时序失败: {str(e)}")
            return 0.0
            
    def _evaluate_constraints(self, data: Dict) -> float:
        """评估约束满足度
        
        Args:
            data: 布局数据
            
        Returns:
            float: 约束满足度得分
        """
        try:
            # 1. 获取约束信息
            constraints = data.get('constraints', {})
            if not constraints:
                return 0.0
                
            # 2. 计算各项约束的满足度
            area_satisfaction = self._evaluate_area_constraint(
                data['layout'],
                constraints.get('area', '100x100')
            )
            power_satisfaction = self._evaluate_power_constraint(
                data['layout'],
                constraints.get('power', '1W')
            )
            timing_satisfaction = self._evaluate_timing_constraint(
                data['layout'],
                constraints.get('timing', '1ns')
            )
            
            # 3. 计算总体约束满足度
            return (area_satisfaction + power_satisfaction + timing_satisfaction) / 3.0
            
        except Exception as e:
            logger.error(f"评估约束满足度失败: {str(e)}")
            return 0.0
            
    def _evaluate_area_constraint(self, layout: Dict, constraint: str) -> float:
        """评估面积约束
        
        Args:
            layout: 布局数据
            constraint: 面积约束
            
        Returns:
            float: 面积约束满足度
        """
        try:
            # 1. 解析约束
            width, height = map(float, constraint.split('x'))
            max_area = width * height
            
            # 2. 计算实际面积
            placement = layout.get('placement', [])
            if not placement:
                return 0.0
                
            actual_area = 0.0
            for component in placement:
                w, h = component[2:4]  # 组件尺寸
                actual_area += w * h
                
            # 3. 计算满足度
            return min(1.0, max_area / (actual_area + 1e-6))
            
        except Exception as e:
            logger.error(f"评估面积约束失败: {str(e)}")
            return 0.0
            
    def _evaluate_power_constraint(self, layout: Dict, constraint: str) -> float:
        """评估功耗约束
        
        Args:
            layout: 布局数据
            constraint: 功耗约束
            
        Returns:
            float: 功耗约束满足度
        """
        try:
            # 1. 解析约束
            max_power = float(constraint.replace('W', ''))
            
            # 2. 计算实际功耗
            placement = layout.get('placement', [])
            routing = layout.get('routing', [])
            if not placement or not routing:
                return 0.0
                
            # 3. 简化的功耗计算
            actual_power = len(placement) * 0.01 + len(routing) * 0.005
            
            # 4. 计算满足度
            return min(1.0, max_power / (actual_power + 1e-6))
            
        except Exception as e:
            logger.error(f"评估功耗约束失败: {str(e)}")
            return 0.0
            
    def _evaluate_timing_constraint(self, layout: Dict, constraint: str) -> float:
        """评估时序约束
        
        Args:
            layout: 布局数据
            constraint: 时序约束
            
        Returns:
            float: 时序约束满足度
        """
        try:
            # 1. 解析约束
            max_timing = float(constraint.replace('ns', ''))
            
            # 2. 获取实际时序
            timing = layout.get('timing', {})
            if not timing:
                return 0.0
                
            actual_timing = timing.get('max_delay', 0.0)
            
            # 3. 计算满足度
            return min(1.0, max_timing / (actual_timing + 1e-6))
            
        except Exception as e:
            logger.error(f"评估时序约束失败: {str(e)}")
            return 0.0
            
    def _generate_report(self, wirelength_score: float,
                        congestion_score: float,
                        timing_score: float,
                        overall_score: float,
                        constraint_satisfaction: float) -> str:
        """生成评估报告
        
        Args:
            wirelength_score: 线长得分
            congestion_score: 拥塞度得分
            timing_score: 时序得分
            overall_score: 总体得分
            constraint_satisfaction: 约束满足度
            
        Returns:
            str: 评估报告
        """
        report = []
        report.append("布局评估报告")
        report.append("=" * 50)
        report.append(f"1. 总体得分: {overall_score:.2f}")
        report.append(f"2. 约束满足度: {constraint_satisfaction:.2f}")
        report.append("\n详细指标:")
        report.append(f"- 线长得分: {wirelength_score:.2f}")
        report.append(f"- 拥塞度得分: {congestion_score:.2f}")
        report.append(f"- 时序得分: {timing_score:.2f}")
        report.append("\n建议:")
        
        if overall_score < 0.6:
            report.append("- 布局质量较差，建议重新生成")
        elif overall_score < 0.8:
            report.append("- 布局质量一般，可以考虑优化")
        else:
            report.append("- 布局质量良好")
            
        if constraint_satisfaction < 0.8:
            report.append("- 约束满足度不足，需要调整布局")
            
        return "\n".join(report) 