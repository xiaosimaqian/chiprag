import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import Dict, List, Any
import logging
import os

logger = logging.getLogger(__name__)

class LayoutVisualizer:
    def __init__(self, font_family=None):
        # 支持中文字体容错
        if font_family is None:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'STHeiti', 'Microsoft YaHei', 'Arial Unicode MS']
        else:
            plt.rcParams['font.sans-serif'] = [font_family]
        plt.rcParams['axes.unicode_minus'] = False
        self.colors = plt.cm.tab20.colors  # 使用tab20颜色映射
        
    def visualize(self, layout: Dict[str, Any], save_path: str = None):
        """可视化布局
        
        Args:
            layout: 布局信息
            save_path: 保存路径
        """
        try:
            # 1. 创建图形
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 2. 绘制组件
            self._draw_components(ax, layout.get('placement', []))
            
            # 3. 绘制布线
            self._draw_routing(ax, layout.get('routing', []))
            
            # 4. 设置图形属性
            self._set_plot_properties(ax)
            
            # 5. 添加图例和标题
            self._add_legend_and_title(ax, layout)
            
            # 6. 保存或显示图形
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"布局图已保存到: {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"布局可视化失败: {str(e)}")
            raise
            
    def _draw_components(self, ax, placement: List[List[float]]):
        """绘制组件
        
        Args:
            ax: matplotlib轴对象
            placement: 组件放置信息
        """
        for i, component in enumerate(placement):
            x, y, width, height = component
            
            # 创建矩形
            rect = patches.Rectangle(
                (x, y),
                width,
                height,
                linewidth=1,
                edgecolor='black',
                facecolor=self.colors[i % len(self.colors)],
                alpha=0.6
            )
            
            # 添加组件
            ax.add_patch(rect)
            
            # 添加组件标签
            ax.text(
                x + width/2,
                y + height/2,
                f'C{i+1}',
                ha='center',
                va='center',
                fontsize=8
            )
            
    def _draw_routing(self, ax, routing: List[List[float]]):
        """绘制布线
        
        Args:
            ax: matplotlib轴对象
            routing: 布线信息
        """
        for i, wire in enumerate(routing):
            start_x, start_y, end_x, end_y = wire
            
            # 绘制线段
            ax.plot(
                [start_x, end_x],
                [start_y, end_y],
                color='gray',
                linestyle='--',
                linewidth=1,
                alpha=0.5
            )
            
            # 添加连接点
            ax.scatter(
                [start_x, end_x],
                [start_y, end_y],
                color='red',
                s=20,
                zorder=3
            )
            
    def _set_plot_properties(self, ax):
        """设置图形属性
        
        Args:
            ax: matplotlib轴对象
        """
        # 设置坐标轴范围
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        
        # 设置网格
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # 设置等比例
        ax.set_aspect('equal')
        
    def _add_legend_and_title(self, ax, layout: Dict[str, Any]):
        """添加图例和标题
        
        Args:
            ax: matplotlib轴对象
            layout: 布局信息
        """
        # 添加标题
        ax.set_title('布局可视化', fontsize=14, pad=20)
        
        # 添加布局信息
        info_text = (
            f"组件数量: {len(layout.get('placement', []))}\n"
            f"布线数量: {len(layout.get('routing', []))}\n"
        )
        
        if 'timing' in layout:
            timing = layout['timing']
            info_text += (
                f"最大延迟: {timing.get('max_delay', 0):.2f}ns\n"
                f"建立时间: {timing.get('setup_time', 0):.2f}ns\n"
                f"保持时间: {timing.get('hold_time', 0):.2f}ns"
            )
            
        ax.text(
            1.02,
            0.5,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8)
        ) 