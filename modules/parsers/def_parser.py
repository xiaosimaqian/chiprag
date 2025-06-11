import re
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class DEFParser:
    def __init__(self):
        self.current_section = None
        self.data = {}
        self.file_path = None
        
    def parse_def(self, file_path: str) -> Dict[str, Any]:
        """解析DEF文件
        
        Args:
            file_path: DEF文件路径
            
        Returns:
            解析后的数据字典
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        self.file_path = os.path.abspath(file_path)
        logger.info(f"开始解析DEF文件: {self.file_path}")
        
        try:
            with open(self.file_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            logger.error(f"DEF文件不存在: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"读取DEF文件时出错: {str(e)}")
            raise
            
        try:
            # 解析版本和设计名称
            version_match = re.search(r'VERSION\s+(\S+)', content)
            design_match = re.search(r'DESIGN\s+(\S+)', content)
            
            if not version_match or not design_match:
                raise ValueError("DEF文件缺少VERSION或DESIGN声明")
                
            self.data['VERSION'] = version_match.group(1)
            self.data['DESIGN'] = design_match.group(1)
            self.data['FILE'] = self.file_path
            
            # 解析各个部分
            self._parse_diearea(content)
            self._parse_units(content)
            self._parse_components(content)
            self._parse_nets(content)
            self._parse_pins(content)
            self._parse_regions(content)
            self._parse_rows(content)
            self._parse_tracks(content)
            self._parse_gcellgrid(content)
            self._parse_vias(content)
            self._parse_specialnets(content)
            
            logger.info(f"DEF文件解析完成: {self.file_path}")
            return self.data
            
        except Exception as e:
            logger.error(f"解析DEF文件时出错: {str(e)}")
            raise
            
    def _parse_diearea(self, content: str):
        """解析DIEAREA部分"""
        diearea_match = re.search(r'DIEAREA\s+\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)', content)
        if not diearea_match:
            logger.warning("DEF文件缺少DIEAREA声明")
            return
            
        try:
            self.data['DIEAREA'] = {
                'x1': int(diearea_match.group(1)),
                'y1': int(diearea_match.group(2)),
                'x2': int(diearea_match.group(3)),
                'y2': int(diearea_match.group(4)),
                'width': int(diearea_match.group(3)) - int(diearea_match.group(1)),
                'height': int(diearea_match.group(4)) - int(diearea_match.group(2))
            }
        except ValueError as e:
            logger.error(f"解析DIEAREA坐标时出错: {str(e)}")
            raise
            
    def _parse_units(self, content: str):
        """解析UNITS部分"""
        units_match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
        if not units_match:
            logger.warning("DEF文件缺少UNITS声明")
            return
            
        try:
            self.data['UNITS'] = {
                'distance': int(units_match.group(1)),
                'unit': 'MICRONS'
            }
        except ValueError as e:
            logger.error(f"解析UNITS时出错: {str(e)}")
            raise
            
    def _parse_components(self, content: str):
        """解析COMPONENTS部分"""
        components_section = re.search(r'COMPONENTS\s+(\d+)\s*;.*?END\s+COMPONENTS', content, re.DOTALL)
        if not components_section:
            logger.warning("DEF文件缺少COMPONENTS声明")
            return
            
        try:
            num_components = int(components_section.group(1))
            self.data['COMPONENTS'] = {
                'count': num_components,
                'instances': {}
            }
            
            component_matches = re.finditer(
                r'-\s+(\S+)\s+(\S+)\s*'  # 组件ID和类型
                r'(?:\+\s*PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+))?'  # 位置和方向（可选）
                r'(?:\+\s*FIXED\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+))?'  # 固定位置和方向（可选）
                r'(?:\+\s*COVER\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+))?'  # 覆盖位置和方向（可选）
                r'(?:\+\s*UNPLACED)?'  # 未放置标记（可选）
                r'(?:\s*\+\s*SOFT)?',  # 软约束标记（可选）
                components_section.group(0)
            )
            
            for match in component_matches:
                comp_id = match.group(1)
                comp_type = match.group(2)
                
                # 解析位置信息
                placement = None
                if match.group(3):  # PLACED
                    placement = {
                        'type': 'PLACED',
                        'x': int(match.group(3)),
                        'y': int(match.group(4)),
                        'orientation': match.group(5)
                    }
                elif match.group(6):  # FIXED
                    placement = {
                        'type': 'FIXED',
                        'x': int(match.group(6)),
                        'y': int(match.group(7)),
                        'orientation': match.group(8)
                    }
                elif match.group(9):  # COVER
                    placement = {
                        'type': 'COVER',
                        'x': int(match.group(9)),
                        'y': int(match.group(10)),
                        'orientation': match.group(11)
                    }
                
                self.data['COMPONENTS']['instances'][comp_id] = {
                    'type': comp_type,
                    'placement': placement,
                    'is_soft': bool(match.group(12))
                }
                
        except Exception as e:
            logger.error(f"解析COMPONENTS时出错: {str(e)}")
            raise
            
    def _parse_nets(self, content: str):
        """解析NETS部分"""
        nets_section = re.search(r'NETS\s+(\d+)\s*;.*?END\s+NETS', content, re.DOTALL)
        if not nets_section:
            logger.warning("DEF文件缺少NETS声明")
            return
            
        try:
            num_nets = int(nets_section.group(1))
            self.data['NETS'] = {
                'count': num_nets,
                'nets': []
            }
            
            net_matches = re.finditer(
                r'-\s+(\S+)\s*'  # 网络ID
                r'(?:\+\s*USE\s+(\S+))?'  # 用途（可选）
                r'(?:\s*\(.*?\))*',  # 连接列表
                nets_section.group(0),
                re.DOTALL
            )
            
            for match in net_matches:
                net_id = match.group(1)
                net_use = match.group(2)
                
                # 解析连接
                connections = []
                conn_matches = re.finditer(
                    r'\(\s*(\S+)\s+(\S+)\s*\)',  # 组件和引脚
                    match.group(0)
                )
                
                for conn_match in conn_matches:
                    connections.append({
                        'component': conn_match.group(1),
                        'pin': conn_match.group(2)
                    })
                
                self.data['NETS']['nets'].append({
                    'id': net_id,
                    'use': net_use,
                    'connections': connections
                })
                
        except Exception as e:
            logger.error(f"解析NETS时出错: {str(e)}")
            raise
            
    def _parse_pins(self, content: str):
        """解析PINS部分"""
        pins_section = re.search(r'PINS\s+(\d+)\s*;.*?END\s+PINS', content, re.DOTALL)
        if not pins_section:
            logger.warning("DEF文件缺少PINS声明")
            return
            
        try:
            num_pins = int(pins_section.group(1))
            self.data['PINS'] = {
                'count': num_pins,
                'pins': {}
            }
            
            pin_matches = re.finditer(
                r'-\s+(\S+)\s+(\S+)\s*'  # 引脚ID和方向
                r'(?:\+\s*USE\s+(\S+))?'  # 用途（可选）
                r'(?:\+\s*PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+))?'  # 位置和方向（可选）
                r'(?:\+\s*FIXED\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+))?'  # 固定位置和方向（可选）
                r'(?:\+\s*COVER\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+))?',  # 覆盖位置和方向（可选）
                pins_section.group(0)
            )
            
            for match in pin_matches:
                pin_id = match.group(1)
                direction = match.group(2)
                pin_use = match.group(3)
                
                # 解析位置信息
                placement = None
                if match.group(4):  # PLACED
                    placement = {
                        'type': 'PLACED',
                        'x': int(match.group(4)),
                        'y': int(match.group(5)),
                        'orientation': match.group(6)
                    }
                elif match.group(7):  # FIXED
                    placement = {
                        'type': 'FIXED',
                        'x': int(match.group(7)),
                        'y': int(match.group(8)),
                        'orientation': match.group(9)
                    }
                elif match.group(10):  # COVER
                    placement = {
                        'type': 'COVER',
                        'x': int(match.group(10)),
                        'y': int(match.group(11)),
                        'orientation': match.group(12)
                    }
                
                self.data['PINS']['pins'][pin_id] = {
                    'direction': direction,
                    'use': pin_use,
                    'placement': placement
                }
                
        except Exception as e:
            logger.error(f"解析PINS时出错: {str(e)}")
            raise
            
    def _parse_regions(self, content: str):
        """解析REGIONS部分"""
        regions_section = re.search(r'REGIONS\s+(\d+)\s*;.*?END\s+REGIONS', content, re.DOTALL)
        if not regions_section:
            logger.warning("DEF文件缺少REGIONS声明")
            return
            
        try:
            num_regions = int(regions_section.group(1))
            self.data['REGIONS'] = {
                'count': num_regions,
                'regions': []
            }
            
            region_matches = re.finditer(
                r'-\s+(\S+)\s*'  # 区域ID
                r'(?:\+\s*TYPE\s+(\S+))?'  # 类型（可选）
                r'(?:\s*\(.*?\))*',  # 矩形列表
                regions_section.group(0),
                re.DOTALL
            )
            
            for match in region_matches:
                region_id = match.group(1)
                region_type = match.group(2)
                
                # 解析矩形
                rectangles = []
                rect_matches = re.finditer(
                    r'\(\s*(\d+)\s+(\d+)\s*\)\s*\(\s*(\d+)\s+(\d+)\s*\)',
                    match.group(0)
                )
                
                for rect_match in rect_matches:
                    rectangles.append({
                        'x1': int(rect_match.group(1)),
                        'y1': int(rect_match.group(2)),
                        'x2': int(rect_match.group(3)),
                        'y2': int(rect_match.group(4))
                    })
                
                self.data['REGIONS']['regions'].append({
                    'id': region_id,
                    'type': region_type,
                    'rectangles': rectangles
                })
                
        except Exception as e:
            logger.error(f"解析REGIONS时出错: {str(e)}")
            raise
            
    def _parse_rows(self, content: str):
        """解析ROWS部分"""
        rows_section = re.search(r'ROWS\s+(\d+)\s*;.*?END\s+ROWS', content, re.DOTALL)
        if not rows_section:
            logger.warning("DEF文件缺少ROWS声明")
            return
            
        try:
            num_rows = int(rows_section.group(1))
            self.data['ROWS'] = {
                'count': num_rows,
                'rows': []
            }
            
            row_matches = re.finditer(
                r'-\s+(\S+)\s+(\S+)\s*'  # 行名称和类型
                r'\(\s*(\d+)\s+(\d+)\s*\)\s*'  # 起始位置
                r'(?:DO\s+(\d+)\s+BY\s+(\d+)\s+STEP\s+(\d+)\s+(\d+))?'  # 重复信息（可选）
                r'(?:\s*\+\s*ORIENTATION\s+(\S+))?',  # 方向（可选）
                rows_section.group(0)
            )
            
            for match in row_matches:
                row_name = match.group(1)
                row_type = match.group(2)
                x = int(match.group(3))
                y = int(match.group(4))
                
                # 解析重复信息
                repeat = None
                if match.group(5):
                    repeat = {
                        'count': int(match.group(5)),
                        'x_step': int(match.group(7)),
                        'y_step': int(match.group(8))
                    }
                
                self.data['ROWS']['rows'].append({
                    'name': row_name,
                    'type': row_type,
                    'x': x,
                    'y': y,
                    'repeat': repeat,
                    'orientation': match.group(9)
                })
                
        except Exception as e:
            logger.error(f"解析ROWS时出错: {str(e)}")
            raise
            
    def _parse_tracks(self, content: str):
        """解析TRACKS部分"""
        tracks_section = re.search(r'TRACKS\s+(\S+)\s+(\d+)\s+DO\s+(\d+)\s+STEP\s+(\d+)\s+LAYER\s+(\S+)\s*;.*?END\s+TRACKS', content, re.DOTALL)
        if not tracks_section:
            logger.warning("DEF文件缺少TRACKS声明")
            return
            
        try:
            direction = tracks_section.group(1)
            start = int(tracks_section.group(2))
            count = int(tracks_section.group(3))
            step = int(tracks_section.group(4))
            layer = tracks_section.group(5)
            
            self.data['TRACKS'] = {
                'direction': direction,
                'start': start,
                'count': count,
                'step': step,
                'layer': layer
            }
            
        except Exception as e:
            logger.error(f"解析TRACKS时出错: {str(e)}")
            raise
            
    def _parse_gcellgrid(self, content: str):
        """解析GCELLGRID部分"""
        gcellgrid_section = re.search(r'GCELLGRID\s+(\d+)\s*;.*?END\s+GCELLGRID', content, re.DOTALL)
        if not gcellgrid_section:
            logger.warning("DEF文件缺少GCELLGRID声明")
            return
            
        try:
            num_grids = int(gcellgrid_section.group(1))
            self.data['GCELLGRID'] = {
                'count': num_grids,
                'grids': []
            }
            
            grid_matches = re.finditer(
                r'(\S+)\s+(\d+)\s+DO\s+(\d+)\s+STEP\s+(\d+)',  # 方向和位置信息
                gcellgrid_section.group(0)
            )
            
            for match in grid_matches:
                direction = match.group(1)
                start = int(match.group(2))
                count = int(match.group(3))
                step = int(match.group(4))
                
                self.data['GCELLGRID']['grids'].append({
                    'direction': direction,
                    'start': start,
                    'count': count,
                    'step': step
                })
                
        except Exception as e:
            logger.error(f"解析GCELLGRID时出错: {str(e)}")
            raise
            
    def _parse_vias(self, content: str):
        """解析VIAS部分"""
        vias_section = re.search(r'VIAS\s+(\d+)\s*;.*?END\s+VIAS', content, re.DOTALL)
        if not vias_section:
            logger.warning("DEF文件缺少VIAS声明")
            return
            
        try:
            num_vias = int(vias_section.group(1))
            self.data['VIAS'] = {
                'count': num_vias,
                'vias': []
            }
            
            via_matches = re.finditer(
                r'-\s+(\S+)\s*'  # Via名称
                r'(?:\+\s*VIARULE\s+(\S+))?'  # Via规则（可选）
                r'(?:\s*\(.*?\))*',  # 位置列表
                vias_section.group(0),
                re.DOTALL
            )
            
            for match in via_matches:
                via_name = match.group(1)
                via_rule = match.group(2)
                
                # 解析位置
                positions = []
                pos_matches = re.finditer(
                    r'\(\s*(\d+)\s+(\d+)\s*\)',
                    match.group(0)
                )
                
                for pos_match in pos_matches:
                    positions.append({
                        'x': int(pos_match.group(1)),
                        'y': int(pos_match.group(2))
                    })
                
                self.data['VIAS']['vias'].append({
                    'name': via_name,
                    'rule': via_rule,
                    'positions': positions
                })
                
        except Exception as e:
            logger.error(f"解析VIAS时出错: {str(e)}")
            raise
            
    def _parse_specialnets(self, content: str):
        """解析SPECIALNETS部分"""
        specialnets_section = re.search(r'SPECIALNETS\s+(\d+)\s*;.*?END\s+SPECIALNETS', content, re.DOTALL)
        if not specialnets_section:
            logger.warning("DEF文件缺少SPECIALNETS声明")
            return
            
        try:
            num_nets = int(specialnets_section.group(1))
            self.data['SPECIALNETS'] = {
                'count': num_nets,
                'nets': []
            }
            
            net_matches = re.finditer(
                r'-\s+(\S+)\s*'  # 网络名称
                r'(?:\+\s*USE\s+(\S+))?'  # 用途（可选）
                r'(?:\s*\(.*?\))*',  # 连接列表
                specialnets_section.group(0),
                re.DOTALL
            )
            
            for match in net_matches:
                net_name = match.group(1)
                net_use = match.group(2)
                
                # 解析连接
                connections = []
                conn_matches = re.finditer(
                    r'\(\s*(\S+)\s+(\S+)\s*\)',  # 组件和引脚
                    match.group(0)
                )
                
                for conn_match in conn_matches:
                    connections.append({
                        'component': conn_match.group(1),
                        'pin': conn_match.group(2)
                    })
                
                self.data['SPECIALNETS']['nets'].append({
                    'name': net_name,
                    'use': net_use,
                    'connections': connections
                })
                
        except Exception as e:
            logger.error(f"解析SPECIALNETS时出错: {str(e)}")
            raise 