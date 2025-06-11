import re
import os
import logging
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

class LEFParser:
    def __init__(self):
        self.current_section = None
        self.data = {}
        self.tech_file = None
        self.cells_file = None
        
    def parse_lef(self, tech_lef_file: str, cells_lef_file: str = None) -> Dict[str, Any]:
        """解析LEF文件
        
        Args:
            tech_lef_file: 工艺LEF文件路径
            cells_lef_file: 单元库LEF文件路径（可选）
            
        Returns:
            解析后的数据字典
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        self.tech_file = os.path.abspath(tech_lef_file)
        logger.info(f"开始解析工艺LEF文件: {self.tech_file}")
        
        try:
            content = ""
            with open(self.tech_file, 'r') as f:
                content += f.read() + "\n"
                
            if cells_lef_file:
                self.cells_file = os.path.abspath(cells_lef_file)
                logger.info(f"开始解析单元库LEF文件: {self.cells_file}")
                with open(self.cells_file, 'r') as f:
                    content += f.read() + "\n"
                    
            # 解析版本和设计名称
            version_match = re.search(r'VERSION\s+(\S+)', content)
            design_match = re.search(r'DESIGN\s+(\S+)', content)
            
            if not version_match or not design_match:
                raise ValueError("LEF文件缺少VERSION或DESIGN声明")
                
            self.data['VERSION'] = version_match.group(1)
            self.data['DESIGN'] = design_match.group(1)
            self.data['FILES'] = {
                'tech': self.tech_file,
                'cells': self.cells_file
            }
            
            # 解析各个部分
            self._parse_units(content)
            self._parse_macros(content)
            self._parse_layers(content)
            self._parse_vias(content)
            self._parse_sites(content)
            self._parse_spacing(content)
            self._parse_minfeature(content)
            
            logger.info("LEF文件解析完成")
            return self.data
            
        except FileNotFoundError as e:
            logger.error(f"LEF文件不存在: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"解析LEF文件时出错: {str(e)}")
            raise
            
    def _parse_units(self, content: str):
        """解析UNITS部分"""
        units_match = re.search(r'UNITS\s+DISTANCE\s+MICRONS\s+(\d+)', content)
        if not units_match:
            logger.warning("LEF文件缺少UNITS声明")
            return
            
        try:
            self.data['UNITS'] = {
                'distance': int(units_match.group(1)),
                'unit': 'MICRONS'
            }
        except ValueError as e:
            logger.error(f"解析UNITS时出错: {str(e)}")
            raise
            
    def _parse_macros(self, content: str):
        """解析MACROS部分"""
        self.data['MACROS'] = {}
        macro_sections = re.finditer(r'MACRO\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in macro_sections:
            try:
                macro_name = section.group(1)
                macro_content = section.group(0)
                
                # 解析大小
                size_match = re.search(r'SIZE\s+(\d+\.?\d*)\s+BY\s+(\d+\.?\d*)', macro_content)
                if not size_match:
                    logger.warning(f"宏单元 {macro_name} 缺少SIZE声明")
                    continue
                    
                size = {
                    'width': float(size_match.group(1)),
                    'height': float(size_match.group(2))
                }
                
                # 解析引脚
                pins = {}
                pin_sections = re.finditer(r'PIN\s+(\S+).*?END\s+\1', macro_content, re.DOTALL)
                
                for pin_section in pin_sections:
                    pin_name = pin_section.group(1)
                    pin_content = pin_section.group(0)
                    
                    # 解析引脚方向
                    direction_match = re.search(r'DIRECTION\s+(\S+)', pin_content)
                    direction = direction_match.group(1) if direction_match else 'INOUT'
                    
                    # 解析引脚用途
                    use_match = re.search(r'USE\s+(\S+)', pin_content)
                    use = use_match.group(1) if use_match else None
                    
                    # 解析引脚形状
                    shape_match = re.search(r'SHAPE\s+(\S+)', pin_content)
                    shape = shape_match.group(1) if shape_match else None
                    
                    # 解析引脚位置
                    port_sections = re.finditer(r'PORT.*?END', pin_content, re.DOTALL)
                    port_rects = []
                    
                    for port_section in port_sections:
                        rect_matches = re.finditer(r'RECT\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', port_section.group(0))
                        for rect in rect_matches:
                            port_rects.append({
                                'x1': float(rect.group(1)),
                                'y1': float(rect.group(2)),
                                'x2': float(rect.group(3)),
                                'y2': float(rect.group(4))
                            })
                    
                    pins[pin_name] = {
                        'direction': direction,
                        'use': use,
                        'shape': shape,
                        'ports': port_rects
                    }
                
                # 解析障碍物
                obs_sections = re.finditer(r'OBS.*?END', macro_content, re.DOTALL)
                obs_rects = []
                
                for obs_section in obs_sections:
                    rect_matches = re.finditer(r'RECT\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', obs_section.group(0))
                    for rect in rect_matches:
                        obs_rects.append({
                            'x1': float(rect.group(1)),
                            'y1': float(rect.group(2)),
                            'x2': float(rect.group(3)),
                            'y2': float(rect.group(4))
                        })
                
                # 解析对称性
                symmetry_match = re.search(r'SYMMETRY\s+(\S+)', macro_content)
                symmetry = symmetry_match.group(1) if symmetry_match else None
                
                # 解析类
                class_match = re.search(r'CLASS\s+(\S+)', macro_content)
                macro_class = class_match.group(1) if class_match else None
                
                self.data['MACROS'][macro_name] = {
                    'SIZE': size,
                    'PIN': pins,
                    'OBS': obs_rects,
                    'SYMMETRY': symmetry,
                    'CLASS': macro_class
                }
                
            except Exception as e:
                logger.error(f"解析宏单元 {macro_name} 时出错: {str(e)}")
                continue
                
    def _parse_layers(self, content: str):
        """解析LAYERS部分"""
        self.data['LAYERS'] = {}
        layer_sections = re.finditer(r'LAYER\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in layer_sections:
            try:
                layer_name = section.group(1)
                layer_content = section.group(0)
                
                # 解析层类型
                type_match = re.search(r'TYPE\s+(\S+)', layer_content)
                layer_type = type_match.group(1) if type_match else 'ROUTING'
                
                # 解析层属性
                properties = {}
                property_matches = re.finditer(r'(\S+)\s+(\S+)', layer_content)
                
                for match in property_matches:
                    prop_name = match.group(1)
                    prop_value = match.group(2)
                    properties[prop_name] = prop_value
                
                # 解析层方向
                direction_match = re.search(r'DIRECTION\s+(\S+)', layer_content)
                direction = direction_match.group(1) if direction_match else None
                
                # 解析层间距
                spacing_matches = re.finditer(r'SPACING\s+(\d+\.?\d*)', layer_content)
                spacing = [float(match.group(1)) for match in spacing_matches]
                
                # 解析层宽度
                width_match = re.search(r'WIDTH\s+(\d+\.?\d*)', layer_content)
                width = float(width_match.group(1)) if width_match else None
                
                self.data['LAYERS'][layer_name] = {
                    'TYPE': layer_type,
                    'DIRECTION': direction,
                    'WIDTH': width,
                    'SPACING': spacing,
                    'PROPERTIES': properties
                }
                
            except Exception as e:
                logger.error(f"解析层 {layer_name} 时出错: {str(e)}")
                continue
                
    def _parse_vias(self, content: str):
        """解析VIAS部分"""
        self.data['VIAS'] = {}
        via_sections = re.finditer(r'VIA\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in via_sections:
            try:
                via_name = section.group(1)
                via_content = section.group(0)
                
                # 解析Via规则
                rule_match = re.search(r'VIARULE\s+(\S+)', via_content)
                rule = rule_match.group(1) if rule_match else None
                
                # 解析Via层
                layers = []
                layer_matches = re.finditer(r'LAYER\s+(\S+).*?END\s+\1', via_content, re.DOTALL)
                
                for layer_match in layer_matches:
                    layer_name = layer_match.group(1)
                    layer_content = layer_match.group(0)
                    
                    # 解析层矩形
                    rect_matches = re.finditer(r'RECT\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', layer_content)
                    rects = []
                    
                    for rect in rect_matches:
                        rects.append({
                            'x1': float(rect.group(1)),
                            'y1': float(rect.group(2)),
                            'x2': float(rect.group(3)),
                            'y2': float(rect.group(4))
                        })
                    
                    layers.append({
                        'name': layer_name,
                        'rects': rects
                    })
                
                self.data['VIAS'][via_name] = {
                    'RULE': rule,
                    'LAYERS': layers
                }
                
            except Exception as e:
                logger.error(f"解析Via {via_name} 时出错: {str(e)}")
                continue
                
    def _parse_sites(self, content: str):
        """解析SITES部分"""
        self.data['SITES'] = {}
        site_sections = re.finditer(r'SITE\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in site_sections:
            try:
                site_name = section.group(1)
                site_content = section.group(0)
                
                # 解析站点类
                class_match = re.search(r'CLASS\s+(\S+)', site_content)
                site_class = class_match.group(1) if class_match else None
                
                # 解析站点大小
                size_match = re.search(r'SIZE\s+(\d+\.?\d*)\s+BY\s+(\d+\.?\d*)', site_content)
                if size_match:
                    size = {
                        'width': float(size_match.group(1)),
                        'height': float(size_match.group(2))
                    }
                else:
                    size = None
                
                # 解析站点对称性
                symmetry_match = re.search(r'SYMMETRY\s+(\S+)', site_content)
                symmetry = symmetry_match.group(1) if symmetry_match else None
                
                self.data['SITES'][site_name] = {
                    'CLASS': site_class,
                    'SIZE': size,
                    'SYMMETRY': symmetry
                }
                
            except Exception as e:
                logger.error(f"解析站点 {site_name} 时出错: {str(e)}")
                continue
                
    def _parse_spacing(self, content: str):
        """解析SPACING部分"""
        self.data['SPACING'] = {}
        spacing_sections = re.finditer(r'SPACING\s+(\S+).*?END\s+\1', content, re.DOTALL)
        
        for section in spacing_sections:
            try:
                spacing_name = section.group(1)
                spacing_content = section.group(0)
                
                # 解析间距规则
                rules = []
                rule_matches = re.finditer(r'SPACING\s+(\d+\.?\d*)', spacing_content)
                
                for rule_match in rule_matches:
                    rules.append(float(rule_match.group(1)))
                
                self.data['SPACING'][spacing_name] = {
                    'RULES': rules
                }
                
            except Exception as e:
                logger.error(f"解析间距规则 {spacing_name} 时出错: {str(e)}")
                continue
                
    def _parse_minfeature(self, content: str):
        """解析MINFEATURE部分"""
        minfeature_match = re.search(r'MINFEATURE\s+(\d+\.?\d*)\s+(\d+\.?\d*)', content)
        if minfeature_match:
            try:
                self.data['MINFEATURE'] = {
                    'width': float(minfeature_match.group(1)),
                    'length': float(minfeature_match.group(2))
                }
            except ValueError as e:
                logger.error(f"解析MINFEATURE时出错: {str(e)}")
                raise 