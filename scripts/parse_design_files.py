import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class DesignEntity:
    id: str
    name: str
    type: str
    properties: tuple

@dataclass(frozen=True)
class DesignRelation:
    source: str
    target: str
    type: str
    properties: tuple

class DesignFileParser:
    def __init__(self, design_dir: str):
        self.design_dir = Path(design_dir)
        self.entities: Set[DesignEntity] = set()
        self.relations: Set[DesignRelation] = set()
        
        # 实体类型映射
        self.entity_types = {
            "module": "模块",
            "cell": "单元",
            "net": "网络",
            "pin": "引脚",
            "port": "端口",
            "block": "功能块",
            "constraint": "约束"
        }
        
        # 关系类型映射
        self.relation_types = {
            "connects": "连接",
            "contains": "包含",
            "depends_on": "依赖",
            "implements": "实现"
        }
    
    def parse_all(self):
        """解析所有设计文件"""
        logger.info(f"开始解析设计文件: {self.design_dir}")
        
        # 解析Verilog文件
        verilog_files = list(self.design_dir.glob("**/*.v"))
        for v_file in verilog_files:
            self._parse_verilog(v_file)
        
        # 解析DEF文件
        def_files = list(self.design_dir.glob("**/*.def"))
        for def_file in def_files:
            self._parse_def(def_file)
        
        # 解析LEF文件
        lef_files = list(self.design_dir.glob("**/*.lef"))
        for lef_file in lef_files:
            self._parse_lef(lef_file)
        
        logger.info(f"设计文件解析完成！")
        logger.info(f"提取到 {len(self.entities)} 个实体")
        logger.info(f"提取到 {len(self.relations)} 个关系")
    
    def _parse_verilog(self, file_path: Path):
        """解析Verilog文件"""
        logger.info(f"解析Verilog文件: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 提取模块定义
        module_pattern = r"module\s+(\w+)\s*\(([^;]*)\);"
        for match in re.finditer(module_pattern, content):
            module_name = match.group(1)
            ports = match.group(2)
            
            # 添加模块实体
            self.entities.add(DesignEntity(
                id=f"module_{module_name}",
                name=module_name,
                type="module",
                properties=tuple(sorted({
                    "file": str(file_path),
                    "type": "verilog_module"
                }.items()))
            ))
            
            # 提取端口
            port_pattern = r"(input|output|inout)\s+(?:\[([^\]]+)\])?\s*(\w+)"
            for port_match in re.finditer(port_pattern, ports):
                port_type = port_match.group(1)
                port_width = port_match.group(2)
                port_name = port_match.group(3)
                
                # 添加端口实体
                port_id = f"port_{module_name}_{port_name}"
                self.entities.add(DesignEntity(
                    id=port_id,
                    name=port_name,
                    type="port",
                    properties=tuple(sorted({
                        "direction": port_type,
                        "width": port_width if port_width else "1",
                        "module": module_name
                    }.items()))
                ))
                
                # 添加模块-端口关系
                self.relations.add(DesignRelation(
                    source=f"module_{module_name}",
                    target=port_id,
                    type="contains",
                    properties=tuple(sorted({
                        "type": "module_port"
                    }.items()))
                ))
    
    def _parse_def(self, file_path: Path):
        """解析DEF文件"""
        logger.info(f"解析DEF文件: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 提取组件定义
        component_pattern = r"- (\w+)\s+(\w+)\s+\(([^)]+)\)"
        for match in re.finditer(component_pattern, content):
            comp_name = match.group(1)
            comp_type = match.group(2)
            comp_pins = match.group(3)
            
            # 添加组件实体
            self.entities.add(DesignEntity(
                id=f"cell_{comp_name}",
                name=comp_name,
                type="cell",
                properties=tuple(sorted({
                    "type": comp_type,
                    "file": str(file_path)
                }.items()))
            ))
            
            # 提取引脚
            pin_pattern = r"(\w+)\s+\(([^)]+)\)"
            for pin_match in re.finditer(pin_pattern, comp_pins):
                pin_name = pin_match.group(1)
                pin_net = pin_match.group(2)
                
                # 添加引脚实体
                pin_id = f"pin_{comp_name}_{pin_name}"
                self.entities.add(DesignEntity(
                    id=pin_id,
                    name=pin_name,
                    type="pin",
                    properties=tuple(sorted({
                        "cell": comp_name,
                        "net": pin_net
                    }.items()))
                ))
                
                # 添加组件-引脚关系
                self.relations.add(DesignRelation(
                    source=f"cell_{comp_name}",
                    target=pin_id,
                    type="contains",
                    properties=tuple(sorted({
                        "type": "cell_pin"
                    }.items()))
                ))
    
    def _parse_lef(self, file_path: Path):
        """解析LEF文件"""
        logger.info(f"解析LEF文件: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 提取宏单元定义
        macro_pattern = r"MACRO\s+(\w+)\s*{([^}]*)}"
        for match in re.finditer(macro_pattern, content):
            macro_name = match.group(1)
            macro_props = match.group(2)
            
            # 添加宏单元实体
            self.entities.add(DesignEntity(
                id=f"macro_{macro_name}",
                name=macro_name,
                type="block",
                properties=tuple(sorted({
                    "file": str(file_path),
                    "type": "lef_macro"
                }.items()))
            ))
            
            # 提取引脚定义
            pin_pattern = r"PIN\s+(\w+)\s*{([^}]*)}"
            for pin_match in re.finditer(pin_pattern, macro_props):
                pin_name = pin_match.group(1)
                pin_props = pin_match.group(2)
                
                # 添加引脚实体
                pin_id = f"pin_{macro_name}_{pin_name}"
                self.entities.add(DesignEntity(
                    id=pin_id,
                    name=pin_name,
                    type="pin",
                    properties=tuple(sorted({
                        "macro": macro_name,
                        "type": "lef_pin"
                    }.items()))
                ))
                
                # 添加宏单元-引脚关系
                self.relations.add(DesignRelation(
                    source=f"macro_{macro_name}",
                    target=pin_id,
                    type="contains",
                    properties=tuple(sorted({
                        "type": "macro_pin"
                    }.items()))
                ))

def main():
    design_dir = "/Users/keqin/Documents/workspace/chip-rag/chip_design/ispd_2015_contest_benchmark"
    parser = DesignFileParser(design_dir)
    parser.parse_all()

if __name__ == "__main__":
    main() 