import re
import os
from typing import Dict, Any, List, Tuple

class VerilogParser:
    def __init__(self):
        self.current_module = None
        self.data = {'modules': []}
        self.parameters = {}
        
    def parse_verilog(self, file_path: str) -> Dict[str, Any]:
        """解析Verilog文件
        
        Args:
            file_path: Verilog文件路径
            
        Returns:
            解析后的数据字典
        """
        # 确保文件路径是绝对路径
        file_path = os.path.abspath(file_path)
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        # 移除注释
        content = self._remove_comments(content)
        
        # 解析参数定义
        self._parse_parameters(content)
        
        # 解析模块
        module_sections = re.finditer(r'module\s+(\w+)\s*(?:#\s*\((.*?)\))?\s*\((.*?)\);.*?endmodule', content, re.DOTALL)
        
        for section in module_sections:
            module_name = section.group(1)
            parameters_str = section.group(2)
            ports_str = section.group(3)
            module_content = section.group(0)
            
            # 解析参数
            parameters = self._parse_module_parameters(parameters_str) if parameters_str else {}
            
            # 解析端口
            ports = self._parse_ports(ports_str)
            
            # 解析内部信号
            signals = self._parse_signals(module_content)
            
            # 解析实例
            instances = self._parse_instances(module_content)
            
            # 解析生成语句
            generate_blocks = self._parse_generate_blocks(module_content)
            
            self.data['modules'].append({
                'name': module_name,
                'parameters': parameters,
                'ports': ports,
                'signals': signals,
                'instances': instances,
                'generate_blocks': generate_blocks,
                'file': file_path
            })
        
        return self.data
    
    def _remove_comments(self, content: str) -> str:
        """移除注释
        
        Args:
            content: 文件内容
            
        Returns:
            移除注释后的内容
        """
        # 移除单行注释
        content = re.sub(r'//.*?$', '', content, flags=re.MULTILINE)
        
        # 移除多行注释
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        return content
    
    def _parse_parameters(self, content: str):
        """解析全局参数定义
        
        Args:
            content: 文件内容
        """
        param_matches = re.finditer(r'parameter\s+(\w+)\s*=\s*([^;]+);', content)
        for match in param_matches:
            param_name = match.group(1)
            param_value = match.group(2).strip()
            self.parameters[param_name] = param_value
    
    def _parse_module_parameters(self, parameters_str: str) -> Dict[str, Any]:
        """解析模块参数
        
        Args:
            parameters_str: 参数字符串
            
        Returns:
            参数字典
        """
        parameters = {}
        if not parameters_str:
            return parameters
            
        param_matches = re.finditer(r'parameter\s+(\w+)\s*(?:=\s*([^,\)]+))?', parameters_str)
        for match in param_matches:
            param_name = match.group(1)
            param_value = match.group(2).strip() if match.group(2) else None
            parameters[param_name] = param_value
            
        return parameters
    
    def _parse_ports(self, ports_str: str) -> List[Dict[str, Any]]:
        """解析端口列表
        
        Args:
            ports_str: 端口定义字符串
            
        Returns:
            端口列表
        """
        ports = []
        port_matches = re.finditer(r'(input|output|inout)\s*(?:\[(\d+):(\d+)\])?\s*(\w+)', ports_str)
        
        for match in port_matches:
            port_type = match.group(1)
            msb = int(match.group(2)) if match.group(2) else None
            lsb = int(match.group(3)) if match.group(3) else None
            port_name = match.group(4)
            
            ports.append({
                'name': port_name,
                'type': port_type,
                'width': (msb - lsb + 1) if msb is not None and lsb is not None else 1,
                'msb': msb,
                'lsb': lsb
            })
        
        return ports
    
    def _parse_signals(self, module_content: str) -> List[Dict[str, Any]]:
        """解析内部信号
        
        Args:
            module_content: 模块内容
            
        Returns:
            信号列表
        """
        signals = []
        signal_matches = re.finditer(r'(wire|reg)\s*(?:\[(\d+):(\d+)\])?\s*(\w+)', module_content)
        
        for match in signal_matches:
            signal_type = match.group(1)
            msb = int(match.group(2)) if match.group(2) else None
            lsb = int(match.group(3)) if match.group(3) else None
            signal_name = match.group(4)
            
            signals.append({
                'name': signal_name,
                'type': signal_type,
                'width': (msb - lsb + 1) if msb is not None and lsb is not None else 1,
                'msb': msb,
                'lsb': lsb
            })
        
        return signals
    
    def _parse_instances(self, module_content: str) -> List[Dict[str, Any]]:
        """解析实例
        
        Args:
            module_content: 模块内容
            
        Returns:
            实例列表
        """
        instances = []
        instance_matches = re.finditer(r'(\w+)\s*(?:#\s*\((.*?)\))?\s+(\w+)\s*\((.*?)\);', module_content, re.DOTALL)
        
        for match in instance_matches:
            instance_type = match.group(1)
            parameters_str = match.group(2)
            instance_name = match.group(3)
            connections_str = match.group(4)
            
            # 解析参数
            parameters = self._parse_instance_parameters(parameters_str) if parameters_str else {}
            
            # 解析连接
            connections = {}
            connection_matches = re.finditer(r'\.(\w+)\s*\((.*?)\)', connections_str)
            
            for conn_match in connection_matches:
                port_name = conn_match.group(1)
                net_name = conn_match.group(2).strip()
                connections[port_name] = net_name
            
            instances.append({
                'type': instance_type,
                'name': instance_name,
                'parameters': parameters,
                'connections': connections
            })
        
        return instances
    
    def _parse_instance_parameters(self, parameters_str: str) -> Dict[str, Any]:
        """解析实例参数
        
        Args:
            parameters_str: 参数字符串
            
        Returns:
            参数字典
        """
        parameters = {}
        if not parameters_str:
            return parameters
            
        param_matches = re.finditer(r'\.(\w+)\s*\((.*?)\)', parameters_str)
        for match in param_matches:
            param_name = match.group(1)
            param_value = match.group(2).strip()
            parameters[param_name] = param_value
            
        return parameters
    
    def _parse_generate_blocks(self, module_content: str) -> List[Dict[str, Any]]:
        """解析生成语句
        
        Args:
            module_content: 模块内容
            
        Returns:
            生成语句列表
        """
        generate_blocks = []
        generate_matches = re.finditer(r'generate\s*(.*?)endgenerate', module_content, re.DOTALL)
        
        for match in generate_matches:
            block_content = match.group(1)
            
            # 解析for循环
            for_matches = re.finditer(r'for\s*\((.*?)\)\s*begin\s*(.*?)end', block_content, re.DOTALL)
            for for_match in for_matches:
                loop_var = for_match.group(1).strip()
                loop_content = for_match.group(2)
                
                # 解析循环中的实例
                instances = self._parse_instances(loop_content)
                
                generate_blocks.append({
                    'type': 'for',
                    'loop_var': loop_var,
                    'instances': instances
                })
            
            # 解析if语句
            if_matches = re.finditer(r'if\s*\((.*?)\)\s*begin\s*(.*?)end', block_content, re.DOTALL)
            for if_match in if_matches:
                condition = if_match.group(1).strip()
                if_content = if_match.group(2)
                
                # 解析if中的实例
                instances = self._parse_instances(if_content)
                
                generate_blocks.append({
                    'type': 'if',
                    'condition': condition,
                    'instances': instances
                })
        
        return generate_blocks 