import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import logging
from typing import Dict, Any, List, Union
import gc
import os
import ssl
import certifi
import requests
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class BertTextEncoder(nn.Module):
    """基于BERT的文本编码器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化BERT文本编码器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.max_length = config.get('max_length', 512)
        self.batch_size = config.get('batch_size', 32)
        self.model_path = config.get('model_path', 'models/bert')
        self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
        self.use_ollama = config.get('use_ollama', False)
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 设置SSL验证
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
        
        if self.use_ollama:
            self._init_ollama()
        else:
            self._init_local_model()
            
    def _init_ollama(self):
        """初始化Ollama模型"""
        try:
            # 检查Ollama服务是否可用
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError("无法连接到Ollama服务")
                
            # 获取模型信息
            self.model_name = self.config.get('ollama_model', 'llama2')
            logger.info(f"使用Ollama模型: {self.model_name}")
            
        except Exception as e:
            logger.error(f"初始化Ollama失败: {str(e)}")
            raise
            
    def _init_local_model(self):
        """初始化本地模型"""
        try:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
                
            # 加载模型配置
            config_path = model_path / 'config.json'
            if not config_path.exists():
                raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
                
            with open(config_path) as f:
                model_config = json.load(f)
                
            # 加载模型权重
            weights_path = model_path / 'pytorch_model.bin'
            if not weights_path.exists():
                raise FileNotFoundError(f"模型权重文件不存在: {weights_path}")
                
            # 加载模型
            self.model = torch.load(weights_path, map_location=self.device)
            self.model.eval()
            logger.info(f"成功加载本地模型: {self.model_path}")
            
        except Exception as e:
            logger.error(f"加载本地模型失败: {str(e)}")
            raise
            
    def _validate_text(self, text: str) -> None:
        """验证输入文本
        
        Args:
            text: 输入文本
            
        Raises:
            ValueError: 文本无效
        """
        if not isinstance(text, str):
            raise ValueError("输入必须是字符串")
        if not text.strip():
            raise ValueError("输入文本不能为空")
            
    def _validate_texts(self, texts: List[str]) -> None:
        """验证输入文本列表
        
        Args:
            texts: 输入文本列表
            
        Raises:
            ValueError: 文本列表无效
        """
        if not isinstance(texts, list):
            raise ValueError("输入必须是列表")
        if not texts:
            raise ValueError("输入列表不能为空")
        for text in texts:
            self._validate_text(text)
            
    def _process_batch(self, texts: List[str], start_idx: int) -> torch.Tensor:
        """处理一批文本
        
        Args:
            texts: 文本列表
            start_idx: 起始索引
            
        Returns:
            文本编码向量
        """
        end_idx = min(start_idx + self.batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        
        # 对文本进行编码
        encoded = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 将输入移到模型所在设备
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # 使用[CLS]标记的输出作为文本表示
        text_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # 清理内存
        del encoded, input_ids, attention_mask, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return text_embeddings
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            texts: 文本列表
            
        Returns:
            文本编码向量
        """
        self._validate_texts(texts)
        
        # 分批处理
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_embeddings = self._process_batch(texts, i)
            all_embeddings.append(batch_embeddings)
            
        # 合并所有批次的编码
        return torch.cat(all_embeddings, dim=0)
        
    def encode_text(self, text: str) -> torch.Tensor:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本编码向量
        """
        self._validate_text(text)
        return self.forward([text])[0]
        
    def encode(self, text: str) -> torch.Tensor:
        """编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: 文本编码向量
        """
        try:
            if self.use_ollama:
                return self._encode_with_ollama(text)
            else:
                return self._encode_with_local_model(text)
                
        except Exception as e:
            logger.error(f"文本编码失败: {str(e)}")
            raise
            
    def _encode_with_ollama(self, text: str) -> torch.Tensor:
        """使用Ollama进行编码
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: 文本编码向量
        """
        try:
            # 调用Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API调用失败: {response.text}")
                
            # 解析响应
            result = response.json()
            embedding = torch.tensor(result['embedding'], device=self.device)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Ollama编码失败: {str(e)}")
            raise
            
    def _encode_with_local_model(self, text: str) -> torch.Tensor:
        """使用本地模型进行编码
        
        Args:
            text: 输入文本
            
        Returns:
            torch.Tensor: 文本编码向量
        """
        try:
            # 预处理文本
            inputs = self._preprocess_text(text)
            
            # 编码
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.last_hidden_state.mean(dim=1)
                
        except Exception as e:
            logger.error(f"本地模型编码失败: {str(e)}")
            raise
            
    def _preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """预处理文本
        
        Args:
            text: 输入文本
            
        Returns:
            Dict[str, torch.Tensor]: 预处理后的输入
        """
        # 实现文本预处理逻辑
        # 这里需要根据具体使用的模型来实现
        pass
        
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        批量编码文本
        
        Args:
            texts: 文本列表
            
        Returns:
            文本编码向量矩阵
        """
        return self.forward(texts)
        
    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度
        
        Args:
            text1: 第一段文本
            text2: 第二段文本
            
        Returns:
            float: 相似度分数
        """
        try:
            # 编码文本
            encoding1 = self.encode(text1)
            encoding2 = self.encode(text2)
            
            # 计算余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                encoding1,
                encoding2
            )
            
            return similarity.item()
            
        except Exception as e:
            logger.error(f"计算文本相似度失败: {str(e)}")
            raise
        
    def save_model(self, path: str):
        """保存模型
        
        Args:
            path: 保存路径
        """
        try:
            if not self.use_ollama:
                torch.save(self.model, path)
                logger.info(f"模型已保存到: {path}")
                
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            raise
        
    @classmethod
    def load_model(cls, path: str) -> 'BertTextEncoder':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.tokenizer = checkpoint['tokenizer']
        model.max_length = checkpoint['max_length']
        model.batch_size = checkpoint['batch_size']
        logger.info(f"模型已从 {path} 加载")
        return model 