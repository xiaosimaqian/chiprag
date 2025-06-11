import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import logging
from typing import Dict, Any, List, Union
import gc

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
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载预训练模型和分词器
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"成功加载BERT模型: {self.model_name}")
        except Exception as e:
            logger.error(f"加载BERT模型失败: {str(e)}")
            raise
            
        # 冻结BERT参数
        for param in self.model.parameters():
            param.requires_grad = False
            
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
        """
        encode_text的别名方法
        
        Args:
            text: 输入文本
            
        Returns:
            文本编码向量
        """
        return self.encode_text(text)
        
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
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            相似度分数
        """
        # 编码文本
        emb1 = self.encode_text(text1)
        emb2 = self.encode_text(text2)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
        
        # 清理内存
        del emb1, emb2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return similarity.item()
        
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'max_length': self.max_length,
            'batch_size': self.batch_size
        }, path)
        logger.info(f"模型已保存到: {path}")
        
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