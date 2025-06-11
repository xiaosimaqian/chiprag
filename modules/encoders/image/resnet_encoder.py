import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ResNetImageEncoder(nn.Module):
    """基于ResNet的图像编码器"""
    
    def __init__(self, config=None):
        """初始化ResNet编码器
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config or {}
        self.model_name = self.config.get('model_name', 'resnet50')
        self.pretrained = self.config.get('pretrained', True)
        self.feature_dim = self.config.get('feature_dim', 2048)
        
        # 检查CUDA是否可用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def _load_model(self):
        """加载ResNet模型"""
        if self.model_name == 'resnet50':
            return models.resnet50(pretrained=self.pretrained)
        elif self.model_name == 'resnet34':
            return models.resnet34(pretrained=self.pretrained)
        elif self.model_name == 'resnet18':
            return models.resnet18(pretrained=self.pretrained)
        else:
            raise ValueError(f"不支持的ResNet模型: {self.model_name}")
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像张量 [batch_size, channels, height, width]
            
        Returns:
            图像编码向量
        """
        with torch.no_grad():
            features = self.model(images)
            # 移除batch维度
            features = features.squeeze()
            
        return features
        
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        编码单个图像
        
        Args:
            image: PIL图像
            
        Returns:
            图像编码向量
        """
        # 预处理图像
        image_tensor = self.transform(image).unsqueeze(0)
        
        # 将输入移到模型所在设备
        image_tensor = image_tensor.to(self.device)
        
        return self.forward(image_tensor)
        
    def encode_images(self, images: list) -> torch.Tensor:
        """
        批量编码图像
        
        Args:
            images: PIL图像列表
            
        Returns:
            图像编码向量矩阵
        """
        # 预处理图像
        image_tensors = torch.stack([
            self.transform(image) for image in images
        ])
        
        # 将输入移到模型所在设备
        image_tensors = image_tensors.to(self.device)
        
        return self.forward(image_tensors)
        
    def compute_similarity(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        计算两个图像的相似度
        
        Args:
            image1: 第一个图像
            image2: 第二个图像
            
        Returns:
            相似度分数
        """
        # 编码图像
        emb1 = self.encode_image(image1)
        emb2 = self.encode_image(image2)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0)
        
        return similarity.item()
        
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'transform': self.transform
        }, path)
        logger.info(f"模型已保存到: {path}")
        
    @classmethod
    def load_model(cls, path: str) -> 'ResNetImageEncoder':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            加载的模型实例
        """
        checkpoint = torch.load(path)
        model = cls()
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.transform = checkpoint['transform']
        logger.info(f"模型已从 {path} 加载")
        return model 