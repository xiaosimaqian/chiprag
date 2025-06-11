"""
RAG控制器模块
"""

import logging
from typing import Dict, Optional, Any
from modules.core.rag_controller import RAGController
from modules.core.rag_system import RAGSystem as BaseRAGSystem

logger = logging.getLogger(__name__)

class RAGSystem(BaseRAGSystem):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            knowledge_base=None,
            llm_manager=None,
            embedding_manager=None,
            layout_generator=None,
            evaluator=None,
            config=config
        )
        self._controller = None
        
    @property
    def controller(self):
        if self._controller is None:
            from chiprag.modules.core.rag_controller import RAGController
            self._controller = RAGController(self.config)
        return self._controller

class RAGController:
    """RAG控制器"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.knowledge_base = None
        self.llm_manager = None
        self.embedding_manager = None
        self.rag_system = None
        self.evaluator = None
        self.layout_generator = None
        self.kg_encoder = None
        self._init_components()
        
    def _init_components(self):
        """初始化组件"""
        from modules.knowledge.knowledge_base import KnowledgeBase
        from modules.utils.llm_manager import LLMManager
        from modules.utils.embedding_manager import EmbeddingManager
        from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
        from modules.core.rag_model import LayoutGenerator
        from modules.encoders.graph.kg_encoder import KGEncoder
        
        self.knowledge_base = KnowledgeBase(self.config.get('knowledge_base', {}))
        self.llm_manager = LLMManager(self.config.get('llm_config', {}))
        self.embedding_manager = EmbeddingManager(self.config.get('embedding_config', {}))
        self.rag_system = RAGSystem(self.config)
        self.evaluator = MultiObjectiveEvaluator(self.config.get('evaluator_config', {}))
        
        # 初始化布局生成器
        layout_config = {
            'input_size': self.config.get('hidden_size', 512),
            'hidden_size': self.config.get('hidden_size', 512),
            'num_layers': self.config.get('num_layers', 6),
            'num_heads': self.config.get('num_heads', 8),
            'dropout': self.config.get('dropout', 0.1)
        }
        self.layout_generator = LayoutGenerator(layout_config)
        
        # 初始化知识图谱编码器
        kg_config = {
            'num_entities': self.config.get('num_entities', 1000),
            'num_relations': self.config.get('num_relations', 100),
            'embedding_dim': self.config.get('embedding_dim', 100),
            'hidden_dim': self.config.get('hidden_dim', 256),
            'num_layers': self.config.get('num_layers', 2),
            'dropout': self.config.get('dropout', 0.1),
            'batch_size': self.config.get('batch_size', 32)
        }
        self.kg_encoder = KGEncoder(kg_config)
        logger.info("RAG控制器初始化完成")
        
    def run(self, design_info: Dict) -> Dict:
        """运行RAG系统
        
        Args:
            design_info: 设计信息
            
        Returns:
            布局结果
        """
        return self.controller.run(design_info) 