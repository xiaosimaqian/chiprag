"""
RAG控制器模块
"""

import logging
from typing import Dict, Optional, Any
from modules.core.rag_controller import RAGController

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
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
        self._init_components()
        
    def _init_components(self):
        """初始化组件"""
        from modules.knowledge.knowledge_base import KnowledgeBase
        from modules.utils.llm_manager import LLMManager
        from modules.utils.embedding_manager import EmbeddingManager
        from modules.evaluation.multi_objective_evaluator import MultiObjectiveEvaluator
        from modules.core.rag_model import LayoutGenerator
        
        self.knowledge_base = KnowledgeBase(self.config.get('knowledge_base', {}))
        self.llm_manager = LLMManager(self.config.get('llm_config', {}))
        self.embedding_manager = EmbeddingManager(self.config.get('embedding_config', {}))
        self.rag_system = RAGSystem(self.config)
        self.evaluator = MultiObjectiveEvaluator(self.config.get('evaluator_config', {}))
        self.layout_generator = LayoutGenerator(self.config.get('layout_config', {}), self.llm_manager)
        logger.info("RAG控制器初始化完成")
        
    def run(self, design_info: Dict) -> Dict:
        """运行RAG系统
        
        Args:
            design_info: 设计信息
            
        Returns:
            布局结果
        """
        return self.controller.run(design_info) 