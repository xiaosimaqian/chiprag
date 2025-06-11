import numpy as np
from typing import List, Dict, Any

class BasicRetriever:
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        # 如果没有知识库，返回10个占位组件
        return [
            {
                "id": i,
                "x": float(np.random.uniform(-0.8, 0.8)),
                "y": float(np.random.uniform(-0.8, 0.8)),
                "width": 0.1,
                "height": 0.1,
                "type": "placeholder"
            }
            for i in range(10)
        ] 