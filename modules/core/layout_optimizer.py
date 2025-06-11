import numpy as np
from typing import List, Dict, Any

class LayoutOptimizer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.min_spacing = self.config.get('min_spacing', 0.05)

    def optimize(self, placement: List[List[float]], max_iter: int = 10) -> List[List[float]]:
        # 多轮重叠检测与调整
        for _ in range(max_iter):
            changed = False
            n = len(placement)
            for i in range(n):
                for j in range(i+1, n):
                    if self._check_overlap(placement[i], placement[j]):
                        # 推开重叠
                        dx = placement[j][0] - placement[i][0]
                        dy = placement[j][1] - placement[i][1]
                        if abs(dx) > abs(dy):
                            placement[j][0] += self.min_spacing * (1 if dx > 0 else -1)
                        else:
                            placement[j][1] += self.min_spacing * (1 if dy > 0 else -1)
                        changed = True
            if not changed:
                break
        return placement

    def _check_overlap(self, comp1, comp2) -> bool:
        x1, y1, w1, h1 = comp1[:4]
        x2, y2, w2, h2 = comp2[:4]
        return not (
            x1 + w1 + self.min_spacing <= x2 or
            x2 + w2 + self.min_spacing <= x1 or
            y1 + h1 + self.min_spacing <= y2 or
            y2 + h2 + self.min_spacing <= y1
        ) 