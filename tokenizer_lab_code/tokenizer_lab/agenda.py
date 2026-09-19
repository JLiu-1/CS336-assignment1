"""教师提供：带惰性删除的最大优先队列；学生无需实现 heapq 比较细节。"""

import heapq
from collections import Counter
from dataclasses import dataclass

from .common import Pair


@dataclass(frozen=True)
class _Entry:
    count: int
    pair: Pair

    def __lt__(self, other: "_Entry") -> bool:
        return (self.count, self.pair) > (other.count, other.pair)


class PairAgenda:
    def __init__(self, counts: Counter[Pair]):
        self.counts = counts
        self.heap = [_Entry(c, p) for p, c in counts.items() if c > 0]
        heapq.heapify(self.heap)

    def refresh(self, changed: set[Pair]) -> None:
        for pair in changed:
            count = self.counts.get(pair, 0)
            if count > 0:
                heapq.heappush(self.heap, _Entry(count, pair))
        # 限制大量过期条目的内存占用；这是工程设施，不是 B 版作业题。
        if len(self.heap) > 4 * len(self.counts) + 1024:
            self.heap = [_Entry(c, p) for p, c in self.counts.items() if c > 0]
            heapq.heapify(self.heap)

    def pop_best(self) -> Pair | None:
        while self.heap:
            item = heapq.heappop(self.heap)
            if item.count == self.counts.get(item.pair, 0) and item.count > 0:
                return item.pair
        return None
