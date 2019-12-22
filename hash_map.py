# https://leetcode.com/problems/word-pattern/
# Example 1:
# Input: pattern = "abba", str = "dog cat cat dog"
# Output: true
class Solution:
    def wordPattern(self, pattern: str, str: str) -> bool:
        p=pattern
        s=str.split()
        return len(s) == len(p) and len(set(zip(p,s)))==len(set(s))==len(set(p))


# LRUCache https://leetcode.com/problems/lru-cache/submissions/
import collections
class LRUCache:

    def __init__(self, capacity: int):
        self.dic=collections.OrderedDict()
        self.remain = capacity
        

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1
        v= self.dic.pop(key)
        self.dic[key]=v
        return v

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self.dic.pop(key)
        else:
            if self.remain>0:
                self.remain -=1
            else:
                self.dic.popitem(last=False)
        self.dic[key]=value

# 149. Max Points on a Line https://leetcode.com/problems/max-points-on-a-line/
class Solution:
    def maxPoints(self, points):
        if len(points) == 0: return 0
        m = 1
        for i, (y1,x1) in enumerate(points[:-1]):
            d = {}
            psame = 1
            for (y0,x0) in points[i+1:]:
                if y1==y0 and x1==x0:
                    psame += 1
                    continue
                elif x1==x0:
                    k = 'INF'
                else:
                    k = float(y1-y0)/(x1-x0)
                d[k] = d.get(k,0)+1
            try:
                pmax = max(d.values())
            except:
                pmax = 0
            m = max(m, pmax+psame)
        return m