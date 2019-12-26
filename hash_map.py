# 860. Lemonade Change https://leetcode.com/problems/lemonade-change/
# Input: [5,5,5,10,20]
# Output: true
# Explanation: 
# From the first 3 customers, we collect three $5 bills in order.
# From the fourth customer, we collect a $10 bill and give back a $5.
# From the fifth customer, we give a $10 bill and a $5 bill.
# Since all customers got correct change, we output true.
class Solution:
	def lemonadeChange(self, bills: List[int]) -> bool:

		collected = {
			5: 0,
			10: 0
		}
		for bill in bills:      # for every customer
			if bill == 5:       # when get 5 just add one 5
				collected[5] += 1
			elif bill == 10:        # when get 10, add on 10 and less one 5
				collected[5] -= 1
				collected[10] += 1
			else:       # when get 20, you don't have to add it
				if collected[10] > 0:
					collected[10] -= 1
					collected[5] -= 1
				else:
					collected[5] -= 3       # you have to give back 15 in case of 20

			if collected[5] < 0:        # only 5 might be negative, and when it is it means you didn't have change
				return False        # so return false

		return True     # when all bills pass it means we made it

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