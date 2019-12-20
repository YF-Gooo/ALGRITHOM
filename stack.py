# 是否为有效括号组合
def invalid(s):
    stack=[]
    paren_map={")":"(","]":"[","}":"{"}
    for c in s:
        if c not in paren_map:
            stack.append(c)
        elif not stack or paren_map[c]!=stack.pop():
            return False
    return not stack

# 71. Simplify Path https://leetcode.com/problems/simplify-path/discuss/450290/Simple-Python-O(n)-with-stack
# Input: "/home//foo/"
# Output: "/home/foo"
class Solution(object):
    def simplifyPath(self, path):
        """
        :type path: str
        :rtype: str
        """
        p = path.split('/')
        stack = []
        for node in p:
            if node == "..":
                if stack:
                    stack.pop()
            elif node == "." or node == "":
                continue
            else:
                stack.append(node)
        return '/' + '/'.join(stack)

# 394. Decode String https://leetcode.com/problems/decode-string/
# s = "3[a]2[bc]", return "aaabcbc".
# s = "3[a2[c]]", return "accaccacc".
# s = "2[abc]3[cd]ef", return "abcabccdcdcdef".

class Solution:
    def decodeString(self, s: str) -> str:
        stack=[]
        stack.append(["",1])
        num=""
        for ch in s:
            if ch.isdigit():
                num+=ch
            elif ch == "[":
                stack.append(["",int(num)])
                num=""
            elif ch == "]":
                st,k=stack.pop()
                stack[-1][0]+=st*k
            else:
                stack[-1][0]+=ch
        return stack[0][0]

# 682. Baseball Game https://leetcode.com/problems/baseball-game/
# Input: ["5","2","C","D","+"]
# Output: 30
# "+" (one round's score): Represents that the points you get in this round are the sum of the last two valid round's points.
# "D" (one round's score): Represents that the points you get in this round are the doubled data of the last valid round's points.
# "C" (an operation, which isn't a round's score): Represents the last valid round's points you get were invalid and should be removed.
class Solution:
    def calPoints(self, ops: List[str]) -> int:
        stack=[]
        for s in ops:
            if s=="C":
                stack.pop()
            elif s=="D":
                stack.append(2*stack[-1])
            elif s=="+":
                stack.append(stack[-1]+stack[-2])
            else:
                stack.append(int(s))
        return sum(stack)

# 735. Asteroid Collision https://leetcode.com/problems/asteroid-collision/
# Input: 
# asteroids = [5, 10, -5]
# Output: [5, 10]
# Explanation: 
# The 10 and -5 collide resulting in 10.  The 5 and 10 never collide.
# Input: 
# asteroids = [10, 2, -5]
# Output: [10]
# Explanation: 
# The 2 and -5 collide resulting in -5.  The 10 and -5 collide resulting in 10.
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        ans=[]
        for new in asteroids:
            while ans and new<0 and ans[-1]>0:
                if ans[-1] <-new:
                    ans.pop()
                    continue
                elif ans[-1] ==-new:
                    ans.pop()
                break
            else:
                ans.append(new)
        return ans

# 739. Daily Temperatures https://leetcode.com/problems/daily-temperatures/
# Given a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. 
# If there is no future day for which this is possible, put 0 instead.
# For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].
# Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].
# 留在栈里面的都是没有找到最大值的可怜数，当遇到比他们大的数，大数会作为天使把他们送去天堂，但牺牲自己留在栈中，成为普通数。
# 天使的能力只能接走比自己小的数，越大的数字能力越大却也越难被接走。
# 每个天使在变为凡人的时候会留下自己的编号，我们可以去原数组中查找编号所对应的能力值
# pop即为解放
# stack时不执行解放操作
class Solution(object):
    def dailyTemperatures(self, T):
        ans = [0] * len(T)
        stack = []
        for i, t in enumerate(T):
            while stack and T[stack[-1]] < t:
                cur = stack.pop()
                ans[cur] = i - cur
            stack.append(i)
        return ans

# Next Greater Element
# 找出下面这个第一个比它大的数字，如果没有就返回-1
# Input: [1,2,1]
# Output: [2,-1,-1]
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        ans = [-1] * len(nums)
        stack = []
        for i, t in enumerate(nums):
            while stack and nums[stack[-1]] < t:
                cur = stack.pop()
                ans[cur] = t
            stack.append(i+1)
        return ans

# 503. Next Greater Element II https://leetcode.com/problems/next-greater-element-ii/
# 是个环形列表
# Input: [1,2,1]
# Output: [2,-1,2]
# Explanation: The first 1's next greater number is 2; 
# The number 2 can't find next greater number; 
# The second 1's next greater number needs to search circularly, which is also 2.
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n=len(nums)
        nums=nums*2
        ans = [-1] * len(nums)
        stack = []
        for i, t in enumerate(nums):
            while stack and nums[stack[-1]] < t:
                cur = stack.pop()
                ans[cur] = t
            stack.append(i)
        return ans[:n]



