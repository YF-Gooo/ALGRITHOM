#73. Set Matrix Zeroes https://leetcode.com/problems/set-matrix-zeroes/
# Input: 
# [
#   [1,1,1],
#   [1,0,1],
#   [1,1,1]
# ]
# Output: 
# [
#   [1,0,1],
#   [0,0,0],
#   [1,0,1]
# ]
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row=[]
        col=[]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j]==0:
                    if i not in row:
                        row.append(i)
                    if j not in col:
                        col.append(j)
                        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in row or j in col:
                    matrix[i][j]=0

# 48. Rotate Image https://leetcode.com/problems/rotate-image/:
# Given input matrix = 
# [
#   [1,2,3],
#   [4,5,6],
#   [7,8,9]
# ],

# rotate the input matrix in-place such that it becomes:
# [
#   [7,4,1],
#   [8,5,2],
#   [9,6,3]
# ]
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n=len(matrix)
        for idx, arr in enumerate(list(zip(*matrix))):
            matrix[idx] = list(arr[::-1])
        return matrix

# 485. Max Consecutive Ones https://leetcode.com/problems/max-consecutive-ones/
# Input: [1,1,0,1,1,1]
# Output: 3
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        globl=0
        local=0
        for n in nums:
            if n==0:
                globl=max(globl,local)
                local=0
            elif n==1:
                local+=1
        globl=max(globl,local)
        return globl

# 747. Largest Number At Least Twice of Others https://leetcode.com/problems/largest-number-at-least-twice-of-others/
# Input: nums = [3, 6, 1, 0]
# Output: 1
# Explanation: 6 is the largest integer, and for every other number in the array x,
# 6 is more than twice as big as x.  The index of value 6 is 1, so we return 1.
class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        first=[-1,-1]
        second=[-1,-1]
        for i,n in enumerate(nums):
            if n>first[1]:
                second=first
                first=[i,n]
                continue
            if n<first[1] and n>second[1]:
                second=[i,n]
        if (first[1]//2)>=second[1]:
            return first[0]
        else:
            return -1

# 448. Find All Numbers Disappeared in an Array
# 通过负号法
# Input:
# [4,3,2,7,8,2,3,1]

# Output:
# [5,6]
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        res=[]
        for n in nums:
            if nums[abs(n)-1]<0:
                continue
            nums[abs(n)-1]=-nums[abs(n)-1]
        for i,n in enumerate(nums):
            if n>0:
                res.append(i+1)
        return res


# 66. Plus One https://leetcode.com/problems/plus-one/
# Input: [1,2,3]
# Output: [1,2,4]
# Explanation: The array represents the integer 123.
# 妙哉
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if len(digits)==0:
            return False
        addCarry=1
        for i in range(len(digits)-1,-1,-1):
            digits[i]+=addCarry
            if digits[i]==10:
                digits[i]=0
                if i==0:
                    digits.insert(0,1)
            else:
                break
        return digits
        


# 56. Merge Intervals https://leetcode.com/problems/merge-intervals/
# Input: [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        out = []
        for i in sorted(intervals, key=lambda i: i[0]):
            if out and i[0] <= out[-1][1]:
                out[-1][1] = max(out[-1][1], i[1])
            else:
                out += i
        return out


# 57. Insert Interval https://leetcode.com/problems/insert-interval/
# Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
# Output: [[1,5],[6,9]] 
def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        stack = []
        for i in range(len(intervals) - 1, -1, -1):
            stack.append(intervals[i])
            
        res = []
        while stack:
            cur_interval = stack.pop()
            # Left ouf of bounds
            if cur_interval[0] > newInterval[1]:
                res.append(newInterval)
                # Update newInterval to be cur_interval to ensure we always come into this condition because there is no more merging that will happen after newInterval has been appended so we keep appending the previous interval
                newInterval = cur_interval
            # Right out of bounds
            elif cur_interval[1] < newInterval[0]:
                res.append(cur_interval)
            else:
                newInterval = [min(cur_interval[0], newInterval[0]), max(cur_interval[1], newInterval[1])]
        # Append the last interval which will be the fully merged newInterval or the last cur_interval from the last iteration
        res.append(newInterval)
        return res

