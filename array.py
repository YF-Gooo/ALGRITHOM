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