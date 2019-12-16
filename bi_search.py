# 传统二分搜索
# 并且找到target出现的第一个位置
def  binarysearch(alist,item):
    if len(alist) == 0:
        return -1
    left,right = 0,len(alist)-1
    while left + 1< right:
        mid = left + (right-left)//2
        if alist[mid] == item:
            right=mid
        elif alist[mid] < item:
            left = mid
        elif alist[mid] >item:
            right = mid
    if alist[left]== item:
        return left
    if alist[right]==item:
        return right
    return -1

# 找到出现的最后一个位置
def  binarysearch(alist,item):
    if len(alist) == 0:
        return -1
    left,right = 0,len(alist)-1
    while left + 1< right:
        mid = left + (right-left)//2
        if alist[mid] == item:
            left=mid
        elif alist[mid] < item:
            left = mid
        elif alist[mid] >item:
            right = mid
    if alist[left]== item:
        return left
    if alist[right]==item:
        return right
    return -1

# 33. Search in Rotated Sorted Array https://leetcode.com/problems/search-in-rotated-sorted-array/
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums)==0:
            return -1
        left,right=0,len(nums)-1
        while left+1<right:
            mid = left + (right-left)//2
            if nums[mid]==target:
                return mid
            if nums[left]<nums[mid]:
                if nums[left]<=target and target<=nums[mid]:
                    right=mid
                else:
                    left=mid
            else:
                if nums[mid]<= target and target <=nums[right]:
                    left=mid
                else:
                    right=mid
        if nums[left]==target:
            return left
        if nums[right]==target:
            return right
        return -1

# 153. Find Minimum in Rotated Sorted Array https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
# Input: [3,4,5,1,2] 
# Output: 1
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums)==0:
            return -1
        left,right=0,len(nums)-1
        while left+1<right:
            if nums[left]<nums[right]:
                return nums[left]
            mid = left + (right-left)//2
            if nums[left]<=nums[mid]:
                left=mid+1
            else:
                right=mid
        return min(nums[left],nums[right])

# 35. Search Insert Position https://leetcode.com/problems/search-insert-position/
# Example 1:

# Input: [1,3,5,6], 5
# Output: 2
# Example 2:

# Input: [1,3,5,6], 2
# Output: 1

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1
        left,right = 0,len(nums)-1
        while left + 1< right:
            mid = left + (right-left)//2
            if nums[mid] == target:
                right=mid
            elif nums[mid] < target:
                left = mid
            elif nums[mid] >target:
                right = mid
        if nums[left]>=target:
            return left
        if nums[right]>=target:
            return right
        return right+1

# 34. Find First and Last Position of Element in Sorted Array https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
# Input: nums = [5,7,7,8,8,10], target = 8
# Output: [3,4]
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        return [self.binarysearch(nums,target,l_r="left"),
                self.binarysearch(nums,target,l_r="right")]

    def binarysearch(self,nums,target,l_r):
        if len(nums) == 0:
            return -1
        left,right = 0,len(nums)-1
        while left + 1< right:
            mid = left + (right-left)//2
            if nums[mid] == target:
                if l_r=="left":
                    right=mid
                else:
                    left=mid
            elif nums[mid] < target:
                left = mid
            elif nums[mid] >target:
                right = mid
        if l_r=="left":
            if nums[left]== target:
                return left
            if nums[right]==target:
                return right
        else:
            if nums[right]==target:
                return right
            if nums[left]== target:
                return left
        return -1

# 475. Heaters https://leetcode.com/problems/heaters/
from bisect import bisect
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        heaters.sort()
        ans=0
        for h in houses:
            hi = bisect(heaters,h)
            left = heaters[hi-1] if hi-1>=0 else float("-inf")
            right = heaters[hi] if hi<len(heaters) else float("inf")
            ans=max(ans,min(h-left,right-h))
        return ans

# 69. Sqrt(x) https://leetcode.com/problems/sqrtx/
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        left,right=1,x
        while left<=right:
            mid = left+(right-left)//2
            if x//mid==mid:
                return mid
            elif x//mid>mid:
                left=mid+1
            else:
                right=mid-1
        return right  #返回较小的

# 50. Pow(x, n) https://leetcode.com/problems/powx-n/
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n==0:
            return 1
        if n<=0:
            return 1/self.myPow(x,-n)
        if n%2:
            return x*self.myPow(x,n-1)
        return self.myPow(x*x,n/2)
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n<=0:
            return 1/self.helper(x,-n)
        return self.helper(x,n)
    
    def helper(self,x:float,n:int)->float:
        pow=1
        while n:
            if n&1:
                pow*=x
            x*=x
            n>>=1
        return pow

# 74. Search a 2D Matrix https://leetcode.com/problems/search-a-2d-matrix/
# Input:
# matrix = [
#   [1,   3,  5,  7],
#   [10, 11, 16, 20],
#   [23, 30, 34, 50]
# ]
# target = 3
# Output: true
# 这题最优解就是从左下角，向右或者向上找，O(m+n)解决这问题
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if len(matrix) == 0:
            return False
        row, col = len(matrix) - 1, 0
        while row >= 0 and col <len(matrix[0]):
            if matrix[row][col] == target: return True
            elif matrix[row][col] < target: col += 1
            elif matrix[row][col] > target: row -= 1
        return False


# 378. Kth Smallest Element in a Sorted Matrix https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/
# matrix = [
#    [ 1,  5,  9],
#    [10, 11, 13],
#    [12, 13, 15]
# ],
# k = 8,
# return 13.
import heapq
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        heap = [(row[0],i,0) for i,row in enumerate(matrix)]   #先把[1,10,12]压入堆，然后每次pop，push一个数字
        heapq.heapify(heap)
        res = 0
        for _ in range(k):
            res,i,j=heapq.heappop(heap)
            if j+1<len(matrix[0]):
                heapq.heappush(heap,(matrix[i][j+1],i,j+1))
        return res

