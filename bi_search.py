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

# 162. Find Peak Elementhttps://leetcode.com/problems/find-peak-element/
# Input: nums = [1,2,3,1]
# Output: 2
# Explanation: 3 is a peak element and your function should return the index number 2.


class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        return self.peak_helper(nums,0,len(nums)-1)
    
    def peak_helper(self,nums,start,end):
        if start==end:
            return start
        if (start+1)==end:
            if nums[start]>nums[end]:
                return start
            return end
        mid = (start+end)//2
        if nums[mid]>nums[mid-1] and nums[mid]>nums[mid+1]:
            return mid
        if nums[mid-1]>nums[mid] and nums[mid]>nums[mid+1]:
            return self.peak_helper(nums,start,mid-1)
        return self.peak_helper(nums,mid+1,end)

# 排序：找出第k大的数字 https://leetcode.com/problems/kth-largest-element-in-an-array/description/
# 快速选择
def partition(arr,low,high): 
    i = low-1         # 最小元素索引
    pivot = arr[high]     
    for j in range(low , high): 
        # 当前元素小于或等于 pivot 
        if   arr[j] <= pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return i+1

def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high) 
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high)

# 解法O(n) 测试结果1876 ms，16.3 MB
class Solution(object):
    def partition(self,arr,low,high):
        i=low-1
        pivot=arr[high]
        for j in range(low,high):
            if arr[j]<=pivot:
                i=i+1
                arr[i],arr[j]=arr[j],arr[i]
        arr[i+1],arr[high]=arr[high],arr[i+1]
        return i+1
    
    def helper(self,arr,low,high,k):
        p=self.partition(arr,low,high)
        if p+1==k:
            return arr[p]
        if p+1>k:
            return self.helper(arr,low,p-1,k)
        else:
            return self.helper(arr,p+1,high,k)
        
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        k=len(nums)-k+1
        return self.helper(nums,0,len(nums)-1,k)

# 解法O(nlogk) 虽然测试下来是48ms，12.4 MB我怀疑用了c++的库
from heapq import * 
class Solution(object):            
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heap=nums[:k]
        heapify(heap)
        for n in nums[k:]:
            if heap[0]<n:
                heapreplace(heap, n) #pop出最小的，push进新的元素
        return heap[0] #heap[0]为最小元素