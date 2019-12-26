# 2Sum：https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        if len(numbers)<2:
            return None
        i,j=0,len(numbers)-1
        while(i<j):
            t=numbers[i]+numbers[j]
            if t<target:
                i+=1
            elif t>target:
                j-=1
            else:
                return [i+1,j+1]
        return None 

# 15. 3Sum https://leetcode.com/problems/3sum/
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums)<3:
            return []
        #排序
        nums.sort()
        res=set()
        for i,v in enumerate(nums[:-2]):
            if i>=1 and v==nums[i-1]:
                continue
            d={}
            for x in nums[i+1:]:
                if x not in d:
                    d[-v-x]=1
                else:
                    res.add((v,-v-x,x))
        return map(list,res)

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums)<3:
            return []
        nums.sort()
        res=[]
        for i in range(len(nums)-2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l,r = i+1 ,len(nums)-1
            while l<r:
                s = nums[i]+nums[l]+nums[r]
                if s<0:
                    l+=1
                elif s>0:
                    r-=1
                else:
                    res.append((nums[i],nums[l],nums[r]))
                    while l < r and nums[l]==nums[l+1]:
                        l+=1
                    while l < r and nums[r]==nums[r-1]:
                        r-=1
                    l+=1
                    r-=1
        return res

# 4sum https://leetcode.com/problems/4sum/
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        res=[]
        for i in range(len(nums)):
            if i>0 and nums[i]==nums[i-1]:
                continue
            for j in range(i+1,len(nums)):
                if j>i+1 and nums[j]==nums[j-1]:
                    continue
                l=j+1
                r=len(nums)-1
                while l<r:
                    s = nums[i]+nums[j]+nums[l]+nums[r]
                    if s>target:
                        r-=1
                    elif s<target:
                        l+=1
                    else:
                        res.append([nums[i],nums[j],nums[l],nums[r]])
                        while l < r and nums[l]==nums[l+1]:
                            l+=1
                        while l < r and nums[r]==nums[r-1]:
                            r-=1
                        l+=1
                        r-=1
        return res

# 88. Merge Sorted Array https://leetcode.com/problems/merge-sorted-array/
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        while m>0 and n>0:
            if nums1[m-1]>=nums2[n-1]:
                nums1[m+n-1]=nums1[m-1]
                m=m-1
            else:
                nums1[m+n-1]=nums2[n-1]
                n=n-1
        if n>0:
            nums1[:n]=nums2[:n]

# 两个有序！！列表找出列表中两个元素相减的最小值
def printClosest(ar1,ar2):
    m=len(ar1)
    n=len(ar2)
    diff=float("inf")
    p1=0
    p2=0
    while (p1<m and p2<n):
        if abs(ar1[p1]-ar2[p2]<diff):
            diff =abs(ar1[p1]-ar2[p2])
        if (ar1[p1]>ar2[p2]):
            p2+=1
        else:
            p1+=1
    return diff

# 连续子串的不超过m的最大值
a=[4,7,12,1,2,3,6]
m=17
output=(2,4)
from itertools import accumulate
def max_subarray(numbers,ceiling):
    cum_sum=[0]
    cum_sum=cum_sum+numbers
    cum_sum=list(accumulate(cum_sum))
    l=0
    r=1
    maximun=0
    while l<len(cum_sum):
        while r< len(cum_sum) and cum_sum[r]-cum_sum[l]<=ceiling:
            r+=1
        if cum_sum[r-1]-cum_sum[l]>maximun:
            maximun = cum_sum[r-1]-cum_sum[l]
            pos=(l,r-2)
        l+=1
    return pos

# 多数投票问题：https://leetcode.com/problems/majority-element/description/
# 摩尔投票法
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        result=count=0
        for i in nums:
            if count==0:
                result=i
                count=1
            elif result==i:
                count+=1
            else:
                count-=1
        return result

# 229. Majority Element II https://leetcode.com/problems/majority-element-ii/
# Input: [1,1,1,3,3,2,2,2]
# Output: [1,2]
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        n1=n2=None
        c1=c2=0
        for n in nums:
            if n1==n:
                c1+=1
            elif n2==n:
                c2+=1
            elif c1==0:
                n1,c1=n,1
            elif c2==0:
                n2,c2=n,1
            else:
                c1,c2=c1-1,c2-1
        size=len(nums)
        return [n for n in (n1,n2) if n is not None and nums.count(n)>size/3]

# sort-colors:https://leetcode.com/problems/sort-colors/description/
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        i,j=0,len(nums)-1
        n=0
        while n<=j:
            if nums[n]==0:
                nums[i],nums[n]=nums[n],nums[i]
                n+=1
                i+=1
            elif nums[n]==1:
                n+=1
            else :
                nums[n],nums[j]=nums[j],nums[n]
                j-=1
        return nums

# 658. Find K Closest Elements https://leetcode.com/problems/find-k-closest-elements/
# Input: [1,2,3,4,5], k=4, x=3
# Output: [1,2,3,4]
class Solution:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        left=right=bisect.bisect_left(arr,x)
        while right-left<k:
            if left==0:
                return arr[:k]
            if right==len(arr):
                return arr[-k:]
            if x-arr[left-1]<=arr[right]-x:
                left-=1
            else:
                right+=1
        return arr[left:right] 

# 11. Container With Most Water https://leetcode.com/problems/container-with-most-water/
# Input: [1,8,6,2,5,4,8,3,7]
# Output: 49
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left=0
        right=len(height)-1
        res=0
        while left<right:
            water = min(height[left],height[right])*(right-left)
            res=max(res,water)
            if height[left]<height[right]:
                left+=1
            else:
                right-=1
        return res

# 42. Trapping Rain Water https://leetcode.com/problems/trapping-rain-water/
from itertools import accumulate
class Solution:
    def trap(self, height: List[int]) -> int:
        #求出每个位置之前出现的最大值
        l = list(accumulate(height, max))
        r = list(accumulate(reversed(height), max))
        r.reverse()
        ans = 0
        for h,i,j in zip(height,l,r):
            ans += min(i,j) - h
        return ans