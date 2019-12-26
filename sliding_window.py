# 26. Remove Duplicates from Sorted Array https://leetcode.com/problems/remove-duplicates-from-sorted-array/
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        tail=0
        for j in range(1,len(nums)):
            if nums[j]!=nums[tail]:
                tail+=1
                nums[tail]=nums[j]
        return tail+1


# 80. Remove Duplicates from Sorted Array II https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
# Given nums = [0,0,1,1,1,1,2,3,3],
# Your function should return length = 7, with the first seven elements of nums being modified to 0, 0, 1, 1, 2, 3 and 3 respectively.
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        count=0
        for i in range(len(nums)):
            if count<2 or nums[count-2]!=nums[i]:
                nums[count]=nums[i]
                count+=1
        return count


# 27. Remove Element https://leetcode.com/problems/remove-element/
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i=0
        for j in range(len(nums)):
            if nums[j]!=val:
                nums[i]=nums[j]
                i+=1
        return i
            
# 643. Maximum Average Subarray I https://leetcode.com/problems/maximum-average-subarray-i/

# 239. Sliding Window Maximum https://leetcode.com/problems/sliding-window-maximum/
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if not nums:
            return []
        window,res=[],[]
        for i,x in enumerate(nums):
            if i>=k and window[0]<=i-k:
                window.pop(0)
            while window and nums[window[-1]]<x:
                window.pop()
            window.append(i)
            if i>=k-1:
                res.append(nums[window[0]])
        return res

