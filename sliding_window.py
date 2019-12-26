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
from itertools import accumulate
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        P=[0]
        nums=list(accumulate(nums))
        P=P+nums
        moving_sum=max(P[i+k]-P[i] for i in range(len(nums)-k+1))
        return moving_sum/float(k)

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

# 209. Minimum Size Subarray Sum https://leetcode.com/problems/minimum-size-subarray-sum/
class Solution:
    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        if len(nums)==0:
            return 0
        i=j=sums=0
        mmin=float("inf")
        while j<len(nums):
            sums+=nums[j]
            j+=1
            while sums>=s:
                mmin=min(mmin,j-i)
                sums-=nums[i]
                i+=1
        return 0 if mmin==float("inf") else mmin

# 28. Implement strStr() https://leetcode.com/problems/implement-strstr/
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        if len(haystack)<len(needle):
            return -1
        l1=len(haystack)
        l2=len(needle)
        for i in range(l1-l2+1):
            count=0
            while count<l2 and haystack[i+count]==needle[count]:
                count+=1
            if count==l2:
                return i
        return -1

# 713. Subarray Product Less Than K https://leetcode.com/problems/subarray-product-less-than-k/
# Input: nums = [10, 5, 2, 6], k = 100
# Output: 8
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k<=min(nums):
            return 0
        product=1
        i=0
        ans=0
        for j,num in enumerate(nums):
            product*=num
            while product>=k:
                product/=nums[i]
                i+=1
            ans+=(j-i+1)
        return ans

# 3. Longest Substring Without Repeating Characters https://leetcode.com/problems/longest-substring-without-repeating-characters/
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start=maxlength=0
        usedchar={}
        for i,c in enumerate(s):
            if c in usedchar and start<=usedchar[c]:
                start=usedchar[c]+1
            else:
                maxlength=max(maxlength,i-start+1)
            usedchar[c]=i
        return maxlength

# 76. Minimum Window Substring https://leetcode.com/problems/minimum-window-substring/
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        def isValid(a,b):
            for item in b:
                if item not in a or a[item] < b[item]:
                    return False
            return True
        source = collections.defaultdict(int)
        target = collections.defaultdict(int)
        for e in t:
            target[e] += 1
        ans = ''
        j = 0; n = len(s)
        minLen = float("inf")
        for i in range(n):
            while j < n and (isValid(source, target) == False):
                source[s[j]] += 1
                j += 1
            if isValid(source, target):
                if minLen > j-i+1:
                    minLen = j-i+1
                    ans = s[i:j]
            source[s[i]] -= 1
        return ans