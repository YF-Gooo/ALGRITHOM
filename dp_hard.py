# https://leetcode.com/problems/bitwise-ors-of-subarrays/submissions/
# Input: [1,1,2]
# Output: 3
# Explanation: 
# The possible subarrays are [1], [1], [2], [1, 1], [1, 2], [1, 1, 2].
# These yield the results 1, 1, 2, 1, 3, 3.
# There are 3 unique values, so the answer is 3.
class Solution:
    def subarrayBitwiseORs(self, A: List[int]) -> int:
        cur = set()
        ans = set()
        for a in A:
            cur = {a|b for b in cur} | {a}
            ans |= cur
        return len(ans)