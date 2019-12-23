# 分治（题号：241）：https://leetcode.com/problems/different-ways-to-add-parentheses/description/
class Solution:
    def diffWaysToCompute(self, input: str) -> List[int]:
        if input.isdigit():
            return [int(input)]
        
        ans = []
        for i, c in enumerate(input):
            if not c.isdigit():
                l = self.diffWaysToCompute(input[0:i])
                r = self.diffWaysToCompute(input[i+1:])
                for l1 in l:
                    for r1 in r:
                        if c == '+':
                            ans.append(l1 + r1)
                        elif c == '-':
                            ans.append(l1 - r1)
                        elif c == '*':
                            ans.append(l1 * r1)
        return ans
        
# hard 可以跳过 493. Reverse Pairs https://leetcode.com/problems/reverse-pairs/
# 用了归并排序，可以说是相当巧妙了
class Solution:

    def _merge( self, left_part, right_part ):
        merge_list = sorted( left_part + right_part )
        return merge_list

    def merge_sort_with_rev_pair_count( self, nums: List[int]) -> (List[int], int):
        if len( nums) <= 1:
            return nums, 0

        else:
            mid = len(nums) // 2
            left_part, left_rev_pair_count = self.merge_sort_with_rev_pair_count( nums[ : mid] )
            right_part, right_rev_pair_count = self.merge_sort_with_rev_pair_count( nums[ mid: ] )
            rev_pair_count = left_rev_pair_count + right_rev_pair_count
            cur_rev_pair_count = 0
            for r in right_part:
                cur_rev_pair_count = len( left_part ) - bisect.bisect( left_part, 2*r)
                if cur_rev_pair_count == 0:
                    break
                rev_pair_count += cur_rev_pair_count
            merged_list = self._merge( left_part, right_part)
        return merged_list, rev_pair_count
    def is_strictly_increasing_by_one( self, nums :List[int]) -> bool:
        result = all( ( i < j and (j-1) == 1 ) for i, j in zip(nums, nums[1:]) ) 
        return result
    def reversePairs( self, nums: List[int]) -> int:
        if self.is_strictly_increasing_by_one( nums ) or len(nums) <= 1:
            return 0
        sorted_list, count_of_reverse_pair = self.merge_sort_with_rev_pair_count( nums )
        return count_of_reverse_pair

