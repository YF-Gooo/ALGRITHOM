# 198. House Robber https://leetcode.com/problems/house-robber/
# 两个状态抢和不抢
class Solution:
    def rob(self, nums: List[int]) -> int:
        n=len(nums)
        dp=[[0]*2 for _ in range(n+1)]
        for i in range(1,n+1):
            dp[i][0]=max(dp[i-1][0],dp[i-1][1])
            dp[i][1]=dp[i-1][0]+nums[i-1]
        return max(dp[-1][0],dp[-1][1])

class Solution:
    def rob(self, nums: List[int]) -> int:
        n=len(nums)
        yes,no=0,0
        for i in nums:
            no,yes=max(yes,no),no+i
        return max(no,yes)

# 213. House Robber II https://leetcode.com/problems/house-robber-ii/
# 房子有环
class Solution:
    def rob(self,nums):
        if len(nums)==0:
            return 0
        if len(nums)==1:
            return nums[0]
        return max(self.helper(nums,0,len(nums)-1),self.helper(nums,1,len(nums)))
    
    def helper(self, nums,s,e):
        yes,no=0,0
        for i in nums[s:e]:
            no,yes=max(yes,no),no+i
        return max(no,yes)

# 337. House Robber III https://leetcode.com/problems/house-robber-iii/
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rob(self, root: TreeNode) -> int:
        return max(self.postorder(root))
    
    def postorder(self,root):
        if root == None:
            # yes,no
            return [0,0]
        left=self.postorder(root.left)
        right=self.postorder(root.right)
        yes=root.val+left[1]+right[1]
        no= max(left[0]+right[0],
                left[1]+right[0],
                left[0]+right[1],
                left[1]+right[1]
               )
        return [yes,no]
        
# 分割整数（343）：https://leetcode.com/problems/integer-break/description/
# Input: 10
# Output: 36
# Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
# dfs法+memory，超时了
# dp:
class Solution:
    def integerBreak(self, n: int) -> int:
        f=[0 for i in range(n+1)]
        f[2]=1
        for i in range(3,n+1):
            for j in range(1,i//2+1):
                f[i]=max(f[i],max(f[j],j)*max(f[i-j],i-j))
        return f[n]

# 746. Min Cost Climbing Stairs https://leetcode.com/problems/min-cost-climbing-stairs/
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n=len(cost)
        dp=[0]*(n+1)
        for i in range(2,n+1):
            dp[i]=min(dp[i-2]+cost[i-2],dp[i-1]+cost[i-1])
        return dp[n]

# 91. Decode Ways https://leetcode.com/problems/decode-ways/
# 'A' -> 1
# 'B' -> 2
# ...
# 'Z' -> 26
# Input: "12"
# Output: 2
# Explanation: It could be decoded as "AB" (1 2) or "L" (12).
class Solution:
    def numDecodings(self, s: str) -> int:
        if s=="" or s[0]=="0":
            return 0
        dp=[1,1]
        for i in range(2,len(s)+1):
            result=0
            if 10 <=int(s[i-2:i])<26:
                result=dp[i-2]
            if s[i-1]!="0":
                result+=dp[i-1]
            dp.append(result)
        return dp[len(s)]
        
# 96. Unique Binary Search Trees https://leetcode.com/problems/unique-binary-search-trees/
# Input: 3
# Output: 5
# Explanation:
# Given n = 3, there are a total of 5 unique BST's:

#    1         3     3      2      1
#     \       /     /      / \      \
#      3     2     1      1   3      2
#     /     /       \                 \
#    2     1         2                 3

#这问题是是个卡特兰数
class Solution:
    def numTrees(self, n: int) -> int:
        dp=[0]*(n+1)
        dp[0]=dp[1]=1
        for i in range(2,n+1):
            # i是节点数
            for left in range(0,i):
                # left是分割线
                dp[i]+=dp[left]*dp[i-1-left]
        return dp[n]


# 股票
# 121. Best Time to Buy and Sell Stock https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
# 只能买卖一次
# Input: [7,1,5,3,6,4]
# Output: 5
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        mmin=float("inf")
        mmax=0
        for p in prices:
            mmin=min(p,mmin)
            mmax=max(p-mmin,mmax)
        return mmax

# 122. Best Time to Buy and Sell Stock II https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
# 可以买卖无数次
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res=0
        for i in range(1,len(prices)):
            price_delta=prices[i]-prices[i-1]
            if price_delta>0:
                res+=price_delta
        return res

# 714. Best Time to Buy and Sell Stock with Transaction Fee https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/
# 有手续费
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        cash,hold=0,-prices[0]
        for i in range(1,len(prices)):
            cash,hold=max(cash,hold+prices[i]-fee),max(hold,cash-prices[i])
        return cash


# 188. Best Time to Buy and Sell Stock IV https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/
# 可以k次交易
# 如果一笔交易已经买入，但是并没有卖出，也会被记作一笔交易
# hold[i][k]和cash[i]][k]代表了在时间到达i时，可以进行k笔交易！！！！注意是可以进行k笔交易，不是强制k笔交易
# hold[1][3]=hold[1][2]=hold[1][1],因为在时间1时，你最多只能交易1笔,就算你有交易三笔的次数条件，但也只能交易一笔
class Solution:
    def maxProfit(self, k, prices) :
        if not k or not prices: 
            return 0
        if k > len(prices) >> 1: 
            return sum(prices[i+1]-prices[i] for i in range(len(prices)-1) if prices[i+1]>prices[i])
        l=len(prices)
        hold = [[-float("inf") for _ in range(k+1)] for _ in range(l+1)]
        cash = [[0 for _ in range(k+1)] for _ in range(l+1)]
        print(hold[0])
        for i in range(1,l+1):
            for j in range(1, k+1):
                cash[i][j] = max(cash[i-1][j], hold[i-1][j]+prices[i-1])
                hold[i][j] = max(hold[i-1][j], cash[i-1][j-1]-prices[i-1])
        return cash[-1][k]



# 123. Best Time to Buy and Sell Stock III https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
# 可以买卖2次
# 令k=2即可
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not k or not prices: 
            return 0
        if k > len(prices) >> 1: 
            return sum(prices[i+1]-prices[i] for i in range(len(prices)-1) if prices[i+1]>prices[i])
        l=len(prices)
        hold = [[-float("inf") for _ in range(k+1)] for _ in range(l+1)]
        cash = [[0 for _ in range(k+1)] for _ in range(l+1)]
        for i in range(1,l+1):
            for j in range(1, k+1):
                cash[i][j] = max(cash[i-1][j], hold[i-1][j]+prices[i-1])
                hold[i][j] = max(hold[i-1][j], cash[i-1][j-1]-prices[i-1])
        return cash[-1][k]
        
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if not k or not prices: 
            return 0
        if k > len(prices) >> 1: 
            return sum(prices[i+1]-prices[i] for i in range(len(prices)-1) if prices[i+1]>prices[i])
        hold, cash = [float('-inf')] * (k + 1), [0] * (k + 1)
        for price in prices:
            for j in range(1, k+1):
                hold[j] = max(hold[j], cash[j-1]-price)  # hold->hold, sold->hold
                cash[j] = max(cash[j], hold[j]+price)  # sold->sold, hold->sold
        return cash[k]



# 309. Best Time to Buy and Sell Stock with Cooldown https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l=len(prices)
        if l<2:
            return 0
        hold, cash = [float('-inf')]*(l+1), [0]*(l+1) # (l+1)*(k+1)矩阵
        for i in range(1,l+1):
                cash[i] = max(cash[i-1], hold[i-1]+prices[i-1])
                hold[i] = max(hold[i-1], cash[i-2]-prices[i-1])
        #cash1:0,hold1:-price[0]
        #cash2:0,hold2:max(-price[0],-price[1])
        #cash3:max(0,prices+hold2),hold3:max(hold2,cash1-price[2])
        return cash[l]

# 62. Unique Paths https://leetcode.com/problems/unique-paths/
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        aux=[[1 for x in range(n)] for x in range(m)]
        for i in range(1,m):
            for j in range(1,n):
                aux[i][j]=aux[i][j-1]+aux[i-1][j]
        return aux[-1][-1]

#滚动更新
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        aux=[1 for x in range(n)]
        for i in range(1,m):
            for j in range(1,n):
                aux[j]=aux[j-1]+aux[j]
        return aux[-1]

# 63. Unique Paths II  https://leetcode.com/problems/unique-paths-ii/
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        M,N=len(obstacleGrid),len(obstacleGrid[0])
        dp = [1]+(N-1)*[0]
        for m in range(M):
            for n in range(N):
                if obstacleGrid[m][n]==1:
                    dp[n]=0
                elif n>0:
                    dp[n]+=dp[n-1]
        return dp[N-1]

# Moving on Checkerboard
# We are given a grid of squares or a checkerboard with (n) rows and (n) columns. There is a profit we can get by moving to some square in the checkerboard. 
# Our goal is to find the most profitable way from some square in the first row to some square in the last row. We can always move to the next square on the next row using one of three ways:
# Go to the square on the next row on the previous column (UP then LEFT)
# Go to the square on the next row on the same column (UP)
# Go to the square on the next row on the next column (UP then RIGHT)
def movingBoard(board):
    result = board
    m = len(board)
    n = len(board[0])
    for i in range(1, m):
        for j in range (0, n):
            result[i][j] = max(0 if j == 0 else result[i-1][j-1], \
                               result[i-1][j], \
                               0 if j == n-1 else result[i-1][j+1] ) \
                            + board[i][j]
    return max(result[-1])

# 221. Maximal Square https://leetcode.com/problems/maximal-square/
# Input: 
# 1 0 1 0 0
# 1 0 1 1 1
# 1 1 1 1 1
# 1 0 0 1 0
# Output: 4
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        M,N=len(matrix),len(matrix[0])
        dp=[[0 for _ in range(N)] for _ in range(M)]
        res=0
        for m in range(M):
            for n in range(N):
                if matrix[m][n]=="0":
                    dp[m][n]==0
                else:
                    dp[m][n]=int(min(dp[m-1][n],dp[m][n-1],dp[m-1][n-1]))+int(matrix[m][n]) if (m>0 and n>0) else int(matrix[m][n])
                    res= max(res,dp[m][n])
        return res**2


# 0-1背包问题
val=[5,3,4]
wt=[3,2,1]
W=5
n=len(val)
def knapsack(W,wt,val,n):
    K=[[0 for x in range(W+1)] for x in range(n+1)]
    for i in range(1,n+1):
        for w in range(1,W+1):
            if wt[i-1]<=w:
                # 若i-1无法行成w,但是可以形成w-wt[i-1]
                # val[i-1]+K[i-1][w-wt[i-1]]
                K[i][w] = max(val[i-1]+K[i-1][w-wt[i-1]],K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
    return K[n][W]


# 0-1 背包（416）：https://leetcode.com/problems/partition-equal-subset-sum/description/
# Input: [1, 5, 11, 5]

# Output: true

# Explanation: The array can be partitioned as [1, 5, 5] and [11].
# 重点复习
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s=sum(nums)
        if s%2==1:
            return False
        s=s//2
        n=len(nums)
        dp=[[False for _ in range(s+1)] for _ in range(n+1)]
        dp[0][0]=True
        for i in range(1,n+1):
            dp[i][0]=True
        for j in range(1,s+1):
            dp[0][j]=False
        for i in range(1,n+1):
            for j in range(1,s+1):
                dp[i][j]=dp[i-1][j] #如果前i-1个数字已经有办法构成j，那我们就没必要用第i个数字了
                if j >=nums[i-1]:
                    dp[i][j]=dp[i][j] or dp[i-1][j-nums[i-1]] #如果钱i-1个数字无法构成j，那如果我们加入第i个数字并且dp[i-1,j-nums[i-1]]为True说明dp[i][j]也为true
        return dp[n][s]

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s=sum(nums)
        if s%2==1:
            return False
        s=s//2
        n=len(nums)
        dp=[False for _ in range(s+1)]
        dp[0]=True
        for num in nums:
            for j in reversed(range(1,s+1)):
                if j >=num:
                    dp[j]=dp[j] or dp[j-num]
        return dp[s]
        
# 1143. Longest Common Subsequence https://leetcode.com/problems/longest-common-subsequence/
# Input: text1 = "abcde", text2 = "ace" 
# Output: 3  
# Explanation: The longest common subsequence is "ace" and its length is 3.
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m=len(text1)
        n=len(text2)
        matrix=[[0 for k in range(n+1)] for l in range(m+1)]
        result=0
        for i in range(m+1):
            for j in range(n+1):
                if (i==0 or j==0):
                    matrix[i][j]=0
                elif (text1[i-1]==text2[j-1]):
                    matrix[i][j]=matrix[i-1][j-1]+1
                    result=max(result,matrix[i][j])
                else:
                    matrix[i][j]=max(matrix[i-1][j],matrix[i][j-1])
        return result


# Max Sum of subarray:
# https://leetcode-cn.com/problems/maximum-subarray/
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        presum=0
        res=-float(inf)
        for n in nums:
            if presum<0:
                presum=n
            else:
                presum=n+presum
            res=max(presum,res)
        return res

# 最长递增子序列（300）：https://leetcode.com/problems/longest-increasing-subsequence/description/
# 可以不连续
# O(n^2)解法
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0 
        dp, largest = [1] * len(nums), 1
        for j in range(1,len(nums)):
            for i in range(0, j):
                if nums[i] < nums[j] and dp[i]+1 > dp[j]:
                    dp[j] = dp[i] + 1
                    if dp[j] > largest:
                        largest = dp[j]
        return largest



# O(nlogn)解法
import bisect
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        incr = [nums[0]]
        for i in range(1, len(nums)):
            if nums[i] > incr[-1]:
                incr.append(nums[i])
            else:
                j = bisect.bisect_left(incr, nums[i])
                incr[j] = nums[i]
        return len(incr)

# 152. Maximum Product Subarray  https://leetcode.com/problems/maximum-product-subarray/
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n=len(nums)+1
        dp = [[0 for _ in range(2)] for _ in range(n)]
        for i in range(0,len(nums)):
            if i==0:
                dp[0][0],dp[0][1],res=nums[0],nums[0],nums[0]
            else:
                dp[i][0]=max(dp[i-1][0]*nums[i],dp[i-1][1]*nums[i],nums[i])
                dp[i][1]=min(dp[i-1][0]*nums[i],dp[i-1][1]*nums[i],nums[i])
                res=max(res,dp[i][0])
        return res

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n=len(nums)
        curmin,curmax,res=nums[0],nums[0],nums[0]
        for i in range(1,n):
            curmax,curmin=max(curmax*nums[i],curmin*nums[i],nums[i]),min(curmax*nums[i],curmin*nums[i],nums[i])
            res=max(res,curmax)
        return res
            


# 字符串编辑（583）：https://leetcode.com/problems/delete-operation-for-two-strings/description/
class Solution(object):
    def minDistance(self, word1, word2):
            """
            :type word1: str
            :type word2: str
            :rtype: int
            """
            dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1) ]
            for i in range(len(word2)+1):
                dp[0][i]=i
            for i in range(len(word1)+1):
                dp[i][0] =i

            for i in range(1,len(word1)+1):
                for j in range(1,len(word2)+1):
                    if word1[i-1]==word2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1])+1

            return dp[-1][-1]

# 72. Edit Distance https://leetcode.com/problems/edit-distance/
class Solution(object):
    def minDistance(self, word1, word2):
            """
            :type word1: str
            :type word2: str
            :rtype: int
            """
            dp = [[0]*(len(word2)+1) for _ in range(len(word1)+1) ]
            for i in range(len(word2)+1):
                dp[0][i]=i
            for i in range(len(word1)+1):
                dp[i][0] =i

            for i in range(1,len(word1)+1):
                for j in range(1,len(word2)+1):
                    if word1[i-1]==word2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1],dp[i-1][j-1])+1
            return dp[-1][-1]

# 120. Triangle https://leetcode.com/problems/triangle/
# [
#      [2],
#     [3,4],
#    [6,5,7],
#   [4,1,8,3]
# ]
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        n=len(triangle)
        res=triangle[-1]
        for i in range(n-2,-1,-1):
            for j in range(len(triangle[i])):
                res[j]=min(res[j],res[j+1])+triangle[i][j]
        return res[0]

# 322. Coin Change https://leetcode.com/problems/coin-change/
# Input: coins = [1, 2, 5], amount = 11
# Output: 3 
# Explanation: 11 = 5 + 5 + 1
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        Max=float("inf")
        dp=[Max]*(amount+1)
        dp[0]=0
        for i in range(1,amount+1):
            for c in coins:
                if c<=i:
                    dp[i]=min(dp[i],dp[i-c]+1)
        return dp[amount] if dp[amount]!=Max else -1

# 55. Jump Game https://leetcode.com/problems/jump-game/
# Input: [2,3,1,1,4]
# Output: true
# Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach, n = 0, len(nums)
        for i, x in enumerate(nums):
            if max_reach < i: 
                return False
            if max_reach >= n - 1: 
                return True
            max_reach = max(max_reach, i + x)

# 5. 最长回文子串
# https://leetcode-cn.com/problems/longest-palindromic-substring/
# dp
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = ""
        # 枚举子串的长度 l+1
        for l in range(n):
            # 枚举子串的起始位置 i，这样可以通过 j=i+l 得到子串的结束位置
            for i in range(n):
                j = i + l
                if j >= len(s):
                    break
                if l == 0:
                    dp[i][j] = True
                elif l == 1:
                    dp[i][j] = (s[i] == s[j])
                else:
                    dp[i][j] = (dp[i + 1][j - 1] and s[i] == s[j])
                if dp[i][j] and l + 1 > len(ans):
                    ans = s[i:j+1]
        return ans
# 中心扩散
class Solution:
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    def longestPalindrome(self, s: str) -> str:
        start, end = 0, 0
        for i in range(len(s)):
            left1, right1 = self.expandAroundCenter(s, i, i)
            left2, right2 = self.expandAroundCenter(s, i, i + 1)
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start: end + 1]