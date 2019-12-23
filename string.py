# 409. Longest Palindrome https://leetcode.com/submissions/detail/284545375/
from collections import defaultdict
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        hashmap=defaultdict(int)
        for ss in s:
            hashmap[ss]+=1
        res=0
        tmp=0
        for _,v in hashmap.items():
            if v %2==0:
                res+=v
            elif v%2 == 1:
                if tmp == 0:
                    tmp=1
                    res+=v
                else:
                    res+=v-1
        return res

# 3. Longest Substring Without Repeating Characters   https://leetcode.com/problems/longest-substring-without-repeating-characters/
# Input: "abcabcbb"
# Output: 3 
# Explanation: The answer is "abc", with the length of 3.   
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
            memo = {}
            left = 0
            current_length = 0
            longest_length = 0
            for i, letter in enumerate(s):
                if letter in memo:
                    longest_length = max(current_length, longest_length)
                    if memo[letter] >= left:
                        left = memo[letter]
                    memo[letter] = i
                    current_length = i - left
                else:
                    current_length += 1
                    memo[letter] = i
            longest_length = max(current_length, longest_length)
            return longest_length

# 392. Is Subsequence https://leetcode.com/problems/is-subsequence/
# Example 1:
# s = "abc", t = "ahbgdc"
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        m=len(s)
        n=len(t)
        i=0
        j=0
        while i<m and j<n:
            if s[i]==t[j]:
                i+=1
            j+=1
        return m==i

# 1143. Longest Common Subsequence https://leetcode.com/problems/longest-common-subsequence/
# 可以不连续
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