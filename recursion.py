# print ruler
# 1
# 121
# 1213121
# 121312141213121
def ruler(n):
    if n==0:
        return
    if n==1:
        return "1"
    t=ruler(n-1)
    return t+" "+str(n)+" "+t

def ruler2(n):
    result=""
    for i in range(1,n+1):
        result = result +str(i)+" "+result
    return result

# a to b
# Given two intergers a<=b, transform a into b by add 1 and multiply 2
# a=5
# b=23
# output: 23=((5*2+1)*2+1)
def intSeq(a,b):
    if (a==b):
        return str(a)
    if (b%2==1):
        return "(" +intSeq(a,b-1)+"+1)"
    if (b<a*2):
        return "(" +intSeq(a,b-1)+"+1)"
    return intSeq(a,b/2)+"*2"


# Tower of Hanoi
# 假设我们已经有
def hanoi(n,start,end,by):
    if n ==1:
        print("Move from "+start+" to "+end)
    else:
        hanoi(n-1,start,by,end)
        hanoi(1,start,end,by)
        hanoi(n-1,by,end,start)

# 78. Subsets https://leetcode.com/problems/subsets/
# Input: nums = [1,2,3]
# Output:
# [
#   [3],
#   [1],
#   [2],
#   [1,2,3],
#   [1,3],
#   [2,3],
#   [1,2],
#   []
# ]
class Solution:
    def subsets(self, nums):
        if not nums:
            return []
        self.res=[]
        self.helper(nums,[])
        return self.res
    
    def helper(self,nums,l):
        if not len(nums):
            self.res.append(l)
            return
        self.helper(nums[1:],l+[nums[0]])
        self.helper(nums[1:],l+[])


# 784. Letter Case Permutation https://leetcode.com/problems/letter-case-permutation/
# Examples:
# Input: S = "a1b2"
# Output: ["a1b2", "a1B2", "A1b2", "A1B2"]

# Input: S = "3z4"
# Output: ["3z4", "3Z4"]

# Input: S = "12345"
# Output: ["12345"]
class Solution:
    def letterCasePermutation(self, S: str) -> List[str]:
        if not S:
            return []
        self.res=[]
        self.helper(S,"")
        return self.res
        
    def helper(self,s,l):
        print(l)
        if not s:
            self.res.append(l)
            return
        if s[0].isdigit():
            self.helper(s[1:],l+s[0])
        else:
            self.helper(s[1:],l+s[0].lower())
            self.helper(s[1:],l+s[0].upper())

# 39. Combination Sum https://leetcode.com/problems/combination-sum/
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res=[]
        self.helper(candidates,target,[],0)
        return self.res
    def helper(self,candidates,target,nums,start):
        if target<0:
            return
        if target==0:
            self.res.append(nums)
            return 
        for i in range(start,len(candidates)):
            c=candidates[i]
            self.helper(candidates,target-c,nums+[c],i)

#40. Combination Sum II https://leetcode.com/problems/combination-sum-ii/
# Input: candidates = [10,1,2,7,6,1,5], target = 8,
# A solution set is:
# [
#   [1, 7],
#   [1, 2, 5],
#   [2, 6],
#   [1, 1, 6]
# ]
# 先排序
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        self.res=[]
        candidates=sorted(candidates)
        self.helper(candidates,target,[],0)
        return self.res
    def helper(self,candidates,target,nums,start):
        if target<0:
            return
        if target==0:
            self.res.append(nums)
            return 
        for i in range(start,len(candidates)):
            if (i>start and candidates[i]==candidates[i-1]):continue
            c=candidates[i]
            self.helper(candidates,target-c,nums+[c],i+1)

# https://leetcode.com/problems/generate-parentheses/
# For example, given n = 3, a solution set is:

# [
#   "((()))",
#   "(()())",
#   "(())()",
#   "()(())",
#   "()()()"
# ]
class Solution:
    def generateParenthesis(self, n: int) :
        if not n:
            return []
        self.res=[]
        self.dfs(0,0,n,"")
        return self.res
        
    def dfs(self,left,right,n,s):
        if left==n and right==n:
            self.res.append(s)
            return
        if left<n:
            self.dfs(left+1,right,n,s+"(")
        if left>right and right<n:
            self.dfs(left,right+1,n,s+")")

#   \\
# 51. N-Queens https://leetcode.com/problems/n-queens/
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if not n:
            return []
        self.cols=set()
        self.pie=set()
        self.na=set()
        self.result=[]
        self.dfs(0,n,[])
        return self._generate_result(n)
        
    def dfs(self,raw,n,state):

        if raw>=n:
            self.result.append(state)
            return 
        for col in range(n):
            if col in self.cols or raw+col in self.pie or raw-col in self.na:
                continue
            self.cols.add(col)
            self.pie.add(raw+col)
            self.na.add(raw-col)
            self.dfs(raw+1,n,state+[col])
            self.cols.remove(col)
            self.pie.remove(raw+col)
            self.na.remove(raw-col)
            
                
    def _generate_result(self,n):
        board = []
        for res in self.result:
            for i in res:
                board.append("."*i+"Q"+"."*(n-i-1))
        return [board[i:i+n] for i in range(0,len(board),n)]
    

# 37. Sudoku Solver https://leetcode.com/problems/sudoku-solver/
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        self.res=[]
        self.solver(board)
        print(board)
        
    def solver(self,board):
        if not board or len(board)==0:
            return
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j]==".":
                    for c in range(1,10):
                        c = str(c)
                        if self._isvaild(board,i,j,c):
                            board[i][j]=c
                            if self.solver(board):
                                return True
                            else:
                                board[i][j]="."
                    return False
        return True
    
    def _isvaild(self,board,i,j,c):
        for x in range(0,9):
            if board[x][j]!="." and board[x][j]==c:
                return False
            if board[i][x]!="." and board[i][x]==c:
                return False
            if board[3*(i//3)+x//3][3*(j//3)+x%3]!="." and board[3*(i//3)+x//3][3*(j//3)+x%3]==c:
                return False
        return True

# 212. Word Search II https://leetcode.com/problems/word-search-ii/
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board:
            return []
        self.trie=Trie()
        for w in words:
            self.trie.insert(w)
        self.res=set()
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board,i,j,"")
        return self.res
    
    def dfs(self,board,i,j,state):
        if i<0 or i>=len(board):
            return
        if j<0 or j>=len(board[0]):
            return
        c=board[i][j]
        if c ==".":
            return 
        if self.trie.search(state+c):
            self.res.add(state+c)
        if self.trie.startsWith(state+c):
            board[i][j]="."
            self.dfs(board,i+1,j,state+c)
            self.dfs(board,i,j+1,state+c)
            self.dfs(board,i-1,j,state+c)
            self.dfs(board,i,j-1,state+c)
            board[i][j]=c 
        return 
            
class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root={}
        self.end_of_words="#"
        

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node=self.root
        for c in word:
            if c not in node:
                node = node.setdefault(c,{})
            else:
                node = node[c]
        node[self.end_of_words]=self.end_of_words
        

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node=self.root
        for c in word:
            if c not in node:
                return False
            else:
                node = node[c]
        if self.end_of_words in node:
            return True
        else:
            return False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node=self.root
        for c in prefix:
            if c not in node:
                return False
            else:
                node = node[c]
        return True