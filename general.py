
class ListNode:
    def __init__(self,value):
        self.value=value
        self.next=None

def reverse_linklist(head):
    pre,cur=None,head
    while cur:
        cur.next,pre,cur= pre, cur, cur.next
    return pre

def reverse_linklist_pairs(head):
    pre=ListNode(0)
    pre.next=head
    dummy=pre
    while pre and pre.next:
        a = pre.next
        b = a.next
        pre.next, b.next , a.next= b, a ,b.next
        pre = a 
    return dummy

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
                        
# 第一周
# 归并排序
def MergerSort(lists):
    if len(lists)<=1:
        return lists
    num=int(len(lists)/2)
    left=MergerSort(lists[:num])
    right=MergerSort(lists[num:])
    return Merge(left,right)

def Merge(left,right):
    r,l=0,0
    result=[]
    while l<len(left) and r<len(right):
        if left[l]<=right[r]:
            result.append(left[l])
            l+=1
        else:
            result.append(right[r])
            r+=1
    result+=list(left[l:])
    result+=list(right[r:])
    return result


    


# 双指针(题号：167)：https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/description/
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
    
# 排序：找出第k大的数字
# 快速选择、堆排序（题号：215）：https://leetcode.com/problems/kth-largest-element-in-an-array/description/
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

# 桶排序（题号：347）：https://leetcode.com/problems/top-k-frequent-elements/description/
from collections import defaultdict
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        hashmap=defaultdict(int)
        for n in nums:  
            hashmap[n]+=1
        return [x[0] for x in sorted(hashmap.items(),key=lambda x:x[1],reverse=True)[:k]]


# 荷兰国旗问题（题号：75）：https://leetcode.com/problems/sort-colors/description/
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

# 贪心（题号：455）：https://leetcode.com/problems/assign-cookies/description/
class Solution(object):
    def findContentChildren(self, g, s):
        """
        :type g: List[int]
        :type s: List[int]
        :rtype: int
        """
        g.sort()
        s.sort()
        count=0
        for i in reversed(range(len(g))):
            if len(s)==0:return count
            if g[i]>s[-1]:continue
            s.pop()
            count+=1
        return count

# 二分（题号：69）：https://leetcode.com/problems/sqrtx/description/
# 传统二分搜索
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

# sqrt
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

# 链表（题号：160）：https://leetcode.com/problems/intersection-of-two-linked-lists/description/
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if not headA or not headB:
            return None
        a,b=headA,headB
        done=0
        while a!=b and done<=2:
            a=a.next
            b=b.next
            if a == None:
                a=headB
                done+=1
            if b == None:
                b=headA
                done+=1
        if done>2:
            return None
        return a

# 哈希表（题号：1）：https://leetcode.com/problems/two-sum/description/
from collections import defaultdict
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        hashmap=defaultdict(int)
        for i,n in enumerate(nums):
            if n in hashmap:
                return [hashmap[n],i]
            hashmap[target-n]=i
        return []

# 字符串（题号：242）：https://leetcode.com/problems/valid-anagram/description/
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s)!=len(t):
            return False
        a,b=[0]*26,[0]*26
        for i in range(len(t)):
            a[ord(s[i])-ord("a")]+=1
            b[ord(t[i])-ord("a")]+=1
        return a==b 

#栈和队列（题号：232）：https://leetcode.com/problems/implement-queue-using-stacks/description/
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1=[]
        self.s2=[]

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: None
        """
        self.s1.append(x)
        
    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if len(self.s2)!=0:
            return self.s2.pop()
        while len(self.s1)!=0:
            self.s2.append(self.s1.pop())
        return self.s2.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if len(self.s2)!=0:
            return self.s2[-1]
        while len(self.s1)!=0:
            self.s2.append(self.s1.pop()) 
        return self.s2[-1]

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        if len(self.s1)==0 and len(self.s2)==0:
            return True

# 字符串：https://leetcode.com/problems/longest-palindrome/description/
# Input:
# "abccccdd"
# Output:
# 7
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

# 378. Kth Smallest Element in a Sorted Matrix //leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/
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

# 位运算：https://leetcode.com/problems/single-number-iii/description/
# Input:  [1,2,1,3,2,5]
# Output: [3,5]
# 首先先用异或得到两个数字的异或值
# 比如出现一次的两个数字是3(001)，5(110),r异或值(101)
# r & ~(r-1) 用来截取最后的最开始的一位不一样的数字，比如 r=101 ,r & ~(r-1)=1,可以作为mask分割两个数字
# r & (r-1) 消除最后一个1,比如 r=101 ,r & (r-1)=100,
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        r = 0
        for n in nums:
            r ^= n
        # 1 to the right most
        mask = r & ~(r-1)
        res = [0, 0]
        for n in nums:
            if n & mask:
                res[0] ^= n
            else:
                res[1] ^= n
        return res


# 进制转换：https://leetcode.com/problems/base-7/description/
class Solution:
    def convertToBase7(self, num: int) -> str:
          ans = ""
          if num == 0:
              return '0'
          x = abs(num)
          while(x > 0):
              ans += str(x%7)
              x //= 7
          if (num > 0):
              return ans[::-1]
          else:
              return '-' + ans[::-1]

# 相遇问题： https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/description/
# Input:
# [1,2,3]
# Output:
# 2
# Explanation:
# Only two moves are needed (remember each move increments or decrements one element):
# [1,2,3]  =>  [2,2,3]  =>  [2,2,2]
def minMoves2(self, nums):
        nums.sort()
        res=0
        mid=nums[len(nums)//2]
        for i in range(len(nums)):
            res += abs(nums[i]-mid)
        return res

# 多数投票问题：https://leetcode.com/problems/majority-element/description/
# 摩尔投票法
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        t={}
        pre=0
        for n in nums:
            if t=={}:
                t[n]=1
                pre=n
            elif pre!=n:
                t[pre]-=1
                if t[pre]==0:
                    t={}
            else:
                t[pre]+=1
        return pre

# 递归（110）：https://leetcode.com/problems/balanced-binary-tree/description/
class Solution:
    def __init__(self):
        self.res=True
        
    def isBalanced(self, root: TreeNode) -> bool:
        self.getdepth(root)
        return self.res
            
    def getdepth(self,root):
        if not self.res:
            return 0
        if not root:
            return 0
        l=self.getdepth(root.left)
        r=self.getdepth(root.right)
        if abs(l-r)>1:
            self.res=False
        return max(l,r)+1

# 层次遍历（513）：https://leetcode.com/problems/find-bottom-left-tree-value/description/
# Input:

#         1
#        / \
#       2   3
#      /   / \
#     4   5   6
#        /
#       7

# Output:
# 7
class Solution:
    def __init__(self):
        self.res=[-1,None]

    def findBottomLeftValue(self, root: TreeNode) -> int:
        self.inorder(root,0,1)
        return self.res[1].val
        
    def inorder(self,root,depth,lr):
        if not root and lr:
            return 1
        if not root:
            return 0
        record=self.inorder(root.left,depth+1,1)
        if record:
            if depth>self.res[0]:
                self.res=[depth,root]
        self.inorder(root.right,depth+1,0)

# 前中后序遍历（144）：https://leetcode.com/problems/binary-tree-preorder-traversal/description/
# 这道题目一开始我和bfs按层遍历搞混了，这题要用栈！！！
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        s=[root]
        r=[]
        while(s):
            tempr=s.pop()
            r.append(tempr.val)
            if tempr.right:
                s.append(tempr.right)
            if tempr.left:
                s.append(tempr.left)
            return r
        

# BST（230）：https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/
class Solution:
    def __init__(self):
        self.res=0
        self.count=0
        self.done=0
        
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        self.k=k
        self.inorder(root)
        return self.res
    def inorder(self,root):
        if self.done:
            return 
        if not root:
            return root
        self.inorder(root.left)
        self.count+=1
        if self.count==self.k:
            self.res=root.val
            self.done=1
        self.inorder(root.right)



# 二分图（785）：https://leetcode.com/problems/is-graph-bipartite/description/
# Example 1:
# Input: [[1,3], [0,2], [1,3], [0,2]]
# Output: true
# Explanation: 
# The graph looks like this:
# 0----1
# |    |
# |    |
# 3----2
# 需要重点看看
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        color={}
        stack=[]
        for node in range(len(graph)):
            if node not in color:
                stack.append(node)
                color[node]=1
                while stack:
                    node=stack.pop()
                    for nei in graph[node]:
                        if nei not in color:
                            stack.append(nei)
                            color[nei]=-color[node]
                        else:
                            if color[nei]==color[node]:
                                return False
        return True

# 拓扑排序（207）：https://leetcode.com/problems/course-schedule/description/
# Input: 2, [[1,0],[0,1]]
# Output: false
# Explanation: There are a total of 2 courses to take. 
#              To take course 1 you should have finished course 0, and to take course 0 you should
#              also have finished course 1. So it is impossible.
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        self.graph = {x : [] for x in range(numCourses)}
        self.visited = {x : 0 for x in range(numCourses)}
        for x, y in prerequisites:
            self.graph[x] += [y]
        for i in range(numCourses):
            if not self.dfs(i):
                return False
        return True
                
    def dfs(self,course):
        if self.visited[course] == -1:
            return False
        if self.visited[course] == 1:
            return True
        self.visited[course] = -1
        for i in self.graph[course]:
            if not self.dfs(i):
                return False
        self.visited[course] = 1
        return True



# BFS（279）：https://leetcode.com/problems/perfect-squares/description/
# Input: n = 12
# Output: 3 
# Explanation: 12 = 4 + 4 + 4.
class Solution:
    def numSquares(self, n: int) -> int:
        max_num = int(n ** 0.5) 
        all_nums = [ i **2 for i in range(1, max_num+1)]
        
        visited = set((n,0))
        queue = [(n,0)]
        
        while queue:
            num, ret = queue.pop(0)
            # for i in all_nums:
            # 先凑大数
            for i in all_nums[::-1]: 
                if i <= num:
                    left = num - i
                    if left == 0:
                        return ret+1
                    if (left, ret+1 ) not in visited:
                        visited.add((left, ret+1 ))
                        queue.append((left, ret+1 ))

# DFS（695）：https://leetcode.com/problems/max-area-of-island/description/
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        self.grid=grid
        self.visited=set()
        self.d=[[1,0],[-1,0],[0,1],[0,-1]]
        self.h=len(grid)
        self.w=len(grid[0])
        self.res=-1
        self.count=0
        for m in range(self.h):
            for n in range(self.w):
                self.dfs(m,n)
                self.res=max(self.res,self.count)       
                self.count=0
        return self.res
    
    def dfs(self,m,n):
        if (m,n) in self.visited:
            return
        if m<0 or n<0 or m>=self.h or n>=self.w:
            return
        self.visited.add((m,n))
        if self.grid[m][n]==0:
            return 
        self.count+=1
        for i in range(4):
            self.dfs(m+self.d[i][0],n+self.d[i][1])
        return 
    
# Backtracking（17）：https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
# Input: "23"
# Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
class Solution:
    def __init__(self):
        self.dic = {'2' : ["a","b","c"],'3' : ["d","e","f"], '4' : ["g","h","i"], '5' : ["j","k","l"],'6' : ["m","n","o"], '7' : ["p","q","r","s"], '8' : ["t","u","v"], '9' : ["w","x","y","z"]}
        self.res=[]
    def letterCombinations(self, digits: str) -> List[str]:
        if digits=="":
            return []
        self.dfs("",digits)
        return self.res 
                    
    def dfs(self,pre,digits):
        if len(digits)==1:
            for s in self.dic[digits[0]]:
                self.res.append(pre+s)
            return 
        for s in self.dic[digits[0]]:
            self.dfs(pre+s,digits[1:])



# 斐波那契数列（70）：https://leetcode.com/problems/climbing-stairs/description/
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        a=1
        b=1
        for _ in range(2,n+1):
            a,b=b,a+b
        return b

# 矩阵路径（64）：https://leetcode.com/problems/minimum-path-sum/description/
# Input:
# [
#   [1,3,1],
#   [1,5,1],
#   [4,2,1]
# ]
# Output: 7
# Explanation: Because the path 1→3→1→1→1 minimizes the sum.
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m,n=len(grid),len(grid[0])
        dp=[[0]*(n+1) for i in range(m+1)]
        for i in range(m+1):
            dp[i][0]=float("inf")
        for j in range(n+1):
            dp[0][j]=float("inf")
        dp[1][1]=grid[0][0]
        for i in range(1,m+1):
            for j in range(1,n+1):
                if i==1 and j==1:
                    continue
                dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i-1][j-1]
        return dp[-1][-1]
        
# 数组区间（303）：https://leetcode.com/problems/range-sum-query-immutable/description/
# Given nums = [-2, 0, 3, -5, 2, -1]
# sumRange(0, 2) -> 1
# sumRange(2, 5) -> -1
# sumRange(0, 5) -> -3
class NumArray:
    def __init__(self, nums: List[int]):
        t=0
        for i,n in enumerate(nums):
            t+=n
            nums[i]=t
        self.nums=nums
    def sumRange(self, i: int, j: int) -> int:
        if i>0:
            return self.nums[j]-self.nums[i-1]
        else:
            return self.nums[j]

# 分割整数（343）：https://leetcode.com/problems/integer-break/description/
# Input: 10
# Output: 36
# Explanation: 10 = 3 + 3 + 4, 3 × 3 × 4 = 36.
# dfs法+memory，超时了
class Solution:
    def integerBreak(self, n: int) -> int:
        self.res=0
        self.mem=set()
        self.dfs(n,1,0)
        return self.res
    
    def dfs(self,left,pre,depth):
        if (left,pre,depth) in self.mem:
            return
        self.mem.add((left,pre,depth))
        if left==0:
            self.res=max(self.res,pre)
        if depth==0:
            for n in range(1,left):           #第一层，数字本身不能算进去



                self.dfs(left-n,pre*n,depth+1)
        else:
            for n in range(1,left+1):
                self.dfs(left-n,pre*n,depth+1)
# dp:
class Solution:
    def integerBreak(self, n: int) -> int:
        f=[0 for i in range(n+1)]
        f[2]=1
        for i in range(3,n+1):
            for j in range(1,i//2+1):
                f[i]=max(f[i],max(f[j],j)*max(f[i-j],i-j))
        return f[n]
        
# 最长递增子序列（300）：https://leetcode.com/problems/longest-increasing-subsequence/description/
# 可以不连续
# O(n^2)解法
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0 
        dp, largest = [1] * len(nums), 1
        for j in range(len(nums)):
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
                dp[i][j]=dp[i-1][j] #如果钱i-1个数字已经有办法构成j，那我们就没必要用第i个数字了
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



# LRUCache https://leetcode.com/problems/lru-cache/submissions/
import collections
class LRUCache:

    def __init__(self, capacity: int):
        self.dic=collections.OrderedDict()
        self.remain = capacity
        

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1
        v= self.dic.pop(key)
        self.dic[key]=v
        return v

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self.dic.pop(key)
        else:
            if self.remain>0:
                self.remain -=1
            else:
                self.dic.popitem(last=False)
        self.dic[key]=value

# Trie（208）：https://leetcode.com/problems/implement-trie-prefix-tree/description/
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