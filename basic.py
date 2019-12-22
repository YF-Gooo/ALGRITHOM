class Node:
    def __init__ (self, value = None, next = None):
        self.value = value
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = Node()
        self.tail = None
        self.length = 0

    def get_first(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        return self.head.next
        
    def get_last(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        node = self.head
        while node.next != None:
            node = node.next
        return node
    
    def get(self, index):
        if (index < 0 or index >= self.length):
            raise Outbound( 'index is out of bound' );
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        node = self.head
        for i in range(index):
            node = node.next
        return node.next
                
    def add_first(self, value):
        node = Node(value, None)
        node.next = self.head.next
        self.head.next = node
        self.length += 1   
        
    def add_last(self, value):
        new_node = Node(value)
        node = self.head
        while node.next != None:
            node = node.next
        node.next = new_node
        self.length += 1

    def add(self, index, value):
        if (index < 0 or index > self.length):
            raise Outbound( 'index is out of bound' )
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        new_node = Node(value)
        node = self.head
        for i in range(index):
            node = node.next
        new_node.next = node.next
        node.next = new_node
        self.length += 1     
        
    def remove_first(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        value = self.head.next
        self.head.next = self.head.next.next
        self.length -= 1
        return value    
        
    def remove_last(self):
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        node = self.head.next
        prev = self.head
        while node.next != None:
            prev = node
            node = node.next
        prev.next = None
        return node.value

    def remove(self, index):
        if (index < 0 or index >= self.length):
            raise Outbound( 'index is out of bound' )
        if not self.head.next:
            raise Empty( 'LinkedList is empty' )
        node = self.head
        for i in range(index):
            node = node.next
        result = node.next
        node.next = node.next.next
        self.length -= 1     
        return result      
        

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

# 反转链表
def reverse_linklist(head):
    pre,cur=None,head
    while cur:
        cur.next,pre,cur= pre, cur, cur.next
    return pre

# 成对反转
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

# 链表有环
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head==None:
            return False
        fast=slow=head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
        return False

# 是否为有效括号组合
def invalid(s):
    stack=[]
    paren_map={")":"(","]":"[","}":"{"}
    for c in s:
        if c not in paren_map:
            stack.append(c)
        elif not stack or paren_map[c]!=stack.pop():
            return False
    return not stack
    
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

# 荷兰国旗问题（题号：75）：https://leetcode.com/problems/sort-colors/description/
class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        i,j=0,len(nums)-1
        n=0
        while(n<=j):
            if nums[n]==0:
                nums[n],nums[i]=nums[i],nums[n]
                i=i+1
                n+=1
            if nums[n]==1:
                n+=1
                continue
            if nums[n]==2:
                nums[n],nums[j]=nums[j],nums[n]
                j=j-1
                continue

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

# 快速排序
def partition(arr,low,high): 
    i = low-1         # 最小元素索引
    pivot = arr[high]     
    for j in range(low , high): 
        # 当前元素小于或等于 pivot 
        if  arr[j] <= pivot: 
            i = i+1 
            arr[i],arr[j] = arr[j],arr[i] 
    arr[i+1],arr[high] = arr[high],arr[i+1] 
    return i+1

def quickSort(arr,low,high): 
    if low < high: 
        pi = partition(arr,low,high) 
        quickSort(arr, low, pi-1) 
        quickSort(arr, pi+1, high)

# 找出第k大的数字 https://leetcode.com/problems/kth-largest-element-in-an-array/description/
# 快速选择
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

#最小公共祖先
# 236. Lowest Common Ancestor of a Binary Tree https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return root
        if root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left,p,q)
        right = self.lowestCommonAncestor(root.right,p,q)
        if left==None:
            return right
        elif right == None:
            return left
        else:
            return root

# 3sum
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

# 378. Kth Smallest Element in a Sorted Matrix //leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/
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

# 66. Plus One https://leetcode.com/problems/plus-one/
# Input: [1,2,3]
# Output: [1,2,4]
# Explanation: The array represents the integer 123.
# 妙哉
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        if len(digits)==0:
            return False
        addCarry=1
        for i in range(len(digits)-1,-1,-1):
            digits[i]+=addCarry
            if digits[i]==10:
                digits[i]=0
                if i==0:
                    digits.insert(0,1)
            else:
                break
        return digits

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