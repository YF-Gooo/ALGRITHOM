class Node:
    def __init__(self,val):
        self.val=val
        self.next=None

class Linklist:
    def __init__(self):
        self.head=Node(0)
        self.length=0
    
    def add_first(self,val):
        tmp=self.head.next
        self.head.next=Node(val)
        self.head.next.next = tmp
        self.length+=1
    
    def add_last(self,val):
        node=self.head.next
        while(node.next):
            node=node.next
        node.next=Node(val)
        self.length+=1

    def add(self,index,val):
        if index<0 or index>=length:
        node=self.head
        for i in range(index):
            node=node.next
        new_node=Node(val)
        node.next,new_node.next=new_node,node.next
        self.length+=1
    
    def get(self,index):
        if index<0 or index>=length:
        node=self.head
        for i in range(index):
            node=node.next
        return node.next

    def remove(self,index):
        if index<0 or index>=length:
        node=self.head
        for i in range(index):
            node=node.next
        node.next=node.next.next
        
def  binarysearch(alist,item):
    if len(alist)==0:
        return -1
    left,right=0,len(alist)-1
    while left+1<right:
        mid=left+(right-left)//2
        if alist[mid]==item:
            mid=right
        elif alist[mid]>item:
            right=mid
        else:
            left=mid
    if alist[left]==item:
        return left
    if alist[right]==item:
        return right
    return -1

def mergesort(alist):
    if len(alsit)<=1:
        reurn alist
    mid=len(alist)//2
    left=mergesort(alist[:mid])
    right=mergesort(alist[mid:])
    return merge(left,right)

def merge(left,right):
    i=0
    j=0
    res=[]
    while i<len(left) and j <len(right):
        if left[i]<=right[j]:
            res.append(left[i])
            i+=1
        else:
            res.append(right[j])
            j+=1
    res+=left[i:]
    res+=right[j:]
    return res

        
def quicksort(alist,low,high):
    if low < high:
        pi=partiton(alist,low,high)
        quicksort(alist,low,pi-1)
        quicksort(alist,pi+1,high)

def partiton(alist,low,high):
    i=low-1
    pivot=alist[high]
    for j in range(low,high):
        if alist[j]<=pivot:
            i+=1
            alist[i],alist[j]=alist[j],alist[i]
    alist[i+1],alist[high]=alist[high],alist[i+1]
    return i+1 

def heap_adjust(n,i,array):
    while 2*i<n:
        lchild_index=2*i+1
        max_child_index=lchild_index
        if lchild_index+1<=n and array[lchild_index+1]>array[lchild_index]:
            max_child_index=lchild_index+1
        if array[max_child_index]>array[i]:
            array[i],array[max_child_index]=array[max_child_index],array[i]
            i=max_child_index  
        else:
            break

def max_heap(length,array):
    for i in range(length//2-1,-1,-1):
        heap_adjust(length-1,i,array)
    return array

def sort(length,array):
    length-=1
    while length>0:
        array[0],array[length]=array[length],array[0]
        length-=1
        if length==1 and array[length]>=array[length-1]:
            break
        heap_adjust(length,0,array)
    return array

def reverse_linklist(head):
    pre,cur=None,head
    while cur:
        cur.next,pre,cur,=pre,cur,cur.next
    return pre

def reverse_linklist_pairs(head):
    pre=Node(0)
    dummy=pre
    pre.next=head
    while pre and pre.next:
        a=pre.next
        b=a.next
        pre.next,b.next,a.next=b,a,b.next
        pre=a
    return dummy

def hascycle(head):
    if head==None:
        return False
    fast=slow=head
    while slow and fast and fast.next:
        slow=slow.next
        fast=fast.next.next
        if slow is fast:
            return True
    return False

def hascycle(head):
    if head==None:
        return False
    fast=slow=head
    while slow and fast and fast.next:
        slow=slow.next
        fast=fast.next.next
        if slow is fast:
            fast=head
            break
    if not fast or not fast.next:
        return 
    while slow and fast:
        fast=fast.next
        slow=slow.next
        count+=1 
        if slow is fast:
            return count
    return 0

def invalid(s):
    stack=[]
    paren_map={")":"(","]":"[","}":"{"}
    for ss in s:
        if ss not in paren_map:
            stack.append(ss)
        elif not stack or paren_map[c]!=stack.pop():
            return False
    return len(stack)==0



# sqrt
class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        left=0
        right=x
        mid=left+(right-left)//2
        while left<=right:
            if x//mid==mid:
                return mid
            elif x//mid>mid:
                left=mid+1
            else:
                right=mid-1
        return right
            
# 50. Pow(x, n) https://leetcode.com/problems/powx-n/
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n==0:
            return 1
        if n<=0:
            return 1/self.myPow(x,-n)
        if n&1:
            return self.myPow(x,n-1)*x
        else:
            return self.myPow(x*x,n//2)

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

    def helper(arr,low,high,k):
        if low <high:
            pi=partition(arr,low,high)
            if pi==k-1:
                return arr[pi]
            elif pi>k-1:
                helper(arr,low,pi-1,k)
            else:
                helper(arr,pi+1,high,k)
        
    
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return self.helper(nums,0,len(nums)-1,k)

# 3sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if len(nums)<3:
            return []
        nums.sort()
        n=len(nums)
        res=[]
        for i in range(0,n-2):
            l,r=i+1,n-1
            if i>0 and nums[i]==nums[i-1]:
                continue
            if nums[i]+nums[l]+nums[r]>0:
                r-=1
            elif nums[i]+nums[l]+nums[r]<0:
                l+=1
            else:
                res.append((i,l,r))
                while l<r and nums[l]==nums[l+1]:
                    l+=1
                while l<r and nums[r]==nums[r-1]:
                    r-=1
                l+=1
                r-=1                   
        return res

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
            m=len(word2)
            n=len(word1)
            dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
            for i in range(n):
                dp[0][i]=i
            for j in range(m):
                dp[j][0]=j
            for i in range(1,m+1):
                for j in range(1,n+1):
                    if word1[i]==word2[j]:
                        dp[i][j]=dp[i-1][j-1]
                    else:
                        dp[i][j]=max(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])+1
            return dp[m][n]

import collections
class LRUCache:

    def __init__(self, capacity: int):
        self.dic=collections.OrderedDict()
        self.capacity = capacity
        

    def get(self, key: int) -> int:
        if key in self.dic:
            return None
        result=self.dic.pop(key)
        self.dic[key]=result
        return result


    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self.dic.pop(key)
        else:
            if len(self.dic)>=self.capacity:
                self.dic.popitem(last=False)
        self.dic[key]=value

