def quicksort(nums, low,high):
    if len(nums)<=1:
        return nums
    pi=partiton(nums,low,high)
    quicksort(nums[low,pi-1])
    quciksort(nums[pi+1,high])


def partiton(nums,low,high):
    if len(nums)<=1:
        return nums
    i=low-1
    pivot=nums[high]
    for j in range(low,high):
        if num[j]<=pivot:
            i+=1
            nums[i],nums[j]=nums[j],nums[i]
    i+=1
    nums[i],nums[high]=nums[high],nums[i]
    return i

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

def linklist(head):
    pre=None
    cur=head
    while cur:
        cur.next,pre,cur=pre,cur,cur.next
    return pre

def pairlinklist(head):
    pre=linklist(0)
    pre.next=head
    dummy=pre
    while (pre.next and pre.next.next):
        a=pre.next
        b=a.next
        pre.next,b,a=a.next,a,b.next
    return dummy.next


def circlelist(head):
    slow=fast=head
    while slow.next and fast.next.next:
        slow=slow.next
        fast=fast.next.next
        if slow is fast:
            return True
    return False

def sum3(nums):
    if not nums<=2:
        return []
    nums.sorted()
    res=[]
    n=len(nums)
    for i in range(0,n-2):
        while i >0 and nums[i]==nums[i-1]:
            continue
        l=i+1
        r=n-1
        while(l<r):
            t=nums[i]+nums[l]+nums[r]
            if t==0:
                res.append([i,l,r])
                while l<r and nums[l]==nums[l+1]:
                    l+=1
                while l<r and nums[r]==nums[r-1]:
                    r-=1
                l+=1
                r-=1
            elif t>0:
                r-=1
            else:
                l+=1
    return res


def sqrt(d):
    l=0
    r=d
    while l<=r:
        print(l,r)
        mid=l+(r-l)//2
        if (d//mid)==mid:
            return mid
        elif (d//mid)<mid:
            r=mid-1
        else:
            l=mid+1
    return r

def power(d,n):
    if n=0:
        return 1
    if n=1:
        return d
    if n<0:
        return 1/power(d,-n)
    elif n&1:
        return power(d*d,n//2)*d
    else:
        return power(d*d,n//2)

import heap
def ksort(matrix):
    l=[(matrix[i][0],i,j) for i in range(len(matrix))]
    heap.heapify(l)
    k=len(matrix)*len(matrix[0])
    res=[]
    while l:
        v,i,j=heap.heappop(l)
        res.append(v)
        if j+1<len(matrix[0]):
            heap.heappush(l,(matrix[i][j+1],i,j+1))
    return res

from collections import OrderedDict
class LRUCache:
    def __init__(self,capacity):
        self.cache=OrderedDict()
        self.capacity=capacity
    
    def get(self, key: int) -> int:
        if key in cache:
            v=self.cache.pop(key)
            self.cache[key]=v
            return v
        else:
            return -1

    def put(self,key, value):
        if key in cache:
            value=self.cache.pop(key)
        else:
            if len(cache.keys())<capacity:
                value=self.cache.pop(key)
            else:
                self.cache.popitem(last=False)
        self.cache[key]=value

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
        for w in word+self.end_of_words:
            node = node.setdefault(w,{})

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node=self.root
        for w in word:
            if w in node:
                node =node[w]
            else:
                return False
        if self.end_of_words in node:  
            return True
        else:
            return False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node=self.root
        for w in word:
            if w in node:
                node =node[w]
            else:
                return False
        return True