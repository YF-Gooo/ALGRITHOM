class Node:
    def __init__ (self, value = None, next = None):
        self.value = value
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = Node()
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
    
# 反转链表
def reverse_linklist(head):
    pre,cur=None,head
    while cur:
        cur.next,pre,cur= pre, cur, cur.next
    return pre

class Solution(object):
	def reverseList(self, head):
		"""
		:type head: ListNode
		:rtype: ListNode
		"""
		pre = None
		cur = head
		while cur:
			tmp = cur.next
			cur.next = pre
			pre = cur
			cur = tmp
		return pre	


class Solution(object):
	def reverseList(self, head):
		"""
		:type head: ListNode
		:rtype: ListNode
		"""
		if(head==None or head.next==None):
			return head
		cur = self.reverseList(head.next)
		head.next.next = head
		head.next = None
		return cur
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

# 142. Linked List Cycle II https://leetcode.com/problems/linked-list-cycle-ii/
# Given a linked list, return the node where the cycle begin
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head==None:
            return None
        fast=slow=head
        while slow and fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                fast=head
                break

        if fast is None or fast.next is None:
            return None
        while fast != slow:
            fast = fast.next
            slow = slow.next
        return fast

# 19. Remove Nth Node From End of List https://leetcode.com/problems/remove-nth-node-from-end-of-list/

class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:

        fast = head
        while n>0:
            fast = fast.next
            n = n - 1
        # 下面这一行必须有
        if not fast:
            return head.next
        slow = head
        while fast.next:
            fast = fast.next
            slow = slow.next
        
        result = slow.next
        slow.next = slow.next.next    
        return head

# 876. Middle of the Linked List https://leetcode.com/problems/middle-of-the-linked-list/
class Solution:
    def middleNode(self,head):
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    
# 这题的衍生问题，链表切割
def split(head):
    slow = head
    fast = head
    front_last_node = slow
    while fast and fast.next:
        front_last_node = slow
        slow = slow.next
        fast = fast.next.next
    front_last_node.next = None
    front = head
    back = slow
    return (front, back)

# 21. Merge Two Sorted Lists https://leetcode.com/problems/merge-two-sorted-lists/
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next,l1 = l1,l1.next
            else:
                cur.next,l2 = l2,l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next

# 160. Intersection of Two Linked Lists https://leetcode.com/problems/intersection-of-two-linked-lists/
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

# 147. Insertion Sort List https://leetcode.com/problems/insertion-sort-list/
class Solution:
    def insertionSortList(self, head):
        if not head or not head.next:
            return head
        dummy = ListNode(0)
        dummy.next = cur = head
        while cur.next:
            if cur.val <= cur.next.val:
                cur = cur.next
            else:
                pre = dummy
                while pre.next.val < cur.next.val:
                    pre = pre.next
                tmp = cur.next
                cur.next = tmp.next
                tmp.next = pre.next
                pre.next = tmp
        return dummy.next

# 148. Sort List https://leetcode.com/problems/sort-list/
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        mid = self.getMiddle(head)
        rHead = mid.next
        mid.next = None
        return self.merge(self.sortList(head), self.sortList(rHead))

    def merge(self, l1, l2):
        dummy = cur = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next,l1 = l1,l1.next
            else:
                cur.next,l2 = l2,l2.next
            cur = cur.next
        cur.next = l1 or l2
        return dummy.next
        
    def getMiddle(self, head):
        if head is None:
            return head
        slow = head
        fast = head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        return slow

# 86. Partition List https://leetcode.com/problems/partition-list/
# Input: head = 1->4->3->2->5->2, x = 3
# Output: 1->2->2->4->3->5
class Solution:
    def partition(self,head, x):
        left_head = ListNode(None)  # head of the list with nodes values < x
        right_head = ListNode(None)  # head of the list with nodes values >= x
        left = left_head  # attach here nodes with values < x
        right = right_head  # attach here nodes with values >= x
        # traverse the list and attach current node to left or right nodes
        while head:
            if head.val < x:
                left.next = head
                left = left.next
            else:  # head.val >= x
                right.next = head
                right = right.next
            head = head.next
        right.next = None  # set tail of the right list to None
        left.next = right_head.next  # attach left list to the right
        return left_head.next  # head of a new partitioned list


# 92. Reverse Linked List II https://leetcode.com/problems/reverse-linked-list-ii/
# Input: 1->2->3->4->5->NULL, m = 2, n = 4
# Output: 1->4->3->2->5->NULL
class Solution:
    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode:
        h = buf = ListNode(0)
        buf.next = head
        for _ in range(m-1):
            h = h.next
        pre=h
        cur = h.next
        for _ in range(n-m+1):
            cur.next,pre,cur= pre, cur, cur.next
        h.next.next =cur
        h.next = pre
        return buf.next

# 234. Palindrome Linked List https://leetcode.com/problems/palindrome-linked-list/
class Solution(object):
    def isPalindrome(self, head):
        fast = slow = head
        # find the mid node
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        # reverse the second half
        pre = None
        while slow:
            slow.next,pre,slow=pre,slow,slow.next
        # compare the first and second half nodes
        while pre: # while node and head:
            if pre.val != head.val:
                return False
            pre = pre.next
            head = head.next
        return True

# 83. Remove Duplicates from Sorted List https://leetcode.com/problems/remove-duplicates-from-sorted-list/
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if head == None:
            return head
        node = head
        while node.next:
            if node.val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        return head

# 82. Remove Duplicates from Sorted List II https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
# Input: 1->2->3->3->4->4->5
# Output: 1->2->5
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = pre = ListNode(0)
        dummy.next = head
        while head and head.next:
            if head.value == head.next.value:
                while head and head.next and head.value == head.next.value:
                    head = head.next
                head = head.next
                pre.next = head
            else:
                pre = pre.next
                head = head.next
        return dummy.next