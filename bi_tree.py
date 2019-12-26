# 110. Balanced Binary Tree https://leetcode.com/problems/balanced-binary-tree/
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
            
# 144. Binary Tree Preorder Traversal https://leetcode.com/problems/binary-tree-preorder-traversal/
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
        
# 94. Binary Tree Inorder Traversal https://leetcode.com/problems/binary-tree-inorder-traversal/
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        node=root
        stack=[]
        res=[]
        while True:
            while node:
                stack.append(node)
                node=node.left
            if len(stack)==0:
                return res
            node = stack.pop()
            res.append(node.val)
            node=node.right
# 145. Binary Tree Postorder Traversal https://leetcode.com/problems/binary-tree-postorder-traversal/
class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        
        node=root
        stack=[(node,False)]
        res=[]
        while stack:
            node,visited=stack.pop()
            if node:
                if visited:
                    res.append(node.val)
                else:
                    stack.append((node,True))
                    stack.append((node.right,False))
                    stack.append((node.left,False))
        return res

# 102. Binary Tree Level Order Traversal https://leetcode.com/problems/binary-tree-level-order-traversal/
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        return self.bfs(root)
        
    def bfs(self,root):
        quene=[root]
        res=[]
        while quene:
            tmp_res=[]
            tmp_queue=[]
            while(quene):
                node=quene.pop(0)
                tmp_res.append(node.val)
                if node.left:
                    tmp_queue.append(node.left)
                if node.right:
                    tmp_queue.append(node.right)
            res.append(tmp_res)
            quene=tmp_queue
        return res

# 103. Binary Tree Zigzag Level Order Traversal https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        return self.bfs(root)
        
    def bfs(self,root):
        quene=[root]
        res=[]
        i=1
        while quene:
            tmp_res=[]
            tmp_queue=[]
            while(quene):
                node=quene.pop(0)
                tmp_res.append(node.val)
                if node.left:
                    tmp_queue.append(node.left)
                if node.right:
                    tmp_queue.append(node.right)
            if i&1:
                res.append(tmp_res)
            else:
                res.append(tmp_res[::-1])
            i+=1
            quene=tmp_queue
        return res

# 700. Search in a Binary Search Tree
class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if not root:
            return 
        node= root
        while node:
            if val==node.val:
                return node
            if val>node.val:
                node=node.right
            else:
                node=node.left
        return 

# 105. Construct Binary Tree from Preorder and Inorder Traversal
# preorder = [3,9,20,15,7]
# inorder = [9,3,15,20,7]    
#     3
#    / \
#   9  20
#     /  \
#    15   7
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if inorder:
            ind =inorder.index(preorder.pop(0))
            root = TreeNode(inorder[ind])
            root.left=self.buildTree(preorder,inorder[0:ind])
            root.right=self.buildTree(preorder,inorder[ind+1:])
            return root

# 108. Convert Sorted Array to Binary Search Tree            
# Given the sorted array: [-10,-3,0,5,9],

#       0
#      / \
#    -3   9
#    /   /
#  -10  5
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        mid=len(nums)//2
        root=Node(nums[mid])
        root.left=self.sortedArrayToBST(nums[:mid])
        root.right=self.sortedArrayToBST(nums[mid+1:])
        return root

# 109. Convert Sorted List to Binary Search Tree https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/
class Solution:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        if not head:
            return None
        dummy=ListNode(0)
        dummy.next=head
        head=dummy
        fast,slow,left_tail =head,head,head
        while fast and fast.next:
            fast = fast.next.next
            left_tail =slow
            slow = slow.next
        left_tail.next=None
        node=ListNode(slow.val)
        node.left=self.sortedListToBST(head.next)
        node.right=self.sortedListToBST(slow.next)
        return node     
# 112. Path Sum https://leetcode.com/problems/path-sum/submissions/
# 练！！
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        if not root.left and not root.right and root.val==sum:
            return True
        s=sum-root.val
        return self.hasPathSum(root.left,s) or self.hasPathSum(root.right,s)   

# 113. Path Sum II https://leetcode.com/problems/path-sum-ii/
class Solution:        
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        self.res=[]
        if not root:
            return self.res
        self.value=sum
        self.dfs(root,[])
        return self.res
    
    def dfs(self,root,s):
        if not root.left and not root.right:
            if sum(s+[root.val]) ==self.value:
                self.res.append(s+[root.val])
            return
        if root.left:
            self.dfs(root.left,s+[root.val])
        if root.right:
            self.dfs(root.right,s+[root.val])

# 701. Insert into a Binary Search Tree https://leetcode.com/problems/insert-into-a-binary-search-tree/submissions/
class Solution:
    def insertIntoBST(self, root: TreeNode, val: int) -> TreeNode:
        newNode=TreeNode(val)
        if not root:
            self.root=newNode
            return 
        cur=root
        pre=None
        while True:
            pre=cur
            if val==cur.val:
                return root
            if val<cur.val:
                cur=cur.left
                if not cur:
                    pre.left=newNode
                    return root
            else:
                cur=cur.right
                if not cur:
                    pre.right=newNode
                    return root

# 450. Delete Node in a BST https://leetcode.com/problems/delete-node-in-a-bst/ 
# 以后再说

 
# 98. Validate Binary Search Tree https://leetcode.com/problems/validate-binary-search-tree/
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        self.res=[]
        self.inorder(root)
        return self.res==sorted(set(self.res))
    def inorder(self,root):
        if not root:
            return None
        self.inorder(root.left)
        self.res.append(root.val)
        self.inorder(root.right)

# 101. Symmetric Tree https://leetcode.com/problems/symmetric-tree/
class Solution:        
    def isSymmetric(self, root: TreeNode) -> bool:
        return self.isMirror(root, root)
    
    def isMirror(self, t1, t2):
        if t1 == None and t2 == None:
            return True
        if t1 == None or t2 == None:
            return False
        return (t1.val == t2.val) and self.isMirror(t1.right, t2.left) and self.isMirror(t1.left, t2.right)
    
# 226. Invert Binary Tree https://leetcode.com/problems/invert-binary-tree/
class Solution(object):
    def invertTree(self, root):
        self.helper(root)
        return root
        
    def helper(self, root):
        if root:
            self.helper(root.left)
            self.helper(root.right)
            root.left,root.right=root.right,root.left     

# 100. Same Tree https://leetcode.com/problems/same-tree/
class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p==None and q == None:
            return True
        if p==None or q==None:
            return False
        return p.val==q.val and self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)

# 104. Maximum Depth of Binary Tree https://leetcode.com/problems/maximum-depth-of-binary-tree/
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return max(self.maxDepth(root.left),self.maxDepth(root.right))+1
# 
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
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