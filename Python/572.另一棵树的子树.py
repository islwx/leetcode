#
# @lc app=leetcode.cn id=572 lang=python3
#
# [572] 另一棵树的子树
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def is_subtree(self, node, subRoot):
        if node.val == subRoot.val:
            is_same = self.is_same(subRoot, node)
            # print(is_same )
            # print(subRoot)
            # print(node)
            if is_same:
                return is_same
        if node.left is not None:
            is_same = self.is_subtree(node.left, subRoot) 
            if is_same:
                return is_same
        if node.right is not None:
            is_same = self.is_subtree(node.right, subRoot) 
            if is_same:
                return is_same
        return False
        
        
    def is_same(self, target_node, input_node):
        if target_node is None and input_node is None:
            return True
        if target_node is None or input_node is None:
            return False
        if target_node.val != input_node.val:
            return False
        left_same = self.is_same(target_node.left, input_node.left)
        if not left_same:
            return False
        right_same = self.is_same(target_node.right, input_node.right)
        if not right_same:
            return False
        return True


    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        """
        182/182 cases passed (132 ms)
        Your runtime beats 29.97 % of python3 submissions
        Your memory usage beats 61.98 % of python3 submissions (15.7 MB)
        """
        return self.is_subtree(root, subRoot)
        
        
# @lc code=end

[3,4,5,1,2]\n[4,1,2,1]
[3,4,5,1,2,null,null,null,null,0]\n[4,1,2]