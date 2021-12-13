#
# @lc app=leetcode.cn id=107 lang=python3
#
# [107] 二叉树的层序遍历 II
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def  level_order_bottom(self, nodes):
        """
        34/34 cases passed (36 ms)
        Your runtime beats 49.98 % of python3 submissions
        Your memory usage beats 8.73 % of python3 submissions (15.6 MB)
        """
        cur_vals = []
        new_nodes = []
        for node in nodes:
            if node.left is not None:
                new_nodes.append(node.left)
            if node.right is not None:
                new_nodes.append(node.right)
            cur_vals.append(node.val)
        if new_nodes:
            ret = self.level_order_bottom(new_nodes)
            ret.append(cur_vals)
        else :
            ret = [cur_vals]
        return ret
        

    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        return self.level_order_bottom([root])
# @lc code=end

