#
# @lc app=leetcode.cn id=337 lang=python3
#
# [337] 打家劫舍 III
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def cal_sub_node_val(node):
    return node.left + node.right
class Solution:
    
    def cal(self, root):
        if root is None:
            return 0
        sub_val  = cal_sub_node_val(root)
        root 
        left_val = self.cal(root.left)
        right_val = self.cal(root.right)
        sub_val = left_val+right_val
        if sub_val>root.val:
            return sub_val
        else:
            return root.val
        
        
            
    def rob(self, root: TreeNode) -> int:
        return self.cal(root)

# @lc code=end

[4,1,null,2,null,3]
[3,2,3,null,3,null,1]