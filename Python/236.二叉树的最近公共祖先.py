#
# @lc app=leetcode.cn id=236 lang=python3
#
# [236] 二叉树的最近公共祖先
#

# @lc code=start
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def search_node(self, root, node):
        if root.val == None:
            return None
        if root.val == node.val:
            return [root]
        else:
            if root.left is not None:
                r1 = self.search_node(root.left, node)
                if r1 is not None:
                    r1.append(root)
                    return r1
            if root.right is not None:
                r1 = self.search_node(root.right, node)
                if r1 is not None:
                    r1.append(root)
                    return r1
            return None
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root.val == p.val:
            return root
        if root.val == q.val:
            return root
        pf = self.search_node(root, p)
        qf = self.search_node(root, q)
        range_end = min(len(pf), len(qf))
        for i in range(-1, -range_end-1, -1):
            if qf[i]!=pf[i]:
                return pf[i+1]
        return pf[i]
                
            

        

# @lc code=end
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
    @classmethod
    def create_by_seq(cls, seq:list):
        seq.reverse()
        root = cls(seq.pop())
        cur_pool = [root]
        next_pool = list()
        while seq:
            val = seq.pop()
            tmp = cls(val)
            if val is not None:
                next_pool.append(tmp)
            node = cur_pool[0]
            if node.left is None:
                node.left = tmp
            elif node.right is None:
                node.right = tmp
                cur_pool = cur_pool[1:]
                if not cur_pool:
                    cur_pool = next_pool
                    next_pool = list()
        
        return root
    

            

# root_seq = [-1,0,3,-2,4,None,None,8]
# root = TreeNode.create_by_seq(root_seq)
# p=TreeNode(8)
# q=TreeNode(0)

root_seq = [3,5,1,6,2,0,8,None,None,7,4]
root = TreeNode.create_by_seq(root_seq)
p=TreeNode(5)
q=TreeNode(1)

solution = Solution()
print(solution.lowestCommonAncestor(root, p, q).val)