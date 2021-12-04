#
# @lc app=leetcode.cn id=14 lang=python3
#
# [14] 最长公共前缀
#
from typing import List
# @lc code=start
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        """
        Your runtime beats 93 % of python3 submissions
        Your memory usage beats 65.98 % of python3 submissions (15 MB)
        """
        if "" in strs:
            return ""
        elif len(strs)==1:
            return strs[0]
        st=""
        for t in zip(*strs):
            try:
                [s] = set(t)
                st+=s
            except:
                break
        return st

# @lc code=end
s = Solution()
strs = ["ab", "a"]
print(s.longestCommonPrefix(strs))
