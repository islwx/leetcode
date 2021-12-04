#
# @lc app=leetcode.cn id=20 lang=python3
#
# [20] 有效的括号
#

# @lc code=start
class Solution:
    def isValid(self, s: str) -> bool:
        """
        #TODO 性能太差
        Your runtime beats 22.42 % of python3 submissions
        Your memory usage beats 97.8 % of python3 submissions (14.7 MB)
        """
        last_size = len(s)
        if last_size%2==1:
            return False
        s=s.replace("{}", "").replace("[]", "").replace("()", "")
        cur_size = len(s)
        while last_size!=cur_size:
            s=s.replace("{}", "").replace("[]", "").replace("()", "")
            last_size,cur_size = cur_size,len(s)
        return s==""
    

        
# @lc code=end

s=Solution()
s.isValid("()")