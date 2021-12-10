#
# @lc app=leetcode.cn id=7 lang=python3
#
# [7] 整数反转
#

# @lc code=start
class Solution:
    def reverse(self, x: int) -> int:
        """
        1032/1032 cases passed (32 ms)
        Your runtime beats 79.62 % of python3 submissions
        Your memory usage beats 5.27 % of python3 submissions (15.1 MB)
        """
        if x<0:
            sign = True
            x *= -1
        else:
            sign = False
        v = 0
        while x!=0:
            v=v*10+x%10
            x//=10
        if sign:
            v *= -1
        if -2**31 <= v <= 2**31 - 1:
            return v
        else:
            return 0
# @lc code=end

s= Solution()
print(s.reverse(-123))