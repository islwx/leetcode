#
# @lc app=leetcode.cn id=7 lang=python3
#
# [7] 整数反转
#

# @lc code=start
class Solution:
    def reverse(self, x: int) -> int:
        s = str(x)
        if s[0].isdigit():
            sign = ''
        else:
            sign,s  = s[0], s[1:]
        x = int(sign+''.join(list(reversed(s))))
        if -2**31 <= x <= 2**31 - 1:
            return x
        else:
            return 0
# @lc code=end

