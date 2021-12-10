#
# @lc app=leetcode.cn id=1694 lang=python3
#
# [1694] 重新格式化电话号码
#

# @lc code=start
class Solution:
    def reformatNumber(self, number: str) -> str:
        """
        Your runtime beats 69.5 % of python3 submissions
        Your memory usage beats 72.34 % of python3 submissions (14.9 MB)
        """
        number=number.replace(" ", "").replace("-", "")
        if len(number)%3==0:
            return "-".join([number[start:start+3] for start in range(0, len(number), 3)])
        elif len(number)%3==1:
            return  "-".join([number[start:start+3] for start in range(0, len(number)-4, 3)] + [number[-4:-2], number[-2:]])
        elif len(number)%3==2:
            return  "-".join([number[start:start+3] for start in range(0, len(number)-2, 3)] + [number[-2:]])

# @lc code=end

s = Solution()
print(s.reformatNumber("1-23-45 6"))
print(s.reformatNumber("123 4-567"))
print(s.reformatNumber("123 4-5678"))
print(s.reformatNumber("12"))
print(s.reformatNumber("--17-5 229 35-39475 "))