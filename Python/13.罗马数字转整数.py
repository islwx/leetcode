#
# @lc app=leetcode.cn id=13 lang=python3
#
# [13] 罗马数字转整数
#

# @lc code=start
class Solution:
    map={
        "I":1,
        "v":4,
        "V":5,
        "x":9,
        "X":10,
        "l":40,
        "L":50,
        "c":90,
        "C":100,
        "d":400,
        "D":500,
        "m":900,
        "M":1000
    }
    def romanToInt(self, s: str) -> int:
        """
        Your runtime beats 73.04 % of python3 submissions
        Your memory usage beats 87.2 % of python3 submissions (14.9 MB)

        Your runtime beats 96.49 % of python3 submissions
        Your memory usage beats 74.18 % of python3 submissions (15 MB)
        """
        if len(s)==1:
            return self.map[s]
        s = s.replace("CM","m").replace("CD", "d") \
             .replace("XC", "c").replace("XL",'l') \
             .replace("IX","x").replace("IV", "v")
        v = 0
        for char in s:
            v+=self.map[char]
        return v

# @lc code=end

print(Solution().romanToInt("MCDLXXVI"))