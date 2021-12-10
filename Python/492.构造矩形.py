#
# @lc app=leetcode.cn id=492 lang=python3
#
# [492] 构造矩形
#
import math
# @lc code=start
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        l = int(math.sqrt(area))
        while area%l:
            l-=1
        l1 = int(area/l)
        return [l1, l] if l1 > l else [l1, l]
# @lc code=end

