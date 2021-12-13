#
# @lc app=leetcode.cn id=1089 lang=python3
#
# [1089] 复写零
#

# @lc code=start
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        30/30 cases passed (80 ms)
        Your runtime beats 20 % of python3 submissions
        Your memory usage beats 46.79 % of python3 submissions (15.1 MB)
        """
        if sum(arr) == 0:
            return arr
        size = len(arr)
        i = 0
        while i<size:
            if arr[i]==0:
                if i+1 >= size:
                    break
                if i+2 < size:
                    arr[i+2:] = arr[i+1:-1]
                arr[i+1] = 0
                i+=2
            else:
                i+=1
        return arr[:size]
        """
        Do not return anything, modify arr in-place instead.
        """
# @lc code=end

