#
# @lc app=leetcode.cn id=215 lang=python3
#
# [215] 数组中的第K个最大元素
#

# @lc code=start
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        32/32 cases passed (28 ms)
        Your runtime beats 97.65 % of python3 submissions
        Your memory usage beats 94.63 % of python3 submissions (15.3 MB)
        """
        nums.sort(reverse=True)
        return nums[k-1]
# @lc code=end

