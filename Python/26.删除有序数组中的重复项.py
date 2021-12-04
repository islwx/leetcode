#
# @lc app=leetcode.cn id=26 lang=python3
#
# [26] 删除有序数组中的重复项
#

# @lc code=start
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        """
        Your runtime beats 99.63 % of python3 submissions
        Your memory usage beats 47.65 % of python3 submissions (15.6 MB)
        """
        """
        r = sorted(set(nums))
        l = len(r)
        for i in range(l):
            nums[i] = r[i]
        return l
        """
        index = 1 
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:
                nums[index] = nums[i]
                index += 1
        return index
# @lc code=end