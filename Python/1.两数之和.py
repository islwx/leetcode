#
# @lc app=leetcode.cn id=1 lang=python3
#
# [1] 两数之和
#
from typing import List
# @lc code=start
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            if target>nums[i] or target<0:
                    plus = target - nums[i]
            else:
                    plus = -(nums[i] - target)
            # print(nums[i], plus) 
            if plus in nums[i+1:]:
                return [i, nums.index(plus, i+1)]


# @lc code=end
s = Solution()
nums=[3,2,95,4,-3]
target = 92
s.twoSum(nums, target)