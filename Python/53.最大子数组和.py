#
# @lc app=leetcode.cn id=53 lang=python3
#
# [53] 最大子数组和
#

# @lc code=start
class Solution:
    def simply(self, nums):
        new_nums = []
        single_num_idx = []
        idx = 0
        cur_idx = 1
        while True:
            if nums[idx]*nums[cur_idx]>0:
                cur_idx+=1
            elif (cur_idx-idx)==1:
                single_num_idx.append(idx)
                idx = cur_idx
                cur_idx += 1
            else:
                nums[idx] = sum(nums[idx:cur_idx])
                del nums[idx+1:cur_idx]
                idx = idx+1
                cur_idx = idx + 1
            if cur_idx == len(nums):
                break
        return nums, single_num_idx

    def max_sub_array(self, nums):
        for num in nums:
            i
        
    def maxSubArray(self, nums: List[int]) -> int:
        pass
        
# @lc code=end

