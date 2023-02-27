#
# @lc app=leetcode.cn id=1144 lang=python3
#
# [1144] 递减元素使数组呈锯齿状
#
# @lc code=start
from typing import List
class Solution:
    def calc_cost_v1(self, nums):
        cost = 0
        
        for i in range(1, len(nums), 2):
            # prev cost
            if nums[i - 1] >= nums[i]:
                prev_cost = nums[i - 1] - nums[i] + 1
            else:
                prev_cost = 0
            if nums[i + 1] >= nums[i]:
                next_cost = nums[i + 1] - nums[i] + 1
            else:
                next_cost = 0

            if next_cost == prev_cost == 0:
                cur_cost = 0
            elif next_cost == 0 and prev_cost != 0:
                cur_cost = prev_cost
            elif prev_cost == 0 and next_cost != 0:
                cur_cost = next_cost
                nums[i + 1] -= cur_cost
            elif prev_cost >= next_cost:
                cur_cost = prev_cost
            else:
                cur_cost = next_cost
                sub_ = next_cost - prev_cost
                nums[i + 1] -= sub_ - 1
            cost += cur_cost
        return cost
    
    def calc_cost(self, nums, nums_size):
        cost = 0
        
        for i in range(1, nums_size, 2):
            # prev cost
            if nums[i - 1] >= nums[i]:
                prev_cost = nums[i - 1] - nums[i] + 1
            else:
                prev_cost = 0
            
            if nums[i + 1] >= nums[i]:
                next_cost = nums[i + 1] - nums[i] + 1
            else:
                next_cost = 0
                

            if next_cost == prev_cost == 0:
                cur_cost = 0
            elif prev_cost >= next_cost:  # next_cost == 0 and prev_cost != 0
                cur_cost = prev_cost
                nums[i] += prev_cost
            elif prev_cost == 0 and next_cost != 0:
                cur_cost = next_cost
                nums[i + 1] -= cur_cost
            else:
                if i+2<nums_size and nums[i+2] <= nums[i+1]:
                    # with     
                    
                    cur_cost = next_cost + 1
                    # sub_ = next_cost - prev_cost
                    nums[i] += prev_cost
                    nums[i + 1] -= (next_cost - prev_cost)
                else:
                    cur_cost  = next_cost
            cost += cur_cost
        print(nums)
        return cost

    def movesToMakeZigzag(self, nums: List[int]) -> int:
        nums_size = len(nums)
        if nums_size == 1:
            return 0
        elif nums_size == 2:
            if nums[0] == nums[1]:
                return 1
            else:
                return 0
        elif nums[1:] == nums[:-1]:
            return nums_size // 2
        else:
            if nums_size % 2 == 0:
                nums_size += 1
                cost_0 = self.calc_cost(nums + [-float("inf")], nums_size)
                cost_1 = self.calc_cost([-float("inf")] + nums ,nums_size)
            else:
                cost_0 = self.calc_cost(nums, nums_size)
                cost_1 = self.calc_cost([-float("inf")] + nums + [-float("inf")],nums_size+2)
            print(cost_0, cost_1)
            return min(cost_0,cost_1)
                
                
            
            
                


# @lc code=end

if __name__ == "__main__":
    #"""
    nums = [9,6,1,6,2]
    assert Solution().movesToMakeZigzag(nums) == 4
    
    nums = [1,2,3]
    assert Solution().movesToMakeZigzag(nums) == 2

    #"""
    
    nums = [10,4,4,10,10,6,2,3]
    assert Solution().movesToMakeZigzag(nums) == 13

    nums = [2,7,10,9,8,9]
    assert Solution().movesToMakeZigzag(nums) == 4