#
# @lc app=leetcode.cn id=1765 lang=python3
#
# [1765] 地图中的最高点
#

# @lc code=start
class Solution:
    def update_around(index, radius):
        
        self.ret[index]

    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        # 待实现
        is_water = np.asarray(isWater)
        self.ret = np.where(is_water==1, 0, -1)
        indexs = np.argwhere(self.ret==1)
        while -1 in ret:
            pass
            
# @lc code=end

solution = Solution()
assert solution.highestPeak([[0,1],[0,0]]) == [[1,0],[2,1]]
assert solution.highestPeak([[0,0,1],[1,0,0],[0,0,0]]) == [[1,1,0],[0,1,1],[1,2,2]]

