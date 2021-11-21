#
# @lc app=leetcode.cn id=1764 lang=python3
#
# [1764] 通过连接另一个数组的子数组得到一个数组
#

# @lc code=start
class Solution:
    def canChoose(self, groups: List[List[int]], nums: List[int]) -> bool:
        """
        速度最好，但内存占用高
        """
        str_nums = " "+str(nums)[1:-1]+","
        for cur_seq in groups:
            str_seq = ' ' + str(cur_seq)[1:-1] + ','
            try:
                index = str_nums.index(str_seq)
            except:
                return False
            str_nums = str_nums[index+len(str_seq):]
        return True

    def canChoose11(self, groups: List[List[int]], nums: List[int]) -> bool:
        str_nums = " "+str(nums)[1:-1]+","
        for cur_seq in groups:
            str_seq = ' ' + str(cur_seq)[1:-1] + ','
            try:
                index = str_nums.index(str_seq)
            except:
                return False
            str_nums = str_nums[index+len(str_seq):]
        return True

    def canChoose2(self, groups: List[List[int]], nums: List[int]) -> bool:
        str_nums = str(nums)
        for cur_seq in groups:
            str_seq = '[ \[]' + str(cur_seq)[1:-1] + '[, \]]'
            index = re.search(str_seq, str_nums)
            if index is None:
                return False
            str_nums = str_nums[index.span()[1]:]
        return True
                


    def canChoose1(self, groups: List[List[int]], nums: List[int]) -> bool:
        cursor = 0
        max_size = len(nums)
        for cur_seq in groups:
            is_continue = False
            size = len(cur_seq)
            while (cursor+size)<=max_size:
                if nums[cursor:cursor+size] == cur_seq:
                    is_continue = True
                    cursor += size
                    break
                cursor+=1
            if not is_continue:
                return False
        return is_continue


# @lc code=end
solution = Solution()
tester = unittest.TestCase()

groups = [[1,-1,-1],[3,-2,0]] 
nums = [1,-1,0,1,-1,-1,3,-2,0]
tester.assertEqual(solution.canChoose(groups, nums), True)

groups = [[10, -2],[1,2,3,4]]
nums = [1,2,3,4,10,-2]
tester.assertEqual(solution.canChoose(groups, nums), False)

groups = [[1,2,3],[3,4]]
nums = [7,7,1,2,3,4,7,7]
tester.assertEqual(solution.canChoose(groups, nums), False)

