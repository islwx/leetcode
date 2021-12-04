#
# @lc app=leetcode.cn id=9 lang=python3
#
# [9] 回文数
#

# @lc code=start
class Solution:
    # def isPalindrome(self, x: int) -> bool:
    #     """
    #     Your runtime beats 61.48 % of python3 submissions
    #     Your memory usage beats 79.79 % of python3 submissions (14.9 MB)
    #     """

    #     if (x < 0 or (x % 10 == 0 and x != 0)):
    #         return False
    #     rever_val = 0
    #     while x>rever_val:
    #         rever_val = rever_val*10 + (x%10)
    #         x=x//10
    #     return x==rever_val or x==(rever_val//10)
            

    def isPalindrome(self, x: int) -> bool:
        """
        Your runtime beats 97.37 % of python3 submissions
        Your memory usage beats 95.23 % of python3 submissions (14.8 MB)
        """
        x= str(x)
        return x == x[::-1]
            
# @lc code=end

s=Solution()
print(s.isPalindrome(10))