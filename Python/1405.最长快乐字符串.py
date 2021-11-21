#
# @lc app=leetcode.cn id=1405 lang=python3
#
# [1405] 最长快乐字符串
#

# @lc code=start
from  copy import deepcopy
class Solution:
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        """
        # TODO 待优化
        34/34 cases passed (40 ms)
        Your runtime beats 17.42 % of python3 submissions
        Your memory usage beats 17.42 % of python3 submissions (15.1 MB)
        """
        # 更新初始存量，根据插空规则修正单个字符最大可能的数量
        d = {'a':min(a,2*(b+c+1)),'b':min(b,2*(a+c+1)),'c':min(c,2*(b+a+1))}
        # 修正后的数量确保可以全部用在结果中，求和计算字符串总长
        n = sum(d.values())
        # 维护结果列表
        res = [' ','  ']
        # 单次插入一个字符，根据长度循环
        cand_ = set(['a','b','c'])
        for _ in range(n):
            # 候选的字母
            cand  = deepcopy(cand_)
            # 如果列表最后两个字符相同，根据规则不能插入连续三个，故将该字符从候选中删除
            if res[-1]==res[-2]:
                cand.remove(res[-1])
            # 贪心，在候选中选择存量最大的字符
            tmp = max(cand,key=lambda x:d[x])
            # 将它加到结果里
            res.append(tmp)
            # 把它的剩余计数减去1. 开始下一轮
            d[tmp] -= 1
        return ''.join(res[2:])

# solution = Solution()
# a, b, c = [1, 1, 7]
# print(solution.longestDiverseString(a, b, c))

# @lc code=end
'''
from  copy import deepcopy
class Solution:
    def longestDiverseString_v1(self, a: int, b: int, c: int) -> str:
        """
        34/34 cases passed (40 ms)
        Your runtime beats 17.42 % of python3 submissions
        Your memory usage beats 17.42 % of python3 submissions (15.1 MB)
        """
        # 更新初始存量，根据插空规则修正单个字符最大可能的数量
        d = {'a':min(a,2*(b+c+1)),'b':min(b,2*(a+c+1)),'c':min(c,2*(b+a+1))}
        # 修正后的数量确保可以全部用在结果中，求和计算字符串总长
        n = sum(d.values())
        # 维护结果列表
        res = [' ']
        # 单次插入一个字符，根据长度循环
        cand_ = set(['a','b','c'])
        for _ in range(n):
            # 候选的字母
            cand  = deepcopy(cand_)
            # 如果列表最后两个字符相同，根据规则不能插入连续三个，故将该字符从候选中删除
            if res[-1]==res[-2]:
                cand.remove(res[-1])
            # 贪心，在候选中选择存量最大的字符
            tmp = max(cand,key=lambda x:d[x])
            # 将它加到结果里
            res.append(tmp)
            # 把它的剩余计数减去1. 开始下一轮
            d[tmp] -= 1
        return ''.join(res[1:])


    def longestDiverseString_v2(self, a, b, c):
        """
        error [0,0,7] -> "cc"
        """
        dic = {}
        dic["a"],dic["b"],dic["c"] = a,b,c
        if max(a,b,c) > (a+b+c) + 1: return ""
        ans = ""
        hp = [(-dic[k], k)for k in dic]
        import heapq
        heapq.heapify(hp)
        while len(hp) > 1:
            k1, s1 = heapq.heappop(hp)
            k2, s2 = heapq.heappop(hp)
            if k1 != k2 or len(ans) == 0: # 说明-k1 > -k2
                if -k1 >= 2:
                    ans += s1 * 2
                    dic[s1] -= 2
                elif 0 < -k1 < 2:
                    ans += s1
                    dic[s1] -= 1
                if -k2 >= 1:
                    ans += s2
                    dic[s2] -= 1
            elif k1 == k2:
                if ans[-1] == s1: # 就先写s2
                    if -k2 >= 2:
                        ans += s2 * 2
                        dic[s2] -= 2
                    elif 0 < -k1 < 2:
                        ans += s2
                        dic[s2] -= 1
                    if -k1 >= 2:
                        ans += s1 * 2
                        dic[s1] -= 2
                    elif 0 < -k1 < 2:
                        ans += s1
                        dic[s1] -= 1
                elif ans[-1] != s1:
                    if -k1 >= 2:
                        ans += s1 * 2
                        dic[s1] -= 2
                    elif 0 < -k1 < 2:
                        ans += s1
                        dic[s1] -= 1
                    if -k2 >= 2:
                        ans += s2 * 2
                        dic[s2] -= 2
                    elif 0 < -k1 < 2:
                        ans += s2
                        dic[s2] -= 1
            
            if dic[s1] > 0:
                heapq.heappush(hp, (-dic[s1], s1))
            if dic[s2] > 0:
                heapq.heappush(hp, (-dic[s2], s2))
        cnt,item = 1,ans[-1]
        for k in range(len(ans)-1,0,-1):
            if ans[k-1] == ans[k]:
                cnt += 1
            else:
                break
        if len(hp) >= 1:
            if hp[0][1] != item:
                if -hp[0][0] >= 2:
                    ans += hp[0][1] * 2
                elif -hp[0][0] == 1:
                    ans += hp[0][1]
            elif -hp[0][0] >= 2:
                ans += hp[0][1] * (2-cnt)
            elif -hp[0][0] == 1:
                ans += hp[0][1] * (1-cnt)
        return ans


    def longestDiverseString_v1(self, a: int, b: int, c: int) -> str:
        """
        34/34 cases passed (40 ms)
        Your runtime beats 17.42 % of python3 submissions
        Your memory usage beats 17.42 % of python3 submissions (15.1 MB)
        """
        # 更新初始存量，根据插空规则修正单个字符最大可能的数量
        d = {'a':min(a,2*(b+c+1)),'b':min(b,2*(a+c+1)),'c':min(c,2*(b+a+1))}
        # 修正后的数量确保可以全部用在结果中，求和计算字符串总长
        n = sum(d.values())
        # 维护结果列表
        res = []
        # 单次插入一个字符，根据长度循环
        for _ in range(n):
            # 候选的字母
            cand = set(['a','b','c'])
            # 如果列表最后两个字符相同，根据规则不能插入连续三个，故将该字符从候选中删除
            if len(res)>1 and res[-1]==res[-2]:
                cand.remove(res[-1])
            # 贪心，在候选中选择存量最大的字符
            tmp = max(cand,key=lambda x:d[x])
            # 将它加到结果里
            res.append(tmp)
            # 把它的剩余计数减去1. 开始下一轮
            d[tmp] -= 1
        return ''.join(res)
'''