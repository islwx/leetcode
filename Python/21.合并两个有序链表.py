#
# @lc app=leetcode.cn id=21 lang=python3
#
# [21] 合并两个有序链表
#
from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    @classmethod
    def create(cls, l):
        last_node = None
        for i in l[::-1]:
            node = cls(i)
            node.next=last_node
            last_node = node
        return last_node
# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list1 is None:
            return list2
        elif list2 is None:
            return list1

        if list1.val>list2.val:
            l = list2
            if list2.next is None:
                l.next=list1
                return l
            else:
                list2=list2.next
        else:
            l = list1
            if list1.next is None:
                l.next=list2
                return l
            else:
                list1=list1.next
        last = l
        while True:
            if list1.val>list2.val:
                last.next = list2
                if list2.next is None:
                    last.next.next=list1
                    return l
                else:
                    list2=list2.next
            else:
                last.next = list1
                if list2.next is None:
                    last.next.next=list2
                    return l
                else:
                    list1=list1.next
            last = last.next

# @lc code=end

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
    @classmethod
    def create(cls, l):
        last_node = None
        for i in l[::-1]:
            node = cls(i)
            node.next=last_node
            last_node = node
        return last_node
l1 = ListNode.create([-9,3])
l2 = ListNode.create([5,7])
s = Solution()
r = s.mergeTwoLists(l1, l2)
import pdb;pdb.set_trace()
print(1)