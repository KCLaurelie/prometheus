import timeit

"""
1. find missing number in array
You are given an array of positive numbers from 1 to n, such that all numbers from 1 to n are present except one number x. You have to find x.
"""


def find_missing(input):
    # calculate sum of all elements
    # in input list
    sum_of_elements = sum(input)

    # There is exactly 1 number missing
    n = len(input) + 1
    actual_sum = (n * (n + 1)) / 2
    return actual_sum - sum_of_elements

starttime = timeit.default_timer()
print(find_missing([3,7,1,2,8,4,5]))
print("The time difference is :", timeit.default_timer() - starttime)

"""
2. Determine if the sum of two integers is equal to the given value
Given an array of integers and a value, determine if there are any two integers in the array whose sum is equal to the given value. 
Return true if the sum exists and return false if it does not.
"""

def find_sum_of_two(A, val):
    found_values = set()
    for a in A:
        if val - a in found_values:
            return True

        found_values.add(a)

    return False

print(find_sum_of_two([3,7,1,2,8,4,5],10))


"""3. Merge two sorted linked lists
Given two sorted linked lists, merge them so that the resulting linked list is also sorted. 
Consider two sorted linked lists and the merged list below them as an example.
https://realpython.com/linked-lists-python/
https://dbader.org/blog/python-linked-list
"""
import collections
llst = collections.deque()  # create linked list
llst1 = collections.deque(['A', 'C', 'M', 'X'])
llst2 = collections.deque(['B', 'D', 'E', 'Y'])
llst_merged=(list(llst1)+list(llst2))
llst_merged.sort() # surely that's cheating
llst.pop()  # remove last element

def merge_sorted_llst(llst1, llst2):
    merged_llst = collections.deque()
    while llst1:
        if llst2: # there's still stuff in llst2
            el1 = llst1[0]
            el2 = llst2[0]
            if el1 <= el2:
                merged_llst.append(el1)
                llst1.popleft()
            else:
                merged_llst.append(el2)
                llst2.popleft()
        else:
            merged_llst += llst1

    if llst2: merged_llst += llst2
    return merged_llst

