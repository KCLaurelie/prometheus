# https://www.educative.io/blog/crack-amazon-coding-interview-questions#overview
import timeit
import collections
import math
from random import randint
import re
from collections import Counter
from collections import defaultdict
from itertools import groupby

#region 1. find missing number in array
"""
1. find missing number in array
You are given an array of positive numbers from 1 to n, such that all numbers from 1 to n are present except one number x. Y
ou have to find x.

Runtime Complexity: Linear, O(n)
Memory Complexity: Constant, O(1)
"""

class Solution1(object):
    def find_missing(self,input):
        # calculate sum of all elements
        # in input list
        sum_of_elements = sum(input)

        # There is exactly 1 number missing
        n = len(input) + 1
        actual_sum = (n * (n + 1)) / 2
        return actual_sum - sum_of_elements

starttime = timeit.default_timer()
print(Solution1().find_missing([3,7,1,2,8,4,5]))
print("The time difference is :", timeit.default_timer() - starttime)
#endregion

#region 2. Determine if the sum of two integers is equal to the given value
"""
2. Determine if the sum of two integers is equal to the given value
Given an array of integers and a value, determine if there are any two integers in the array whose sum is equal to the given value. 
Return true if the sum exists and return false if it does not.

Runtime Complexity: Linear, O(n)
Memory Complexity: Linear, O(n)
"""
class Solution2(object):
    def find_sum_of_two(self, A, val):
        found_values = set()
        for a in A:
            if val - a in found_values:
                return True

            found_values.add(a)

        return False

print(Solution2().find_sum_of_two([3,7,1,2,8,4,5],10))
#endregion

#region 3. Merge two sorted linked lists
"""3. Merge two sorted linked lists
Given two sorted linked lists, merge them so that the resulting linked list is also sorted. 
Consider two sorted linked lists and the merged list below them as an example.

https://realpython.com/linked-lists-python/
https://dbader.org/blog/python-linked-list

Runtime Complexity: Linear, O(m + n) where m and n are lengths of both linked lists
Memory Complexity: Constant, O(1)
"""

# METHOD 1: using deque
llst = collections.deque()  # create linked list
llst1 = collections.deque(['A', 'C', 'M', 'X'])
llst2 = collections.deque(['B', 'D', 'E', 'Y'])
llst_merged=(list(llst1)+list(llst2))
llst_merged.sort() # surely that's cheating
llst.pop()  # remove last element

def merge_sorted_llst(llst1, llst2):
    merged_llst = collections.deque()
    while llst1:
        if llst2: # if there's still stuff in llst2
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

merge_sorted_llst(collections.deque(['A', 'C', 'M', 'X']), collections.deque(['A', 'B', 'D', 'E', 'Y', 'Z']))
merge_sorted_llst(collections.deque(['A', 'B', 'C', 'D','E','F','G','H','I', 'M', 'X']), collections.deque(['A', 'B', 'D', 'E', 'Y', 'Z']))


# METHOD 2: building linked lists objects from scratch
class LLNode:
    def __init__(self, data):
        self.data = data
        self.next = None

    def __repr__(self):
        return self.data


class LinkedList:
    def __init__(self, nodes=None):
        self.head = None
        if nodes is not None:
            node = LLNode(data=nodes.pop(0))
            self.head = node
            for elem in nodes:
                node.next = LLNode(data=elem)
                node = node.next

    def __repr__(self):
        node = self.head
        nodes = []
        while node is not None:
            nodes.append(node.data)
            node = node.next
        nodes.append("None")
        return " -> ".join(nodes)

    def __iter__(self): # to traverse the llist
        node = self.head
        while node is not None:
            yield node
            node = node.next

    def add_first(self, node):  # inserting at the beginning
        node.next = self.head
        self.head = node

    def add_last(self, node):  # inserting at the end
        if self.head is None:
            self.head = node
            return
        for current_node in self:  # traverse the whole list until you reach the end
            pass
        current_node.next = node

    def append(self, llst2):  # inserting at the end
        if self.head is None:
            self = llst2
            return
        for current_node in self:  # traverse the whole list until you reach the end
            pass
        for current_nodellst2 in llst2:
            current_node.next = current_nodellst2
            current_node = current_nodellst2

    def pop_left(self):  # removing first node
        if self.head is None:
            raise Exception("List is empty")
        else:
            self.head = self.head.next
            return

    def pop_right(self):  # removing last node
        if self.head is None:
            raise Exception("List is empty")
        else:
            current_node = self.head
            while current_node.next:
                prev_node = current_node
                current_node = current_node.next
            prev_node.next = None
            return

    def merge_sorted_llst(self, llst2):
        llmerged = LinkedList()
        while self.head is not None:
            if llist2.head is not None and self.head.data <= llist2.head.data:
                llmerged.add_last(LLNode(self.head.data))
                self.pop_left()
            elif llist2.head is not None and self.head.data > llist2.head.data:
                llmerged.add_last(LLNode(llist2.head.data))
                llist2.pop_left()
            else:
                break
        # append remaining bits
        if self.head is not None: llmerged.append(self)
        if llist2.head is not None: llmerged.append(llist2)
        return llmerged


llist1 = LinkedList(["a", "b", "c", "d", "e"])
llist2 = LinkedList(["b", "e", "f", "g", "h"])

llist1.add_last(LLNode('f'))
llist1.append(llist2)
llist1.merge_sorted_llst(llist2)
print(llist1, llist1.head, llist1.pop_left())

#endregion

#region 4. Deep copy linked list with arbitrary pointer TODO
"""
4. Copy linked list with arbitrary pointer
You are given a linked list where the node has two pointers. 
The first is the regular next pointer. 
The second pointer is called arbitrary_pointer and it can point to any node in the linked list. 
Your job is to write code to make a deep copy of the given linked list. 
Here, deep copy means that any operations on the original list should not affect the copied list.

Runtime Complexity: Linear, O(n)
Memory Complexity: Linear, O(n)
"""

# TODO

#endregion

#region 5. Level order Binary Tree Traversal
"""
5. Level order Binary Tree Traversal in python

https://stephanosterburg.gitbook.io/scrapbook/coding/python/breadth-first-search/level-order-tree-traversal
Runtime Complexity: Linear, O(n) where n is number of nodes in the binary tree
Memory Complexity: Linear, O(n)
"""


# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution5(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        if root is None: return  # Base Case
        queue = [root]  # Create an empty queue for level order traversal

        while len(queue) > 0:
            # Print front of queue and remove it from queue
            print(queue[0].val)
            node = queue.pop(0)

            # Enqueue left child
            if node.left is not None: queue.append(node.left)
            # Enqueue right child
            if node.right is not None: queue.append(node.right)

root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
Solution5().levelOrder(root)

#endregion

#region 6. Determine if a binary tree is a binary search tree
"""
6. Determine if a binary tree is a binary search tree
Given a Binary Tree, figure out whether it’s a Binary Search Tree. 
In a binary search tree, each node’s key value is smaller than the key value of all nodes in the right subtree, 
and is greater than the key values of all nodes in the left subtree
        4
    2       5
1       3

The optimal approach is a regular in-order traversal and in each recursive call, 
pass maximum and minimum bounds to check whether the current node’s value is within the given bounds.

https://www.geeksforgeeks.org/a-program-to-check-if-a-binary-tree-is-bst-or-not/

Runtime Complexity: Linear, O(n) where n is number of nodes in the binary tree
Memory Complexity: Linear, O(n)
"""

class TreeNode:
    # Constructor to create a new node
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class Solution6(object):
    def is_bst_rec(self, root, min_value, max_value):
        if root is None: return True
        if root.data < min_value or root.data > max_value: return False

        return self.is_bst_rec(root.left, min_value, root.data) and self.is_bst_rec(root.right, root.data, max_value)

    def is_bst(self, root, INT_MAX=4294967296):
        return self.is_bst_rec(root, -INT_MAX - 1, INT_MAX)


root = TreeNode(4)
root.left = TreeNode(2)
root.right = TreeNode(5)
root.left.left = TreeNode(1)
root.left.right = TreeNode(3)
Solution6().is_bst(root)

#endregion

#region 7. String segmentation
"""
You are given a dictionary of words and a large input string. 
You have to find out whether the input string can be completely segmented into the words of a given dictionary. 
e.g.
Given a dictionary of words: apple apple pear pie
Input string of “applepie” can be segmented into dictionary words. apple pie

Runtime Complexity: Exponential, O(2^n)
Memory Complexity: Polynomial, O(n^2)
"""

class Solution7(object):
    def wordBreak(self,dict, str, out=""):
        # if the end of the string is reached,
        # print the output string
        if not str:
            print(out)
            return

        for i in range(1, len(str) + 1):
            # consider all prefixes of the current string
            prefix = str[:i]

            # if the prefix is present in the dictionary, add it to the
            # output string and recur for the remaining string
            if prefix in dict:
                self.wordBreak(dict, str[i:], out + " " + prefix)

dict = ["self", "th", "is", "famous", "Word", "break", "b", "r", "e", "a", "k", "br","bre", "brea", "ak", "problem"]
Solution7().wordBreak(dict, "Wordbreakproblem")

#endregion

#region 8. Reverse words in a sentence

class Solution8(object):
    def rev_sentence(self, sentence):
        # first split the string into words
        words = sentence.split(' ')

        # then reverse the split string list and join using space
        reverse_sentence = ' '.join(reversed(words))

        # finally return the joined string
        return reverse_sentence

Solution8().rev_sentence('hello world')
#endregion

#region 9. Coin change problem (dynamic programming solution)
"""
You are given coins of different denominations and a total amount of money. 
Write a function to compute the number of combinations that make up that amount. 
You may assume that you have an infinite number of each kind of coin.

Input:
  - Amount: 5
  - Coins: [1, 2, 5]
Output: 4
Explanation: There are 4 ways to make up the amount:
  - 5 = 5
  - 5 = 2 + 2 + 1
  - 5 = 2 + 1 + 1 + 1
  - 5 = 1 + 1 + 1 + 1 + 1

https://www.geeksforgeeks.org/coin-change-dp-7/
  
Runtime Complexity: Quadratic, O(m*n), m=number of coins, m=amount
Memory Complexity: Linear, O(n)O(n)
"""

class Solution9(object):
    def change(self, amount, coins):
        solution = [0]*(amount+1)
        solution[0] = 1 # base case (given value is 0)
        for coin in coins: #pick all coins one by one
            for i in range(coin, amount+1):
                solution[i] += solution[i-coin]
        return solution

Solution9().change(amount=5,coins=[1,2,5])

class Solution9b(object):
    def make_change(self, goal, coins):
        wallets = [[coin] for coin in coins]
        new_wallets = []
        collected = []

        while wallets:
            for wallet in wallets:
                s = sum(wallet)
                for coin in coins:
                    if coin >= wallet[-1]:
                        if s + coin < goal:
                            new_wallets.append(wallet + [coin])
                        elif s + coin == goal:
                            collected.append(wallet + [coin])
            wallets = new_wallets
            new_wallets = []
        return collected
Solution9b().make_change(goal=5,coins=[1,2,5])

#endregion

#region X. Finding 2 numbers from given list that add to a total

class SolutionX(object):
    def find_2_nbs_giving_total(self, total, numbers):
        n2 = total//2
        goodnums = {total-x for x in numbers if x<=n2} & {x for x in numbers if x>n2}
        pairs = {(total-x, x) for x in goodnums}
        return pairs
SolutionX().find_2_nbs_giving_total(total=181, numbers= [80, 98, 83, 92, 1, 38, 37, 54, 58, 89])
#endregion

#region 10. Find k_th permutation
"""
given a set of n elements, find the k-th permutation (given permutations are in ascending order
e.g. for the set 123
the ordered permutations are: 123, 132, 213, 231, 312, 321

Runtime Complexity: Linear, O(n)
Memory Complexity: Linear, O(n)
"""
set=[1,2,3,4,5,6]
k = 4 # we want the 4th permutation
nb_permutations = math.factorial(len(set)) # number of permutations using 6 numbers
nb_ind_permutations = nb_permutations/(len(set))# nb of permutations for each number (=factorial n-1)
first_nb_perm= math.floor(k/nb_ind_permutations) # the kth permutation starts with that number


class Solution10(object):
    def kth_permutation(self, k, set, res):
        print('set',set,'res',res)
        if not set: # if set is empty we reached the end of the algo
            return res
        n = len(set)
        nb_ind_permutations = math.factorial(n-1) if n > 0 else 1 # nb of permutations starting with each number
        perm_group = (k-1)//nb_ind_permutations # the kth permutation starts with that number
        res = res + str(set[perm_group])
        # now we want to find permutations in reduced set
        set.pop(perm_group)
        k = k - (nb_ind_permutations*perm_group)
        self.kth_permutation(k, set, res)
Solution10().kth_permutation(k=4, set=[1,2,3,4,5,6], res='')

#endregion

#region 11. Find all subsets of a given set of integers
"""
given the set [1,2,3]
the possible subsets are 1, 2, 3, [1,2], [1,3],[2,3],[1,2,3]
https://www.geeksforgeeks.org/find-the-k-th-permutation-sequence-of-first-n-natural-numbers/

Runtime Complexity: Exponential, O(2^n*n)
Memory Complexity: Exponential, O(2^n*n)
"""

# solution using recursion
class Solution11(object):
    def all_subsets(self, set, res=[], to_return=[]):
        for size_subset in range(len(set)):
            print('res',res)
            new_res = res.copy()
            new_res.append(set[size_subset])
            new_set = set[size_subset+1:]
            print('new_res', new_res)
            to_return.append(new_res)
            self.all_subsets(new_set, new_res)
        return to_return
Solution11().all_subsets(set=[2,3,4])
#endregion

#region 12. Print balanced brace combinations TODO
"""
Input: n=1
Output: {}
This the only sequence of balanced 
parenthesis formed using 1 pair of balanced parenthesis. 

Input : n=2
Output: 
{}{}
{{}}

Runtime Complexity: Exponential, 2^n2
Memory Complexity: Linear, O(n)
"""
class Solution12(object):
    def print_all_braces(self, n, left_count=0, right_count=0, output=[], result=[]):
        if left_count >= n and right_count >= n:
            result.append(output.copy());

        if left_count < n:
            output += '{'
            self.print_all_braces(n, left_count + 1, right_count, output, result)
            output.pop()

        if right_count < left_count:
            output += '}'
            self.print_all_braces(n, left_count, right_count + 1, output, result)
            output.pop()
        return result
Solution12().print_all_braces(2,0,0,[],[])
#endregion

#region 13. Deep copy of a directed graph  TODO
"""
In a diagram of a graph, a vertex is usually represented by a circle with a label, 
and an edge is represented by a line or arrow extending from one vertex to another
https://www.educative.io/m/clone-directed-graph

Runtime Complexity: Linear, O(n)
Memory Complexity: Logarithmic, O(logn)
"""
class DGNode:
    def __init__(self, d):
        self.data = d
        self.neighbors = []

class Solution13(object):
    def clone_rec(self, root, nodes_completed={}):
        if root == None: return None

        pNew = DGNode(root.data)
        nodes_completed[root] = pNew

        for p in root.neighbors:
            x = nodes_completed.get(p)
        if x is None:
            pNew.neighbors += [self.clone_rec(p, nodes_completed)]
        else:
            pNew.neighbors += [x]
        return pNew

root = DGNode(1)
root.neighbors = DGNode(2)
root.neighbors = DGNode(3)

Solution13().clone_rec(root)
#endregion

#region 16. find K largest elements from an array
"""
https://www.geeksforgeeks.org/k-largestor-smallest-elements-in-an-array/
"""
#METHOD1 Using sorting
#Runtime complexity: O(nlogn) (n: size of array)
class Solution16(object):
    def klargest(self, array, k):
        size=len(array)
        array.sort()
        return array[len(array)-k:len(array)]

#METHOD2 Usining min heap (optimization of method 1)
#Runtime complexity: O(k+(n-k)logk) (n: size of array)
class Solution16b(object):
    def klargest(self, array, k):
        size = len(array)

        # create min heap of k elements with priority queue
        minHeap = array[0:k].copy()

        for i in range(k, size):
            minHeap.sort()
            if array[i] > minHeap[0]:
                minHeap.pop(0)
                minHeap.append(array[i])

        return minHeap

Solution16().klargest(array=[1, 23, 12, 9, 30, 2, 50], k=3)
Solution16b().klargest(array=[1, 23, 12, 9, 30, 2, 50], k=3)
#endregion

#region 17 Convert a Binary tree to a Doubly Linked List TODO
#endregion

#region 21. Implement a stack with push(), min(), and pop() in O(1)O(1) time TODO
#endregion

#region 22. Rotate an array by K

def rotLeft(a, d):
    if d > len(arr):
        d = d % len(arr)
    print(d)
    return arr[d:]+arr[0:d]

arr = [1, 2, 3, 4, 5, 6]
rotLeft(arr, 14)

#endregion

#region 25 Implement a queue using a linked list TODO
"""
https://www.geeksforgeeks.org/queue-linked-list-implementation/

QUEUE: FIFO
In a Queue data structure, we maintain two pointers, front and rear. The front points the first item of queue and rear points to last item.
enQueue() This operation adds a new node after rear and moves rear to the next node.
deQueue() This operation removes the front node and moves front to the next node.

Runtime Complexity: Time complexity of both operations enqueue() and dequeue() is O(1)
"""
# Linked list node
class LLNode:
    def __init__(self, data):
        self.data = data
        self.next = None
class Queue:
    def __init__(self):
        self.front = self.rear = None

class Solution25(object):
    def enQueue(self, queue, item):
        nodeitem = LLNode(item)
        if queue.rear is None:
            queue.front = queue.rear = nodeitem
        else:
            queue.rear.next = nodeitem
            queue.rear = nodeitem
        print("Queue Rear ", str(queue.rear.data))

    def deQueue(self, queue):
        if (queue.rear is None) or (queue.front is None):
            print('queue is empty')
            return
        else:
            tmp = queue.front
            queue.front = tmp.next
        print("Queue Rear ", str(queue.rear.data))

q = Queue()
qobj = Solution25()
qobj.enQueue(queue=q, item=10)
qobj.enQueue(queue=q, item=20)
qobj.deQueue(queue=q)

#endregion

#region 36. egg dropping puzzle for dynamic programming [**TO REVISE**]
"""
https://www.geeksforgeeks.org/egg-dropping-puzzle-dp-11/
"""
# METHOD1 brut force recursion
# Runtime complexity: O(2^k) (many subproblems solved repeatedly)
# Memory Complexity: O(1)
class Solution36(object):
    def egg_drops(self, n_eggs, n_floors):
        if n_floors <= 1 or n_eggs <= 1: return n_floors # base case
        min = 9223372036854775807
        for floor in range(1, n_floors+1):
            res = max(self.egg_drops(n_eggs-1, floor-1),
                      self.egg_drops(n_eggs, n_floors-floor))
            if res < min: min = res
        return min+1


# METHOD2 recursion + memoization
# Runtime complexity: O(n*k^2)
# Memory Complexity: O(n*k)
class Solution36b(object):
    def egg_drops_rec(self, n_eggs, n_floors, memo):
        if memo[n_eggs][n_floors] != -1: return memo[n_eggs][n_floors]
        if n_floors <= 1 or n_eggs <= 1: return n_floors

        # recursion
        min = 9223372036854775807
        for floor in range(1, n_floors+1):
            res = max(self.egg_drops_rec(n_eggs-1, floor-1, memo),
                      self.egg_drops_rec(n_eggs, n_floors-floor, memo))
            if res < min: min = res
        memo[n_eggs][n_floors] = min + 1
        return min+1

    def egg_drops(self, n_eggs, n_floors):
        memo = [[-1 for x in range(n_floors + 1)] for x in range(n_eggs + 1)]
        return self.egg_drops_rec(n_eggs, n_floors, memo)


# METHOD3 dynamic programming
# Runtime complexity: O(n*k^2)
# Memory Complexity: O(n*k)
class Solution36c(object):
    def egg_drops(self, n_eggs, n_floors, min = 32767):
        memo = [[0 for x in range(n_floors + 1)] for x in range(n_eggs + 1)]

        for egg in range(1, n_eggs+1): # for 1 floor, 1 trial is needed, for 0 floors, 0 trial needed
            memo[egg][1] = 1
            memo[egg][0] = 0
        for floor in range(1, n_floors+1): # for 1 egg, we need n_floors trials
            memo[1][floor] = floor
        # recursion

        for egg in range(2, n_eggs+1):
            for floor in range(2, n_floors+1):
                memo[egg][floor] = min
                for floor_int in range(1, floor):
                    res = 1 + max(memo[egg-1][floor_int-1], memo[egg][floor-floor_int])
                    if res < memo[egg][floor]: memo[egg][floor] = res

        return memo[egg][floor]


Solution36().egg_drops(n_eggs=2, n_floors=10)
Solution36b().egg_drops(n_eggs=2, n_floors=10)

#endregion

#region 39. knapsack problem [**TO REVISE**]
"""
https://www.educative.io/blog/0-1-knapsack-problem-dynamic-solution
https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/
to trace elements: https://codereview.stackexchange.com/questions/125374/solution-to-the-0-1-knapsack-in-python
"""


# METHOD1: using recursion (slow)
# Runtime complexity: O(2^n), due to the number of calls with overlapping subcalls
# Memory Complexity: Constant, O(1)
class Solution39(object):
    def knapsack_rec(self, profits, weights, capacity, curr_item):
        # stop condition
        if curr_item >= len(profits) or capacity <= 0:
            return 0
        weight_curr_item = weights[curr_item]
        if weight_curr_item > capacity: # if weight of nth item bigger than the capacity then we exclude it
            return self.knapsack_rec(profits, weights, capacity, curr_item+1)
        else: # take the best solution between including curr_item or not
            profit_with_curr_item = profits[curr_item] \
                                    + self.knapsack_rec(profits, weights, capacity - weight_curr_item, curr_item + 1)

            profit_wo_curr_item = self.knapsack_rec(profits, weights, capacity, curr_item + 1)
            return max(profit_wo_curr_item, profit_with_curr_item)

    def solve_knapsack(self, profits, weights, capacity):
        return self.knapsack_rec(profits, weights, capacity, curr_item=0)


# METHOD2: using dynamic programming
# Runtime complexity: O(n*capacity)
# Memory Complexity: Constant, O(n*capacity)
class Solution39b(object):
    def solve_knapsack(self, profits, weights, capacity):
        nb_items = len(profits)
        states = [[0 for i in range(capacity + 1)] for j in range(nb_items + 1)]
        # build states table in bottle up manner
        for item in range(nb_items+1):
            for curr_C in range(capacity+1):
                if item <= 0 or curr_C <= 0: states[item][curr_C] = 0
                elif weights[item-1] <= curr_C:
                    profit_with_new_item = profits[item - 1] + states[item - 1][curr_C - weights[item - 1]]
                    if profit_with_new_item > states[item-1][curr_C]:
                        print('item picked', item -1, profits[item-1], weights[item-1])
                        states[item][curr_C] = profit_with_new_item
                    else: states[item][curr_C] = states[item-1][curr_C]
                else: states[item][curr_C] = states[item-1][curr_C]
        print('final state:', states)
        return states[nb_items][curr_C]


# METHOD3: using recursion + memoization technique to remove redundant states
# # uses 2D arrays to store particular states (nb_irems, weights) to avoid computing redundant states
# Runtime complexity: O(n*capacity)
# Memory Complexity: Constant, O(n*capacity)
class Solution39c(object):
    def knapsack_rec(self, profits, weights, capacity, curr_item, states):
        if curr_item <= 0 or capacity <= 0:
            return 0
        if states[curr_item][capacity] != -1:
            pass
        if weights[curr_item-1] <= capacity:
            states[curr_item][capacity] = max(
                profits[curr_item-1] + self.knapsack_rec(profits, weights, capacity-weights[curr_item-1], curr_item-1, states),
                self.knapsack_rec(profits, weights, capacity, curr_item - 1, states))
        else:
            states[curr_item][capacity] = self.knapsack_rec(profits, weights, capacity, curr_item - 1, states)

        print(states)
        return states[curr_item][capacity]

    def solve_knapsack(self, profits, weights, capacity):
        nb_items = len(profits)
        # We initialize the matrix with -1 at first.
        states_init = [[-1 for i in range(capacity + 1)] for j in range(nb_items + 1)]
        return self.knapsack_rec(profits, weights, capacity, curr_item=nb_items, states=states_init)


profits, weights, capacity = [[60, 100, 20], [1, 2, 3], 5]
Solution39().solve_knapsack(profits, weights, capacity)
Solution39b().solve_knapsack(profits, weights, capacity)
Solution39c().solve_knapsack(profits, weights, capacity)



#endregion

#region 42. Print nth number in the Fibonacci series
"""
https://www.geeksforgeeks.org/python-program-for-n-th-fibonacci-number/
F(n) = F(n-1) + F(n-2)
F(0) = 0
F(1) = 1
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
"""

# METHOD1: recursion
# Runtime complexity: O(2^n)
# Memory complexity: O(1)
class Solution42(object):
    def fibonacci(self, n):
        if n< 0:
            pass
        elif n <= 2:
            return n-1
        else:
            return self.fibonacci(n-1) + self.fibonacci(n-2)

# METHOD2: recursion with memoization
# Runtime complexity:
# Memory complexity: O(n)
class Solution42b(object):
    def fibonacci(self, n, FibSequence=[0,1]):
        if n< 0:
            pass
        elif n <= len(FibSequence):
            return FibSequence[n-1]
        else:
            newFibNb = self.fibonacci(n-1, FibSequence) + self.fibonacci(n-2, FibSequence)
            FibSequence.append(newFibNb)
            return newFibNb

#METHOD3: dynamic programming
# Runtime complexity: O(n)
# Memory complexity: O(1)
class Solution42c(object):
    def fibonacci(self, n):
        fib0 = 0
        fib1 = 1
        if n < 0:
            pass
        elif n <= 2:
            return n-1
        else:
            for i in range(2, n):
                fibnew = fib0 + fib1
                fib0 = fib1
                fib1 = fibnew
            return fibnew

Solution42().fibonacci(9)
Solution42b().fibonacci(9)
Solution42c().fibonacci(9)

#endregion

#region Divide and Conquer (binary search of given element in an array)

"""
EARCH ELEMENT IN A SORTED ARRAY
Binary Search works on a divide-and-conquer approach and relies on the fact that the array is sorted to eliminate half of possible candidates in each iteration. 
More specifically, it compares the middle element of the sorted array to the element it's searching for in order to decide where to continue the search.
Runtime complexity: O(logn)
"""


class SolutionBinarySearch(object):
    def binary_search_rec(self, element, array, start=0, end=None):
        if end is None: end = len(array) # initializing
        # recursive loop
        mid = (start + end) //2
        if element == array[mid]:
            return mid
        elif element <= array[mid]:
            return self.binary_search_rec(element, array, start, mid-1)
        else:
            return self.binary_search_rec(element, array, mid+1, end)

element = 18
array = [1, 2, 5, 7, 13, 15, 16, 18, 24, 28, 29]
SolutionBinarySearch().binary_search_rec(element, array)
#endregion

#region Common sorting algos
class SolutionSorting(object):

    # Runtime complexity: O(n2) on average
    def BubbleSort(self, array):
        n = len(array)
        if n < 2: return array  # nothing to do

        for i in range(n):
            already_sorted = True

            for j in range(n-i-1):
                if array[j] > array[j+1]: # swap elements if not sorted
                    array[j], array[j + 1] = array[j + 1], array[j]
                    already_sorted = False

            if already_sorted:
                break

        return array

    # Runtime complexity: O(n2) on average
    def InsertionSort(self, array):
        for i in range(1, len(array)):
            key_item = array[i]
            left_ix = i - 1

            # go through left portion of matrix
            while left_ix >= 0 and array[left_ix] > key_item:
                array[left_ix+1] = array[left_ix] # shift values
                left_ix -=1
            array[left_ix + 1] = key_item

    # Runtime complexity: O(nlogn)
    def QuickSort(self, array):
        if len(array) < 2: return array # nothing to do

        low, same, high = [], [], [] # elements smaller than pivot go to low, bigger in high etc...
        pivot = array[randint(0, len(array)-1)] # select pivot randomly

        for item in array:
            if item < pivot: low.append(item)
            elif item == pivot: same.append(item)
            else: high.append(item)
        return self.QuickSort(low) + same + self.QuickSort(high)


    # Runtime complexity: O(nlogn)
    # like quicksort but uses middle element as pivot
    def MergeSort(self, array):
        if len(array) < 2: return array  # nothing to do
        midpoint = len(array) // 2
        left = self.MergeSort(array[:midpoint])
        right = self.MergeSort(array[midpoint:])
        # TODO merge left and right
        return 0

array_to_sort = [randint(0, 1000) for i in range(42)]
SolutionSorting().QuickSort(array)

#endregion

#region HeapSort TODO
"""
https://www.geeksforgeeks.org/heap-sort/
A Binary Heap is a Complete Binary Tree where items are stored in a special order 
such that the value in a parent node is greater(or smaller) than the values in its two children nodes. 
The former is called max heap and the latter is called min-heap.

Runtime complexity: O(nlogn) (complexity of heapify: O(logn)
Memory complexity: 
"""
#endregion

#region Simple Negex
"""
find mentions of suicide in text and check if they are negated or not"""
target = 'suicid'
context_words = 5
context_chars=20
neg_terms = ['no', "n't", 'deni', 'deny']
string = 'he patient denies having suicidal thoughts. This was not an intentional overdose. ' \
         'She has been suicidal in the past. Suicidal ideation was not intentional. ' \
         'blabla random sentence???? ' \
         'another random stuff'


class SolutionNegex(object):
    # WORKING ON WORDS
    def negex_words(self, string, target, neg_terms=['no', "n't"], context_words=5):
        sentences = [snt.lower().split() for snt in re.split('[.,!?;]', string)]
        neg=[]
        for word_list in sentences:
            idx_lst = [i for i, word in enumerate(word_list) if target in word] # find if suicide words used
            if idx_lst == []:
                neg.append('no_word')
            else:
                for idx in idx_lst:
                    context_list = word_list[max(0,idx-context_words):min(len(word_list), idx+context_words)]
                    context_str = ''.join(context_list)
                    neg.append(max([neg in context_str for neg in neg_terms]))
                    print(context_str, neg)
        return [sentences, neg]

    #WORKING ON STRINGS
    def negex_chars(self, string, target, neg_terms=['no', "n't"], context_chars=20):
        idx_lst = [i for i in range(len(string)) if string.lower().startswith(target, i)]
        print(idx_lst)
        res=[]
        for ix in idx_lst:
            context_str = string[max(0,ix-context_chars):min(len(string),ix+context_chars+len(target))]
            neg = max([neg in context_str for neg in neg_terms])
            print(context_str, neg)
            res.append([context_str, neg])
        return 0

SolutionNegex().negex_words(string, target, neg_terms = ['no', "n't", 'deni', 'deny'])
SolutionNegex().negex_chars(string, target, neg_terms = ['no', "n't", 'deni', 'deny'])

#endregion

#region FizzBuzz
def fizz_buzz(num):
    string = ''
    if num % 3 == 0: string +='Fizz'
    if num % 5 == 0: string +='Buzz'
    return string if string else num


if __name__ == "__main__":
    for n in range(1, 5):
        print(fizz_buzz(n))

def fizzBuzz(n):
    # Write your code here
    for i in range(1, n+1):
        print("Fizz"*(i%3==0) + "Buzz"*(i%5==0) or i)

if __name__ == '__main__':
    n = int(input().strip())

    fizzBuzz(n)
#endregion

#region bribes in queue - NY Chaos
def minimumBribes(q):
    orig_queue=list(range(1,len(q)+1))
    diff = [o_i - i for o_i, i in zip(orig_queue, q)]
    #print(diff)
    if max(diff) <=-2:
        print('Too chaotic')
    else:
        print(-(sum(x for x in diff if x < 0)))

# optimized
def minimumBribesb(q):
    # Write your code here
    bribes = 0
    for i in range(len(q)-1,-1,-1):
        if q[i] - (i + 1) > 2:
            print('Too chaotic')
            return
        for j in range(max(0, q[i] - 2),i):
            if q[j] > q[i]:
                bribes += 1
    print(bribes)

q = [1,2,3,5,4,6,7,8]
q = [4,1,2,3,5,6,7,8]
q = [2, 1, 5, 3, 4]

#endregion

#region minimum swaps to order a list
def minimumSwaps(arr):
    swaps = 0
    n = len(arr)

    for idx in range(n):
        while arr[idx] - 1 != idx:  # check if element in its right place
            ele = arr[idx]  # this is the misplaced element
            arr[ele - 1], arr[idx] = arr[idx], arr[ele - 1]  # we swap it back where it belongs
            swaps += 1  # and increase the swap counter
    return swaps
minimumSwaps([1,2,3,5,4,6,7,8])
#endregion

#region hourglass array TODO
arr = [[1, 2, 3, 0, 0],
       [0, 0, 0, 0, 0],
       [2, 1, 4, 0, 0],
       [0, 0, 0, 0, 0],
       [1, 1, 0, 1, 0]]
#endregion

#region Ransom pb (find if string could have been written using another string)
note = 'hello i am zelda'.split()
magazine = 'hello my name is zelda and i am cool and awesome'.split()

def checkMagazine0(magazine, note):
    for word in note:
        if word not in magazine:
            print('No')
            return
        else:
            magazine.remove(word)
    print('Yes')

# faster
def checkMagazine(magazine, note):
    if Counter(note) - Counter(magazine) == {}:
        print('Yes')
    else:
        print('No')

checkMagazine(magazine, note)
#endregion

#region Count anagrams in string
def get_all_substrings(input_string):
    length = len(input_string)
    return [input_string[i:j + 1] for i in range(length) for j in range(i, length)]

def get_unordered_anagram_count(string):
    allsubs = get_all_substrings(string)
    subsorted = [sorted(x) for x in allsubs]
    subsorted.sort()
    print(subsorted)
    count = 0;
    curr_count = 0;
    for j in range(len(subsorted) - 1):
        if subsorted[j] == subsorted[j + 1]:
            curr_count += 1
            count += curr_count
        else:
            curr_count = 0

    return count

get_unordered_anagram_count('mom')


from collections import Counter

def sherlockAndAnagrams(s):
    all_subs = []
    # get all possible substrings of s
    # for anagrams, order doesn't matter so we will sort all substrings
    # if 2 substrings are anagrams they will now be the same
    for i in range(1,len(s)):
        for j in range(0,len(s)-i+1):
            sub = s[j:j+i]
            all_subs.append(''.join(sorted(sub)))  #to sort substring alphabetically

    # now we count how many times appears each substring. if >1 then it has an anagram
    count = Counter(all_subs)
    count_ana = {k:v for k,v in count.items() if v>1}
    # if a substring appears v times, then we can make 1+2+...+v-1 anagrams out of it
    # 1+2+...+v-1 = sum(range(v))
    return sum(sum(range(v)) for v in count.values())

s = 'abba'
sherlockAndAnagrams(s)
#endregion

#region Count geometric progression triplets in list
"""
You are given an array and you need to find number of tripets of indices  
such that the elements at those indices are in geometric progression of ratio r
https://www.geeksforgeeks.org/number-gp-geometric-progression-subsequences-size-3/
"""
from collections import defaultdict
def countTriplets(arr, r):
    res = 0
    # keep track of left and right elements
    left = defaultdict(lambda:0) #to store arra[elem]/r
    right = defaultdict(lambda:0) #to store arra[elem]*r

    # count the nb f occurences of each element present in the arrays
    for elem in arr:
        right[elem] +=1

    for elem in arr:
        cl, cr = 0, 0  # initialize counters
        if elem % r == 0: # if divisible by ratio
            cl = left[elem//r]  # count elements in left hash
        right[elem] -= 1  # remove from right hash
        left[elem] += 1  # increase count in left hash
        cr = right[elem*r]  # count candidate elements in right hash

        res += cl * cr
    return res

arr = [3, 1, 2, 6, 2, 3, 6, 9, 18, 3, 9]
countTriplets(arr, r=3)
#endregion

#region Common element in 2 strings?
# METHOD 1 This is too slow
def twoStrings(s1, s2):
    # Write your code here
    l1 = len(s1)
    l2 = len(s2)

    for i in range(l1):
        for j in range(i + 1, l1 + 1):
            stem = s1[i:j]
            if stem in s2:
                return "YES"
    return "NO"
# METHOD 2 using set
def twoStrings(s1, s2):
    set1 = set(s1) #converting string to set
    set2 = set(s2)
    if set.intersection(set1,set2):
        return "YES"
    else:
        return "NO"
#endregion

#region Parse queries

def freqQuery(queries):
    res = []
    cnt = dict()
    freq = defaultdict(int)

    for x in queries:
        ops, value = x
        initial = cnt.get(value, 0)

        if ops == 3:
                res.append(1 if freq.get(value) else 0)
        elif ops == 1: # insert the element
              freq[initial] -= 1
              cnt[value] = initial + 1
              freq[cnt.get(value,0)] += 1
        else: # remove 1 occurence of the element if exists
                freq[initial] -= 1
                if initial: cnt[value] -= 1
                freq[cnt.get(value,0)] += 1
        print('cnt', cnt, 'freq', freq)
    return res
queries=[[1,1],[2,2],[3,2],[1,1],[1,1],[2,1],[3,2]]
#endregion

#region Fraud detection

# SUPER SLOW
def med(expenditure, d):
    lastd = sorted(expenditure[:d])
    return lastd[d//2] if d % 2 == 1 else ((lastd[d//2] + lastd[d//2-1])/2)

def activityNotifications(expenditure, d):
    # Write your code here
    notif = 0
    for i in range(d, len(expenditure)-1):
        if expenditure[i] >= 2*med(expenditure, i):
            notif += 1
    return notif
res = activityNotifications(expenditure=[2, 3, 4, 2, 3 ,6 ,8 ,4, 5], d=5)
res = activityNotifications(expenditure=[10,20,30,40,50], d=3)

# FASTER
# https://www.martinkysel.com/hackerrank-fraudulent-activity-notifications-solution/
# https://shareablecode.com/snippets/python-solution-for-hackerrank-problem-fraudulent-activity-notifications-FuHR-WX84
from bisect import bisect_left, insort_left
def activityNotifications(expenditure, d):
    # Write your code here
    notif = 0
    lastd = sorted(expenditure[:d]) # sort expenditures for first d-days window
    for day, exp in enumerate(expenditure):
        if day < d: continue
        # compute the median
        median = lastd[d//2] if d % 2 == 1 else ((lastd[d//2] + lastd[d//2-1])/2)
        if exp >= median*2:
            notif +=1
        # remove previous element and add new element in median window
        del lastd[bisect_left(lastd, expenditure[day - d])]
        insort_left(lastd, exp)

    return notif

#endregion

#region Lily's homework
# https://www.martinkysel.com/hackerrank-lilys-homework-solution/
# Problem reformulation:
# The sum is minimal if the array is sorted
# So we need to count the number of swaps to sort the array (both ascending and descending)

def cntSwaps(arr):
    pos = sorted(list(enumerate(arr)), key=lambda e: e[1]) # sort the array and key indices
    swaps = 0
    for idx in range(len(arr)):
        while True: #loop until everything is at the right place
            if pos[idx][0] == idx: # already at the right position, exit the loop
                break
            else:
                swaps += 1
                swapped_idx = pos[idx][0]
                pos[idx], pos[swapped_idx] = pos[swapped_idx], pos[idx]
    return swaps

def lilysHomework(arr):
    # Write your code here
    # Run the count of swaps on both ascending and descending arrays
    return min(cntSwaps(arr), cntSwaps(arr[::-1]))

arr = [2, 5, 3, 1]
#endregion

#region how many elements t delete to make anagrams?
def makeAnagram(a, b):
    cnt_a = Counter(a)
    cnt_b = Counter(b)
    delete_from_a = cnt_a - cnt_b
    delete_from_b = cnt_b - cnt_a
    deletions = delete_from_a + delete_from_b
    return len(list(deletions.elements()))
#endregion

#region longest common substring
def longestSubstring(s1, s2):
    # Write your code here
    answer = ""
    len1, len2 = len(s1), len(s2)
    for i in range(len1):
        match = ""
        for j in range(len2):
            if (i + j < len1 and s1[i + j] == s2[j]):
                match += s2[j]
            else:
                if (len(match) > len(answer)): answer = match
                match = ""
    return len(answer)
#endregion

#region common child between strings (longest common substring if deleting elems allowed
# NOT THE FASTEST
def commonChild(s1, s2):
    # Write your code here
    l1, l2 = len(s1), len(s2)
    lengths = [[0 for j in range(l2 + 1)] for i in range(l1 + 1)]
    for i1, el1 in enumerate(s1):
        for i2, el2 in enumerate(s2):
            if el1 == el2:
                lengths[i1 + 1][i2 + 1] = lengths[i1][i2] + 1
            else:
                lengths[i1 + 1][i2 + 1] = \
                    max(lengths[i1 + 1][i2], lengths[i1][i2 + 1])

    return lengths[-1][-1]

#endregion

#region Count deletions needed to remove matching adjacent characters in string
def alternatingCharacters(s):
    # Write your code here
    i=0
    num_del=0
    while i < len(s)-1:
        if s[i]==s[i+1]: # while consecutive elements are equal
            s = s[0 : i : ] + s[i + 1 : :] # remove second element
            num_del+=1 #increment number of deletions
        else: # otherwise move to next element
            i+=1
    return num_del
#endregion

#region Sherlock valid string
#https://www.martinkysel.com/hackerrank-sherlock-and-valid-string-solution/
def isValid(s):
    char_map = Counter(s)
    char_occurence_map = Counter(char_map.values())

    if len(char_occurence_map) == 1:
        return "YES"

    if len(char_occurence_map) == 2:
        k1, k2 = char_occurence_map.keys()
        v1, v2 = char_occurence_map.values()

        # there is exactly 1 extra symbol and it can be deleted
        if (k1 == 1 and v1 == 1) or (k2 == 1 and v2 == 1):
            return "YES"

        # the is exactly 1 symbol that occurs an extra 1 time
        if (k1 == k2 + 1 and v1 == 1) or (k2 == k1 + 1 and v2 == 1):
            return "YES"
    return "NO"
#endregion

#region All substrings of string
def subString(s):
    n = len(s)
    res = []
    for i in range(n):
        for _len in range(i + 1, n + 1):
            res.append(s[i: _len])
            print(s[i: _len])
s = "abcd"
subString(s)
#endregion

#region All Special substrings of string
"""
Special string defined by
All of the characters are the same, e.g. aaa. (case A)
All characters except the middle one are the same, e.g. aadaa (case B)
"""
# METHOD1
def substrCount(s):
    count = n = len(s)
    for i, char in enumerate(s):
        diff_char_idx = None
        for j in range(i + 1, n):
            if char == s[j]:
                if diff_char_idx is None:
                    count += 1
                elif j - diff_char_idx == diff_char_idx - i:
                    count += 1
                    break
            else:
                if diff_char_idx is None:
                    diff_char_idx = j
                else:
                    break
    return count
#METHOD2 using groupby

def substrCount(s):
    n = len(s)
    cpt_a = cpt_b = 0

    # count the number of cases A
    a_cases = [len(list(g)) for k, g in groupby(s)]
    cpt_a = sum([i*(i+1)//2 for i in a_cases])

    # count the number of cases B
    for i in range(1, n-1):
        skip = 1
        if s[i-skip] == s[i] or s[i+skip] == s[i]:
            continue # already counted in case A, move back to top of lfor oop
        match = s[i-skip]
        while i-skip > -1 and i+skip < n and s[i-skip] == s[i+skip] == match:
            cpt_b += 1
            skip += 1
    return cpt_a+cpt_b

substrCount('mnonopoo')
#endregion

#region Luck balance
"""
Lena passes a series of test of [Luck_i, Importance_i]
Initially, her luck balance is 0.
For each test she fails, ker luck increases by L[i] to her balance
She can fail at max k important tests
If Lena fails k important contests, what is the max amount of luck she can have
"""
contests = [[5, 1], [2, 1], [1, 1], [8, 1], [10, 0], [5, 0]]
nb_lost_ones = k = 3
contests = [[13, 1],[10, 1],[9, 1],[8, 1],[13 ,1],[12, 1],[18, 1],[13, 1]]
nb_lost_ones = k = 5
def luckBalance(nb_lost_ones, contests):
    zeros = sum([el[0] for el in contests if el[1]==0])
    ones = [el for el in contests if el[1]==1]
    ones.sort(reverse=True)
    ones_won = sum([el[0] for el in ones[nb_lost_ones:][:]])
    ones_lost = sum([el[0] for el in ones[:nb_lost_ones][:]])
    return zeros+ones_lost-ones_won

#endregion

#region Greedy florist
"""
Group of k people want to buy all flowers in shop
After the flower bought, the price of each following flower increases by i 
(price of nth bough flower = n*price)
"""
costs = [1, 3, 5, 7, 9, 10, 10]
k=3
# expected answer: 10+10+9 + 2*7 + 2*5 + 2*3 + 3*1 = 62
def getMinimumCost(k, costs):
    total_cost=0
    incr=1
    costs.sort(reverse=True)
    total_cost += sum(costs[0:k])
    for idx, c in enumerate(costs[k:]):
        if idx % k == 0: incr += 1
        print(idx, incr,c)
        total_cost += incr*c
    return total_cost
#endregion

#region Min Max array
"""
You are given a list of integers arr and a single integer n.
You must create an array of length n from arr such that the difference between its max and min is minimum
"""
arr = [1,2,3,4,10,20,30,40,100,200]
k=4
# solution: selecting [1,2,3,4] gives max diff=3

arr = [100,200,300,350,400,401,402]
k=3
# solution: selecting [400,401,402] gives max diff=2
def maxMin(k, arr):
    arr.sort()
    min_diff=arr[len(arr)-1]-arr[0]
    for idx in range(len(arr)-k+1):
        curr_diff = arr[idx+k-1]-arr[idx]
        print(idx, arr[idx:idx+k], curr_diff)
        if curr_diff<min_diff:
            min_diff=curr_diff
    return min_diff
#endregion

#region Icecream parlor
"""
Each time Sunny and Johnny take a trip to the Ice Cream Parlor, 
they pool their money to buy ice cream. 
On any given day, the parlor offers a line of flavors, each flavor has a cost c.
Help Sunny and Johnny choose 2 distinct flavors such that they spend their entire pool of money during each visit.
Note: there is always a unique solution
Output: int int: the indices of the two flavors they will purchase as two space-separated integers on a line
"""
cost, money = [2,1,3,5,6], 5  # exp output [1, 3] (2+3)
cost, money = [1, 4, 5, 3, 2], 4 # exp output [1,4]
cost, money = [2, 2, 4, 3], 4 # exp output [1,2]

sorted_costs = dict(sorted(enumerate(cost), key=lambda e: e[1])) # need to swap keys, values
costs_dic = {k: v for v, k in enumerate(cost)} # not good because removes duplicates

def whatFlavors(cost, money):
    saved_values = {}
    for counter, value in enumerate(cost):
        print(saved_values)
        if money-value in saved_values:
            print(saved_values[money-value] + 1, counter + 1)
        elif value not in saved_values:
            saved_values[value] = counter

#endregion

#region Candies
"""
a teacher gives candies to her students.
if student has bigger score than their neighbor, they must receive more candies
when two children have equal score, they can have different number of candies.
minimum amount of candies?
"""
arr = [4,2,6,1,7,8,9,2,1]
# candies distributed: [1,2,1,2,1,2,3,4,2,1] total=18
def candies(arr):
    n=len(arr)
    candies = [1]*n
    for i in range(n-1):
        if arr[i+1]>arr[i]:
            candies[i+1] = candies[i]+1
        print("first pass", candies)
    for i in range(n-1,0,-1):
        if arr[i-1]>arr[i] and candies[i-1]<=candies[i]:
            candies[i-1] = candies[i]+1
        print("second pass", candies)
    return sum(candies)





#endregion
