# https://www.educative.io/blog/crack-amazon-coding-interview-questions#overview
import timeit
import collections
import math

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
https://www.geeksforgeeks.org/coin-change-dp-7/

Input:
  - Amount: 5
  - Coins: [1, 2, 5]
Output: 4
Explanation: There are 4 ways to make up the amount:
  - 5 = 5
  - 5 = 2 + 2 + 1
  - 5 = 2 + 1 + 1 + 1
  - 5 = 1 + 1 + 1 + 1 + 1
  
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

#region 17 Convert a Binary tree to a Doubly Linked List TODO
#endregion

#region 21. Implement a stack with push(), min(), and pop() in O(1)O(1) time TODO
#endregion

#region 25 Implement a queue using a linked list TODO
#endregion

#region 36. egg dropping puzzle for dynamic programming TODO
#endregion

#region 39. knapsack problem TODO
"""
https://www.educative.io/blog/0-1-knapsack-problem-dynamic-solution
"""
#endregion

#region 42. Print nth number in the Fibonacci series TODO
#endregion

