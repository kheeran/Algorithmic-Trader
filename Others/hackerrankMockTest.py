#!/bin/python3

import math
import os
import random
import re
import sys



#
# Complete the 'palindromeIndex' function below.
#
# The function is expected to return an INTEGER.
# The function accepts STRING s as parameter.
#

def palindromeIndex(s):
    # Write your code here
    n = len(s)
    if len(s) == 1:
        return -1
    else:
        index_first = -1
        index_last = -1
        ii = 0
        jj = 0
        for i in range(int(n/2)):
            cur_i = i + ii
            cur_j = n - 1 - i - jj
            
            first = s[cur_i]
            last = s[cur_j]
            
            if first != last:
                if index_first == -1 and index_last == -1:
                    if first == s[cur_j - 1]:
                        index_last = cur_j
                        jj = 1
                    elif last == s[cur_i + 1]:
                        index_first = cur_i
                        ii = 1
                    else:
                        return -1
                else:
                    return -1
            
        if index_first != -1:
            return index_first
        elif index_last != -1:
            return index_last
        else:
            return -1
            

s = 'hgygsvlfcwnswtuhmyaljkqlqjjqlqkjlaymhutwsnwcflvsgygh'
pindex = palindromeIndex(s)
print(pindex)
print(s[:pindex] + s[pindex + 1:])