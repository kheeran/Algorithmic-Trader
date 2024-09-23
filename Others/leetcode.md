# Overview

Recall that we are given a stock's price over $n$ consecutive days (in array $P$) and the goal is to find the maximum profit when allowed **at most two** disjoint transactions. 

$\textbf{Goal}: \text{maxProfit}_2(P)$

where the subscript represents the limit on the number of transactions allowed.

The key technique required to solve this problem efficiently is to use **Dynamic Programming** (DP), but applied **twice**. As a quick reminder, DP is the process of breaking down the problem into smaller and simpler subproblems that can be efficiently recombined to solve the original problem.

# Approach

To gain a good intuition of the problem, we first consider a *brute force* approach, which requires $O(n^4)$ time. Then, the *efficient algorithm* is obtained by applying two different DP steps, which brings the time complexity down to $O(n)$ time!

## Brute Force

Each transaction can be represented as a buy-sell pair $(i,j)$ where $i < j \in [n] := \{1, 2, \dots n\}$. 
This means  buying on day $i$ and selling on day $j$ to obtain a (potentially negative) profit of $P[j] - P[i]$.
We also have that any two disjoint transactions $(i_1, j_1), (i_2, j_2)$ must be such that $1 \leq i_1 < j_1 < i_2 < j_2 \leq n$.

Now, since we are allowed **at most** two transactions to maximise profits, an optimal solution can consist of either 0, 1 or 2 disjoint transactions. By counting the number of these possibilities, we have that
- no transactions requires picking 0 indices (or days) from $[n]$ $\implies$ $\binom{n}{0} = 1$ choice;
- a single transaction requires picking 2 indices from $[n]$ $\implies$ $\binom{n}{2} = \Theta(n^2)$ choices; and
- exactly two transactions requires picking 4 indices from $[n]$ $\implies$ $\binom{n}{4} = \Theta(n^4)$ choices.

Therefore, a brute force approach would simply enumerate all of these possibilities and compute the profit in $O(1)$ time for each one, requiring $O(n^4)$ time in total. Then, the one that obtains the largest profit, which requires $O(n^4)$ time to find, solves the problem.

## Efficient Algorithm

The first DP step splits the problem into *well-structured* sets of smaller subarrays, where knowing the solution to the **at most one** transaction variant of the problem ($\text{maxProfit}_1$) in each subarray is enough to solve the main problem ($\text{maxProfit}_2$). However, Computing $\text{maxProfit}_1$ using a brute force approach still requires $O(N^2)$ time where $N \leq n$ is the size of the subarray, which is still computationally expensive. Hence, the second DP step shows how to use the structure of the sets to elegantly compute this for *all* the required subarrays in just $O(n)$ time.

### DP Step 1

Split the array $P$ of $n$ prices into the subarrays $\text{prefix}_s := P[1 \dots s]$ and $\text{suffix}_s := P[s+1 \dots n]$ for $s \in [n]$.

$\textbf{Split}: \underbrace{P[1],P[2], \dots, P[s]}_{\text{prefix}_s = P[1 \dots s]}, \underbrace{P[s+1], \dots, P[n]}_{\text{suffix}_s = P[s+1 \dots n]} ~~~~ \text{for} ~~~ s \in [n]$

Now, for any split $s$, the maximum profit using **at most two** transactions where the transactions *do not* cross the split at $s$ is given by the sum of the maximum profits achievable using **at most one** transaction in $\text{prefix}_s$ and $\text{suffix}_s$. Therefore, $\text{maxProfit}_2(P)$ is found by finding the split $s$ that maximises this sum.

$\textbf{Goal}: \text{maxProfit}_2(P) = \max_{s \in [n]} \{ \text{maxProfit}_1(\text{prefix}_s) + \text{maxProfit}_1(\text{suffix}_s)\}$

**Sanity Check.** Consider any array $P$ of $n$ prices where the two optimal transactions are the buy-sell pairs $(i_1,j_1)$ and $(i_2, j_2)$ such that $1 \leq i_1 < j_1 < i_2 < j_2 \leq n$. Any split $s^* \in [n]$ such that $j_1 \leq s^* < i_2$ (at least one must exist) will definitely be such that $\text{maxProfit}_2(P) = \text{maxProfit}_1(\text{prefix}_s^*) + \text{maxProfit}_1(\text{suffix}_s^*)$. Hence, this strategy finds the write answer. Note that if the optimal transaction has a single buy-sell pair, set $i_2, j_2 = \infty$, and if it has no buy-sell pairs, further set $i_1, j_1 = - \infty$.

### DP Step 2


# Bottleneck
<!-- Describe your approach to solving the problem. -->
The difficulty in implementing this solution comes from finding the maximum profit using at most a single transaction for $\text{prefix}_i$ and $\text{suffix}_i$ for every $i \in \{0,1, \dots, n\}$. Solving this for a single sequence of prices is exactly the problem  [Best Time to Buy And Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/), but we need to solve it efficiently for all prefixes and suffixes of $P$.

**Brute Force.** This can be done for a sequence (prefix or suffix) of $N$ prices by trying all possible buy-sell pairs, i.e., $\binom{N}{2} = \Theta(N^2)$ different pairs. This is, however, computationally expensive.

**Efficient Solution.** Again, dynamic programming. Imagine the prices are revealed in sequence one-by-one. Whenever price $P[i]$ is revealed, the goal is to compute $\text{maxProfit}_1(\text{prefix}_i)$ given $\text{maxProfit}_1(\text{prefix}_{i-1})$ is known. This is not possible since if price $P[i]$ is part of an optimal transaction, it must be the selling price and thus its corresponding buying price must be the minimum price in $\text{prefix}_{i-1}$. Hence, we resolve this by also tracking exactly this, i.e., when price $P[i]$ is revealed 

**Benefit.** This strategy allows us to compute, using **a single pass** of the prices $P$, the maximum profit with a single transaction $\text{maxProfit}_1(\text{prefix}_i)$ for every $i \in \{0,1, \dots n\}$. Similarly, using **a single pass** of $P$ in the **reverse order**, $\text{maxProfit}_1(\text{suffix}_i)$ can be efficiently computed for every $i \in \{0,1, \dots n\}$. Note that, in my code, I have done this in parallel using a single for loop.





# Complexity
## Time complexity:
<!-- Add your time complexity here, e.g. $$O(n)$$ -->
The bottlenecks in speed come from 
1. the single pass of the array $P$ to compute $\text{prefix}_s$ (and $\text{suffix}_s$) for every $s \in [n]$, which requires $O(n)$ time; and 
2. summing the maximum profit using at most a single transaction of $\text{prefix}_s$ and $\text{suffix}_s$ for each $s \in [n]$, which also requires $O(n)$ time.

This totals $O(n)$ time.

## Space complexity:
<!-- Add your space complexity here, e.g. $$O(n)$$ -->
The bottleneck in space comes from the need to store $\text{maxProfit}_1(\text{prefix}_s)$ and $\text{maxProfit}_1(\text{suffix}_s)$ for each $s \in [n]$ in arrays of length $n$ each.

This totals $O(n)$ words of space (assuming the stock's price in each day can be represented with $O(1)$ words).

# Code
```python3 []
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        n = len(prices)

        ### Step 1: Create an array storing the max profit in 
        # all prefixes and suffixes of the list of prices.

        # Initialise the arrays with a 0 entry to represent the 
        # empty (considering no days) prefix and suffix respectively,
        # which makes the indexing simpler.
        profit_prefixes = [0]
        profit_suffixes_rev = [0] # this array will be in reverse 
        # (since it is faster to append to a list than add to 
        # a specific position of a list)

        # the idea is to compute the at most one transaction variant
        # (version I) of this problem forwards and also backwards.

        # initialisation
        min_buy_pre, max_sell_suf = prices[0], prices[-1]
        max_profit_pre, max_profit_suf = 0, 0

        # O(n) bottleneck
        for i in range(n):

            # prefixes (using i)
            if prices[i] < min_buy_pre:
                min_buy_pre = prices[i]

            cur_profit = prices[i] - min_buy_pre # compute the current profit

            if cur_profit > max_profit_pre:
                max_profit_pre = cur_profit

            # store max profit of current prefix
            profit_prefixes.append(max_profit_pre)

            
            # suffixes (using j)
            j = n - 1 - i

            if prices[j] > max_sell_suf:
                max_sell_suf = prices[j]

            cur_profit = max_sell_suf - prices[j]  # compute current profit

            if cur_profit > max_profit_suf:
                max_profit_suf = cur_profit

            # store max profit of current suffix
            profit_suffixes_rev.append(max_profit_suf)


        ### Step 2: compute the max profit with 2 transactions 
        # by considering every possible split of the days
        #  into two groups, i.e., days 1 to k and days k+1 
        # to n for all k in {0, 1, ..., n}.

        max_profit2 = 0

        # O(n) bottleneck
        for i in range(n+1):
            
            # compute the max profit of at most two transactions 
            # at the split occuring after the ith day
            cur_max_profit = profit_prefixes[i] + profit_suffixes_rev[-1 - i]

            if cur_max_profit > max_profit2:
                max_profit2 = cur_max_profit
       
        return max_profit2
```