# Overview
*It turns out that, along side being pretty intuitive (in my opinion), this solution works really well.*


<!-- Describe your first thoughts on how to solve this problem. -->
**Problem Statement.** Recall that we are given a stock's price over $n$ consecutive days (in array $P$) and the goal is to find the maximum profit when allowed **at most two** disjoint transactions. 

$\text{Goal}: \text{maxProfit}_2(P[1 \dots n])$

**Key Idea.** Essentially, dynamic programming. Split the $n$ prices into $\text{prefix}_i := P[1 \dots i]$ and $\text{suffix}_i := P[i+1 \dots n]$) for $i \in \{0, 1, \dots, n\}$.

$\text{Split}: \underbrace{P[1],P[2], \dots, P[i]}_{\text{prefix}_i = P[1 \dots i]}, \underbrace{P[i+1], \dots, P[n]}_{\text{suffix}_i = P[i+1 \dots n]} ~~~ \text{for} ~~~ i \in \{0, 1, \dots n\}$

Now, we only need to solve the 'easier' **at most one** transaction variant ($\text{maxProfit}_1$) of the problem on $\text{prefix}_i$ and $\text{suffix}_i$ for every $i$ instead. 

**Reason.** For any split $i$, the maximum profit using **at most two** transactions where no transaction crosses the split at $i$ is given by the sum of $\text{maxProfit}_1(\text{prefix}_i)$ and $\text{maxProfit}_1(\text{suffix}_i)$. Therefore, $\text{maxProfit}_2(P[1 \dots n])$ is found by finding the split $i$ that maximises the sum.

$\text{Goal}: \text{maxProfit}_2(P[1 \dots n]) = \max_{i \in \{0, 1, \dots, n\}} \{ \text{maxProfit}_1(\text{prefix}_i) + \text{maxProfit}_1(\text{suffix}_i)\}$

**Sanity Check.** Consider any array $P[1 \dots n]$ of prices where the (at most) two optimal transactions are the buy-sell pairs $(j_1,k_1)$ and $(j_2, k_2)$ such that $1 \leq j_1 < k_1 < j_2 < k_2 \leq n$, i.e., $(j,k)$ means buy on day $j$ and sell on day $k$. Any split $i^* \in \{0,1 ,\dots n\}$ such that $k_1 \leq i^* < j_2$ (at least one must exist) will definitely be such that $\text{maxProfit}_2(P[1 \dots n]) = \text{maxProfit}_1(\text{prefix}_i^*) + \text{maxProfit}_1(\text{suffix}_i^*)$. Hence, this strategy finds the write answer.


# Bottleneck
<!-- Describe your approach to solving the problem. -->
The difficulty in implementing this solution comes from finding the maximum profit using at most a single transaction for $\text{prefix}_i$ and $\text{suffix}_i$ for every $i \in \{0,1, \dots, n\}$. Solving this for a single sequence of prices is exactly the problem  [Best Time to Buy And Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/), but we need to solve it efficiently for all prefixes and suffixes of $P$.

**Brute Force.** This can be done for a sequence (prefix or suffix) of $N$ prices by trying all possible buy-sell pairs, i.e., $\binom{N}{2} = \Theta(N^2)$ different pairs. This is, however, computationally expensive.

**Efficient Solution.** Again, dynamic programming. Imagine the prices are revealed in sequence one-by-one. Whenever price $P[i]$ is revealed, the goal is to compute $\text{maxProfit}_1(\text{prefix}_i)$ given $\text{maxProfit}_1(\text{prefix}_{i-1})$ is known. This is not possible since if price $P[i]$ is part of an optimal transaction, it must be the selling price and thus its corresponding buying price must be the minimum price in $\text{prefix}_{i-1}$. Hence, we resolve this by also tracking exactly this, i.e., when price $P[i]$ is revealed 

**Benefit.** This strategy allows us to compute, using **a single pass** of the prices $P$, the maximum profit with a single transaction $\text{maxProfit}_1(\text{prefix}_i)$ for every $i \in \{0,1, \dots n\}$. Similarly, using **a single pass** of $P$ in the **reverse order**, $\text{maxProfit}_1(\text{suffix}_i)$ can be efficiently computed for every $i \in \{0,1, \dots n\}$. Note that, in my code, I have done this in parallel using a single for loop.





# Complexity
- Time complexity:
<!-- Add your time complexity here, e.g. $$O(n)$$ -->
The bottlenecks in speed come from 
1. the pass of the array $P$, which requires $O(n)$ time; and 
2. summing the max profit (using a single transaction) of $\text{prefix}_i$ and $\text{suffix}_i$ for each $i \in \{0,1, \dots n\}$, which also requires $O(n)$ time.

This totals $O(n)$ time.

- Space complexity:
<!-- Add your space complexity here, e.g. $$O(n)$$ -->
The bottleneck in space comes from the need to store $\text{maxProfit}_1(\text{prefix}_i)$ and $\text{maxProfit}_1(\text{suffix}_i)$ for each $i \in \{0,1, \dots n\}$ in arrays of length $n + 1$ each.

This totals $O(n)$ words of space (assuming the stock's price in each day can be represented with $O(1)$ words).

# Code
```python3 []
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        n = len(prices)

        ### Step 1: Create an array storing the max profit in 
        # all prefixes and suffixes of the list of prices.

        # Initialise the arrays with a 0 entry to represent the 
        # empty (considering no days) prefix and suffix respectively
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