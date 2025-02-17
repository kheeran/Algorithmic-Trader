import math 
import itertools
import matplotlib.pyplot as plt

def func(n):
    myn = n
    c = 0
    i = 0
    while n >= 0:
        i += 1
        n = n-2
        c = c + (n - 2)
    # print(i == math.ceil(myn/2))
    return c

xs = list(range(-50, 150+1))
ys = [ func(x) for x in xs ]

plt.plot(xs,ys)
plt.show()

xs = list(range(-10, 10+1))
ys = [ func(x) for x in xs ]

plt.plot(xs,ys)
plt.show()

def isTrue(d,r):
    if d == 'O':
        return True
    if d == 'A':
        if r == 'A' or r == 'AB':
            return True
    if d == 'B':
        if r == 'B' or r == 'AB':
            return True
    if d == 'AB':
        if r == 'AB':
            return True
    return False

ds = ['A', 'B', 'O']
rs = ['A', 'B', 'AB']

trueCount = 0
for dperm in itertools.permutations(ds):
    for rperm in itertools.permutations(rs):
        pairs = zip(dperm,rperm)
        allTrue = True
        for pair in pairs:
            if not isTrue(*pair):
                allTrue = False
        if allTrue:
            trueCount += 1

print("Prob all true:", trueCount/36)

