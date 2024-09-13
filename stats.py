import random

## Plug-in estimates
def plugin_mean(xs):
    n = len(xs)
    sum = 0
    for x in xs:
        sum += x
    return sum/n

def plugin_var(xs):
    n = len(xs)
    mu = plugin_mean(xs)
    sum = 0
    for x in xs:
        sum += (x - mu)**2
    return sum/n

def plugin_cdf(y, xs):
    n = len(xs)
    count = 0
    for x in xs:
        if x < y:
            count += 1
    return count/n

def plugin_surv(y, xs):
    return 1 - plugin_cdf(y,xs)


# Qs and random variables

# Q1: Roll a fair 100-sided die three times. What is the expectation and variance of the minimum value of the three rolls?

def roll_die():
    return random.randint(1,100)

if __name__ == '__main__':

    # Q1
    xs = [min(roll_die(), roll_die(), roll_die()) for _ in range(1000)]

    # plugin
    mu = plugin_mean(xs)
    var = plugin_var(xs)

    mu2=0
    var2 = 0
    
    for y in range(100):
        Sy = plugin_surv(y, xs)
        mu2 += Sy
        var2 += 2 * y * Sy
    
    var2 -= mu2**2

    print('Q1')
    print('computed:', 25.5025, 400.4608338)
    print('plugin (direct):', mu, var)
    print('plugin (survival):', mu2, var2)

    # bootstrap variance for mu
    B = 1000
    boot_mus = []
    for _ in range(B):
        sample_xs = []
        for _ in range(3000):
            sample_i = random.randint(0,len(xs)-1)
            sample_xs.append(xs[sample_i])
        boot_mus.append(plugin_mean(sample_xs))
    boot_mus_var = plugin_var(boot_mus)

    print('bootstrap var of mean:', boot_mus_var)
    print('95% (normal) interval for mean:', mu - 1.96 * boot_mus_var**0.5, mu + 1.96 * boot_mus_var**0.5)


        




    
    

    








