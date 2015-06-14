
import random
import numpy as np

K = 100000
N = 1000
v1 = []
vrand = []
vmin = []

# run experiment K times
for k in range(0, K):
    y = []

    # N coins
    for i in range(0, N):
        # flip each coin 10 times
        tosses = []
        for j in range(0, 10):
            tosses.append(random.choice([0, 1]))

        # record the fraction of heads
        y.append(np.mean((tosses)))


    # remember the first coin's result
    v1.append(y[0])
    # remember the random coin's result
    vrand.append(y[random.randint(0, N - 1)])
    # remember the result of a coin that has the minimum fraction of heads
    vmin.append(min(y))


print "Avg of v1: ", np.mean(v1), '\n'
print "Avg of vrand: ", np.mean(vrand), '\n'
print "Avg of vmin: ", np.mean(vmin), '\n'

#Avg of v1:  0.501327

#Avg of vrand:  0.50014

#Avg of vmin:  0.037314

