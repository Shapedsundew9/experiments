import matplotlib.pyplot as plt
from numpy import float32, arange, sqrt, arange
from numpy.random import uniform
from scipy.spatial.distance import cdist
from tqdm import trange

POPULATION = 100
MIN = 0.0
MAX = 1.0
DIMS = 2**16

"""
data = []
for d in trange(1, DIMS + 1, ascii=True):
    p = uniform(MIN, MAX, (POPULATION, d)).astype(float32)
    s = cdist(p, p, 'euclidean').sum() / (POPULATION * (POPULATION - 1))
    data.append(s)
"""
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1,1,1, title='Mean Euclidean Distance between Uniformly Distributed Points')
#ax.plot(data)

# My Q: https://math.stackexchange.com/questions/4435332/what-is-the-relationship-between-mean-seperation-and-number-of-dimensions-for-un
# Answer: https://math.stackexchange.com/questions/1976842/how-is-the-distance-of-two-random-points-in-a-unit-hypercube-distributed
ax.plot(sqrt(arange(10000, DIMS + 1)/6 - 7/120)/sqrt(arange(10000, DIMS + 1)))

ax.set_xlabel('Number of Dimensions')
ax.set_ylabel('Mean Distance')
plt.show()



ax.plot(sqrt(arange(1, DIMS+1)*1/9))
ax.plot(arange(DIMS)**(1/2.12))
