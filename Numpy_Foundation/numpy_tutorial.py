import numpy as np
import matplotlib.pyplot as plt
a= np.array([(1,2,3),(4,5,6),(7,8,9)])
b=np.array([(2,3,4),(5,6,9),(2,5,7)])
print(a)

import time
import sys

s=range(1000)

print(sys.getsizeof(5)*len(s))

d=np.arange(1000)
print(d.size*d.itemsize)

size=100000

l1=range(size)
l2=range(size)

A1=np.arange(size)
A2=np.arange(size)

start=time.time()

result=[(x,y) for x,y in zip(l1,l2)]

print((time.time()-start)*1000)

result=A1+A2
start=time.time()
print((time.time()-start)*1000)

a.shape
a.reshape(9,1)

## axis concept
##Sum of Columns
print(a.sum(axis=1))
## Square root
print(np.sqrt(a))
##Standard Deviation
print(np.std(a))
## Addition
print(a+b)
### Stacking the arrays on top of the other

print(np.hstack((a,b)))

x=np.arange(0,3*np.pi,0.1)
y=np.tan(x)
plt.plot(x,y)
plt.show()

ar=np.array([1,2,3,4])

## Print exponential value for the above array

t=print(np.exp(ar))
t.dtype
## Index

for i  in range(len(t)-1):
    print(ar[i+1]-ar[i])
    















