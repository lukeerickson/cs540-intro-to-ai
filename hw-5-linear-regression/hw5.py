import sys
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

filename = sys.argv[1]

with open(filename, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    
initial = np.array(data)
    
#arr = np.array(data, dtype = float)
initial = np.delete(initial, (0), axis = 0)
arr = np.array(initial, dtype = float)

x = arr[:, 0]
y = arr[:, 1]

plt.plot(x, y)
plt.xticks(x)
#plt.yticks(y)
plt.xlabel('Year')
plt.ylabel('Number of Frozen Days')
#plt.show()
plt.savefig("plot.jpg")

n = len(arr)
X = np.ones((n, 2), dtype = np.int64)
Y = np.ones(n, dtype = np.int64)
index = 0

for row in arr:
    X[index][1] = arr[index][0]
    Y[index] = arr[index][1]
    index += 1

print("Q3a:")
#print(type(X[0][0]))
print(X)

print("Q3b:")
print(Y)

Z = np.matmul(np.transpose(X), X)

print("Q3c:")
print(Z)

I = np.linalg.inv(Z)

print("Q3d:")
print(I)

#PI = np.zeros()
#PI = np.matmul(I, np.transpose(X), dtype = np.int64)
PI = np.matmul(I, np.transpose(X))

print("Q3e:")
print(PI)

B = np.matmul(PI, Y)

print("Q3f:")
print(B)

print("Q4: " + str(B[0] + B[1]*2022))

print("Q5a: <")
print("Q5b: A negative Beta-one indicates that based on the given data, the number of days that Mendota is frozen over is expected to decrease each year. For a positive Beta-one, Mendota is expected be frozen over an increasing amount of days each year. For a Beta-one of 0, Lake Mendota is expected to be frozen over the same amount of days every year.")

year = -B[0] / B[1]
print("Q6a: " + str(year))

print("Q6b: An x* of 2463 is a compelling prediction based on the data. From viewing the data, it is evident that the number of ice days has dwindled significantly over the years, especially in the past four decades. It also appears that Mendota was frozen over 20-30 more days per year in the 1850s and 1860s, as compared to the 2010s and 2020s. At this rate, I would expect Mendota to not freeze at all in the 2400s.")


    
