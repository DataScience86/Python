import matplotlib.pyplot as plt
import numpy as np

n = 50
x = [1,2,3,4,5]#np.random.randn(n)
y = [3,3,4,5,6]#x * np.random.randn(n)

fig, ax = plt.subplots()
fit = np.polyfit(x, y, deg=1)
print fit[0]
print fit[1]

#assert False

ax.plot(x, fit[0] * x + fit[1], color='red')
ax.scatter(x, y)

fig.show()
raw_input()
