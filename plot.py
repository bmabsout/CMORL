# This python code alayzes the velocity.txt file
# and plots a histogram of the velocities
with open("velocity.txt", "r") as f:
    data = f.readlines()
    data = [float(x) for x in data]

import matplotlib.pyplot as plt

plt.hist(data, bins=100)
plt.xlabel("Velocity")
plt.ylabel("Frequency")
plt.title("Histogram of Velocities")
plt.show()
