from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

plt.plot(range(1, 13), [0.4, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.8, 0.6, 0.5, 0.9, 0.9], color='green', label="Empirical")
plt.plot(range(1, 13), [0.4] * 12, linestyle="dashed", alpha=0.4, color="black", label="Baseline")
plt.legend()
plt.xlabel("Semester week")
plt.ylabel("SCP Index")
plt.title("SCP Index during project writing")
plt.show()
