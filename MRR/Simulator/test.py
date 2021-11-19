import numpy as np
from generate_figure import generate_figure
import matplotlib.pyplot as plt


plt.plot(np.arange(2000),generate_figure(1000,100,1,0,2000))
plt.show()