import matplotlib.pyplot as plt
import numpy as np

def base_array(n,bottom):
    return [bottom]*n

def set_3db(center,length,data):
    for i in range(length):
        data[int(center-int(length/2)+i)] = -3

    # for j in range(300-int(length/2)):
    #     data[center-int(length/2)-j] = -3+(-27/(300-(length/2)))*j
    #     data[center+int(length/2)+j] = -3+(-27/(300-(length/2)))*j


def set_cross(center,length,cross,data):
    if cross == 0:
        return
    for i in range(length):
        data[int(center-int(length/2)+i+600)] = cross
        data[int(center-int(length/2)+i-600)] = cross

def generate_figure(center,length1,length2,cross,n,bottom):
    data = base_array(n,bottom)
    set_3db(center,length1,data)
    set_cross(center,length2,cross,data)

    return data  

# plt.plot(np.arange(2001),generate_figure(1000,100,1,0,2001,-30))
# plt.show()

