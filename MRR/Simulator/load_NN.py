import numpy as np
from transfer_function import TransferFunction
import csv
from mymath import minus_and_round
from control_data import read_K, load_NN
from generate_figure import generate_figure


#予測開始
pre_number = 10
xaxis = np.arange(1540e-9,1560e-9,0.01e-9)
number_of_rings = 4
n = 100000
model_DNN = load_NN(n,number_of_rings)
L=np.array([82.4e-6,82.4e-6,55.0e-6,55.0e-6])                             #予測個数
pre_data_K = np.array(read_K(pre_number,number_of_rings)) #予測用のK
file_name = "C:\\Users\\taipa\\Documents\\研究室\\プログラム\\MRR\\data\\pred_K" + str(number_of_rings) + "_" + str(n) +".csv"
with open(file_name,"w",newline="") as file:
    writer = csv.writer(file)
    for i in range(pre_number):
        pre_data = TransferFunction(L,pre_data_K[i],config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
        temp = np.array(list(map(minus_and_round,pre_data.simulate(np.arange(1540e-9,1560e-9,0.01e-9)))))
        pred_Y = model_DNN.predict(temp.reshape(1,len(xaxis)))
        print(pred_Y[0],pre_data_K[i])
        writer.writerow(np.append(pred_Y[0],pre_data_K[i]))
data = np.array(generate_figure(1000,100,1,0,2001,-30))
print(model_DNN.predict(data.reshape(1,2001)))