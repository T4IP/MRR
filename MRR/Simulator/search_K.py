from mymath import graph_integrate
from transfer_function import TransferFunction
import numpy as np

def search_K(L,K,xaxis,trans_data1):
    true_K = list(K)
    data = TransferFunction(L,K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
    trans_data2 = data.simulate(xaxis)
    S = graph_integrate(trans_data1,trans_data2)
    temp_K = list(K)
    true_K = list(K)
    for i in range(len(K)):
        if i != 0:
            temp_K[i-1] = true_K[i-1]
        temp_K[i]=true_K[i]
        if 0.7<K[i]:
            temp = 9
        elif 0.6<K[i]<=0.7:
            temp = 8
        else:
            temp = 6
        for j in range(temp*2+1):

            temp_K[i] = K[i] + 0.01*(j-temp)
            data = TransferFunction(L,temp_K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
            trans_data2 = data.simulate(xaxis)
            temp_S = graph_integrate(trans_data1,trans_data2)
            print(temp_S)
            if j == 0:
                S=temp_S
            if S >= temp_S:
                S=temp_S
                true_K[i] = temp_K[i]
                print(temp_K)
    data = TransferFunction(L,true_K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
    trans_data2 = data.simulate(xaxis)
    temp_S = graph_integrate(trans_data1,trans_data2)
    print(temp_S)
    return true_K

L=np.array([82.4e-6,82.4e-6,55.0e-6,55.0e-6])
K=np.array([0.16,0.79,0.72,0.38,0.49])
xaxis = np.arange(1540e-9,1560e-9,0.01e-9)          #シミュレーション範囲1.54µ~1.56µ
data = TransferFunction(L,K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
trans_data1 = data.simulate(xaxis)

K=np.array([0.16,0.79,0.71,0.39,0.49])
print(search_K(L,K,xaxis,trans_data1))