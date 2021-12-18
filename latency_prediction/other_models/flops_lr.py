import pickle
from sklearn.linear_model import LinearRegression
import numpy as np

def evaluate(prediction,latency, error_percentage=[0.01,0.05,0.1,0.2]):
    error_percentage = np.array(error_percentage).reshape(-1,1)
    delta = np.abs(latency-prediction)/latency
    #delta = np.abs(latency-prediction)/prediction
    c = np.sum(delta<=error_percentage, axis=1)
    n = len(latency)
    c = c/n
    return c

path_dir='../datasets/'
path_root=path_dir+'mobile-cpu-snapdragon-450-cortex-a53-int8'
#path_root=path_dir+'desktop-gpu-gtx-1080ti-fp32'
#train = pickle.load(open(path_root+'-train-3-2000.pickle','rb'))
train = pickle.load(open(path_root+'-train.pickle','rb'))
test = pickle.load(open(path_root+'-test.pickle','rb'))

"""
data= pickle.load(open(path_root+'.pickle','rb'))
archs = list(data.keys())
import random
random.shuffle(archs)
All = len(archs)
T = int(All*0.7)
train = {k:data[k] for k in archs[:T]}
test = {k:data[k] for k in archs[T:]}
"""

flops= pickle.load(open(path_dir+'flops.pickle','rb'))
#flops= pickle.load(open(path_dir+'flops_normalized.pickle','rb'))

N = len(train)
train_flops = [flops[_] for _ in train.keys()]
train_flops = np.array(train_flops).reshape(N,1)
train_ltc = [__ for _,__ in train.items()]
train_ltc = np.array(train_ltc).reshape(N,1)
reg=LinearRegression().fit(train_flops, train_ltc)
test_flops = [flops[_] for _ in test.keys()]
test_flops = np.array(test_flops).reshape(len(test),1)
test_ltc = np.array([__ for _,__ in test.items()])
test_ltc_hat = reg.predict(test_flops).reshape(-1)
print(evaluate(test_ltc_hat, test_ltc))

