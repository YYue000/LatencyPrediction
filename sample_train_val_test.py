import random
import pickle

flops = pickle.load(open('datasets/flops.pickle','rb'))
gpu = pickle.load(open('datasets/desktop-gpu-gtx-1080ti-fp32.pickle','rb'))


train_num = 900
val_num = 100

archs = list(gpu.keys()) 
random.seed(100)
random.shuffle(archs)

train_set = {arch: gpu[arch] for arch in archs[:train_num]}
val_set = {arch: gpu[arch] for arch in archs[train_num:train_num+val_num]}
test_set = {arch: gpu[arch] for arch in archs[train_num+val_num:]}

pickle.dump(train_set, open('datasets/desktop-gpu-gtx-1080ti-fp32-train-2.pickle','wb'))
pickle.dump(val_set, open('datasets/desktop-gpu-gtx-1080ti-fp32-val-2.pickle','wb'))
pickle.dump(test_set, open('datasets/desktop-gpu-gtx-1080ti-fp32-test-2.pickle','wb'))