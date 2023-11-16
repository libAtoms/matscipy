import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import params
from matscipy import parameter

folder_name = parameter('folder_name','data')
pid = parameter('data_load_pid')
traj_files = [file for file in os.listdir(folder_name) if pid in file]

data_sets = {}
#now read each file and append the data to the data_sets for each key

for file in traj_files:
    hf = h5py.File(os.path.join(folder_name,file),'r')
    keys = list(hf.keys())
    for key in keys:
        #if key is not in data sets, add it
        if key not in data_sets:
            data_sets[key] = []
        data_sets[key].append(hf[key][:])
    hf.close()
# print(data_sets)

#now save each data set to new h5 files in the folder
#set keys to data_sets.keys()
keys = data_sets.keys()
for key in keys:
    hf = h5py.File(os.path.join(folder_name,f'{pid}_fin_{key}.h5'),'w')
    hf.create_dataset('x',(0, len(data_sets[key][0])),maxshape=(None,len(data_sets[key][0])),compression='gzip')
    #write each entry of data_sets[key] to the file
    for data in data_sets[key]:
        hf['x'].resize((hf['x'].shape[0]+1),axis=0)
        hf['x'][-1,:] = data
    hf.close()


##open all traj files in folder that have just been made and plot the k-alpha curves
#traj_files = [file for file in os.listdir(folder_name) if f'{pid}_fin' in file]
#for i,traj_file in enumerate(traj_files):
#    hf = h5py.File(os.path.join(folder_name,traj_file),'r')
#    x_traj = hf['x'][:]
#    #sort x_traj by second to last column
#    x_traj = x_traj[x_traj[:,-2].argsort()]
#    Ks = x_traj[:,-1]
#    print(len(Ks))
#    alphas = x_traj[:,-2]
#    hf.close()
#    print(Ks,alphas)
#    plt.plot(Ks,alphas,label=traj_file)###

#plt.legend()
#plt.xlabel('K')
#plt.ylabel('alpha')
#plt.title('NCFlex diamond (1-10) solution curve')
#plt.tight_layout()
#plt.grid()
#plt.savefig('rI_conv_curve.png')