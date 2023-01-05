import numpy as np
import os
import subprocess
import json

size = 2**15

regions = np.arange(1,33)
times = np.ones((len(regions), len(regions)))*99999
auxdict = {} 
for i, NREGIONS_V in enumerate(regions):
    for j,NREGIONS_H in enumerate(regions):
        subprocess.run(['make', 'clean'], stdout=subprocess.PIPE)
        subprocess.run(['make', '-j', '8', 'NREGIONS_H='+str(NREGIONS_H), 'NREGIONS_V='+str(NREGIONS_V)], stdout=subprocess.PIPE)
        result = subprocess.run(['./bin/prog', "0", str(size), "4", "10", "0.5", "1"], stdout=subprocess.PIPE).stdout.decode('utf-8')
        if not 'GPUassert' in result:
            times[i,j] = float(result.split()[0][:-1])
            auxdict[i*len(regions) + j] = str(NREGIONS_H)+', '+str(NREGIONS_V)+', ' +result

np.save("out.npy", times)
with open('out.txt','w') as data: 
    data.write(str(auxdict))