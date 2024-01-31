import numpy as np
import os
import subprocess
import json
import sys


if len(sys.argv) != 3:
    print("Run with args:\n<GPUid> <GPUname>")
    exit()

GPUid = sys.argv[1]
GPUName = sys.argv[2]
sizes = [1024 + 2048*i for i in range(35)]
methods = [1, 2, 5, 6, 7, 8]
method_names = ['global', 'shared', 'tensor', 'millan', 'topa', 'cagigas']
repeats = 5
blocksizes_x = [32, 16, 16, 16, 16, 16]
blocksizes_y = [16, 16, 16, 16, 16, 16]
#nregions_x = [1, 30, 1]
#nregions_y = [17, 1, 31]
nregions_x = [1]
nregions_y = [12]
radiuses = [i for i in range(1,16)]
smin = [2]
smax = [3]
bmin = [3]
bmax = [3]   

r0 = (2*1+1)**2
for r in radiuses[1:]:
    N = ((2*r+1)**2)/r0
    smin.append(round(smin[0] * N))
    smax.append(round(smax[0] * N))
    bmin.append(round(bmin[0] * N))
    bmax.append(round(bmax[0] * N))

# 1: passed
# 0: failed
results = {}
auxdict = {} 

for r, radius in enumerate(radiuses):
    for k, method in enumerate(methods):
        for l, size in enumerate(sizes):
            blocksize = [blocksizes_x[k], blocksizes_y[k]]
            print("Cleaning...")
            subprocess.run(['make', 'clean'], stdout=subprocess.PIPE, stderr=None, cwd="../")
            print(f"Compiling... NREGIONS_H: {str(nregions_x[0])}, NREGIONS_V: {str(nregions_y[0])}, BSIZE: {blocksize[0]}x{blocksize[1]}, RADIUS: {radius}")
            subprocess.run(['make', '-j', '8', 'NREGIONS_H='+str(nregions_x[0]), 'NREGIONS_V='+str(nregions_y[0]), 'BSIZE3DX='+str(blocksize[0]), 'BSIZE3DY='+str(blocksize[1]), 'RADIUS='+str(radius),], stdout=subprocess.PIPE, cwd="../")
            print(f"    Running... GPU: {GPUid}, size: {size}, method: {method}, repeats: {repeats}")
            result = subprocess.run(['../bin/prog', str(GPUid), str(size), str(method), str(repeats), "0.4", "0", "0"], stdout=subprocess.PIPE).stdout.decode('utf-8')
            print(result)
            if not 'GPUassert' in result:
                with open("../benchmark_results-"+GPUName+"-"+str(method_names[k])+".txt","a") as data: 
                    res = {'radius' : radius, 'method' : method, 'size' : size}
                    res['time'] = str(result[:-2])
                    data.write(str(res) + '\n')

