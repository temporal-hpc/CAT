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
sizes = [1024 + 2048*i for i in range(29)]
methods = [1,2,6,7,8]
method_names = ['global', 'shared', 'millan', 'topa', 'cagigas']
#method_names = ['millan-int16x16', 'millan-int32x32']
blocksizes_x = [32, 16, 16, 32, 16]
blocksizes_y = [32, 16, 16, 32, 16]
#nregions_x = [1, 30, 1]
#nregions_y = [17, 1, 31]
nregions_x = [1]#a100
nregions_y = [13]
# nregions_x = [2]#V100
# nregions_y = [5]
# nregions_x = [1]#H100
# nregions_y = [14]
radiuses = [i for i in range(1,16)]
# radiuses = [1, 5, 10, 15]
smin = [2, 7, 15, 40, 35, 49, 65, 85, 108, 122, 156, 181, 213, 245, 281]
smax = [3, 12, 23, 80, 59, 81, 111, 143, 181, 211, 265, 312, 364, 420, 481]
bmin = [3, 8, 14, 41, 34, 46, 63, 80, 100, 123, 147, 175, 203, 234, 267]
bmax = [3, 11, 17, 80, 46, 65, 87, 110, 140, 170, 205, 243, 283, 326, 373]   
densities = [0.07, 0.2, 0.2, 0.5, 0.21, 0.22, 0.23, 0.23, 0.24, 0.25, 0.25, 0.25, 0.26, 0.26, 0.26]
# smin = [2, 35, 122, 281]
# smax = [3, 59, 211, 481]
# bmin = [3, 34, 123, 267]
# bmax = [3, 46, 170, 373]   
# densities = [0.07, 0.21,  0.25, 0.26]
repeats = [12, 12, 12, 12, 12, 12, 12, 12, 10, 10, 
           10, 10,  8,  8,  6,  6,  6,  6,  6,  6,
           4,   4,  4,  4,  4,  4,  4,  4,  4,  2]
# 1: passed
# 0: failed
results = {}
auxdict = {} 

for r, radius in enumerate(radiuses):
    for k, method in enumerate(methods):
        blocksize = [blocksizes_x[k], blocksizes_y[k]]
        print("Cleaning...")
        subprocess.run(['make', 'clean'], stdout=subprocess.PIPE, stderr=None, cwd="../")
        print(f"Compiling... NREGIONS_H: {str(nregions_x[0])}, NREGIONS_V: {str(nregions_y[0])}, BSIZE: {blocksize[0]}x{blocksize[1]}, RADIUS: {radius}")
        print('make', '-j', '8', 'NREGIONS_H='+str(nregions_x[0]), 'NREGIONS_V='+str(nregions_y[0]), 'BSIZE3DX='+str(blocksize[0]), 'BSIZE3DY='+str(blocksize[1]), 'RADIUS='+str(radius), 'SMIN='+str(smin[r]), 'SMAX='+str(smax[r]), 'BMIN='+str(bmin[r]), 'BMAX='+str(bmax[r]))
        subprocess.run(['make', '-j', '8', 'NREGIONS_H='+str(nregions_x[0]), 'NREGIONS_V='+str(nregions_y[0]), 'BSIZE3DX='+str(blocksize[0]), 'BSIZE3DY='+str(blocksize[1]), 'RADIUS='+str(radius), 'SMIN='+str(smin[r]), 'SMAX='+str(smax[r]), 'BMIN='+str(bmin[r]), 'BMAX='+str(bmax[r])], stdout=subprocess.PIPE, cwd="../")
        for l, size in enumerate(sizes):
#            if l<3:
#                continue
            print(f"    Running... GPU: {GPUid}, size: {size}, method: {method}, repeats: {repeats[l]}")
            print(['../bin/prog', str(GPUid), str(size), str(method), str(repeats[l]), str(densities[r]), str(0), "0"])
            result = subprocess.run(['../bin/prog', str(GPUid), str(size), str(method), str(repeats[l]), str(densities[r]), str(0), "0"], stdout=subprocess.PIPE).stdout.decode('utf-8')
            print(result)
            if not 'GPUassert' in result:
                with open("../data/benchmark_results-"+GPUName+"-"+str(method_names[k])+".txt","a") as data: 
                    res = {'radius' : radius, 'method' : method, 'size' : size}
                    res['time'] = str(result[:-2])
                    data.write(str(res) + '\n')

