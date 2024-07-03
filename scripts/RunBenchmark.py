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
#sizes = [1024 + 2048*i for i in range(29)]
sizes = [1024 + 2048*i for i in range(30)]
methods = [1,2,5,6,7,8]
method_names = ['global', 'shared', 'CAT', 'millan', 'topa', 'cagigas']
#method_names = ['millan-int16x16', 'millan-int32x32']
blocksizes_x = [32, 16, 16, 16, 32, 16]
blocksizes_y = [32, 16, 16, 16, 32, 16]
#nregions_x = [1, 30, 1]
#nregions_y = [17, 1, 31]
#nregions_x = [1]#A100
#nregions_y = [13]
# nregions_x = [2]#V100
# nregions_y = [5]
nregions_x = [1]#H100
nregions_y = [14]
radiuses = [i for i in range(1,17)]
smin = [2, 7,  15, 40, 35, 49, 101,  163, 108, 122, 156, 170, 213, 245, 170, 170]
smax = [3, 12, 23, 80, 59, 81, 201,  223, 181, 211, 265, 296, 364, 420, 296, 296]
bmin = [3, 8,  14, 41, 34, 46, 75,   74,  100, 123, 147, 170, 203, 234, 170, 170]
bmax = [3, 11, 17, 80, 45, 65, 170,  252, 140, 170, 205, 240, 283, 326, 240, 300]
densities = [0.07, 0.15, 0.25, 0.50, 0.21, 0.22, 0.29, 0.23, 0.24, 0.25, 0.24, 0.25, 0.25, 0.25, 0.28, 0.26]
#radiuses = [16]
#smin = [170]
#smax = [296]
#bmin = [170]
#bmax = [300]
#densities = [0.26]
repeats = [32, 32, 32, 32, 16, 16, 16, 16, 8, 8,
            8,  8,  8,  8,  4,  4,  4,  4, 4, 4,
            4,  4,  4,  4,  2,  2,  2,  2, 2, 2]
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

