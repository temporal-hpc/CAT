import numpy as np
import os
import subprocess
import json
import sys


if len(sys.argv) != 4:
    print("Run with args:\n<GPUid> <GPUname> <CAsize>")
    exit()

GPUid = sys.argv[1]
GPUName = sys.argv[2]
CAsize = sys.argv[3]
sizes = [CAsize]
methods = [1,2,5,6,7,8]
method_names = ['global', 'shared', 'CAT', 'millan', 'topa', 'cagigas']
blocksizes_x = [32, 16, 16, 16, 32, 16]
blocksizes_y = [32, 16, 16, 16, 32, 16]

# V100 GPU
#nregions_x = [2]
#nregions_y = [5]

# A100 GPU
nregions_x = [1]
nregions_y = [13]

# H100 GPU
#nregions_x = [1]
#nregions_y = [14]

radiuses = [1, 4, 8, 16]
smin = [2, 40, 163, 170]
smax = [3, 80, 223, 296]
bmin = [3, 41, 74, 170]
bmax = [3, 80, 252, 300]
densities = [0.07, 0.50, 0.23, 0.26]
repeats = [4]
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
            print(f"    Running... GPU: {GPUid}, size: {size}, method: {method}, repeats: {repeats[l]}")
            print(['../bin/prog', str(GPUid), str(size), str(method), str(repeats[l]), str(densities[r]), str(0), "0"])
            result = subprocess.run(['../bin/prog', str(GPUid), str(size), str(method), str(repeats[l]), str(densities[r]), str(0), "0"], stdout=subprocess.PIPE).stdout.decode('utf-8')
            print(result)
            if not 'GPUassert' in result:
                with open("../data/scaling-results-"+GPUName+"-"+str(method_names[k])+".txt","a") as data:
                    res = {'radius' : radius, 'method' : method, 'size' : size}
                    res['time'] = str(result[:-2])
                    data.write(str(res) + '\n')
