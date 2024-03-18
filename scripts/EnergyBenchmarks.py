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
sizes = [1024 + 2048*28]
methods = [1, 2, 6, 7]#, 1, 2, 6, 7]
method_names = ['global-char', 'shared-char', 'millan-char', 'topa-char'] #, 'global-int', 'shared-int', 'millan-int', 'topa-int']
blocksizes_x = [32, 16, 32, 16]#, 32, 16, 32, 16]
blocksizes_y = [16, 16, 32, 16]#, 16, 16, 32, 16]
#nregions_x = [1, 30, 1]
#nregions_y = [17, 1, 31]
nregions_x = [1]
nregions_y = [12]
radiuses = [i for i in range(1,16)]
smin = [2, 7, 15, 40, 35, 49, 65, 85, 108, 122, 156, 181, 213, 245, 281]
smax = [3, 12, 23, 80, 59, 81, 111, 143, 181, 211, 265, 312, 364, 420, 481]
bmin = [3, 8, 14, 41, 34, 46, 63, 80, 100, 123, 147, 175, 203, 234, 267]
bmax = [3, 11, 17, 80, 46, 65, 87, 110, 140, 170, 205, 243, 283, 326, 373]   
densities = [0.07, 0.2, 0.2, 0.5, 0.21, 0.22, 0.23, 0.23, 0.24, 0.25, 0.25, 0.25, 0.26, 0.26, 0.26]
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
        subprocess.run(['make', '-j', '8', 'MEASURE_POWER=MEASURE_POWER', 'NREGIONS_H='+str(nregions_x[0]), 'NREGIONS_V='+str(nregions_y[0]), 'BSIZE3DX='+str(blocksize[0]), 'BSIZE3DY='+str(blocksize[1]), 'RADIUS='+str(radius), 'SMIN='+str(smin[r]), 'SMAX='+str(smax[r]), 'BMIN='+str(bmin[r]), 'BMAX='+str(bmax[r])], stdout=subprocess.PIPE, cwd="../")
        for l, size in enumerate(sizes):
            print(f"    Running... GPU: {GPUid}, size: {size}, method: {method}, repeats: 1")
            print(['../bin/prog', str(GPUid), str(size), str(method), str(1), str(densities[r]), str(0), "0"])
            result = subprocess.run(['../bin/prog', str(GPUid), str(size), str(method), "1", str(densities[r]), str(0), "0"], stdout=subprocess.PIPE).stdout.decode('utf-8')
            print(result)
            result = subprocess.run(["ls"], stdout=subprocess.PIPE).stdout.decode('utf-8')
            if 'power-0.dat' in result:
                subprocess.run(['mv', "power-0.dat", "../energy/power-"+str(blocksize[0]) +"x"+str(blocksize[1])+"-"+GPUName+"-"+str(method_names[k])+"-RADIUS"+str(radius)+".dat"], stdout=subprocess.PIPE).stdout.decode('utf-8')
            else:
                print(result)
