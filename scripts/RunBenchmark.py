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
sizes = [2**i for i in range(5, 16)]
#methods = [10]
methods = [0, 1, 3, 5, 7]
repeats = 10
blocksizes_x = [16, 32, 32]
blocksizes_y = [16, 16, 32]
#nregions_x = [1, 30, 1]
#nregions_y = [17, 1, 31]
nregions_x = [1, 1, 1]
nregions_y = [10, 13, 18]
radiuses = [1, 2, 4, 8, 15]

# 1: passed
# 0: failed
results = np.ones((len(blocksizes_x), len(radiuses), len(methods), len(sizes)))
auxdict = {} 

for i, blocksize in enumerate(zip(blocksizes_x, blocksizes_y)):
    for r, radius in enumerate(radiuses):
        print("Cleaning...")
        subprocess.run(['make', 'clean'], stdout=subprocess.PIPE, stderr=None, cwd="../")
        print(f"Compiling... NREGIONS_H: {str(nregions_x[i])}, NREGIONS_V: {str(nregions_y[i])}, BSIZE: {blocksize[0]}x{blocksize[1]}, R: {radius}")
        subprocess.run(['make', '-j', '8', 'NREGIONS_H='+str(nregions_x[i]), 'NREGIONS_V='+str(nregions_y[i]), 'BSIZE3DX='+str(blocksize[0]), 'BSIZE3DY='+str(blocksize[1]), 'R='+str(radius),], stdout=subprocess.PIPE, cwd="../")
        for k, method in enumerate(methods):
            for l, size in enumerate(sizes):
                print(f"    Running... GPU: {GPUid}, size: {size}, method: {method}, repeats: {repeats}")
                result = subprocess.run(['../bin/prog', str(GPUid), str(size), str(method), str(repeats), "0.5", "1"], stdout=subprocess.PIPE).stdout.decode('utf-8')
                if not 'GPUassert' in result:
                    try:
                        results[i,r,k,l] = float(result.split()[0][:-1])
                    except:
                        print(result)

                    #     try:
                    #         times[i,j] = float(result.split()[0][:-1])
                    #     except:
                    #         print(result)
                    #     auxdict[i*len(regions) + j] = str(NREGIONS_H)+', '+str(NREGIONS_V)+', ' +result

np.save("../benchmark_results-"+GPUName+"-"+".npy", results)
# with open("../landscape-"+GPUname+"-"+GPUid+"-"+sizes+"-"+methods+"-"+blocksizes_x+"x"+blocksizes_y+".txt","w") as data: 
#     data.write(str(auxdict))
