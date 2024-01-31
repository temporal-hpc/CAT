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
sizes = [2**i for i in range(5, 15)]
methods = [1,2,5,6,7,8]
# methods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
repeats = 2
blocksizes_x = [32, 16, 16, 32, 16, 16]
blocksizes_y = [16, 16, 16, 32, 16, 16]
nregions_x = [1]
nregions_y = [13]
radiuses = [1, 2, 4, 8, 15]

# 1: passed
# 0: failed
results = np.ones((len(blocksizes_x), len(nregions_x), len(radiuses), len(methods), len(sizes)))
auxdict = {} 

for i, blocksize in enumerate(zip(blocksizes_x, blocksizes_y)):
    for j, nregions in enumerate(zip(nregions_x, nregions_y)):
        for r, radius in enumerate(radiuses):
            print("Cleaning...")
            subprocess.run(['make', 'clean'], stdout=subprocess.PIPE, stderr=None, cwd="../")
            print(f"Compiling... NREGIONS_H: {str(nregions[0])}, NREGIONS_V: {str(nregions[1])}, BSIZE: {blocksize[0]}x{blocksize[1]}, RADIUS: {radius}")
            subprocess.run(['make', 'debug', '-j', '8', 'NREGIONS_H='+str(nregions[0]), 'NREGIONS_V='+str(nregions[1]), 'BSIZE3DX='+str(blocksize[0]), 'BSIZE3DY='+str(blocksize[1]), 'RADIUS='+str(radius),], stdout=subprocess.PIPE, cwd="../")
            for k, method in enumerate(methods):
                for l, size in enumerate(sizes):
                    print(f"    Running... GPU: {GPUid}, size: {size}, method: {method}, repeats: {repeats}")
                    result = subprocess.run(['../debug/prog', str(GPUid), str(size), str(method), str(repeats), "0.5", "1"], stdout=subprocess.PIPE).stdout.decode('utf-8')
                    if not 'successful' in result:
                        print('Failed!')
                        results[i, j, r, k, l] = 0
                    else:
                        print('Sucess!')
                        results[i, j, r, k, l] = 1

                    #     try:
                    #         times[i,j] = float(result.split()[0][:-1])
                    #     except:
                    #         print(result)
                    #     auxdict[i*len(regions) + j] = str(NREGIONS_H)+', '+str(NREGIONS_V)+', ' +result

np.save("../verification_results-"+GPUName+"-"+".npy", results)
# with open("../landscape-"+GPUname+"-"+GPUid+"-"+sizes+"-"+methods+"-"+blocksizes_x+"x"+blocksizes_y+".txt","w") as data: 
#     data.write(str(auxdict))
