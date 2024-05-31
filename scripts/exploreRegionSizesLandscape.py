import numpy as np
import os
import subprocess
import json
import sys

size = 0 

if len(sys.argv) != 8:
    print("Run with args:\n<GPUid>\n<size>\n<method>\n<repeats>\n<bsize x>\n<bsize y>\n<GPUName>")
    exit()

GPUid = sys.argv[1]
size = sys.argv[2]
method = sys.argv[3]
repeats = sys.argv[4]
bsizex = sys.argv[5]
bsizey = sys.argv[6]
GPUname = sys.argv[7]

regions = np.arange(1,31)

times = np.ones((len(regions), len(regions)))*99999
auxdict = {} 
for i, NREGIONS_V in enumerate(regions):
    for j,NREGIONS_H in enumerate(regions):
        print("Cleaning...")
        subprocess.run(['make', 'clean'], stdout=subprocess.PIPE, stderr=None, cwd="../")
        print(f"Compiling... NREGIONS_H: {str(NREGIONS_H)}, NREGIONS_V: {str(NREGIONS_V)}, BSIZE: {bsizex}x{bsizey}")
        subprocess.run(['make', '-j', '8', 'NREGIONS_H='+str(NREGIONS_H), 'NREGIONS_V='+str(NREGIONS_V), 'BSIZE3DX='+str(bsizex), 'BSIZE3DY='+str(bsizey)], stdout=subprocess.PIPE, cwd="../")
        print(f"Running... GPU: {GPUid}, size: {size}, method: {method}, repeats: {repeats}")
        result = subprocess.run(['../bin/prog', str(GPUid), str(size), str(method), str(repeats), "0.5", "1", '0'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        print(result)
        if not 'GPUassert' in result:
            try:
                times[i,j] = float(result.split()[0][:-1])
            except:
                print(result)
            auxdict[i*len(regions) + j] = str(NREGIONS_H)+', '+str(NREGIONS_V)+', ' +result

np.save("../landscape-"+GPUname+"-"+GPUid+"-"+size+"-"+method+"-"+bsizex+"x"+bsizey+".npy", times)
with open("../landscape-"+GPUname+"-"+GPUid+"-"+size+"-"+method+"-"+bsizex+"x"+bsizey+".txt","w") as data: 
    data.write(str(auxdict))
