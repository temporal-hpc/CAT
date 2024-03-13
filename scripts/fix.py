import glob
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

GPUName = 'H100'
folder = f'../'
if len(sys.argv) != 2:
    print("Run with args:\n<GPUName>")
    exit()

GPUName = sys.argv[1]

files = [f for f in glob.glob(f'../benchmark_results-{GPUName}-*.txt') if 'HIGH' not in f]
highfiles = []

for f in files:
    string = f.replace(f'benchmark_results-{GPUName}', f'benchmark_results-{GPUName}-HIGH')
    string = string.replace(f'.txt', f'-22.txt')
    highfiles.append(string)


print(files)
print(highfiles)

for i, highfileName in enumerate(highfiles):
    print (f'Fixing {highfileName}')
    print (f'With {files[i]}')
    with open(highfileName, "r") as highfile:
        with open(files[i], "r") as file:
            newdata = highfile.readlines()
            newdata = [eval(x) for x in newdata]
            data = file.readlines()
            data = [eval(x) for x in data]
            l = 0
            for j in range(15,0,-1):
                for k in range(5):
                    try:
                        if newdata[l*5 + k]['time'].split(',')[2] < data[(j-1)*29 + k]['time'].split(',')[2]:
                            data[(j-1)*29 + k] = newdata[l*5 + k]
                    except:
                        pass
                l+=1
            
            with open(files[i], "w") as file:
                for line in data:
                    file.write(str(line) + "\n")
