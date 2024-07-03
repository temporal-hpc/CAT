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

files = [f for f in glob.glob(f'../benchmark_results-{GPUName}-*.txt') if 'millan' in f]

print(files)

for i, filename in enumerate(files):
    print (f'With {files[i]}')
    with open(files[i], "r") as file:
        data = file.readlines()
        data = [eval(x) for x in data]
        print(data)
        # with open(files[i], "w") as file:
        #     for line in data:
        #         file.write(str(line) + "\n")
