import glob
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

N_SIZES = 29
N_RADIUSES = 15
ORDER = []


if len(sys.argv) != 2:
    print("Run with args:\n<GPUName>")
    exit()

GPUName = sys.argv[1]

plotsFolder = f'../plots/{GPUName}'
sizes = []
radiuses = []


files = [f for f in glob.glob(f'../benchmark_results-{GPUName}*.txt') if 'HIGH' not in f]
method_names = [f.split(f'{GPUName}-')[1].split('.txt')[0] for f in files]
print(method_names)

#plot data
sns.set_theme()
#sns.set_style("whitegrid")
sns.set_palette("tab10")

sns.set_context("paper")

allData = np.zeros((len(files), N_RADIUSES, N_SIZES))

for i,f in enumerate(files):
    with open(f, "r") as file:
        data = file.readlines()
        data = [eval(x) for x in data]
        for j in range(N_RADIUSES):
            radiuses.append(data[j]['radius'])
            for k in range(N_SIZES):
                sizes.append(data[j*N_SIZES+k]['size'])
                try:
                    allData[i][j][k] = float(data[j*N_SIZES+k]['time'].split(',')[0])
                except:
                    allData[i][j][k] = None

        # # turn all columns 'time' into a single array
        # try:
        #     times = [float(x['time'].split(',')[0]) for x in data]
        # except:
        #     times = [0.0 for x in data]
        # radiuses = [x['radius'] for x in data]
        # sizes = [x['size'] for x in data]
        # print(sizes)
        # # plot
        # sns.lineplot(x=sizes, y=times, label=method_names[i], errorbar=None)
radiuses = radiuses[:N_RADIUSES]
sizes = sizes[:N_SIZES]
# plot by radius
for i in range(N_RADIUSES):
    data = allData[:,i,:]

    # plot
    #log scale
    plt.yscale('log')
    for j in range(len(method_names)):
        if 'char' in method_names[j]:
            continue
        sns.lineplot(x=sizes, y=data[j], label=method_names[j], errorbar=None)
    plt.title(f'Radius {radiuses[i]}')
    plt.xlabel('Size')
    plt.ylabel('Time (ms)')
    plt.legend()

    plt.savefig(f'{plotsFolder}/radius_{i+1}.png')
    plt.clf()
index = 0
for m in method_names:
    if 'global-int' in m:
        print(m, index)
        break
    index +=1
baseline = allData[index]
allData_Speedup = baseline/allData
for i in range(N_RADIUSES):
    data = allData_Speedup[:,i,:]

    # plot
    #log scale
    #plt.yscale('log')
    for j in range(len(method_names)):
        if 'char' in method_names[j]:
            continue
        if i<7:
            if '16x16' in method_names[j] and 'millan' in method_names[j]:
                continue
        sns.lineplot(x=sizes, y=data[j], label=method_names[j], errorbar=None)
    plt.title(f'Radius {i+1}')
    plt.xlabel('Size')
    plt.ylabel('Speedup')
    plt.legend()

    plt.savefig(f'{plotsFolder}/speedup-radius_{i+1}.png')
    plt.clf()

plt.yscale('log')
for j in range(len(method_names)):
    if 'char' in method_names[j]:
        continue

    sns.lineplot(x=np.linspace(1,15,N_RADIUSES), y=allData[j, :, -1], label=method_names[j], errorbar=None)
plt.title(f'Impact of radius')
plt.xlabel('Radius')
plt.ylabel('Time (ms)')
plt.legend()

plt.savefig(f'{plotsFolder}/radius_impact.png')
plt.clf()


for R in range(1,16):
    files = [f for f in glob.glob(f'../energy/power-*{GPUName}*.dat') if 'HIGH' not in f]
    rfiles = [f for f in files if f'RADIUS{R}.dat' in f]

    for i,f in enumerate(rfiles):
        with open(f, "r") as file:
            data = file.readlines()
            val = float(data[-1].split()[2])
            label = f.split('-')[-3] + '-' + f.split('-')[-2]
            sns.barplot(x=[label], y=[val], label=label)
    plt.title(f'Energy consumption')
    plt.xlabel('Method')
    plt.ylabel('Energy (J)')
    plt.legend()

    plt.savefig(f'{plotsFolder}/energy/bar-radius{R}.png')
    plt.clf()

import pandas as pd

for R in range(1,16):
    files = [f for f in glob.glob(f'../energy/power-*{GPUName}*.dat') if 'HIGH' not in f]
    rfiles = [f for f in files if f'RADIUS{R}.dat' in f]

    for i,f in enumerate(rfiles):
        with open(f, 'r') as file:
            data = pd.read_csv(file, delimiter='\s+')
            print(data)
            X = data['acc-time'][3:]
            Y = data['power'][3:]
            label = f.split('-')[-3] + '-' + f.split('-')[-2]
            sns.lineplot(x=X, y=Y, label=label)
    plt.title(f'Energy consumption')
    plt.xlabel('time')
    plt.ylabel('Power (W)')
    plt.legend()

    plt.savefig(f'{plotsFolder}/energy/time_series-radius{R}.png')
    plt.clf()