import glob
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

N_SIZES = 29
N_RADIUSES = 15
ORDER = []
#r>7
#obs
# Only use millan 32x32
lineswidth = 2.5
def shouldExclude(methodName):
    return ('millan' in methodName and '32x32' not in methodName) #or 'char' in methodName

if len(sys.argv) != 2:
    print("Run with args:\n<GPUName>")
    exit()

GPUName = sys.argv[1]

plotsFolder = f'../plots/{GPUName}'
sizes = []
radiuses = []

titleSuffix = f', NVIDIA {GPUName}'
files = [f for f in glob.glob(f'../benchmark_results-{GPUName}*.txt') if 'HIGH' not in f]
method_names = [f.split(f'{GPUName}-')[1].split('.txt')[0] for f in files]
print(method_names)

methods = ['Base', 'Base-sh', 'CAT', 'Millan', 'Topa', 'Cagigas']
line_styles = ['--', '-.', '-', ':', (0, (5,2,20,2)), (0, (3, 1, 1,1))]
suffix = ['', '', '', '(*)']
# suffix = ['(int)', '(char)', '', '(Overflow)']
names_pre = ['global-char', 'shared-char', 'tensor', 'millan-char32x32', 'topa-char', 'cagigas']
suffix_pre = [1, 1, 2, 1, 1, 2]

names_post = ['global-int', 'shared-int', 'tensor', 'millan-int32x32',  'topa-int', 'cagigas']
suffix_post = [0, 0, 2, 0, 0, 3]

#create a color for each method name
colors = sns.color_palette("bright", len(methods))
color_dict = {methods[i]: colors[i] for i in range(len(methods))}
print(color_dict)
# CAT     --> #006e76   (temporal)
# Base    --> #333333   (gris bien oscuro)
# Base-sh --> #df6b4d   (salmon)
# Millan  --> #bc8600   (amarillo mostaza)
# Topa    --> #ea6e8f   (rosado)
# Cagigas --> #7379ca   (azul medio morado)

color_dict['CAT'] = (0, 0.43137254901960786, 0.4627450980392157)
color_dict['Base'] = (0.2, 0.2, 0.2)
color_dict['Base-sh'] = (0.8745098039215686, 0.4196078431372549, 0.30196078431372547)
color_dict['Millan'] = (0.7372549019607844, 0.5254901960784314, 0.0)
color_dict['Topa'] = (0.9176470588235294, 0.43137254901960786, 0.5607843137254902)
color_dict['Cagigas'] = (0.45098039215686275, 0.4745098039215686, 0.792156862745098)
#plot data
sns.set_theme()
sns.set_style("whitegrid")
sns.set_palette("bright")
sns.set_context("paper")

sizes = [1024 + 2048*i for i in range(29)]
allData = np.zeros((len(files), N_RADIUSES, N_SIZES))
plt.figure(dpi=350)
for i,f in enumerate(files):
    with open(f, "r") as file:
        data = file.readlines()
        data = [eval(x) for x in data]
        for j in range(N_RADIUSES):
            radiuses.append(data[j]['radius'])
            for k in range(N_SIZES):
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
       
    for j in range(len(methods)):
        if i < 7: 
            fileIndex = method_names.index(names_pre[j])
            suff = suffix[suffix_pre[j]]
        else:
            fileIndex = method_names.index(names_post[j])
            suff = suffix[suffix_post[j]]

        sns.lineplot(x=sizes, y=data[fileIndex], label=methods[j] + "" + suff, errorbar=None, color=color_dict[methods[j]], linestyle=line_styles[j], linewidth=lineswidth)

            # if i >= 7 and 'cagigas' in method_names[j]:
            #     sns.lineplot(x=sizes, y=data[j], label=method_names[j].split('-')[0].capitalize() + " (imprecise)", errorbar=None, color=color_dict[method_names[j]], linestyle='dashed')
            # else:
            #     sns.lineplot(x=sizes, y=data[j], label=method_names[j].split('-')[0].capitalize(), errorbar=None, color=color_dict[method_names[j]])
    #change font size of the axis
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(18)  # Adjust the fontsize as needed
    ax.xaxis.get_offset_text().set_fontsize(18)  # Adjust the fontsize as needed
    # Move y-   axis and its label to the left
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    #add the lines minor
    ax.yaxis.tick_left()
    ax.yaxis.set_tick_params(width=1, which='major', color='lightgrey')
    ax.yaxis.set_tick_params(which='minor', color='lightgrey')

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'CA step time, $r$ = {i+1}', fontsize=21)
    plt.xlabel('n', fontsize=18)
    plt.ylabel('Time (ms)', fontsize=18)
    legend = plt.legend(loc='upper left',# bbox_to_anchor=(0.5, 1.19),
        ncol=1, fontsize=14)
    # legend.get_frame().set_alpha(None)
    # legend.get_frame().set_facecolor((0, 0, 1, 0.03))
    plt.ylim([0.01, 30000])
    plt.savefig(f'{plotsFolder}/radius_{i+1}.pdf', bbox_inches='tight')
    plt.clf()

# exit()
baseline_int = allData[method_names.index('global-int')]
baseline_char = allData[method_names.index('global-char')]
print("nidex int", method_names.index('global-int'))
print("nidex int", method_names.index('global-char'))
allData_Speedup_char = baseline_char/allData
allData_Speedup_int = baseline_int/allData

for i in range(N_RADIUSES):
    if i < 7:
        data = allData_Speedup_char[:,i,:]
    else:
        data = allData_Speedup_int[:,i,:]

    # plot
    #log scale
    plt.yscale('log')
    for j in range(len(methods)):
        if i < 7: 
            fileIndex = method_names.index(names_pre[j])
            suff = suffix[suffix_pre[j]]
        else:
            fileIndex = method_names.index(names_post[j])
            suff = suffix[suffix_post[j]]
        if (methods[j] == 'Base'):
            continue
        sns.lineplot(x=sizes, y=data[fileIndex], label=methods[j] + "" + suff, errorbar=None, color=color_dict[methods[j]], linestyle=line_styles[j], linewidth=lineswidth)
    sns.lineplot(x=sizes, y=1, errorbar=None, color="black", linestyle='-', linewidth=lineswidth*0.5)
        
    #add scientific notation in X
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #change font size of the axis
    # Get the current axes
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(19)  # Adjust the fontsize as needed
    ax.xaxis.get_offset_text().set_fontsize(19)  # Adjust the fontsize as needed

    ax.yaxis.tick_left()
    #change color of the ticks
    ax.yaxis.set_tick_params(width=1, which='major', color='lightgrey')
    ax.yaxis.set_tick_params(which='minor', color='lightgrey')

    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.ylim([0.06, 200])





    plt.title(f'Speedup over Base, $r$ = {i+1}', fontsize=22)
    plt.xlabel('n', fontsize=19)
    plt.ylabel('Speedup', fontsize=19)
    if i==0:
        legend = plt.legend(loc='upper left',# bbox_to_anchor=(0.5, 1.19),
            ncol=1, fontsize=14)
    elif i==4:
        legend = plt.legend(loc='upper left',# bbox_to_anchor=(0.5, 1.19),
            ncol=1, fontsize=14)
    elif i==9:
        legend = plt.legend(loc='upper left',# bbox_to_anchor=(0.5, 1.19),
            ncol=1, fontsize=14)
    elif i==14:
        legend = plt.legend(loc='upper left',#, bbox_to_anchor=(0.0, 0.3),
            ncol=1, fontsize=14)
    else: 
        legend = plt.legend(loc='upper left',# bbox_to_anchor=(0.5, 1.19),
            ncol=1, fontsize=14)    

    
    bestCat = (data[method_names.index('tensor')][-1])

    ax_right = ax.twinx()
    ax_right.set_yscale('log')
    ax_right.set_ylim(ax.get_ylim())
    ax_right.tick_params(axis='y', which='both', length=0, labelsize=19, colors="#004d52")
    ax_right.set_yticks([bestCat])
    #the label must be in decimal notation
    if bestCat < 10:
        ax_right.set_yticklabels([f'{bestCat:.1f}x'])
    else:
        ax_right.set_yticklabels([f'{bestCat:.0f}x'])
    # ax_right.yaxis.set_tick_params(width=1, which='major')
    
    #set the color of the grid
    # ax_right.grid(True, which="major",linewidth=0.1, color="#004d52")
    ax_right.grid(False)






    # legend.get_frame().set_alpha(None)
    # legend.get_frame().set_facecolor((0, 0, 1, 0.03))
        
    # # Shrink current axis's height by 10% on the bottom
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])

    # # Put a legend below current axis
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncols=5)

    plt.savefig(f'{plotsFolder}/speedup-radius_{i+1}.pdf', bbox_inches='tight')
    plt.clf()

data = []

for j in range(len(methods)):
    d = []
    for i in range(N_RADIUSES):
        if i < 7: 
            fileIndex = method_names.index(names_pre[j])
            suff = suffix[suffix_pre[j]]
        else:
            fileIndex = method_names.index(names_post[j])
            suff = suffix[suffix_post[j]]
        d.append(allData[fileIndex, i, -1])
    data.append(d)
plt.yscale('log')
for j in range(len(methods)):

    sns.lineplot(x=np.linspace(1,15,N_RADIUSES), y=data[j], label=methods[j], errorbar=None, color=color_dict[methods[j]], linestyle=line_styles[j], linewidth=lineswidth)
# ax.yaxis.tick_right()
# ax.yaxis.set_label_position("right")
ax = plt.gca()

ax.yaxis.tick_left()
#change color of the ticks
ax.yaxis.set_tick_params(width=1, which='major', color='lightgrey')
ax.yaxis.set_tick_params(which='minor', color='lightgrey')


plt.xticks(np.arange(1, 16, 1), fontsize=19)
#xticks from 1 to 15
# plt.xticks()
plt.yticks(fontsize=19)
plt.title(f'Impact of $r$ in Time, $n$ = 58368', fontsize=22)
plt.xlabel('$r$', fontsize=19)
plt.ylabel('Time (ms)', fontsize=19)
legend = plt.legend(ncols=1, fontsize=14)
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor((0, 0, 1, 0.02))

plt.savefig(f'{plotsFolder}/radius_impact.pdf', bbox_inches='tight')
plt.clf()


data = []

for j in range(len(methods)):
    d = []
    for i in range(N_RADIUSES):
        if i < 7: 
            fileIndex = method_names.index(names_pre[j])
            suff = suffix[suffix_pre[j]]
        else:
            fileIndex = method_names.index(names_post[j])
            suff = suffix[suffix_post[j]]
        d.append(allData[fileIndex, i, -1])
    data.append(d)
data = np.array(data[0])/np.array(data)
plt.yscale('log')
for j in range(1,len(methods)):
    sns.lineplot(x=np.linspace(1,15,N_RADIUSES), y=data[j], label=methods[j], errorbar=None, color=color_dict[methods[j]], linestyle=line_styles[j], linewidth=lineswidth)
sns.lineplot(x=np.linspace(1,15,N_RADIUSES), y=1, color='black', linestyle='-', linewidth=lineswidth*0.5)
ax = plt.gca()

ax.yaxis.tick_left()
#change color of the ticks
ax.yaxis.set_tick_params(width=1, which='major', color='lightgrey')
ax.yaxis.set_tick_params(which='minor', color='lightgrey')

plt.xticks(np.arange(1, 16, 1), fontsize=23)
plt.yticks(fontsize=23)
plt.title(f'Impact of $r$ in Speedup, $n$ = 58368', fontsize=24)
plt.xlabel('$r$', fontsize=23)
plt.ylabel('Speedup', fontsize=23)
legend = plt.legend(ncols=1, fontsize=16)
# legend.get_frame().set_alpha(None)
# legend.get_frame().set_facecolor((0, 0, 1, 0.02))

plt.savefig(f'{plotsFolder}/radius_impact_speedup.pdf', bbox_inches='tight')
plt.clf()


# exit()

for R in range(1,16):
    # plt.yscale('log')
    # plt.figure(dpi=300)
    # plt.figure(figsize=(6.4, 7))

    # print(R)
    files = [f for f in glob.glob(f'../energy/power-*{GPUName}*.dat') if 'HIGH' not in f]
    rfiles = [f for f in files if f'RADIUS{R}.dat' in f]

    xs = ['' for i in range(6)]
    ys = [0 for i in range(6)]
    orders = ['' for i in range(6)]
    colors = ['' for i in range(6)]

    for i,f in enumerate(rfiles):
        with open(f, "r") as file:
            # print(f)
            data = file.readlines()
            val = float(data[-1].split()[2])
            label = f.split('-')[-3] + '-' + f.split('-')[-2] + f.split('-')[1]
            if 'millan' not in label:
                label = label.split('x')[0][:-2]
            if GPUName in label:
                label = label.split('-')[1]            
            if R < 8:
                if not (label in names_pre):
                    continue
                lab = names_pre.index(label)
                suff = suffix[suffix_pre[lab]]
            else:
                if not (label in names_post):
                    continue
                lab = names_post.index(label)
                suff = suffix[suffix_post[lab]]  
            print(f'{methods[lab]} {suff}', val)
            orders[lab] = f'{methods[lab]}{suff}'
            colors[lab] = color_dict[methods[lab]]
            xs[lab] = f'{methods[lab]}{suff}'
            ys[lab] = (58368.0+2*R)**2/val
    ax = sns.barplot(x=xs, y=ys, order=orders, palette=colors)
    # Annotate each bar with its value
    for i, p in enumerate(ax.patches):
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%.1e' % ys[i],
            fontsize=17, color='#2f2f2f', ha='center', va='bottom')
        #now in scientific notation
    #increase the size of the scale in top of the y axis

    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax = plt.gca()
    ax.yaxis.get_offset_text().set_fontsize(22)  # Adjust the fontsize as needed
    ax.xaxis.get_offset_text().set_fontsize(22)  # Adjust the fontsize as needed

    plt.title(f'Energy efficiency, $n$ = 58368, $r$ = {R}', fontsize=24)
    # plt.xlabel('Method', fontsize=18)
    plt.ylabel(r'$\frac{\text{Cells}}{J}$',rotation=0, fontsize=22, labelpad=20)
    # ax.yaxis.set_label_coords(-0.11, 0.475)  # Adjust position

    # plt.tight_layout()
    # plt.legend()

    plt.xticks(rotation=15, ha='center', fontsize=19)
    plt.yticks(fontsize=22)
    # plt.tight_layout()
    plt.savefig(f'{plotsFolder}/energy/bar-radius{R}.pdf', bbox_inches='tight')
    plt.clf()



yss = []
tots = []
for R in range(1,16):
    # plt.yscale('log')
    # plt.figure(dpi=300)
    # plt.figure(figsize=(6.4, 7))

    print(R)
    files = [f for f in glob.glob(f'../energy/power-*{GPUName}*.dat') if 'HIGH' not in f]
    rfiles = [f for f in files if f'RADIUS{R}.dat' in f]

    xs = ['' for i in range(6)]
    ys = [0 for i in range(6)]
    tot = [0 for i in range(6)]
    orders = ['' for i in range(6)]
    colors = ['' for i in range(6)]

    for i,f in enumerate(rfiles):
        with open(f, "r") as file:
            # print(f)
            data = file.readlines()
            val = float(data[-1].split()[2])
            label = f.split('-')[-3] + '-' + f.split('-')[-2] + f.split('-')[1]
            if 'millan' not in label:
                label = label.split('x')[0][:-2]
            if GPUName in label:
                label = label.split('-')[1]            
            if R < 8:
                if not (label in names_pre):
                    continue
                lab = names_pre.index(label)
                suff = suffix[suffix_pre[lab]]
            else:
                if not (label in names_post):
                    continue
                lab = names_post.index(label)
                suff = suffix[suffix_post[lab]]  
            print(f'{methods[lab]} {suff}', val)
            orders[lab] = f'{methods[lab]}{suff}'
            colors[lab] = color_dict[methods[lab]]
            xs[lab] = f'{methods[lab]}{suff}'
            ys[lab] = (58368.0+2*R)**2/val
            tot[lab] = val
    yss.append(ys)
    tots.append(tot)
    # ax = sns.barplot(x=xs, y=ys, order=orders, palette=colors)
    # Annotate each bar with its value
    # for i, p in enumerate(ax.patches):
    #     ax.text(p.get_x() + p.get_width() / 2., p.get_height(), '%.1e' % ys[i],
    #         fontsize=13, color='#2f2f2f', ha='center', va='bottom')
        #now in scientific notation
    #increase the size of the scale in top of the y axis

    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
yss = np.array(yss)
tots = np.array(tots)
print(yss)
#log y
# plt.yscale('log')
for i in range(6):
    sns.lineplot(x=np.linspace(1,15,15), y=yss[:,i], label=f'{methods[i]}', color=color_dict[methods[i]],
                  linestyle=line_styles[i], linewidth=lineswidth)
ax = plt.gca()

ax.yaxis.get_offset_text().set_fontsize(21)  # Adjust the fontsize as needed
ax.xaxis.get_offset_text().set_fontsize(21)  # Adjust the fontsize as needed
#move it to the left
# ax.yaxis.get_offset_text().set_x(-0.11)

# set the y ticks in scientific notation


plt.title(f'Impact of $r$ in Energy Efficency, $n$ = 58368', fontsize=23)
# plt.xlabel('Method', fontsize=21)
plt.ylabel(r'$\frac{\text{Cells}}{J}$',rotation=0, fontsize=22, labelpad=30)
plt.xlabel('$r$', fontsize=22)
#set the ticks in x from 1 to 15
plt.xticks(np.arange(1, 16, 1), fontsize=22)
# ax.yaxis.set_label_coords(-0.11, 0.475)  # Adjust position

# plt.tight_layout()
# plt.legend()
legend = plt.legend(ncols=1, fontsize=17)

plt.xticks(ha='center', fontsize=22)
plt.yticks(fontsize=22)
# plt.tight_layout()
plt.savefig(f'{plotsFolder}/energy/efficiency_radius.pdf', bbox_inches='tight')
plt.clf()

#############################################################################################################################################3
#############################################################################################################################################3
#############################################################################################################################################3
#############################################################################################################################################3

# plt.yscale('log')
for i in range(6):
    sns.lineplot(x=np.linspace(1,15,15), y=tots[:,i], label=f'{methods[i]}', color=color_dict[methods[i]],
                  linestyle=line_styles[i], linewidth=lineswidth)
ax = plt.gca()
#log y
plt.yscale('log')
ax.yaxis.get_offset_text().set_fontsize(21)  # Adjust the fontsize as needed
ax.xaxis.get_offset_text().set_fontsize(21)  # Adjust the fontsize as needed

plt.title(f'Impact of $r$ in Energy, $n$ = 58368', fontsize=24)
# plt.xlabel('Method', fontsize=21)
plt.ylabel('J',rotation=0, fontsize=21, labelpad=20)
plt.xlabel('$r$', fontsize=21)
#set the ticks in x from 1 to 15
plt.xticks(np.arange(1, 16, 1), fontsize=21)
# ax.yaxis.set_label_coords(-0.11, 0.475)  # Adjust position

# plt.tight_layout()
# plt.legend()
legend = plt.legend(ncols=1, fontsize=16)

plt.xticks(ha='center', fontsize=21)
plt.yticks(fontsize=21)
plt.tight_layout()
plt.savefig(f'{plotsFolder}/energy/total_energy_radius.pdf', bbox_inches='tight')
plt.clf()

import pandas as pd
import matplotlib.ticker as mticker



for R in range(1,16):
    plt.xscale('log')
    files = [f for f in glob.glob(f'../energy/power-*{GPUName}*.dat') if 'HIGH' not in f]
    rfiles = [f for f in files if f'RADIUS{R}.dat' in f]
    xs = ['' for i in range(6)]
    ys = [0 for i in range(6)]
    labs = [0 for i in range(6)]
    labels = ['' for i in range(6)]
    for i,f in enumerate(rfiles):
        with open(f, 'r') as file:
            data = pd.read_csv(file, delimiter='\s+')
            start = 10
            X = (data['acc-time'][start:]- data['acc-time'][start]*0.9)*1000.0/250.0
            Y = data['power'][start:]
            label = f.split('-')[-3] + '-' + f.split('-')[-2] + f.split('-')[1]
            if 'millan' not in label:
                label = label.split('x')[0][:-2]
            if GPUName in label:
                label = label.split('-')[1]
            if R < 8:
                if not (label in names_pre):
                    continue
                lab = names_pre.index(label)
                suff = suffix[suffix_pre[lab]]
            else:
                if not (label in names_post):
                    continue
                lab = names_post.index(label)
                suff = suffix[suffix_post[lab]]
            xs[lab] = X
            ys[lab] = Y
            labs[lab] = lab
            labels[lab] = methods[lab] + "" + suff

    for i in range(len(xs)):
        index = labs[i]
        ax = sns.lineplot(x=xs[index], y=ys[index], label=labels[index], color=color_dict[methods[index]], linestyle=line_styles[index], linewidth=lineswidth)
        ax.grid(True, which="both",linewidth=0.1, color='#efefef')



    
    plt.xlim([0.4, 20000.0])
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.title(f'Power consumption, $n$ = 58368, $r$ = {R}', fontsize=24)
    plt.xlabel('Time (ms)', fontsize=21)
    plt.ylabel('Power (W)', fontsize=21)
    legend = plt.legend(loc='upper right',# bbox_to_anchor=(0.5, 1.19),
        ncol=1, fontsize=16)
    # legend.get_frame().set_alpha(None)
    # legend.get_frame().set_facecolor((0, 0, 1, 0.03))
    plt.savefig(f'{plotsFolder}/energy/time_series-radius{R}.pdf', bbox_inches='tight')
    plt.clf()
exit()

# plt.figure(dpi=350)
# files = [f for f in glob.glob(f'../data/landscape-{GPUName}*.txt')]
# lands = np.zeros((31,31))
# for i,f in enumerate(files):
#     print(f)
#     with open(f, "r") as file:
#         data = file.readlines()
#         for line in data:
#             inter = line.split(',')
#             if len(inter) < 2:
#                 continue

#             x = int(inter[0].strip())-1
#             y = int(inter[1].strip())-1

#             if len(inter) < 4:
#                 lands[y,x] = None
#                 continue
                
            

#             val = (float(inter[2].strip()))
#             lands[y, x] = (val)

#         # sns.heatmap(lands, cmap='plasma')
#         # sns.heatmap(lands, cmap='viridis')
#         # sns.heatmap(lands, cmap='cividis')
#         # sns.heatmap(lands, cmap='coolwarm')
#         from matplotlib.ticker import LogFormatterSciNotation
#         class MF(LogFormatterSciNotation):
#             def set_locs(self, locs=None):
#                 self._sublabels = set([])
#                 # self.format_ticks = None

#         print(np.nanmin(lands), np.nanmax(lands))
#         # Apply logarithmic normalization to colormap
#         import matplotlib.colors as mcolors

#         norm = mcolors.LogNorm(vmin=np.nanmin(lands), vmax=np.nanmax(lands))


#         # Plot the heatmap with logarithmic colormap normalization
#         ax = sns.heatmap(lands, cmap='RdYlBu_r', norm=norm, cbar_kws={"label": "Time (ms)"})
#         for _, spine in ax.spines.items():
#             spine.set_visible(True)

#         # Manually set ticks and labels for the color bar
#         # Define linear ticks and labels

#         linear_ticks = np.linspace(np.nanmin(lands), np.nanmax(lands), num=10)
#         linear_labels = ["{:.2f}".format(val) for val in linear_ticks]

#         cbar = ax.collections[0].colorbar
#         import matplotlib.ticker as ticker

#         cbar.ax.yaxis.set_minor_formatter(ticker.NullFormatter())
#         cbar.ax.yaxis.set_minor_locator(ticker.NullLocator())
#         cbar.ax.yaxis.set_major_formatter(ticker.NullFormatter())

#         # Set linear ticks and labels
#         # cbar.set_format(MF())
#         # cbar.set_norm(norm)
#         # cbar.set_ticks(linear_ticks)# Remove the logarithmic ticks
#         cbar.set_ticks([])

#         # cbar.set_ticklabels(linear_labels)
#         # Define linear ticks and labels
#         linear_ticks = np.linspace(np.nanmin(lands), np.nanmax(lands), num=10)
#         linear_labels = ["{:.0f}".format(val) for val in linear_ticks]

#         # Set linear ticks and labels
#         cbar.set_ticks(linear_ticks)
#         cbar.set_ticklabels(linear_labels)
#         # plt.tight_layout()



#         # # Adjust the axis labels
#         # plt.xticks(np.arange(0.5, 31.5, 1), np.arange(1, 32))
#         # plt.yticks(np.arange(0.5, 31.5, 1), np.arange(1, 32))
#         # plt.gca().set_xticklabels(np.arange(1, 32, 4))
#         # plt.gca().set_yticklabels(np.arange(1, 32, 4))
#         plt.xticks(np.array([1, 4, 8, 12, 16, 20, 24, 28, 31])-0.5, [1, 4, 8, 12, 16, 20, 24, 28, 31])
#         plt.yticks(np.array([1, 4, 8, 12, 16, 20, 24, 28, 31])-0.5, [1, 4, 8, 12, 16, 20, 24, 28, 31], rotation=0)

#         plt.title("Optimal tile size, $n$ = 58368")
#         plt.xlabel("Size H")
#         plt.ylabel("Size V");
#         plt.tight_layout()
#         plt.savefig(f'{plotsFolder}/landscape-{f.split('-')[5]}.pdf', bbox_inches='tight')
#         plt.clf()


