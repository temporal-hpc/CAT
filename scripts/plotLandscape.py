import numpy as np
import sys

if len(sys.argv) != 2:
    print("Run with args:\n<filename.npy>")
    exit()

filename_npy = sys.argv[1]
filename_txt = filename_npy[:-4]+".txt"
landscape = np.load(filename_npy)
with open(filename_txt, "r") as f:
    details = eval(f.read())


landscape = np.log(landscape)
lim = np.partition(np.unique(landscape.flatten()), -2)[-2]

import matplotlib.pyplot as plt
# Plot the heatmap
plt.imshow(landscape, vmax=lim)
plt.colorbar()
plt.xlabel("NREGIONS_H")
plt.ylabel("NREGIONS_V")
plt.savefig(filename_npy[:-4]+"_heatmap.png")

