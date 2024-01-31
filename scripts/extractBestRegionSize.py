import numpy as np
import sys

if len(sys.argv) != 2:
    print("Run with args:\n<filename.npy>")
    exit()

filename = sys.argv[1]
landscape = np.load(filename)

print(landscape)
best =np.unravel_index(landscape.argmin(), landscape.shape) 
print(best)
print(landscape.min())


with open(filename[:-4]+".txt", "r") as f:
    data = dict(eval(f.read()))
    print(data[best[0]*32 + best[1]])
