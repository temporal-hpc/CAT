import numpy as np
import sys

if len(sys.argv) != 2:
    print("Run with args:\n<filename.npy>")
    exit()

filename = sys.argv[1]
landscape = np.load(filename)

print(landscape.shape)
print(np.unravel_index(landscape.argmin(), landscape.shape))
print(landscape.min())
