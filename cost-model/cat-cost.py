import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def compute_CAT(n, p, q, w, h, C, c, u, P_sm, Z_sm, P, d, r):

    #a1=1.38238039e-01
    #a2=-2.89361723e-02
    #a3=-4.56041880e-02
    #a4=2.27145466e-03
    #a5=4.20920360e-03
    #a6=6.14753339e-03
    #a7=-6.68874545e-05
    #a8=-1.54315806e-04
    #a9=-1.87263528e-04
    #a10=-1.66422110e-04
    #a11=6.93991400e-07
    #a12=2.03570388e-06
    #a13=2.74914534e-06
    #a14=1.20793990e-06
    #a15=2.13375485e-06
    a1=  2.43024186e-01
    a2= -4.47182865e-02
    a3= -8.20349265e-02
    a4=  3.27250347e-03
    a5=  7.98429639e-03
    a6=  9.54041156e-03
    a7= -8.63804245e-05
    a8= -3.11990690e-04
    a9= -1.99175457e-04
    a10=-1.89184273e-04
    a11= 7.04926203e-07
    a12= 4.36420144e-06
    a13= 3.02194010e-06
    a14=-2.66749914e-06
    a15= 2.42283670e-06
    # tile efficiency factor
    b=10
    E=1.0 + b*(a1 + a2*w + a3*h + a4*w**2 + a5*h**2 + a6*w*h + a7*w**3 + a8*h**3+ a9*w**2*h + a10*w*h**2  + a11*w**4 + a12*h**4 + a13*w**3*h + a14*w**2*h**2 + a15*w*h**3)
    # maximum benefit is an order of magnitude as shown by the heatmap
    # shared band fragments
    S_pi = c*ceil((p*q) / P_sm)
    # tile H
    Q_H = ceil((w*(h + 2)) / Z_sm) * (3*C + 3*u + 4*c)
    # tile R
    Q_R = ceil((w*h) / Z_sm) * (6*c + 3*u + C)
    # transition function
    f = (d + 3*C) * ceil(((w*h) * (p*q)) / P_sm)
    # tile cost
    Q = 3*S_pi + E*(Q_H + Q_R + f)
    #U = ceil((n**2 / (p * q * w * h))/P)
    U = ceil( (ceil(ceil(n/p)/h)*ceil(ceil(n/q)/w))/P )
    CAT=U*Q
    print(f"CAT: {n=}, {U=}, E={round(E,2)}, {S_pi=}, {Q_H=}, {Q_R=}, {f=}, Q={round(Q,1)}, => {round(CAT,1)}")
    return CAT

def compute_Baseline(n, P, P_sm, C, d, r):
    result = ceil(n**2 / (P * P_sm)) * (((1 + 2*r)**2 * C) + ((1 + 2*r)**2 - 1) + d + C)
    factor = ceil(n**2 / (P * P_sm))
    thread = (((1 + 2*r)**2 * C) + ((1 + 2*r)**2 - 1) + d + C)
    print(f"BASELINE: {n=}, {factor=}, {thread=}, {result=}")
    return ceil(n**2 / (P * P_sm)) * (((1 + 2*r)**2 * C) + ((1 + 2*r)**2 - 1) + d + C)

def plot_curves(n_values, p, q, w, h, C, c, u, P_sm, Z_sm, P, d, r):
    CAT_values = [compute_CAT(n, p, q, w, h, C, c, u, P_sm, Z_sm, P, d, r) for n in n_values]
    Baseline_values = [compute_Baseline(n, P, P_sm, C, d, r) for n in n_values]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(n_values, CAT_values, label='CAT')
    plt.plot(n_values, Baseline_values, label='Baseline')
    plt.xlabel('n')
    plt.ylabel('Parallel Time')
    plt.title('Parallel Time vs n')
    plt.legend()

    plt.subplot(1, 2, 2)
    speedup_values = np.divide(Baseline_values, CAT_values)
    plt.plot(n_values, speedup_values, label='Speedup', color='orange')
    plt.xlabel('n')
    plt.ylabel('Speedup')
    plt.title('Speedup vs n')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and plot CAT and Baseline expressions")
    parser.add_argument("-n", type=int, help="Value for n")
    parser.add_argument("-p", type=int, help="Value for p")
    parser.add_argument("-q", type=int, help="Value for q")
    parser.add_argument("-w", type=int, help="Value for w")
    parser.add_argument("-hh", type=int, help="Value for h")
    parser.add_argument("-C", type=int, help="Value for C")
    parser.add_argument("-c", type=int, help="Value for c")
    parser.add_argument("-u", type=int, help="Value for u")
    parser.add_argument("-P_sm", type=int, help="Value for P_sm")
    parser.add_argument("-Z_sm", type=int, help="Value for Z_sm")
    parser.add_argument("-P", type=int, help="Value for P")
    parser.add_argument("-d", type=int, help="Value for d")
    parser.add_argument("-r", type=int, help="Value for r")
    args = parser.parse_args()

    if not all(vars(args).values()):
        parser.error("All arguments are required")

    n_values = np.arange(2**10, 2**20 + 1, 1024)  # You can adjust this range accordingly
    plot_curves(n_values, args.p, args.q, args.w, args.hh, args.C, args.c, args.u, args.P_sm, args.Z_sm, args.P, args.d, args.r)
