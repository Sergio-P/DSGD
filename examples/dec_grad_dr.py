import random
import numpy as np
import matplotlib.pyplot as plt
from dsgd.core import compute_gradient_dempster_rule

ax = random.random() * 0.5
ay = random.random() * 0.5
ae = 1 - ax - ay
bx = random.random() * 0.5
by = random.random() * 0.5
be = 1 - bx - by

P = 0.5

axs = []
ays = []
bxs = []
bys = []
aes = []
bes = []
Js = [1]


def normalize(a, b, c):
    a = 0 if a < 0 else 1 if a > 1 else a
    b = 0 if b < 0 else 1 if b > 1 else b
    c = 0 if c < 0 else 1 if c > 1 else c
    n = a + b + c
    return a/n, b/n, c/n


def print_maf():
    axs.append(ax)
    ays.append(ay)
    bxs.append(bx)
    bys.append(by)
    aes.append(ae)
    bes.append(be)
    # print "phi\t %.3f" % 0.
    print(" X\t %.3f \t %.3f" % (ax, bx))
    print(" Y\t %.3f \t %.3f" % (ay, by))
    print("X,Y\t %.3f \t %.3f" % (ae, be))
    print("-" * 18)


alpha = 0.01
gamma = 0.01
certain_J = 0.1

print_maf()
J = 0
last_J = 1

BATCH_SIZE = 20
MIN_DELTA_J = 0.0005

data = [[0, 1] if random.random() > P else [1, 0] for _ in range(BATCH_SIZE)]

# Optimization using belief and dempster rule
i = 0
while abs(last_J - J) > MIN_DELTA_J:
    last_J = J
    J = 0.
    for i in range(BATCH_SIZE):
        # Computing
        px = (ax + bx - ax * bx - ax * by - ay * bx) / (ax + bx + ay + by - ax * bx - ay * by - 2 * ax * by - 2 * ay * bx)
        py = (ay + by - ay * by - ax * by - ay * bx) / (ax + bx + ay + by - ax * bx - ay * by - 2 * ax * by - 2 * ay * bx)
        y = [px, py]
        y_cup = data[i]
        rx, ry = y_cup
        # Optimization
        dJdax, dJday, dJdbx, dJdby = compute_gradient_dempster_rule(ax, ay, bx, by, rx, ry)
        # Updating
        ax -= alpha * dJdax
        ay -= alpha * dJday
        bx -= alpha * dJdbx
        by -= alpha * dJdby
        Jn = .5 * ((y[0] - y_cup[0])**2 + (y[1] - y_cup[1])**2)
        # ae = 1 - ax - ay
        # be = 1 - bx - be
        ae += gamma * (Jn - certain_J)
        ax, ay, ae = normalize(ax, ay, ae)
        be += gamma * (Jn - certain_J)
        bx, by, be = normalize(bx, by, be)

        J += Jn / BATCH_SIZE
    Js.append(J)
    # Debugging
    print_maf()
    i += 1
    alpha *= .95

print("Iters: %d" % i)

plt.figure(figsize=(14, 4))

plt.subplot(131)
plt.plot(np.arange(len(axs)), axs, label="X")
plt.plot(np.arange(len(ays)), ays, label="Y")
plt.plot(np.arange(len(aes)), aes, label="X,Y")
plt.axis([0, len(axs)-1, -0.02, 1.0])
plt.xlabel("Iterations")
plt.ylabel("Mass values")
plt.title("Optimization for mA")
plt.legend()
plt.grid(True)

plt.subplot(132)
plt.plot(np.arange(len(bxs)), bxs, label="X")
plt.plot(np.arange(len(bys)), bys, label="Y")
plt.plot(np.arange(len(bes)), bes, label="X,Y")
plt.axis([0, len(bxs)-1, -0.02, 1.0])
plt.xlabel("Iterations")
plt.ylabel("Mass values")
plt.title("Optimization for mB")
plt.legend()
plt.grid(True)

plt.subplot(133)
plt.plot(np.arange(len(Js)), Js)
plt.title("Error")
plt.xlabel("Iterations")
plt.ylabel("Sqauared mean error")
plt.axis([0, len(Js)-1, 0.0, 1.0])
plt.show()
