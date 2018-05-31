import random
import numpy as np
import matplotlib.pyplot as plt

w1 = 0.98
w2 = 0.01
we = 1 - w1 - w2
P = 0.48

a = []
b = []
ab = []
Js = [1]


def normalize(a, b, c):
    a = 0 if a < 0 else 1 if a > 1 else a
    b = 0 if b < 0 else 1 if b > 1 else b
    c = 0 if c < 0 else 1 if c > 1 else c
    n = a + b + c
    return a/n, b/n, c/n


def print_maf():
    a.append(w1)
    b.append(w2)
    ab.append(we)
    # print "phi\t %.3f" % 0.
    print " A\t %.3f" % w1
    print " B\t %.3f" % w2
    print "A,B\t %.3f" % we
    print "-" * 11

alpha = 0.01
gamma = 0.01
certain_J = 0.1

print_maf()
J = 0
last_J = 1

batch_size = 100

# Optimization using belief
i = 0
while abs(last_J - J) > 0.0005:
    last_J = J
    J = 0.
    for _ in range(batch_size):
        # Computing
        y = [w1/(w1+w2), w2/(w1+w2)]
        y_cup = [0, 1] if random.random() > P else [1, 0]
        # Optimization for w1
        dJdw1 = -(y_cup[0] - w1/(w1 + w2))*(1/(w1 + w2) - w1/(w1 + w2)**2) + w2*(y_cup[1] - w2/(w1 + w2))/(w1 + w2)**2
        # Optimization for w2
        dJdw2 = -(y_cup[1] - w2/(w1 + w2))*(1/(w1 + w2) - w2/(w1 + w2)**2) + w1*(y_cup[0] - w1/(w1 + w2))/(w1 + w2)**2
        # Updating
        w1 -= alpha * dJdw1
        w2 -= alpha * dJdw2
        Jn = .5 * ((y[0] - y_cup[0])**2 + (y[1] - y_cup[1])**2)
        we += gamma * (Jn - certain_J)
        w1, w2, we = normalize(w1, w2, we)
        J += Jn/batch_size
    Js.append(J)
    # Debugging
    print_maf()
    i += 1
    # alpha *= .95

print "Iters: %d" % i

plt.subplot(121)
plt.plot(np.arange(len(a)), a, label="A")
plt.plot(np.arange(len(a)), b, label="B")
plt.plot(np.arange(len(a)), ab, label="A,B")
plt.axis([0, len(a)-1, -0.02, 1.0])
plt.xlabel("Iterations")
plt.ylabel("Mass values")
plt.title("Optimization of masses")
plt.legend()
plt.grid(True)
plt.subplot(122)

plt.plot(np.arange(len(Js)), Js)
plt.title("Error")
plt.xlabel("Iterations")
plt.ylabel("Sqauared mean error")
plt.axis([0, len(a)-1, 0.0, 1.0])
plt.show()
