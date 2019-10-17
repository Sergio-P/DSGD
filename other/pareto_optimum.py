import numpy as np
import matplotlib.pyplot as plt

d = np.matrix([[0.3672, 0.0382],
     [0.2195, 0.0382],
     [0.1062, 0.0476],
     [0.0522, 0.0524],
     [0.0363, 0.0762],
     [0.0155, 0.1095]])

# d[:, 0] = d[:, 0] + (1./162 * (5 - np.arange(6))).reshape(-1,1)
print(d[:,0])

s = d.sum(1)
print(s)
plt.plot(d[:, 0], d[:, 1], ".", ms=10, c="blue")
plt.plot([0, 0.1046], [0.1046, 0], "--", c="grey")
plt.plot(d[3, 0], d[3, 1], ".", ms=15, c="red")
plt.xlabel("$Q_{CPLX}$")
plt.ylabel("$Error$")
plt.axis([0, 0.38, 0, 0.12])
plt.grid(True)
plt.show()
