import matplotlib.pyplot as plt
import numpy as np

fig, axFcGc = plt.subplots()

Gc = np.linspace(0.1,10, 50)

Fc_lin = Gc*9
Fc_nonLin = Gc**2
axFcGc.plot(Gc, Fc_lin)

axFcGc.plot(Gc, Fc_nonLin)
# plt.rcParams['text.usetex'] = True
axFcGc.set_xlabel('$G_c(\mu, \delta)$', fontsize=14)
axFcGc.set_ylabel('$F_c(\mu, \delta)$', fontsize=14)
axFcGc.grid()


plt.show()