import PostTraitement as PostTraitement
import matplotlib.pyplot as plt
import numpy as np

fig, axFcGc = plt.subplots()

Gc = np.linspace(0,10, 50)

Fc_lin = lambda Gc : Gc*9
Fc_nonLin = lambda Gc : Gc**2

axFcGc.plot(Gc, Fc_lin(Gc))
axFcGc.plot(Gc, Fc_nonLin(Gc))
# mat = lambda x: np.array([[x**3,x**2,x,1],[x**3,x**2,x,1],[x**3,x**2,x,1],[x**3,x**2,x,1]])

# vect = np.linalg.solve(mat([0,2,3,4]),np.array([0,2*Fc_lin(2)-Fc_nonLin(2),Fc_lin(9)]))

# Fc_nonLin2 = vect[0]*Gc**3 + vect[1]*Gc**2 + vect[2]*Gc + vect[3]

# axFcGc.plot(Gc, Fc_nonLin2)
# axFcGc.plot(Fc_nonLin(Gc), Gc)
# plt.rcParams['text.usetex'] = True
axFcGc.set_xlabel('$G_c$', fontsize=14)
axFcGc.set_ylabel('$F_c$', fontsize=14)
axFcGc.grid()

# PostTraitement.Save_fig("C:\\Users\\Matthieu\\OneDrive\\__Doctorat\\Presentations\\COPIL 3\\fig","Identification Gc")

plt.show()