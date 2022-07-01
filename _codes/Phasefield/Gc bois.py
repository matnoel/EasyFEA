import numpy as np
import matplotlib.pyplot as plt

E=11580e6
v=0.44

# Valeurs de 10.1098/rspa.1985.0034
Kic = np.linspace(1e-2,1) # MPa m^(1/2)
Kic *= 1e6 # Pa m^(1/2)

# Formule dans "Rupture et Plasticité_ Pierre Suquet.pdf" eq 3.19

Gc = Kic**2 * (1-v**2)/E

print(f"mean(Gc) = {np.mean(Gc)}")

plt.rc('font', size=13) 
plt.plot(Kic, Gc)
plt.xlabel("$K_{ic} \ [Pa \ \sqrt{m}]$")
plt.ylabel("$G_c \ [J \ m^{-2}]$")
plt.show()