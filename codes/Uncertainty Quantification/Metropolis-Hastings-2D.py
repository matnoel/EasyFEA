"""this is an example from
https://doi.org/10.1007/978-3-319-54339-0
Soize 2017 Uncertainty Quantification : An Accelerated Course with Advanced Applications in Computational Engineering.
4.2.2
"""


import Folder
import Display

from Display import np, plt

import scipy.integrate as integrate
from scipy import stats

Display.Clear()

N = 1000
z1_array = np.linspace(-1.5, 1.5, N)
z2_array = np.linspace(-1.5, 2, N)

Z1, Z2 = np.meshgrid(z1_array, z2_array)

gauss = lambda mu, sig, x: 1/(sig*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sig**2))

pZ = lambda z1, z2: np.exp(-15*(z1**3-z2)**2 - (z2-0.3)**4)

# normalization constant
k_num, error = integrate.dblquad(pZ, z1_array.min(), z1_array.max(), z2_array.min(), z2_array.max())

ax1 = plt.subplots()[1]
ax1.contourf(Z1, Z2, pZ(Z1,Z2))
ax1.set_xlabel("z1"); ax1.set_ylabel("z2"); ax1.set_title('pdf')


# metropolis-hastings algorithm
z_0 = np.array([1.4, -0.75]) # starting point
sig = 0.1 # standart deviation 0.1 is good
cov = sig**2 * np.eye(2)

def Metropolis_Hastings_2D(z_0: np.ndarray, cov: np.ndarray, burn_in: int, nSamples: int):
    # in this function we use a normal distribution as our guess function

    # pdf multivariate normal with mean t0 and covariance cov evaluated in t1    
    q = lambda t1, t0: stats.multivariate_normal.pdf(t1, t0, cov)

    z_t = z_0
    samples = []
    for i in range(burn_in+nSamples):
        z_tp1 = stats.multivariate_normal.rvs(z_t, cov)

        accept_prob = (pZ(*z_tp1) * q(z_t, z_tp1))/(pZ(*z_t) * q(z_tp1, z_t))
        test = q(z_t, z_tp1)/q(z_tp1, z_t) # = 1 if the guess function is symmetric
        # then accept_prob = pZ(z_tp1)/pX(z_t)

        isAccepted = np.random.uniform(0, 1) < accept_prob
        
        if isAccepted:
            z_t = z_tp1

        if isAccepted and i >= burn_in:
            samples.append(z_t)

    assert len(samples) > 0

    rejectRatio: float = 1 - len(samples)/nSamples

    return np.array(samples), rejectRatio


nSamples = 10000
burn_in = 100

samples, rejectRatio = Metropolis_Hastings_2D(z_0, sig, burn_in, nSamples)



print(f"rejectRatio = {rejectRatio*100:.3f}%")

# [20, 25] is a good rejectRatio
# [20, 25] is a good rejectRatio





ax2 = plt.subplots()[1]
ax2.contourf(Z1, Z2, pZ(Z1,Z2))
ax2.scatter(*samples.T,c='red',marker='.')
ax2.set_xlabel("z1"); ax2.set_ylabel("z2"); ax2.set_title('tirages')


plt.show()