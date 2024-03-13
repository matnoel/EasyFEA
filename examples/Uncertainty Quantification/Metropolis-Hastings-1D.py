import Folder
import Display

from Display import np, plt

import scipy.integrate as integrate
from scipy import stats

if __name__ == '__main__':

    Display.Clear()

    x_array = np.linspace(-10, 10, 1000)


    # target distribution
    pX = lambda x: 5 * np.exp(-x*x/2) + np.exp(-(x - 4)**2/2)

    # gauss = lambda mu, sig, x: 1/(sig*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2/(2*sig**2))
    # pX = lambda x: 0.1 * gauss(-2,1,x) + 0.3 * gauss(2,1,x) + 0.5 * gauss(6,0.5,x)

    # normalization constant
    k_num, error = integrate.quad(pX, x_array.min(), x_array.max())
    # testK = integrate.quad(lambda x: pX(x)/k_num, x_array.min(), x_array.max())[0]

    y_array = pX(x_array)

    ax1 = Display.init_Axes()
    ax1.plot(x_array, y_array, label='p(x)')
    ax1.legend()


    # metropolis-hastings algorithm
    x_0 = 5 # starting value
    sig = 1 # standart deviation

    def Metropolis_Hastings_1D(x_0: float, sig: float, burn_in: int, nSamples: int):
        # in this function we use a normal distribution as our guess function

        # pdf normal with mean t0 and standard deviation sig evaluated in t1
        q = lambda t1, t0: np.exp(-(t1 - t0)**2/(2*sig**2)) * 1/(sig*np.sqrt(2 * np.pi))
        # same as q = stats.norm.pdf(t1, t0, sig)

        x_t = x_0
        samples = []
        for i in range(burn_in+nSamples):        
            # proposal / guess
            x_tp1 = np.random.normal(x_t, sig)
            # same as x_tp1 = stats.norm.rvs(x_t, sig)
            
            # accept ratio a
            a = (pX(x_tp1) * q(x_t, x_tp1)) / (pX(x_t) * q(x_tp1, x_t))
            # a = pX(x_tp1) / pX(x_t) if q(x_t, x_tp1)/q(x_tp1, x_t) == 1 if the guess function is symmetric
            
            if np.random.uniform(0, 1) < a:
                x_t = x_tp1
                if i >= burn_in:
                    samples.append(x_t)

        assert len(samples) > 0

        rejectRatio = 1 - len(samples)/nSamples

        return np.array(samples), rejectRatio


    nSamples = 200000
    burn_in = 1000

    samples, rejectRatio = Metropolis_Hastings_1D(x_0, sig, burn_in, nSamples)



    print(f"rejectRatio = {rejectRatio*100:.3f}%")

    # [20, 25] is a good rejectRatio
    # [20, 25] is a good rejectRatio





    ax2 = Display.init_Axes()
    ax2.plot(x_array, y_array/k_num, label='p(x)/k')
    ax2.hist(samples, bins=x_array.size//2, histtype='bar', density=True, label='bins')
    ax2.legend()


    plt.show()