from scipy import stats
from scipy.special import gamma
from scipy.optimize import least_squares, minimize

import Display
from Display import plt, np
import Folder

Display.Clear()

if __name__ == '__main__':

    importGc = True

    if importGc:

        import sys
        sys.path.append(Folder.Join(Folder.Get_Path(), 'codes'))
        from FCBA.Functions import dfGc as data
        
        dfGc = data.copy()
        dfGc = dfGc[(dfGc['solveur']==1)&(dfGc['ftol']==1e-5)]
        dfGc = dfGc.sort_values(by=['Essai'])
        dfGc = dfGc.set_index(np.arange(dfGc.shape[0]))
        # print(dfGc.describe())

        Gc = dfGc['Gc'].values
        n = Gc.size
    
        x = np.linspace(1e-12,Gc.max()*1.5,1000)

        # samples
        y = Gc

    else:
        # np.random.seed(2)
        # n = 20000
        n = 18

        x = np.linspace(1e-12,450,1000)

        a = 90 # shape
        b = 2.32 # scale
        y1 = stats.gamma.pdf(x, a, scale=b)

        y = stats.gamma.rvs(a, scale=b, size=n)
    
    x0 = np.array([50,2])
    mean = np.mean(y)
    std = np.std(y, ddof=1) # non bias sigma

    # --------------------------------------------------------------------------------------------
    # Maximal likehood
    # --------------------------------------------------------------------------------------------
    # V1
    a_ml_f, _, b_ml_f = stats.gamma.fit(y, floc=0)

    # V2
    def J_ml(v):
        a,b = tuple(v)
        J = -n*(a*np.log(b) + np.log(gamma(a))) + (a-1)*np.sum(np.log(y)) - 1/b*np.sum(y)
        return -J
    res_ml = minimize(J_ml, x0, bounds=((2,np.inf),(1e-12,np.inf)), tol=1e-12)
    a_ml, b_ml = tuple(res_ml.x)

    # Test
    test_a_ml = np.abs(a_ml - a_ml_f)/a_ml
    test_b_ml = np.abs(b_ml - b_ml_f)/b_ml

    y2 = stats.gamma.pdf(x, a_ml, scale=b_ml)

    if importGc:
        print("maximum likehood :")
        print(f'a = {a_ml:.3f}')
        print(f'b = {b_ml:.3f}')
    else:
        print("maximum likehood errors :")
        print(f'err a = {np.abs(a_ml-a)/a*100:.3f} %')
        print(f'err b = {np.abs(b_ml-b)/b*100:.3f} %')

    # --------------------------------------------------------------------------------------------
    # Least squares
    # --------------------------------------------------------------------------------------------

    def J_ls(v: np.ndarray, option:int):
        a, b = tuple(v)

        mean_num = stats.gamma.mean(a, scale=b)
        std_num = stats.gamma.std(a, scale=b)
        if option==1:
            # dont work for least squares
            J = (mean_num-mean)**2/mean**2 + (std_num-std)**2/std**2
        elif option==2:        
            J = np.array([mean_num, std_num]) - np.array([mean, std])

        return J

    # # V1
    # # J must return vector values in this case
    # res_ls = least_squares(J_ls, x0, bounds=((2,0),(np.inf, np.inf)), args=(2,))

    # V2 
    # J must return scalar values when you use minimize
    res_ls = minimize(J_ls, x0, bounds=((2,np.inf),(0, np.inf)), args=(1,))

    a_ls, b_ls = tuple(res_ls.x)

    y3 = stats.gamma.pdf(x, a_ls, scale=b_ls)

    if importGc:
        print("\nleast squares :")
        print(f'a = {a_ls:.3f}')
        print(f'b = {b_ls:.3f}')
    else:
        print("\nleast squares erros :")
        print(f'err a = {np.abs(a_ls-a)/a*100:.3f} %')
        print(f'err b = {np.abs(b_ls-b)/b*100:.3f} %')

    # --------------------------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------------------------
    ax = plt.subplots()[1]
    ax.set_title('pdf')
    if importGc:
        print("\nmaximum likehood vs least squares :")
        print(f'err a = {np.abs(a_ls-a_ml)/a_ml*100:.3f} %')
        print(f'err b = {np.abs(b_ls-b_ml)/b_ml*100:.3f} %')
        # pdfGc = stats.gamma.pdf(Gc, a_ml, scale=b_ml)
        # pdfGc = stats.gamma.pdf(Gc, a_ls, scale=b_ls)
        # ax.plot(Gc,pdfGc,label='exp',ls='',marker='+', zorder=10, c='red')        
    else:
        ax.plot(x,y1,label='exp')

    ax.plot(x,y2,label='maximum likehood')
    ax.plot(x,y3,label='least squares')
    ax.legend()

    plt.show()
    pass