from scipy import stats
from scipy.special import gamma
from scipy.optimize import least_squares, minimize

import Display
from Display import plt, np

Display.Clear()

n = 20000
x = np.linspace(1e-12,500,1000)

a = 90 # shape
b = 2.32 # scale
y1 = stats.gamma.pdf(x, a, scale=b)

x0 = np.array([50,2])

# samples
y = stats.gamma.rvs(a, scale=b, size=n)
mean = np.mean(y)
std = np.std(y, ddof=1) # non bias sigma


# ----------------------------------------------
# Maximal likehood
# ----------------------------------------------
# V1
a_ml, _, b_ml = stats.gamma.fit(y, floc=0)

# V2
def J_ml(v):
    a,b = tuple(v)
    J = -n*(a*np.log(b) + np.log(gamma(a))) + (a-1)*np.sum(np.log(y)) - 1/b*np.sum(y)
    return -J
res_ml = minimize(J_ml, x0, bounds=((2,np.inf),(1e-12,np.inf)), tol=1e-12)
a_ml, b_ml = tuple(res_ml.x)

# Test
test_a_ml = np.abs(a_ml - a_ml)/a_ml
test_b_ml = np.abs(b_ml - b_ml)/b_ml

y2 = stats.gamma.pdf(x, a_ml, scale=b_ml)

print("maximum likehood errors :")
print(f'err a = {np.abs(a_ml-a)/a*100:.3f} %')
print(f'err b = {np.abs(b_ml-b)/b*100:.3f} %')

# ----------------------------------------------
# Least squares
# ----------------------------------------------

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
# res_ls = least_squares(J, x0, bounds=((2,0),(np.inf, np.inf)), args=(2,))

# V2 
# J must return scalar values when you use minimize
res_ls = minimize(J_ls, x0, bounds=((2,np.inf),(0, np.inf)), args=(1,))


a_ls, b_ls = tuple(res_ls.x)

y3 = stats.gamma.pdf(x, res_ls.x[0], scale=res_ls.x[1])

print("\nleast squares erros :")
print(f'err a = {np.abs(a_ls-a)/a*100:.3f} %')
print(f'err b = {np.abs(b_ls-b)/b*100:.3f} %')


# ----------------------------------------------
# Plot
# ----------------------------------------------
ax = plt.subplots()[1]
ax.set_title('pdf')
ax.plot(x,y1,label='exp')
ax.plot(x,y2,label='maximum likehood')
ax.plot(x,y3,label='least squares')
ax.legend()


# x = stats.norm.rvs()

plt.show()
pass