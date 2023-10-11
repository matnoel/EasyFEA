import Display
import Materials
from Display import plt, np

from scipy import stats

Display.Clear()

material_str = "Elas_Isot"

if material_str == "Elas_Isot":
    material = Materials.Elas_Isot(3, 210)

    C = material.C
    C1 = material.get_bulk()
    C2 = material.get_mu()

    Ivect = np.array([1,1,1,0,0,0])

    I = np.einsum('i,j->ij', Ivect, Ivect)
    Isym = np.eye(6)

    E1 = 1/3 * I
    E2 = Isym - E1

    test_C = np.linalg.norm((3*C1*E1  + 2*C2*E2) - C)/np.linalg.norm(C)
    assert test_C <= 1e-12

else:
    raise Exception("Not implemented")


x = np.linspace(0,500,1000) #

shape1, loc1, scale1 = stats.gamma.fit(np.random.normal(C1,0.1*C1, 1000))
shape2, loc2, scale2 = stats.gamma.fit(np.random.normal(C2,0.1*C2, 1000))

y1 = stats.gamma.pdf(x, shape1, loc=loc1, scale=scale1)
y2 = stats.gamma.pdf(x, shape2, loc=loc2, scale=scale2)
# y = stats.gamma.pdf(x, shape, scale=scale)

ax_pdf = plt.subplots()[1]
ax_pdf.plot(x,y1, label="$p_{C1}(c)$")
ax_pdf.plot(x,y2, label="$p_{C2}(c)$")
ax_pdf.legend()
ax_pdf.set_xlabel("$c$")

pass