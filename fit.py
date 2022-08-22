import lmfit
from lmfit.models import LinearModel
import numpy as np
import matplotlib.pyplot as plt

# def ideal(r, T = 35e3, I0 = 18e3):
#     return T/(r**2 + I0)
    
# model = lmfit.Model(ideal)
lin = LinearModel()
lin.set_param_hint('sound', expr = '2*pi*intercept/slope')
lin.set_param_hint('intercept', value = 11225, vary = True)

radii = np.array([55,45,35,25,25, 35, 45, 55])/100
fan_freq = np.array([-1.724837662337662, -1.826298701298701, -1.92775974025974,-1.92775974025974,1.92775974025974,1.92775974025974,1.826298701298701,1.724837662337662])
freqs = np.array([11062.5, 11068.359375, 11091.796875,11121.09375,11296.875,11343.75,11390.625,11414.0625])
# result = model.fit(freqs, r = radii)
# result.plot()
# plt.grid()
# print(result.fit_report())
# plt.show()

lin_fit = lin.fit(freqs, x = radii*fan_freq)
lin_fit.plot()
plt.grid()
print(lin_fit.fit_report())
plt.show()