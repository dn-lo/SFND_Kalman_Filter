import numpy as np

def f(mu, sigma2, x):
    return 1/np.sqrt(2.*np.pi*sigma2) * np.exp(-.5*(x-mu)**2 / sigma2)

print(f(10.,4.,10.)) #Change the 8. to something else!
