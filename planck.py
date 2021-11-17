import numpy as np
import matplotlib.pyplot as plt


k=1.38e-23 # J K-1
h=6.626e-34 #J s
c=3e8*1e9       #um/s

T=[20000]
x = np.linspace(10,3000,1000)    #um

for ti in T:
    dis=1/(np.exp(h*c/(x*k*ti))-1)
    B = (2*h*c**2/x**5)*dis
    plt.plot(x,B,'y-',label=str(ti))
plt.plot([121.6,121.6],[0,1.3e-11],'k--',label=str('121.6nm'))

plt.legend()
plt.grid()
plt.title("Planck's radiation law")
plt.xlabel('WaveLength(nm)')
plt.ylabel('Spectral radiance')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,1))
plt.show()