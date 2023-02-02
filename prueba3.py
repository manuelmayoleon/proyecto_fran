import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
import pandas as pd
import numpy as np
import scipy . stats as ss
import math
import matplotlib . mlab as mlab
import matplotlib.animation as manimation
from scipy import optimize
import os as os 
import pandas as pd
from re import search
import pycircular
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib.transforms import (
    Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
#! Define function to read all sheets of excel file  
def readAllSheets(filename):
    if not os.path.isfile(filename):
        return None
    
    xls = pd.ExcelFile(filename)
    sheets = xls.sheet_names
    results = {}
    for sheet in sheets:
        results[sheet] = xls.parse(sheet)
        
    xls.close()
    
    return results, sheets

sheets ,   names  = readAllSheets("Cuantificacion_cilios_2.xlsx")

print(names)


wt = list()
ko = list()
plx = list()
for i in names: 
    for j in sheets[i]:
        if search("WT_", j):
                wt.append(sheets[i][j][:])
        if search("KO_", j):
            ko.append(sheets[i][j][:])
        if search("PLX_",j):
            plx.append(sheets[i][j][:])

# print(sheets['WT'])


# for i in sheets['WT']['WT_345']:
#     wt.append(i)
    # wt.append(sheets['WT']['WT_346'][0:])
    #     wt.append(i)
# wt.append(sheets['WT']['WT_345'][:])
# wt.append(sheets['WT']['WT_346'][:])
# wt.append(sheets['WT']['WT_347'][:])
# wt.append(sheets['WT']['WT_348'][:])
# wt.append(sheets['WT']['WT_646'][:])

# wt.append(sheets['WT']['WT_345'][:])
# wt.append(sheets['WT']['WT_346'][:])
# wt.append(sheets['WT']['WT_347'][:])
# wt.append(sheets['WT']['WT_348'][:])
# wt.append(sheets['WT']['WT_646'][:])

num_bins =10 

fig22 = plt.subplots (1,1,figsize=(10,10))

plt.xlabel ( r' $W_T$ ', fontsize=20)
plt.ylabel ( r' Frecuencia ',fontsize=20)

plt.title ( r'Histograma de WT   ',fontsize=30)

n,bins,patches = plt.hist(wt ,num_bins,density ='false',facecolor ='C1',edgecolor='white')
n2,bins2,patches2 = plt.hist(ko ,num_bins,density ='false',facecolor ='C1',edgecolor='white')
n3,bins3,patches3 = plt.hist(plx ,num_bins,density ='false',facecolor ='C1',edgecolor='white')


# print(bins)
# print(n)
# Fixing random state for reproducibility
np.random.seed(19680801)

# Compute pie slices
N = 10
# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# radii = 



#! Radio == Frecuencia 
# ! Angulo == Valor 
# print(n)
# print(sum(n))

radii =sum(n) / sum(sum(n))
theta   =  bins[1:] * 2*  np.pi / 360 

radii2 =sum(n2) / sum(sum(n2))
theta2   =  bins2[1:] * 2*  np.pi / 360 

radii3 =sum(n3) / sum(sum(n3))
theta3   =  bins3[1:] * 2*  np.pi / 360 
# print(len(theta) )
# print(len(radii) )

width = np.pi / num_bins
# colors = plt.cm.viridis( np.random.rand(N))

colors2 = plt.cm.viridis( np.random.rand(20))
colors3 = plt.cm.viridis( np.random.rand(30))




ax= plt.subplot(projection='polar')

#  ! KO 


# ax.bar((np.pi - theta2), (radii2), width=width, bottom=0.0, color="C1", alpha=0.5)

# ax.set_title( r' \textbf {KO}' ,fontsize=40)

#  ! WT 

# ax.bar(theta, radii, width=width, bottom=0.0, color='C0', alpha=0.5)


# ax.set_title( r' \textbf {WT}' ,fontsize=40)

#  ! PLX 


ax.bar((np.pi-theta3), radii3, width=width, bottom=0.0, color='C3', alpha=0.5)

ax.set_title( r' \textbf {PLX}' ,fontsize=40)



ax.set_rticks([0.05, 0.10, 0.15, 0.20,0.25])  # Less radial ticks


ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_ylim(0.0,0.25)


# plt.savefig('WT.pdf', dpi=1000)
# plt.savefig('WT.png', dpi=1000)

# plt.savefig('KO.pdf', dpi=1000)
# plt.savefig('KO.png', dpi=1000)

plt.savefig('PLX.pdf', dpi=1000)
plt.savefig('PLX.png', dpi=1000)

plt.tight_layout()


plt.show()