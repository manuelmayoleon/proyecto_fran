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

import seaborn as sns

micro_ac= pd.read_csv("microglia_AC.csv", index_col=0,header=0,sep=',' )

micro_cc= pd.read_csv("microglia_CC.csv", index_col=0,header=0,sep=',' )



# ! Delete wam cell
del micro_ac['WAM']
del micro_cc['WAM']

array_dot_product =   np.zeros((len(micro_ac.head(0).columns),len(micro_ac.head(0).columns)))
array_dot_product_cc = np.zeros((len(micro_cc.head(0).columns),len(micro_cc.head(0).columns)))



array_dist = np.zeros((len(micro_ac.head(0).columns),len(micro_ac.head(0).columns)))

array_dist_cc = np.zeros((len(micro_cc.head(0).columns),len(micro_cc.head(0).columns)))



micro = list()



i =0 
j = 0
for x in micro_ac:

    j=0
    for y in micro_ac:

      
            
        dist = np.linalg.norm(micro_ac[x]-micro_ac[y]) 
        dot_prod = abs( np.dot(micro_ac[x],micro_ac[y]) /np.sqrt( np.dot(micro_ac[x],micro_ac[x]) * np.dot(micro_ac[y],micro_ac[y]) ) ) 
        array_dist[i][j] = dist
        array_dot_product[i][j] = dot_prod
          
        j+=1

    i+=1

i =0 
for x in micro_cc:

    j=0
    for y in micro_cc:
        
        dist = np.linalg.norm(micro_cc[x]-micro_cc[y]) 
        dot_prod = abs( np.dot(micro_cc[x],micro_cc[y]) /np.sqrt( np.dot(micro_cc[x],micro_cc[x]) * np.dot(micro_cc[y],micro_cc[y]) ) ) 
        array_dist_cc[i][j] = dist
        array_dot_product_cc[i][j] = dot_prod
          
        j+=1

    i+=1


# #!c sequence
d = array_dist
c = array_dot_product
col1 = list(micro_ac.head(0).columns)

d_cc= array_dist_cc
c_cc = array_dot_product_cc
col1_cc = list(micro_cc.head(0).columns)



# #!plot



#!plotting the heatmap for correlation


f = plt.figure(figsize=(16, 12))

ax = sns.heatmap(c_cc, annot=True,xticklabels=col1, yticklabels=col1,cmap="crest")

plt.title(r'\bf CORRELATION MATRIX ', fontsize=30)

plt.tight_layout()
 
 
plt.savefig("correlation_ac.png",dpi=1200)
plt.savefig("correlation_ac.pdf",dpi=1200)


f2 = plt.figure(figsize=(16, 12))


ax2 = sns.heatmap(c_cc, annot=True,xticklabels=col1_cc, yticklabels=col1_cc,cmap="crest")

plt.title(r'\bf CORRELATION MATRIX ', fontsize=30)

plt.tight_layout()


plt.savefig("correlation_cc.png",dpi=1200)
plt.savefig("correlation_cc.pdf",dpi=1200)




plt.show()