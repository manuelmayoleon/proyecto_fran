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


# micro_ac.drop(labels='WAM', axis=1)
# ! Delete wam cell
del micro_ac['WAM']
del micro_cc['WAM']
# micro_ac = micro_ac.set_index(['column1'])

# print(list(micro_ac).index('Aged'))

# print(np.dot(micro_ac['Microglía AQP4 AC neonatal'],micro_ac['Aged'])/(np.dot(micro_ac['Microglía AQP4 AC neonatal'],micro_ac['Microglía AQP4 AC neonatal'])*np.dot(micro_ac['Aged'],micro_ac['Aged'])))

array_dot_product =   np.zeros((len(micro_ac.head(0).columns),len(micro_ac.head(0).columns)))
array_dot_product_cc = np.zeros((len(micro_cc.head(0).columns),len(micro_cc.head(0).columns)))

# print(len(micro_ac.head(0).columns))

array_dist = np.zeros((len(micro_ac.head(0).columns),len(micro_ac.head(0).columns)))

array_dist_cc = np.zeros((len(micro_cc.head(0).columns),len(micro_cc.head(0).columns)))


# print(array_dist)

micro = list()



i =0 
j = 0
for x in micro_ac:
    # print(list(micro_ac).index(x))
    # micro = micro_ac.drop(micro_ac.iloc[:,0:list(micro_ac).index(x)],axis =1)
    # print(x)
    j=0
    for y in micro_ac:
        # print(np.isnan(micro_ac[x]))
        # if(x!=y):
      
            
        dist = np.linalg.norm(micro_ac[x]-micro_ac[y]) 
        dot_prod = abs( np.dot(micro_ac[x],micro_ac[y]) /np.sqrt( np.dot(micro_ac[x],micro_ac[x]) * np.dot(micro_ac[y],micro_ac[y]) ) ) 
        array_dist[i][j] = dist
        array_dot_product[i][j] = dot_prod
          
        j+=1

    i+=1

i =0 
for x in micro_cc:
    # print(list(micro_ac).index(x))
    # micro = micro_ac.drop(micro_ac.iloc[:,0:list(micro_ac).index(x)],axis =1)
    # print(x)
    j=0
    for y in micro_cc:
        
        dist = np.linalg.norm(micro_cc[x]-micro_cc[y]) 
        dot_prod = abs( np.dot(micro_cc[x],micro_cc[y]) /np.sqrt( np.dot(micro_cc[x],micro_cc[x]) * np.dot(micro_cc[y],micro_cc[y]) ) ) 
        array_dist_cc[i][j] = dist
        array_dot_product_cc[i][j] = dot_prod
          
        j+=1

    i+=1

#figure
# fig, ax1 = plt.subplots()
# fig.set_size_inches(13, 10)

# #labels
# # ax1.set_xlabel('Alcohol')
# # ax1.set_ylabel('Color Intensity')
# # ax1.set_title('Relationship Between Color Intensity and Alcohol Content in Wines')

# #c sequence
d = array_dist
c = array_dot_product
col1 = list(micro_ac.head(0).columns)

d_cc= array_dist_cc
c_cc = array_dot_product_cc
col1_cc = list(micro_cc.head(0).columns)



# #plot
# # plt.scatter( col1, col1 , c=c, 
# #             cmap = 'RdPu', alpha =0.5)

# # cbar.set_label('Color Intensity')
# plt.contourf(col1,col1,c,levels = 11)
# plt.colorbar()


 
# checking correlation using heatmap
#Loading dataset
# flights = sns.load_dataset("flights")
 
#plotting the heatmap for correlation


f = plt.figure(figsize=(16, 12))

ax = sns.heatmap(c_cc, annot=True,xticklabels=col1, yticklabels=col1,cmap="crest")

plt.title(r'\bf CORRELATION MATRIX ', fontsize=30)

plt.tight_layout()
 
 
plt.savefig("correlation_ac.png",dpi=1200)
plt.savefig("correlation_ac.pdf",dpi=1200)


f2 = plt.figure(figsize=(16, 12))

# ax2 = sns.heatmap(d_cc, annot=True,xticklabels=col1_cc, yticklabels=col1_cc,cmap="crest")

ax2 = sns.heatmap(c_cc, annot=True,xticklabels=col1_cc, yticklabels=col1_cc,cmap="crest")

plt.title(r'\bf CORRELATION MATRIX ', fontsize=30)

plt.tight_layout()


plt.savefig("correlation_cc.png",dpi=1200)
plt.savefig("correlation_cc.pdf",dpi=1200)

# f = plt.figure(figsize=(19, 15))
# plt.matshow(array_dist, fignum=f.number)
# # plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# # plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)


plt.show()