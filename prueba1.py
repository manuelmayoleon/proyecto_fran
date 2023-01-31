# !pip install pycircular
import pandas as pd
import matplotlib.pyplot as plt
import os as os 
import pycircular



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

print(sheets['WT'])


df = pycircular.datasets.load_transactions()['data']
df['date']= pd.to_datetime(df['date'])
dates = df.loc[df['user'] == 1, 'date']

print(dates.head())

print(dates.describe(datetime_is_numeric=True))

dates.groupby(dates.dt.hour).count().plot(kind="bar")


time_segment = 'hour'  # 'hour', 'dayweek', 'daymonth
freq_arr, times = pycircular.utils.freq_time(dates , time_segment=time_segment)
fig, ax1 = pycircular.plots.base_periodic_fig(freq_arr[:, 0], freq_arr[:, 1], time_segment=time_segment)
ax1.legend(bbox_to_anchor=(-0.3, 0.05), loc="upper left", borderaxespad=0)

plt.show()

# dates_mean = times.values.mean()
# fig, ax1 = pycircular.plots.base_periodic_fig(freq_arr[:, 0], freq_arr[:, 1], time_segment=time_segment)
# ax1.bar([dates_mean], [1], width=0.1, label='Arithmetical Mean Hour')
# ax1.legend(bbox_to_anchor=(-0.3, 0.05), loc="upper left", borderaxespad=0)

# plt.show()