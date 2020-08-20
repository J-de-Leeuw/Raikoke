
#####################################################################################################################
#
#   File name: NAME_TROPOMI_mass_burden_plot.py
#
#   Written by: Hans de Leeuw, University of Cambridge, UK
#
#   Calculation done by the program: Plotting the mass burden evolution from NAME and TROPOMI.
#
#Code last updated: 30-07-2020
#
#####################################################################################################################

####  Imported libraries  ##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import pandas as pd
from netCDF4 import Dataset
import datetime, time
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# reading in the NAME SO2 daily mass files:

SO2_mass_VR15=pd.read_csv('SO2_mass_daily_AK15km_VolRes15.csv')
SO2_mass_VR15=np.array(SO2_mass_VR15)

SO2_mass_VR20=pd.read_csv('SO2_mass_daily_AK15km_VolRes20.csv')
SO2_mass_VR20=np.array(SO2_mass_VR20)

SO2_mass_SPF=pd.read_csv('SO2_mass_daily_AK15km_StratProfile.csv')
SO2_mass_SPF=np.array(SO2_mass_SPF)

SO2_mass_VR15[SO2_mass_VR15==0]=np.nan
SO2_mass_VR20[SO2_mass_VR20==0]=np.nan
SO2_mass_SPF[SO2_mass_SPF==0]=np.nan

# read in a file to plot the TROPOMI estimate and the AAI index:

SO2_mass_TROP=pd.read_csv('TROP_SO2_mass_daily_AK15km.csv')
SO2_mass_TROP=np.array(SO2_mass_TROP)

SO2_mass_TROP[SO2_mass_TROP==0]=np.nan

# reading in the NAME SO4 daily mass files:

SO4_mass_VR15=pd.read_csv('SO4_mass_daily_AK15km_VolRes15.csv')
SO4_mass_VR15=np.array(SO4_mass_VR15)

SO4_mass_VR20=pd.read_csv('SO4_mass_daily_AK15km_VolRes20.csv')
SO4_mass_VR20=np.array(SO4_mass_VR20)

SO4_mass_SPF=pd.read_csv('SO4_mass_daily_AK15km_StratProfile.csv')
SO4_mass_SPF=np.array(SO4_mass_SPF)

SO4_mass_VR15[SO4_mass_VR15==0]=np.nan
SO4_mass_VR20[SO4_mass_VR20==0]=np.nan
SO4_mass_SPF[SO4_mass_SPF==0]=np.nan

length_long=24

datum_long=np.zeros(length_long, dtype='datetime64[h]')

# make the date array for the x-axis plotting
for e in range(length_long):
  maand=6
  dag=22+e
  if dag >30:
    dag=dag-30
    maand=7
  datum_long[e]=(datetime.datetime(year=2019, month=maand, day=dag))

datum_start=(datetime.datetime(year=2019, month=6, day=21, hour=18))
datum_legend=(datetime.datetime(year=2019, month=6, day=21, hour=15))

#The style parameters for the plot

plt.style.use(['seaborn-darkgrid','tableau-colorblind10', 'seaborn-poster']) #,'seaborn-poster'

plt.rcParams['axes.edgecolor']='black'
plt.rcParams['legend.edgecolor']='black'
plt.rcParams['legend.fontsize']=18
plt.rcParams['axes.linewidth']='0.8'
plt.rcParams['xtick.bottom']= 'True'
plt.rcParams['ytick.left']= 'True'
plt.rcParams['xtick.color']='black'
plt.rcParams['ytick.color']='black'
plt.rcParams['xtick.major.size']='3.5'
plt.rcParams['ytick.major.size']='3.5'
plt.rcParams['xtick.major.width']='0.8'
plt.rcParams['ytick.major.width']='0.8'

colors_plot1 = plt.cm.jet(np.linspace(0, 1,9))

fig, ax = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

ax[0].set_title("Evolution of SO$_2$ mass (Tg of SO$_2$) during the 2019 Raikoke eruption ")
# plot the start eruption dashed line
ax[0].plot([datum_start,datum_start],[0,0.35], linestyle='--', color='k', linewidth=1)
ax[0].plot([datum_start,datum_start],[1.05,2.25], linestyle='--', color='k', linewidth=1)
ax[0].annotate('Start eruption', xy=(datum_legend, 0.92), rotation=90, size=13, color='k')

# plot the AAI index as background bar
ax[0].bar(datum_long[:5], SO2_mass_TROP[:5,5]/10, color='grey', alpha=0.5, width=0.4)

# plot the NAME SO2 estimates
ax[0].plot(datum_long, SO2_mass_VR15[:length_long,3], color=colors_plot1[6],marker='o', markersize=5, linestyle='-', linewidth=2, label='VolRes1.5')
ax[0].plot(datum_long, SO2_mass_VR20[:length_long,3], color=colors_plot1[7],marker='o', markersize=5, linestyle='-', linewidth=2, label='VolRes2.0')
ax[0].plot(datum_long, SO2_mass_SPF[:length_long,3],color=colors_plot1[8],marker='o', markersize=5, linestyle='-', linewidth=2, label='StratProfile')

#plot the TROPOMI SO2 estimates
ax[0].plot(datum_long, SO2_mass_TROP[:length_long,3],color=colors_plot1[2],marker='o', markersize=5, linestyle='-.', linewidth=2, label='TROPOMI')
for j in range(200):
   value=j/400.0
   value2=j/200.
   value3=(j+1)/200.
   ax[0].fill_between(datum_long, SO2_mass_TROP[:length_long,3]+value2*SO2_mass_TROP[:length_long,4], SO2_mass_TROP[:length_long,3]+value3*SO2_mass_TROP[:length_long,4] ,alpha=(0.5-value), facecolor=colors_plot1[2])
   ax[0].fill_between(datum_long, SO2_mass_TROP[:length_long,3]-value3*SO2_mass_TROP[:length_long,4] ,SO2_mass_TROP[:length_long,3]-value2*SO2_mass_TROP[:length_long,4],alpha=(0.5-value), facecolor=colors_plot1[2])

# plot the SO2 deposition
ax[0].plot(datum_long[:0], SO2_mass_VR15[:0,4], color='k',marker='o', markersize=5, linestyle='-', linewidth=2)
ax[0].plot(datum_long[:0], SO2_mass_VR15[:0,4], color='k',marker='o', markersize=5, linestyle=':', linewidth=2)
ax[0].plot(datum_long, SO2_mass_VR15[:length_long,4], color=colors_plot1[6],marker='o', markersize=5, linestyle=':', linewidth=2)
ax[0].plot(datum_long, SO2_mass_VR20[:length_long,4], color=colors_plot1[7],marker='o', markersize=5, linestyle=':', linewidth=2)
ax[0].plot(datum_long, SO2_mass_SPF[:length_long,4], color=colors_plot1[8],marker='o', markersize=5, linestyle=':', linewidth=2)

lines = ax[0].get_lines()

#add the additional legend for the deposition/mass burden split
legend2 = plt.legend([lines[i] for i in [6,7]], ['mass burden','deposition'], scatterpoints=1, frameon=True, labelspacing=1,loc='upper center', bbox_to_anchor=(0.5, 4.45), fancybox=True, ncol=2)

ax[0].legend()
ax[0].set_ylabel("Mass SO$_2$ (Tg)")
ax[0].set_ylim([0,2.25])
#box = ax[0].get_position()
#ax[0].set_position([box.x0-0.015, box.y0+0.05, box.width*1.1, box.height])

#second subplot

ax[1].set_xticklabels(datum_long, rotation=-45, ha="left")
ax[1].set_xticks(datum_long)
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %B"))

ax[1].set_title("Evolution of SO$_4$ mass (Tg) during the 2019 Raikoke eruption ")
# plot the start eruption dashed line
ax[1].plot([datum_start,datum_start],[0,0.75], linestyle='--', color='k', linewidth=1)
#ax[1].plot([datum_start,datum_start],[0.43,0.75], linestyle='--', color='k', linewidth=1)
#ax[1].annotate('Start eruption', xy=(datum_legend, 0.4), rotation=90, size=13, color='k')

# plot the NAME SO4 estimates
ax[1].plot(datum_long, SO4_mass_VR15[:length_long,3], color=colors_plot1[6],marker='o', markersize=5, linestyle='-', linewidth=2, label='VolRes1.5 mass')
ax[1].plot(datum_long, SO4_mass_VR20[:length_long,3], color=colors_plot1[7],marker='o', markersize=5, linestyle='-', linewidth=2, label='VolRes2.0 mass')
ax[1].plot(datum_long, SO4_mass_SPF[:length_long,3],color=colors_plot1[8],marker='o', markersize=5, linestyle='-', linewidth=2, label='StratProfile mass')

# plot the SO4 deposition
ax[1].plot(datum_long, SO4_mass_VR15[:length_long,4], color=colors_plot1[6],marker='o', markersize=5, linestyle=':', linewidth=2, label='VolRes1.5 deposition')
ax[1].plot(datum_long, SO4_mass_VR20[:length_long,4], color=colors_plot1[7],marker='o', markersize=5, linestyle=':', linewidth=2, label='VolRes2.0 deposition')
ax[1].plot(datum_long, SO4_mass_SPF[:length_long,4], color=colors_plot1[8],marker='o', markersize=5, linestyle=':', linewidth=2, label='StratProfile deposition')

ax[1].set_ylabel("Mass SO$_4$ (Tg)")
ax[1].set_ylim([0,0.75])

plt.show()
