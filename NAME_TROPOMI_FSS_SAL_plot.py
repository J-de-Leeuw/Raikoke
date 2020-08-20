#! /usr/bin/env python2.7

#####################################################################################################################
#
#   File name: NAME_TROPOMI_FSS_SAL_plot.py
#
#   Written by: Hans de Leeuw, University of Cambridge, UK
#
#   Calculation done by the program: Plotting the FSS score and the SAL score output
#
#
#   Code last updated: 31-07-2020
#
#####################################################################################################################


####  Imported libraries  ##############################################################################

import os
import sys
import iris
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import pandas as pd

from netCDF4 import Dataset
from general_func import swd_JASMIN, polar_stere
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import datetime, time
from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Reading all the FSS, SAL and mass values obtained from the NAME_TROPOMI_FSS_SAL.py program
table_data=pd.read_csv('FSS_03DU_N1_15TG_15km_AK_VolRes15.csv')
table_data=np.array(table_data)

table_data2=pd.read_csv('mass_03DU_N1_15TG_15km_AK_VolRes15.csv')
table_data2=np.array(table_data2)

table_data3=pd.read_csv('SAL_03DU_N1_15TG_15km_AK_VolRes15.csv')
table_data3=np.array(table_data3)

length=np.shape(np.where(table_data[:,0]>0))[1]

# Make array to use for the date on the x-axis in the plot

datum=np.zeros(length, dtype='datetime64[h]')
datum2=np.zeros(length, dtype='datetime64[h]')
datum3=np.zeros(length, dtype='datetime64[h]')

for e in range(length):
  datum[e]=(datetime.datetime(year=int(table_data[e,0]), month=int(table_data[e,1]), day=int(table_data[e,2]), hour=int(table_data[e,3])))

for e in range(length):
  datum2[e]=(datetime.datetime(year=int(table_data2[e,0]), month=int(table_data2[e,1]), day=int(table_data2[e,2]), hour=int(table_data2[e,3])))

for e in range(length):
  datum3[e]=(datetime.datetime(year=int(table_data3[e,0]), month=int(table_data3[e,1]), day=int(table_data3[e,2]), hour=int(table_data3[e,3])))

datum_plot=np.zeros(24, dtype='datetime64[h]')
for w in range(24):
  maand=6
  dag=22+w
  if dag >30:
    dag=dag-30
    maand=7
  datum_plot[w]=(datetime.datetime(year=2019, month=maand, day=dag, hour=0))

# define array to show the SO2 thresholds used (DU) to calculate the FSS. Should be the same as used in the NAME_TROPOMI_FSS_SAL.py program.

legendlist=[0.1,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,45,50,60,70,80,90,100]

# remove any overpasses where the total SO2 mass is below 0.2 Tg as the FSS data will be very noisy and doesn't give much information (edges of the plume)
table_data[(table_data2[:,4]<0.2)]=np.nan
table_data3[(table_data2[:,4]<0.2)]=np.nan
table_data2[(table_data2[:,4]<0.2)]=np.nan

#selecting specific threshold to highlight in the FSS plot
concentration=table_data[0:length,5:]	#all the thresholds
concentration2=table_data[0:length,5]	# 0.3 DU
concentration3=table_data[0:length,12]	# 1 DU
concentration4=table_data[0:length,16]	# 5 DU
concentration5=table_data[0:length,21]	# 10 DU
concentration6=table_data[0:length,27]	# 20 DU
concentration7=table_data[0:length,29]	# 30 DU
concentration8=table_data[0:length,31]	# 40 DU
concentration9=table_data[0:length,33]	# 50 DU

# making sure we don't plot all the zeros to make the plot look better
concentration2[concentration2==0]=np.nan
concentration3[concentration3==0]=np.nan
concentration4[concentration4==0]=np.nan
concentration5[concentration5==0]=np.nan
concentration6[concentration6==0]=np.nan
concentration7[concentration7==0]=np.nan
concentration8[concentration8==0]=np.nan
concentration9[concentration9==0]=np.nan

mass_fraction=table_data2[0:length,4]
mass_fraction2=table_data2[0:length,5]
mass_fraction3=table_data2[0:length,6]

# Set plot parameters 
plt.style.use(['seaborn-darkgrid','tableau-colorblind10', 'seaborn-poster'])

plt.rcParams['axes.edgecolor']='black'
plt.rcParams['legend.edgecolor']='black'
plt.rcParams['axes.linewidth']='0.8'
plt.rcParams['xtick.bottom']= 'True'
plt.rcParams['ytick.left']= 'True'
plt.rcParams['xtick.color']='black'
plt.rcParams['ytick.color']='black'
plt.rcParams['xtick.major.size']='3.5'
plt.rcParams['ytick.major.size']='3.5'
plt.rcParams['xtick.major.width']='0.8'
plt.rcParams['ytick.major.width']='0.8'
plt.rc('font', size=24)          # controls default text sizes
plt.rc('axes', titlesize=24)     # fontsize of the axes title
plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.rc('legend', fontsize=24)    # legend fontsize
plt.rc('figure', titlesize=28)  # fontsize of the figure title

#defining the last date we want to plot
datend=199

#Plotting the FSS
fig, ax = plt.subplots(figsize=(21, 7.5))
ax.set_xticklabels(datum, rotation=-35, ha="left")
ax.set_xticks(datum)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b-%y"))

colors_plot1 = plt.cm.plasma_r(np.linspace(0, 1, 8))
ax.plot(datum[0:datend],concentration[0:datend],color='grey', linewidth=1, alpha=0.4, linestyle='None',marker='o', markersize=7)
ax.plot(datum[0:0],concentration2[0:0],color='grey', linewidth=2, alpha=0.9, linestyle='None',marker='o', markersize=10, label='> 0.3 DU')
ax.plot(datum[0:datend],concentration3[0:datend],color=colors_plot1[1], linewidth=2, alpha=0.9, linestyle='None',marker='o', markersize=10, label='> 1.0 DU')
ax.plot(datum[0:datend],concentration5[0:datend],color=colors_plot1[3], linewidth=2, alpha=0.9, linestyle='None',marker='o', markersize=10, label='> 10 DU')
ax.plot(datum[0:datend],concentration7[0:datend],color=colors_plot1[5], linewidth=2, alpha=0.9, linestyle='None',marker='o', markersize=10, label='> 30 DU')
ax.plot(datum[0:datend],concentration9[0:datend],color=colors_plot1[7], linewidth=2, alpha=0.9, linestyle='None',marker='o', markersize=10, label='> 50 DU')
ax.axhline(0.5, color='k', linewidth=1, alpha=0.7, linestyle='--')
ax.set_ylim([-1.0,1.0])
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ax.yaxis.set_label_coords(-0.07, 0.75) 

ax2 = ax.twinx()
ax2.set_xticks(datum_plot)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
ax2.xaxis.set_tick_params(pad=2)
ax.xaxis.set_tick_params(pad=2)

# plotting the mass for each overpass included for TROPOMI and NAME
ax2.bar(datum2[0:datend],mass_fraction[0:datend],linewidth=1,alpha=0.7, label='TROPOMI',width=0.08, yerr=mass_fraction3[0:datend],error_kw=dict(lw=1, capsize=3, capthick=1))
ax2.bar(datum2[0:datend],mass_fraction2[0:datend],linewidth=1,alpha=0.7, label='VolRes1.5',width=0.08)
#ax2.bar(datum2,mass_fraction2,linewidth=1,alpha=0.7, label='NAME 1.5Tg',width=0.08)
ax2.set_ylim([0,5])
ax2.set_yticks([0,0.5,1.0,1.5,2.0])
ax2.yaxis.set_label_coords(1.07, 0.2) 
 
# Add titles
ax.set_title("Evolution of the FSS value during the 2019 Raikoke eruption ")
ax.set_ylabel("Fraction Skill Score")
ax.legend(bbox_to_anchor=(0.99, 0.99),handletextpad=0.1)
ax2.set_ylabel("Total SO$_2$ Mass (Tg)")
ax2.legend(bbox_to_anchor=(0.7, 0.20))

box = ax.get_position()
ax.set_position([box.x0-0.015, box.y0+0.02, box.width, box.height])
box = ax2.get_position()
ax2.set_position([box.x0-0.015, box.y0+0.02, box.width, box.height])


################## Plot to show the SAL-score diagram 

plt.rc('legend', fontsize=18)    # legend fontsize

fig3= plt.figure()
ax3 = fig3.gca()

length=199

S=table_data3[0:length,4]
S[np.isnan(S)]=0
A=table_data3[0:length,5]
L=table_data3[0:length,6]+table_data3[0:length,7]
L[L==0]=0.001 #logarithmic scale, so has to be larger than 0 for all points

mass_fraction[np.isnan(mass_fraction)]=0

# The size of each dot is scaled by the total mass of that day 
size=20*2**(mass_fraction/1.5*5)

# number of days we want to include in the plot
dagen=18

S_mean=np.zeros(18)
A_mean=np.zeros(18)
L_mean=np.zeros(18)

# select the correct date for each overpass to calculate the daily average and calculate the daily means
for i in range(np.shape(datum)[0]):
  if table_data[i,1]==7:
     table_data[i,2]= table_data[i,2]+30
  if table_data[i,3] < 12:
     if table_data[i,1]==6:
        table_data[i,2]= table_data[i,2]-1
     if table_data[i,1]==7:
        table_data[i,2]= table_data[i,2]-1

for time in range(18):
   S_mean[time]=np.sum(table_data3[(table_data[:,2]==time+21)][:,4]*table_data2[(table_data[:,2]==time+21)][:,4])/ np.sum(table_data2[(table_data[:,2]==time+21)][:,4])
   A_mean[time]=np.sum(table_data3[(table_data[:,2]==time+21)][:,5]*table_data2[(table_data[:,2]==time+21)][:,4])/ np.sum(table_data2[(table_data[:,2]==time+21)][:,4])
   L_mean[time]=np.sum(table_data3[(table_data[:,2]==time+21)][:,6]*table_data2[(table_data[:,2]==time+21)][:,4])/ np.sum(table_data2[(table_data[:,2]==time+21)][:,4])

# Plotting of the SAL-score data

#adding the arrows and labels
cax = fig3.add_axes([0.15, 0.15, 0.3, 0.025])
cax.set_title('L(ocation)', size=24)
ax3.axhline(0, color='k', linewidth=1, alpha=0.7, linestyle='--')
ax3.axvline(0, color='k', linewidth=1, alpha=0.7, linestyle='--')
ax3.annotate('', xy=(-1.75, 1.5), xytext=(-1.75, 0.5),
            arrowprops={'arrowstyle': '-|>','color': 'grey','lw': 2,'connectionstyle':'arc3'}, va='center')
ax3.annotate('Simulation', xy=(-1.97, 1.25), rotation=90, size=18)
ax3.annotate('overestimates mass ', xy=(-1.87, 1.65), rotation=90, size=18)
ax3.annotate('', xy=(-1.75, -1.4), xytext=(-1.75, -0.4),
            arrowprops={'arrowstyle': '-|>','color': 'grey','lw': 2,'connectionstyle':'arc3'}, va='center')
ax3.annotate('Simulation', xy=(-1.97, -0.65), rotation=90, size=18)
ax3.annotate('underestimates mass ', xy=(-1.87, -0.15), rotation=90, size=18)

ax3.annotate('', xy=(-1.5, 1.70), xytext=(-0.5, 1.70),
            arrowprops={'arrowstyle': '-|>','color': 'grey','lw': 2,'connectionstyle':'arc3'}, va='center')
ax3.annotate('Simulated SO$_2$ cloud is leptokurtic', xy=(-1.68, 1.82), size=18)
ax3.annotate('', xy=(1.5, 1.70), xytext=(0.5, 1.70),
            arrowprops={'arrowstyle': '-|>','color': 'grey','lw': 2,'connectionstyle':'arc3'}, va='center')
ax3.annotate('Simulated SO$_2$ cloud is platykurtic', xy=(0.25, 1.82), size=18)

ax3.annotate('22 June', xy=(S_mean[0], A_mean[0]), xytext=(S_mean[0]-0.4, A_mean[0]-0.3),arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.2", linewidth=1), size=20)
ax3.annotate('27 June', xy=(S_mean[5], A_mean[5]), xytext=(S_mean[5]+0.2, A_mean[5]+0.5),arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.2", linewidth=1), size=20)
ax3.annotate('5 July', xy=(S_mean[13], A_mean[13]), xytext=(S_mean[13]+0.17, A_mean[13]+0.35),arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=.2", linewidth=1), size=20)

# if you want to add arrow to show general direction the score moves within the plot
#ax3.annotate('',
 #           xy=(0.50, -1.25), xycoords='data',
  #          xytext=(-1.25, 0.75), textcoords='data',
   #         size=20,
    #        arrowprops=dict(arrowstyle="fancy",
     #                       fc="0.6", ec="none",
      #                      connectionstyle="arc3,rad=0.3"))


bounds = np.linspace(0,1, 100)
#bounds = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
norm = colors.LogNorm(vmin=0.001, vmax=1)
#norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)

#Plot the SAL data
cs=ax3.scatter(S,A,c=L,marker="o",cmap=plt.cm.plasma,vmin=0.001, vmax=1,alpha=0.9, norm=norm, s=size)
ax3.plot(S_mean, A_mean, color='black', linestyle='--', linewidth=1, marker='s', markersize=0)
ax3.scatter(S_mean,A_mean,c=L_mean, marker="s",cmap=plt.cm.plasma,vmin=0.001, vmax=1,alpha=0.9, norm=norm, s=100,edgecolors='black', linewidths=2)

for area in [0.2, 0.5, 1, 1.5]:
    ax3.scatter([], [], c=cs.cmap(0.1),alpha=0.7, s=20*2**(area/1.5*5),
                label=str(area) + ' Tg')
leg=ax3.legend(bbox_to_anchor=(1.215, 1.0), scatterpoints=1, frameon=True, labelspacing=1, loc=1, borderpad=0.5)
leg.set_title('Mass TROPOMI',prop={'size':16})

box = ax3.get_position()
ax3.set_position([box.x0-0.015, box.y0, box.width * 0.95, box.height])

#cbar=fig3.colorbar(cs, cax=cax, ticks=[0,0.25,0.5,0.75,1], format='%.2f', orientation="horizontal")
cbar=fig3.colorbar(cs, cax=cax, ticks=[0.001,0.01,0.1,1], orientation="horizontal")
cbar.ax.tick_params(labelsize=18) 
cbar.ax.xaxis.set_tick_params(pad=3)
ax3.set_xlabel("S(tructure)")
ax3.set_ylabel("A(mplitude)")
ttl=ax3.set_title("Evolution of the SAL-values during the 2019 Raikoke eruption")
ttl.set_position([.5, 1.03])
ax3.set_ylim(-2,2)
ax3.set_xlim(-2,2)

plt.show()

