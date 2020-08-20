# Raikoke
Code and data for NAME Raikoke simulations

### NAME_TROPOMI_FSS_SAL_plot.py 
Python code to plot FSS and SAL-score.This program uses the following data:

*SAL_03DU_N1_15km_AK..._csv*: Data for the S score, A score and L score for the various NAME simulations (VolRes1.5, VolRes2.0, StratProfile and StratProfile_rd)

*FSS_03DU_N1_15km_AK..._csv*: Data for the Fractional Skill Score for the various NAME simulations

*mass_03DU_N1_15km_AK..._csv*: Data for SO2 mass estimates (Tg of SO2) from TROPOMI, NAME and the TROPOMI error estimates for the various NAME simulations

### NAME_TROPOMI_mass_burden_plot.py 
Python code to plot the mass burden and deposition plots.This program uses the following data:

*SO2_mass_daily_AK15km_...csv*: Data for the daily SO2 mass burden (Tg of SO2) and depostion for the various NAME simulations (VolRes1.5, VolRes2.0, StratProfile).

*SO4_mass_daily_AK15km_...csv*: Data for the daily SO4 mass burden (Tg) and depostion for the various NAME simulations.

*TROP_SO2_mass_daily_AK15km.csv*: Daily mass estimates from TROPOMI and the corresponding max AAI values for each day.
