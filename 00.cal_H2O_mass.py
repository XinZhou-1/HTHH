## calculate the H2O mass using grided MLS level3 zonal mean data
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime 

def make_xarray(odata,ntimes,vname,sdate,unit1):
    
    time1 = pd.date_range(sdate, periods=ntimes, freq="1D")#+ pd.DateOffset(days=-16)
    ds = xr.Dataset({vname: (['time'], odata.astype('float32'), {'units':unit1})},
                 coords={
                         'time': (['time'], time1)
                })
    
    ds.attrs['Conventions'] = 'CF-1.7'
    ds.attrs['title'] = 'MLS'
    ds.attrs['nc.institution'] = 'University of Leeds'
    ds.attrs['source'] = 'MLS'
    ds.attrs['history'] = str(datetime.utcnow()) + ' Python'
    ds.attrs['references'] = 'x.zhou1@leeds.ac.uk'
    ds.attrs['comment'] = 'H2O mass calculated from zonal mean MLS grided data'
    return ds

### constants
R = 8.3143                       ## Constante des gazs parfaits (J.K-1.mol-1)
N_avo = 6.022e23                 ## Avogadro
M_air = 28.97e-3                 ## kg/mol
M_CO = 28.01e-3                  ## kg/mol
M_H2O = 18.01528e-3              ## kg/mol
M_CH3 = 41.05e-3                 ## kg/mol
g = 9.80665                      ## m.s−2
R_earth = 6.371e6                ## Rayon de la Terre m

### Read data
ipath= "YOUR_PATH_TO_MLS_DATA/" # path to the MLS level3 grided data. 
##Here I'm using the filtered data processed by Hugh Pumphrey (Uni. Edinburgh)
# ds_v5 = xr.open_dataset(ipath+'l2gpts_H2O_V5.nc', decode_times = False)
ds_v5 = xr.open_dataset(ipath+'l2gpts_H2O_V5_filt.nc', decode_times = False)
h_v5 = ds_v5['L2GPzm']*1.e6 #unit:ppmv
days_since_1993 = ds_v5['Time']
start_date = pd.to_datetime('1993-01-01')
date_array = start_date + pd.to_timedelta(days_since_1993.values, unit='D')
h_v5["Time"]=date_array
del days_since_1993,start_date,date_array
hmm_v5 = h_v5.resample(Time="1MS").mean(dim="Time") # monthly mean

### Calculation
P_levels = list(100*np.array([1.0000000e+03, 8.2540417e+02, 6.8129205e+02, 5.6234131e+02,
       4.6415887e+02, 3.8311868e+02, 3.1622775e+02, 2.6101572e+02,
       2.1544347e+02, 1.7782794e+02, 1.4677992e+02, 1.2115276e+02,
       1.0000000e+02, 8.2540421e+01, 6.8129204e+01, 5.6234131e+01,
       4.6415890e+01, 3.8311867e+01, 3.1622776e+01, 2.6101572e+01,
       2.1544348e+01, 1.7782795e+01, 1.4677993e+01, 1.2115276e+01,
       1.0000000e+01, 8.2540417e+00, 6.8129206e+00, 5.6234131e+00,
       4.6415887e+00, 3.8311868e+00, 3.1622777e+00, 2.6101573e+00,
       2.1544347e+00, 1.7782794e+00, 1.4677993e+00, 1.2115277e+00,
       1.0000000e+00, 6.8129206e-01, 4.6415889e-01, 3.1622776e-01,
       2.1544346e-01, 1.4677992e-01, 1.0000000e-01, 4.6415888e-02,
       2.1544347e-02, 9.9999998e-03, 4.6415888e-03, 2.1544348e-03,
       1.0000000e-03, 4.6415889e-04, 2.1544346e-04, 9.9999997e-05,
       4.6415887e-05, 2.1544347e-05, 9.9999997e-06]))
P0 = 101325
P_demi_levels = []
for i in range(len(P_levels)-1):
    P_demi_levels.append(np.exp((np.log(P_levels[i])+np.log(P_levels[i+1]))/2))

ind_bas = 14 # 13-83hPa; 14-68hPa
ind_haut = 36 # 36:1 Pa, 
P_lvl = P_levels[ind_bas]
P_lvl_demi = P_demi_levels[ind_bas-1]
P_lvl2 = P_levels[ind_haut]
P_lvl2_demi = P_demi_levels[ind_haut]
print('From '+str(int(P_lvl/100))+' hPa to '+str(P_lvl2/100)+' hPa') # 68-1hPa

for h,fname in zip([h_v5, hmm_v5],['v5','v5.mm']): # h_v5, hmm_v5 are for the daily and monthly mean data respectively
    latitude = h.Latitude
    time = h.Time

    mass_H2O = []
    for itime in range(len(time)):
        mass_H2O_col = []
        for ilat in range(len(latitude)):

            lat_min = latitude[ilat]-2.5
            lat_max = latitude[ilat]+2.5
            surface_grid = 2*np.pi*(R_earth**2)*(np.sin(lat_max*np.pi/180)-np.sin(lat_min*np.pi/180))
            partial_air_col = (P_lvl_demi - P_lvl2_demi)*N_avo/(g*M_air)
            total_air_molecules_col = partial_air_col*surface_grid
            
            vmr_grid = h[:,ilat,itime]
            H2O_conc_col, air_conc_col = [], []
            for pr in range(P_levels.index(P_lvl),P_levels.index(P_lvl2)+1):
                H2O_conc_col.append( vmr_grid[pr]*(P_demi_levels[pr-1] - P_demi_levels[pr])/(g*M_air) )
                air_conc_col.append( (P_demi_levels[pr-1] - P_demi_levels[pr])/(g*M_air) )
            VMR_col_H2O = sum(H2O_conc_col)/sum(air_conc_col)
            N_H2O_molecules_col = total_air_molecules_col * VMR_col_H2O
            mass_H2O_col.append( N_H2O_molecules_col*M_H2O/(N_avo*1000*1000) )
        mass_H2O.append( sum(mass_H2O_col) )
        
    # save the data
    ofile = ipath + "h2omass.mls."+fname+'.68to1hPa.202409.nc'

    ## create a xarray
    ntimes = len(time)
    sdate = str(time.values[0])[0:10]
    unit = 'Kg'
    vname = 'h2omass'
    ds = make_xarray(np.array(mass_H2O), ntimes, vname, sdate, unit)
    print('saving the file: '+ ofile)
    ds.to_netcdf(ofile)
 