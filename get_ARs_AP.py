import xarray as xr
import pandas as pd


filename = '~/ARs_DATA/Global_AR_dataset_Guan.nc'
ds = xr.open_dataset(filename)


de_sel = ds[['axismap']].sel(time=slice('1980-01-01', '2022-12-31'), ens = 1, lev = 0)

de_sel.coords['lon'] = (de_sel.coords['lon'] + 180) % 360 - 180
de_sel = de_sel.sortby(de_sel.lon)
de_sel = de_sel.sel(lon=slice(-65, -55), lat=slice(-59, -65))
de_sel = de_sel.sel(time=de_sel['time'].dt.month.isin([11, 12, 1, 2, 3]))

ARs_mask = []
for i in range(0, len(de_sel.axismap)): #len(de_sel.axismap)
    ARs_mask.append(de_sel.axismap[i].max().values)


AR_inf = pd.DataFrame(ARs_mask, columns=['ARs'])#.plot()
AR_inf['date'] = de_sel.axismap.time

AR_inf = AR_inf.dropna().reset_index(drop=True)
AR_inf.index = AR_inf.date

AR_inf = AR_inf.resample('1d').max().dropna()

filename = '/home/cr2/cmtorres/repos/ARs_SSI_FEL/data/ARs_Guan_AP_day.csv'
AR_inf.to_csv(filename, index=True, sep='\t')