import xarray as xr

filename = '/home/cr2/cmtorres/ERA5/Z500_daily_anomalies.nc'
Z500_an = xr.open_dataset(filename)/9.98



Z500_index = Z500_an.sel(latitude=slice(-55,-62.5), longitude=slice(-85,-40), pressure_level=500).mean(('latitude', 'longitude')).to_dataframe()[['z_anomaly']]

print(Z500_index)

Z500_index.to_csv('data/Z500_index_daily.csv', sep='\t')
