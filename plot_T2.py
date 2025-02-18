import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset, num2date
import glob
import xarray as xr


files = sorted(glob.glob('./ERA5/ERA5_SL_*.nc'))
files1 = './ERA5_T2_daily_climatology.nc'

print(files)

#breakpoint()

for i in range(len(files)):


	ds = xr.open_dataset(files[i]) 
	print(ds)

	ds1 = xr.open_dataset(files1) - 273.15
	

	data = (ds['t2m'][0])- 273.15#.sel(latitude=slice(ds1.latitude.max(),ds1.latitude.min()),
		                #    longitude=slice(ds1.longitude.min(),ds1.longitude.max()))) - 273.15

	clim_t2m = (ds1.sel(dayofyear=data.valid_time.dt.dayofyear.values))

	t2_ann = (data-clim_t2m)

	print(t2_ann)

#	print(t2_ann.min())
#	print(t2_ann.max())

#	breakpoint()


	lat = t2_ann['latitude']
	lon = t2_ann['longitude']
	lons, lats = np.meshgrid(lon, lat)

	cart_proj = crs.Stereographic(central_latitude=-62, central_longitude=-62)

	# Create a figure that will have 3 subplots
	fig = plt.figure(figsize=(10,8))
	ax_ctt = plt.axes(projection=cart_proj)
	ax_ctt.coastlines('50m', linewidth=0.8)


	ax_ctt.set_extent([-90, -40, -45, -70], crs=crs.PlateCarree())

	ax_ctt.gridlines(color="black", linestyle="dotted")
	#ax_ctt.set_title('Precipitation'+' on '+timeidx1, {"fontsize" : 14}, weight='bold')

	cmaps=plt.colormaps.get_cmap('coolwarm')
	val_levels= np.arange(-10,10.5,0.50)

	ctt_contours = ax_ctt.contourf(lons, lats, 
	                           t2_ann["t2m"],
	                           extend="both",
	                           cmap=cmaps,
	                           levels=val_levels,
	                           transform=crs.PlateCarree())

	cbar = fig.colorbar(ctt_contours, ax=ax_ctt, orientation='horizontal', pad=0.03, fraction=.04)
	cbar.ax.tick_params(labelsize=12)
	cbar.set_label('°C', fontsize=14, weight='bold')

	#Q = ax_ctt.quiver(lons, lats, u, v, pivot='middle', transform=crs.PlateCarree(), 
	#	              regrid_shape=30)

	#ax_ctt.quiverkey(Q, X=0.3, Y=1.1, U=50, label='Quiver key, length = 50', labelpos='E')

	#qk = ax_ctt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
    #               coordinates='figure')

	#ax_ctt.quiverkey(Q, X=0.7, Y=1.02, U=10, label="10 m/s", labelpos="E", 
	#                 transform=fig.transFigure, coordinates='figure')

	timeidx1 = str(data.valid_time.values)[:-16]

	ax_ctt.set_title('Temperature anomaly'+' on '+timeidx1, {"fontsize" : 14}, weight='bold')


	fig.savefig('T2_ann_'+timeidx1+'.png', dpi = 500, facecolor='w', bbox_inches = 'tight', 
	            pad_inches = 0.1)
