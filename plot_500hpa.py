import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
import cartopy.feature as cfeature
from netCDF4 import Dataset, num2date
import glob
import xarray as xr


files = sorted(glob.glob('./ERA5/ERA5_PL_*.nc'))

for i in files:


	ds = xr.open_dataset(i)

	data = ds['z'][0,4,:,:]/10

	lat = ds['latitude']
	lon = ds['longitude']
	lons, lats = np.meshgrid(lon, lat)

	cart_proj = crs.Stereographic(central_latitude=-62, central_longitude=-62)

	# Create a figure that will have 3 subplots
	fig = plt.figure(figsize=(10,8))
	ax_ctt = plt.axes(projection=cart_proj)
	ax_ctt.coastlines('50m', linewidth=0.8)


	ax_ctt.set_extent([-90, -40, -45, -70], crs=crs.PlateCarree())

	ax_ctt.gridlines(color="black", linestyle="dotted")
	#ax_ctt.set_title('Precipitation'+' on '+timeidx1, {"fontsize" : 14}, weight='bold')

	cmaps=plt.colormaps.get_cmap('RdBu_r')
	val_levels= np.arange(4750,5850,50)

	ctt_contours = ax_ctt.contourf(lons, lats, 
	                           data,
	                           extend="both",
	                           cmap=cmaps,
	                           levels=val_levels,
	                           transform=crs.PlateCarree())

	cbar = fig.colorbar(ctt_contours, ax=ax_ctt, orientation='horizontal', pad=0.03, fraction=.04)
	cbar.ax.tick_params(labelsize=12)
	cbar.set_label('m', fontsize=14, weight='bold')



	timeidx1 = str(data.valid_time.values)[:-16]

	ax_ctt.set_title('geopotential height at 500mb'+' on '+timeidx1, {"fontsize" : 14}, weight='bold')


	fig.savefig('Z500_'+timeidx1+'.png', dpi = 500, facecolor='w', bbox_inches = 'tight', 
	            pad_inches = 0.1)
