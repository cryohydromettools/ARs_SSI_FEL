import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
import pandas as pd
from scipy.ndimage import label
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


filename = '/home/cr2/cmtorres/ERA5/T2_daily_anomalies.nc'
T2_an = xr.open_dataset(filename)
filename = '/home/cr2/cmtorres/ERA5/Z500_daily_anomalies.nc'
Z500_an = xr.open_dataset(filename)/9.98

filename = '/home/cr2/cmtorres/ERA5/V10_daily_anomalies.nc'
V10_an = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/U10_daily_anomalies.nc'
U10_an = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/precip_daily_sum.nc'
PRECIP = xr.open_dataset(filename)*1000

filename = '/home/cr2/cmtorres/ERA5/IVTu_12UTC.nc'
IVTu = xr.open_dataset(filename)
IVTu['time'] = IVTu['time'].dt.strftime('%Y-%m-%d')

filename = '/home/cr2/cmtorres/ERA5/IVTv_12UTC.nc'
IVTv = xr.open_dataset(filename)
IVTv['time'] = IVTv['time'].dt.strftime('%Y-%m-%d')

filename = '/home/cr2/cmtorres/ERA5/MSLP_daily_anomalies.nc'
mslp_an = xr.open_dataset(filename)/100

filename = 'data/ARs_EMEs_SSI.csv'
days_EC = pd.read_csv(filename, sep= '\t')
print(days_EC)

print(IVTu)
print(IVTv)
print(Z500_an)


date_events = pd.read_csv('data/HW_ARs_days_SSI.csv', sep='\t')
days_events = date_events.label.drop_duplicates()
print(days_events)
date_event = date_events[date_events.label==151]


# --- Subset and average data over time range ---
time_range = date_event.date.values
date_start = pd.to_datetime(min(time_range)).strftime('%Y-%m-%d')
date_end = pd.to_datetime(max(time_range)).strftime('%Y-%m-%d')

t2m_anom = T2_an.sel(time=time_range).mean('time').t2m_anomaly
mslp_anom = mslp_an.sel(time=time_range).mean('time').msl_anomaly
u_anom = U10_an.sel(time=time_range).mean('time').u10_anomaly  # U wind anomaly (e.g., at 500 hPa)
v_anom = V10_an.sel(time=time_range).mean('time').v10_anomaly  # V wind anomaly

# --- Define Mercator projection ---
proj = ccrs.Mercator()

# --- Create figure and axes with Mercator projection ---
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})

# --- Add geographic features ---
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Optional: Set geographical extent (adjust as needed)
ax.set_extent([-120, -30, -70, -30], crs=ccrs.PlateCarree())

# --- Plot T2m anomalies as shaded contours ---
levels_t2m = np.arange(-6, 7, 1)
cmap = plt.get_cmap("RdBu_r")
t2m_plot = ax.contourf(
    mslp_anom.longitude, mslp_anom.latitude, t2m_anom,
    levels=levels_t2m,
    cmap=cmap,
    extend='both',
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(t2m_plot, ax=ax, orientation='vertical', pad=0.02, label="T2m Anomaly (°C)")

# --- Plot Z500 anomalies as black contours (dashed for negative, no zero line) ---
levels_z500 = [lev for lev in np.arange(-40, 41, 2.5) if lev != 0]  # Exclude zero
linestyles = ['dashed' if lev < 0 else 'solid' for lev in levels_z500]
contour = ax.contour(
    mslp_anom.longitude, mslp_anom.latitude, mslp_anom,
    levels=levels_z500,
    colors='black',
    linewidths=1.0,
    linestyles=linestyles,
    transform=ccrs.PlateCarree()
)
ax.clabel(contour, fmt='%d', fontsize=9)

# --- Plot wind vectors (thinned) ---

step = 5
ref_value = 10  # reference vector value

# Draw the main vector field
Q = ax.quiver(
    u_anom.longitude[::step], u_anom.latitude[::step],
    u_anom[::step, ::step], v_anom[::step, ::step],
    transform=ccrs.PlateCarree(), scale=250, width=0.0020, regrid_shape=20,
    color='grey'
)


# Create a small inset area inside the main axes
ax_inset = inset_axes(ax, width="18%", height="8%", loc='lower left',
                      bbox_to_anchor=(0.02, 0.02, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)

# Set white background and black border for the inset box
ax_inset.set_facecolor('white')
for spine in ax_inset.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# Remove ticks and unnecessary borders
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_xlim(0, 1)
ax_inset.set_ylim(0, 1)

# Add the reference arrow inside the inset box
qk = ax_inset.quiver(
    0.2, 0.5, 1, 0, angles='xy', scale_units='xy', scale=4,
    color='grey'
)

# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.6, 0.5, f'{ref_value}', va='center', fontsize=9)

# Add the label below the arrow
ax_inset.text(0.5, 0.1, 'Reference vector', ha='center', fontsize=9)

# Ensure nothing outside the box is clipped
ax_inset.set_clip_on(False)

# --- Title and layout ---
ax.set_title(f"T2m (shaded), Z500 (black contours), and Wind (grey vectors) Anomalies\n{date_start} to {date_end}")

fig.savefig('fig/T2.png', dpi = 500, facecolor='w', bbox_inches = 'tight', 
	            pad_inches = 0.1)


# --- Subset and average data over time range ---
u_anom = U10_an.sel(time=time_range).mean('time').u10_anomaly
v_anom = V10_an.sel(time=time_range).mean('time').v10_anomaly
mslp_anom = mslp_an.sel(time=time_range).mean('time').msl_anomaly

# --- Compute wind speed ---
wind_speed = np.sqrt(u_anom**2 + v_anom**2)

# --- Define Mercator projection ---
proj = ccrs.Mercator()

# --- Create figure and axes ---
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})

# --- Add geographic features ---
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Optional: set geographical extent
ax.set_extent([-120, -30, -70, -30], crs=ccrs.PlateCarree())

# --- Plot wind speed as color fill ---
levels_ws = np.arange(0, 16, 1)
cmap_ws = plt.get_cmap("viridis")
ws_plot = ax.contourf(
    wind_speed.longitude, wind_speed.latitude, wind_speed,
    levels=levels_ws,
    cmap=cmap_ws,
    extend='max',
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(ws_plot, ax=ax, orientation='vertical', pad=0.02, label="Wind Speed (m/s)")

# --- Plot Z500 anomalies as black contours (dashed for negative, no zero line) ---
levels_z500 = [lev for lev in np.arange(-40, 41, 2.5) if lev != 0]  # Exclude zero
linestyles = ['dashed' if lev < 0 else 'solid' for lev in levels_z500]
contour = ax.contour(
    mslp_anom.longitude, mslp_anom.latitude, mslp_anom,
    levels=levels_z500,
    colors='black',
    linewidths=1.0,
    linestyles=linestyles,
    transform=ccrs.PlateCarree()
)
ax.clabel(contour, fmt='%d', fontsize=9)

# --- Plot wind vectors (thinned) ---

step = 5
ref_value = 10  # reference vector value

# Draw the main vector field
Q = ax.quiver(
    u_anom.longitude[::step], u_anom.latitude[::step],
    u_anom[::step, ::step], v_anom[::step, ::step],
    transform=ccrs.PlateCarree(), scale=250, width=0.0020, regrid_shape=20,
    color='grey'
)


# Create a small inset area inside the main axes
ax_inset = inset_axes(ax, width="18%", height="8%", loc='lower left',
                      bbox_to_anchor=(0.02, 0.02, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)

# Set white background and black border for the inset box
ax_inset.set_facecolor('white')
for spine in ax_inset.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# Remove ticks and unnecessary borders
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_xlim(0, 1)
ax_inset.set_ylim(0, 1)

# Add the reference arrow inside the inset box
qk = ax_inset.quiver(
    0.2, 0.5, 1, 0, angles='xy', scale_units='xy', scale=4,
    color='grey'
)

# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.6, 0.5, f'{ref_value}', va='center', fontsize=9)

# Add the label below the arrow
ax_inset.text(0.5, 0.1, 'Reference vector', ha='center', fontsize=9)

# Ensure nothing outside the box is clipped
ax_inset.set_clip_on(False)

# --- Title and layout ---
ax.set_title(f"Wind Speed (shaded), MSLP (contours) and Wind Vectors\n{date_start} to {date_end}")
fig.savefig('fig/WS.png', dpi = 500, facecolor='w', bbox_inches = 'tight', 
	            pad_inches = 0.1)

precip = PRECIP.sel(time=time_range).mean('time').tp_daily_sum  # Use .tp_anomaly se for anomalia
mslp_anom = mslp_an.sel(time=time_range).mean('time').msl_anomaly
u_anom = U10_an.sel(time=time_range).mean('time').u10_anomaly
v_anom = V10_an.sel(time=time_range).mean('time').v10_anomaly

# --- Define projection ---
proj = ccrs.Mercator()

# --- Create figure ---
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})

# --- Add features ---
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# --- Set extent ---
ax.set_extent([-120, -30, -70, -30], crs=ccrs.PlateCarree())

# --- Plot precipitation (mm) ---
levels_precip = np.arange(0, 11, 1)
cmap_precip = plt.get_cmap("Blues")
precip_plot = ax.contourf(
    precip.longitude, precip.latitude, precip,  # Convert from m to mm
    levels=levels_precip,
    cmap=cmap_precip,
    extend='max',
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(precip_plot, ax=ax, orientation='vertical', pad=0.02, label="Precipitation (mm/day)")

# --- Plot Z500 anomalies as black contours (dashed for negative, no zero line) ---
levels_z500 = [lev for lev in np.arange(-40, 41, 2.5) if lev != 0]  # Exclude zero
linestyles = ['dashed' if lev < 0 else 'solid' for lev in levels_z500]
contour = ax.contour(
    mslp_anom.longitude, mslp_anom.latitude, mslp_anom,
    levels=levels_z500,
    colors='black',
    linewidths=1.0,
    linestyles=linestyles,
    transform=ccrs.PlateCarree()
)
ax.clabel(contour, fmt='%d', fontsize=9)

# --- Plot wind vectors (thinned) ---

step = 5
ref_value = 10  # reference vector value

# Draw the main vector field
Q = ax.quiver(
    u_anom.longitude[::step], u_anom.latitude[::step],
    u_anom[::step, ::step], v_anom[::step, ::step],
    transform=ccrs.PlateCarree(), scale=250, width=0.0020, regrid_shape=20,
    color='grey'
)


# Create a small inset area inside the main axes
ax_inset = inset_axes(ax, width="18%", height="8%", loc='lower left',
                      bbox_to_anchor=(0.02, 0.02, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)

# Set white background and black border for the inset box
ax_inset.set_facecolor('white')
for spine in ax_inset.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# Remove ticks and unnecessary borders
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_xlim(0, 1)
ax_inset.set_ylim(0, 1)

# Add the reference arrow inside the inset box
qk = ax_inset.quiver(
    0.2, 0.5, 1, 0, angles='xy', scale_units='xy', scale=4,
    color='grey'
)

# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.6, 0.5, f'{ref_value}', va='center', fontsize=9)

# Add the label below the arrow
ax_inset.text(0.5, 0.1, 'Reference vector', ha='center', fontsize=9)

# Ensure nothing outside the box is clipped
ax_inset.set_clip_on(False)


# --- Title ---
ax.set_title(f"Precipitation (shaded), MSLP (contours) and Wind Vectors\n{date_start} to {date_end}")
plt.tight_layout()
fig.savefig('fig/RRR.png', dpi = 500, facecolor='w', bbox_inches = 'tight', 
	            pad_inches = 0.1)


# --- Subset and average data over time range ---
IVTu = IVTu.sel(time=time_range).mean('time').viwve_12UTC
IVTv = IVTv.sel(time=time_range).mean('time').viwvn_12UTC
Z500_anom = Z500_an.sel(time=time_range).mean('time').z_anomaly[0]


# --- Compute wind speed ---
IVT = np.sqrt(IVTu**2 + IVTv**2)

# --- Define projection ---
proj = ccrs.Mercator()

# --- Create figure ---
fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})

# --- Add features ---
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# --- Set extent ---
ax.set_extent([-120, -30, -70, -30], crs=ccrs.PlateCarree())

# --- Plot precipitation (mm) ---
levels_precip = np.arange(100, 750, 50)
cmap_precip = plt.get_cmap("rainbow")
precip_plot = ax.contourf(
    IVT.longitude, IVT.latitude, IVT,  # Convert from m to mm
    levels=levels_precip,
    cmap=cmap_precip,
    extend='max',
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(precip_plot, ax=ax, orientation='vertical', pad=0.02, label="IVT (kg m⁻¹ s⁻¹)")

# --- Plot Z500 anomalies as black contours (dashed for negative, no zero line) ---
levels_z500 = [lev for lev in np.arange(-200, 300, 25) if lev != 0]  # Exclude zero
linestyles = ['dashed' if lev < 0 else 'solid' for lev in levels_z500]
contour = ax.contour(
    Z500_anom.longitude, Z500_anom.latitude, Z500_anom,
    levels=levels_z500,
    colors='black',
    linewidths=1.0,
    linestyles=linestyles,
    transform=ccrs.PlateCarree()
)
ax.clabel(contour, fmt='%d', fontsize=9)

# --- Plot wind vectors (thinned) ---
step = 5

ref_value = 300  # reference vector value

# Draw the main vector field
Q = ax.quiver(
    IVTu.longitude[::step], IVTu.latitude[::step],
    IVTu[::step, ::step], IVTv[::step, ::step],
    transform=ccrs.PlateCarree(), scale=7500, width=0.0020,
    regrid_shape=20, color='grey'
)

# Create a small inset area inside the main axes
ax_inset = inset_axes(ax, width="18%", height="8%", loc='lower left',
                      bbox_to_anchor=(0.02, 0.02, 1, 1),
                      bbox_transform=ax.transAxes, borderpad=0)

# Set white background and black border for the inset box
ax_inset.set_facecolor('white')
for spine in ax_inset.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(1.2)

# Remove ticks and unnecessary borders
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.set_xlim(0, 1)
ax_inset.set_ylim(0, 1)

# Add the reference arrow inside the inset box
qk = ax_inset.quiver(
    0.2, 0.5, 1, 0, angles='xy', scale_units='xy', scale=4,
    color='grey'
)

# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.6, 0.5, f'{ref_value}', va='center', fontsize=9)

# Add the label below the arrow
ax_inset.text(0.5, 0.1, 'Reference vector', ha='center', fontsize=9)

# Ensure nothing outside the box is clipped
ax_inset.set_clip_on(False)

# --- Title ---
ax.set_title(f"IVT (shaded), Z500 (contours) and IVTuv Vectors\n{date_start} to {date_end}")
fig.savefig('fig/IVT.png', dpi = 500, facecolor='w', bbox_inches = 'tight', 
	            pad_inches = 0.1)

