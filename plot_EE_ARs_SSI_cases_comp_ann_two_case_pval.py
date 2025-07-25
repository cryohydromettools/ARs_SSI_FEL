import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
import pandas as pd
from scipy.ndimage import label
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def compute_composite_significance(daily_data, event_dates, n_bootstrap=2000, alpha=0.05):
    # Convert to xarray datetime if needed
    event_dates = np.array(event_dates, dtype='datetime64[D]')

    # Select composite mean for event days
    composite = daily_data.sel(time=event_dates).mean(dim='time')

    # Number of days in the composite
    n_event_days = len(event_dates)

    # All possible time indices to sample from (excluding event days to avoid bias)
    all_dates = daily_data.time.values
    other_dates = np.setdiff1d(all_dates, event_dates)

    # Bootstrap: sample N random days, compute mean map
    boot_means = []
    for _ in range(n_bootstrap):
        sample_dates = np.random.choice(other_dates, size=n_event_days, replace=False)
        sample_mean = daily_data.sel(time=sample_dates).mean(dim='time')
        boot_means.append(sample_mean)

    # Stack bootstrap results into one DataArray: shape (n_bootstrap, lat, lon)
    boot_stack = xr.concat(boot_means, dim='sample')

    # Compute lower and upper bounds of the confidence interval
    lower = boot_stack.quantile(alpha / 2, dim='sample')
    upper = boot_stack.quantile(1 - alpha / 2, dim='sample')

    # Determine significance mask
    sig_mask = (composite < lower) | (composite > upper)

    return sig_mask


filename = '/home/cr2/cmtorres/ERA5/T2_daily_anomalies.nc'
T2_an = xr.open_dataset(filename)
filename = '/home/cr2/cmtorres/ERA5/Z500_daily_anomalies.nc'
Z500_an = xr.open_dataset(filename)/9.98

filename = '/home/cr2/cmtorres/ERA5/V10_daily_anomalies.nc'
V10_an = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/U10_daily_anomalies.nc'
U10_an = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/WS10_daily_anomalies.nc'
WS10_an = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/V500_daily_anomalies.nc'
V500_an = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/U500_daily_anomalies.nc'
U500_an = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/precip_daily_anomalies.nc'
PRECIP = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/IVTu_12UTC.nc'
IVTu = xr.open_dataset(filename)
IVTu['time'] = ('time', IVTu['time'].dt.floor('D').data)

filename = '/home/cr2/cmtorres/ERA5/IVTv_12UTC.nc'
IVTv = xr.open_dataset(filename)
IVTv['time'] = ('time', IVTv['time'].dt.floor('D').data)

filename = '/home/cr2/cmtorres/ERA5/IVT_daily_anomalies.nc'
IVT = xr.open_dataset(filename)

filename = '/home/cr2/cmtorres/ERA5/MSLP_daily_anomalies.nc'
mslp_an = xr.open_dataset(filename)/100

time_range = ['1999-11-12', '1999-11-13', '1999-11-14', '2005-02-09', '2005-02-10', '2005-02-11', '2005-02-12']
time_range = pd.to_datetime((time_range)).strftime('%Y-%m-%d')

#time_range = HWs_ARS_KJS['date'].to_list()

t2m_anom = T2_an.sel(time=time_range).mean('time').t2m_anomaly
mslp_anom = mslp_an.sel(time=time_range).mean('time').msl_anomaly
u_anom = U10_an.sel(time=time_range).mean('time').u10_anomaly  # U wind anomaly (e.g., at 500 hPa)
v_anom = V10_an.sel(time=time_range).mean('time').v10_anomaly  # V wind anomaly

# --- Define Mercator projection ---
proj = ccrs.Mercator()

fig, axs = plt.subplots(2, 2, figsize=(16, 10), subplot_kw={'projection': proj})
axs = axs.flatten()
# --- Create figure and axes with Mercator projection ---
#fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})
ax = axs[0]
# --- Add geographic features ---
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Optional: Set geographical extent (adjust as needed)
ax.set_extent([-120, -30, -70, -30], crs=ccrs.PlateCarree())

# --- Plot T2m anomalies as shaded contours ---
levels_t2m = np.arange(-7, 8, 1)
cmap = plt.get_cmap("RdBu_r")
t2m_plot = ax.contourf(
    mslp_anom.longitude, mslp_anom.latitude, t2m_anom,
    levels=levels_t2m,
    cmap=cmap,
    extend='both',
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(t2m_plot, ax=ax, orientation='vertical', pad=0.02, label="T2m Anomaly (°C)")

sig_mask = compute_composite_significance(T2_an, time_range, n_bootstrap=2000, alpha=0.05)
ax.contourf(
    sig_mask.longitude, sig_mask.latitude, sig_mask.t2m_anomaly,
    levels=[0.5, 1],   # somente valores True (significativo)
    colors='none',
    hatches=['..'],  # pontilhado
    transform=ccrs.PlateCarree()
)

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

# --- Plot wind vectors ---
# You can thin the grid for better visibility
step = 5
ax.quiver(
    u_anom.longitude[::step], u_anom.latitude[::step],
    u_anom[::step, ::step], v_anom[::step, ::step],
    transform=ccrs.PlateCarree(), scale=150, width=0.0020, regrid_shape=20,
    color='grey'
)

ax.text(
    0.02, 0.975, '(a)',
    transform=ax.transAxes,
    fontsize=14,
    fontweight='bold',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', linewidth=1)
)

# Create a small inset area inside the main axes
ax_inset = inset_axes(ax, width="24%", height="11%", loc='lower left',
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
ref_value = 10
# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.30, 0.825, 'm/s', va='center', fontsize=9)

# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.6, 0.5, f'{ref_value}', va='center', fontsize=9)

# Add the label below the arrow
ax_inset.text(0.5, 0.1, 'Reference vector', ha='center', fontsize=9)

# Ensure nothing outside the box is clipped
ax_inset.set_clip_on(False)

# --- Title and layout ---
ax.set_title(f"T2m (shaded), MSLP (contours), and UV10 (vectors)")

#fig.savefig(f"fig/T2_comp.png", dpi = 300, facecolor='w', bbox_inches = 'tight', pad_inches = 0.1)

# --- Subset and average data over time range ---
precip = PRECIP.sel(time=time_range).mean('time').tp_anomaly # Use .tp_anomaly se for anomalia
Z500_anom = Z500_an.sel(time=time_range).mean('time').z_anomaly[0]#*100
U500_anom = U500_an.sel(time=time_range).mean('time').u_anomaly[0]
V500_anom = V500_an.sel(time=time_range).mean('time').v_anomaly[0]

# --- Define projection ---
#proj = ccrs.Mercator()

# --- Create figure ---
#fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})
ax = axs[1]
# --- Add features ---
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# --- Set extent ---
ax.set_extent([-120, -30, -70, -30], crs=ccrs.PlateCarree())

# --- Plot precipitation (mm) ---
levels_precip = np.arange(-5, 5.5, 0.5)
cmap_precip = plt.get_cmap("BrBG")
precip_plot = ax.contourf(
    precip.longitude, precip.latitude, precip,  # Convert from m to mm
    levels=levels_precip,
    cmap=cmap_precip,
    extend='both',
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(precip_plot, ax=ax, orientation='vertical', pad=0.02, label="RRR Anomaly (mm/day)")

sig_mask = compute_composite_significance(PRECIP, time_range, n_bootstrap=2000, alpha=0.05)
ax.contourf(
    sig_mask.longitude, sig_mask.latitude, sig_mask.tp_anomaly,
    levels=[0.5, 1],   # somente valores True (significativo)
    colors='none',
    hatches=['..'],  # pontilhado
    transform=ccrs.PlateCarree()
)

# --- Plot Z500 anomalies as black contours (dashed for negative, no zero line) ---
# --- Plot Z500 anomalies as black contours (dashed for negative, no zero line) ---
levels_z500 = [lev for lev in np.arange(-250, 250, 50) if lev != 0]  # Exclude zero
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
ax.quiver(
    U500_anom.longitude[::step], U500_anom.latitude[::step],
    U500_anom[::step, ::step], V500_anom[::step, ::step],
    transform=ccrs.PlateCarree(), scale=300, width=0.0020, regrid_shape=20,
    color='grey'
)

ax.text(
    0.02, 0.975, '(b)',
    transform=ax.transAxes,
    fontsize=14,
    fontweight='bold',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', linewidth=1)
)

# Create a small inset area inside the main axes
ax_inset = inset_axes(ax, width="24%", height="11%", loc='lower left',
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
ref_value = 20
# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.30, 0.825, 'm/s', va='center', fontsize=9)

# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.6, 0.5, f'{ref_value}', va='center', fontsize=9)

# Add the label below the arrow
ax_inset.text(0.5, 0.1, 'Reference vector', ha='center', fontsize=9)

# Ensure nothing outside the box is clipped
ax_inset.set_clip_on(False)

# --- Title ---
ax.set_title(f"RRR (shaded), Z500 (contours) and UV500 (vectors)")

#fig.savefig(f"fig/RRR_comp.png", dpi = 300, facecolor='w', bbox_inches = 'tight', pad_inches = 0.1)
# --- Subset and average data over time range ---
u_anom = U10_an.sel(time=time_range).mean('time').u10_anomaly
v_anom = V10_an.sel(time=time_range).mean('time').v10_anomaly

# --- Compute wind speed ---
wind_speed = WS10_an.sel(time=time_range).mean('time').ws10_anomaly


# --- Define Mercator projection ---
#proj = ccrs.Mercator()
ax = axs[2]

# --- Create figure and axes ---
#fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})

# --- Add geographic features ---
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# Optional: set geographical extent
ax.set_extent([-120, -30, -70, -30], crs=ccrs.PlateCarree())

# --- Plot wind speed as color fill ---
levels_ws = np.arange(-8, 9, 1)
cmap_ws = plt.get_cmap("PuOr")
ws_plot = ax.contourf(
    wind_speed.longitude, wind_speed.latitude, wind_speed,
    levels=levels_ws,
    cmap=cmap_ws,
    extend='both',
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(ws_plot, ax=ax, orientation='vertical', pad=0.02, label="WS10 Anomaly (m/s)")

sig_mask = compute_composite_significance(WS10_an, time_range, n_bootstrap=2000, alpha=0.05)

ax.contourf(
    sig_mask.longitude, sig_mask.latitude, sig_mask.ws10_anomaly,
    levels=[0.5, 1],   # somente valores True (significativo)
    colors='none',
    hatches=['..'],  # pontilhado
    transform=ccrs.PlateCarree()
)

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
ax.quiver(
    u_anom.longitude[::step], u_anom.latitude[::step],
    u_anom[::step, ::step], v_anom[::step, ::step],
    transform=ccrs.PlateCarree(), scale=150, width=0.0020, regrid_shape=20,
    color='grey'
)

ax.text(
    0.02, 0.975, '(c)',
    transform=ax.transAxes,
    fontsize=14,
    fontweight='bold',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', linewidth=1)
)

# Create a small inset area inside the main axes
ax_inset = inset_axes(ax, width="24%", height="11%", loc='lower left',
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
ref_value = 10
# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.30, 0.825, 'm/s', va='center', fontsize=9)

# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.6, 0.5, f'{ref_value}', va='center', fontsize=9)

# Add the label below the arrow
ax_inset.text(0.5, 0.1, 'Reference vector', ha='center', fontsize=9)

# Ensure nothing outside the box is clipped
ax_inset.set_clip_on(False)

# --- Title and layout ---
ax.set_title(f"WS10 (shaded), MSLP (contours) and UV10 (vectors)")

#fig.savefig(f"fig/WS_comp.png", dpi = 300, facecolor='w', bbox_inches = 'tight', pad_inches = 0.1)


# --- Subset and average data over time range ---
IVTv_sel = IVTv.sel(time=time_range).mean('time').viwvn_12UTC
IVTu_sel = IVTu.sel(time=time_range).mean('time').viwve_12UTC
Z500_anom = Z500_an.sel(time=time_range).mean('time').z_anomaly[0]#*100

# --- Compute wind speed ---
IVT_sel  = IVT.sel(time=time_range).mean('time').ivt_anomaly 

# --- Define projection ---
#proj = ccrs.Mercator()

# --- Create figure ---
#fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': proj})
ax = axs[3]
# --- Add features ---
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')

# --- Set extent ---
ax.set_extent([-120, -30, -70, -30], crs=ccrs.PlateCarree())

# --- Plot precipitation (mm) ---
levels_precip = np.arange(-200, 201, 25)
cmap_precip = plt.get_cmap("Spectral")
precip_plot = ax.contourf(
    IVT_sel.longitude, IVT_sel.latitude, IVT_sel,  # Convert from m to mm
    levels=levels_precip,
    cmap=cmap_precip,
    extend='both',
    transform=ccrs.PlateCarree()
)
cbar = plt.colorbar(precip_plot, ax=ax, orientation='vertical', pad=0.02, label="IVT Anomaly (kg/m s)")

sig_mask = compute_composite_significance(IVT, time_range, n_bootstrap=2000, alpha=0.05)

ax.contourf(
    sig_mask.longitude, sig_mask.latitude, sig_mask.ivt_anomaly,
    levels=[0.5, 1],   # somente valores True (significativo)
    colors='none',
    hatches=['..'],  # pontilhado
    transform=ccrs.PlateCarree()
)

# --- Plot Z500 anomalies as black contours (dashed for negative, no zero line) ---
levels_z500 = [lev for lev in np.arange(-250, 250, 50) if lev != 0]  # Exclude zero
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
    IVTu_sel.longitude[::step], IVTu_sel.latitude[::step],
    IVTu_sel[::step, ::step], IVTv_sel[::step, ::step],
    transform=ccrs.PlateCarree(), scale=7500, width=0.0020,
    regrid_shape=20, color='grey'
)

ax.text(
    0.02, 0.975, '(d)',
    transform=ax.transAxes,
    fontsize=14,
    fontweight='bold',
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', linewidth=1)
)
# Create a small inset area inside the main axes
ax_inset = inset_axes(ax, width="24%", height="11%", loc='lower left',
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
ax_inset.text(0.25, 0.825, 'kg/m s', va='center', fontsize=9)

# Add the numeric value (e.g., 300) next to the arrow
ax_inset.text(0.6, 0.5, f'{ref_value}', va='center', fontsize=9)

# Add the label below the arrow
ax_inset.text(0.5, 0.1, 'Reference vector', ha='center', fontsize=9)

# Ensure nothing outside the box is clipped
ax_inset.set_clip_on(False)

# --- Title ---
ax.set_title(f"IVT (shaded), Z500 (contours) and IVTuv (vectors)")

fig.subplots_adjust(wspace=0.02, hspace=0.08)

fig.savefig(f"fig/HWs_ARs_comp_two_cases_pval.png", dpi = 300, facecolor='w', bbox_inches = 'tight', pad_inches = 0.1)
fig.savefig(f"fig/HWs_ARs_comp_two_cases_pval.pdf", dpi = 300, facecolor='w', bbox_inches = 'tight', pad_inches = 0.1)