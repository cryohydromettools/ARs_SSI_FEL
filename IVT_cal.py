import xarray as xr
import numpy as np

# Constants
g = 9.80665  # gravity acceleration (m/s²)

# Load ERA5 data
ds = xr.open_dataset("era5_ivt_data.nc").rename({'valid_time': 'time', 'pressure_level': 'level'})



# Extract variables
q = ds['q']       # specific humidity (kg/kg)
u = ds['u']       # u wind component (m/s)
v = ds['v']       # v wind component (m/s)
p = ds['level'] * 100  # Convert hPa to Pa if variable is 'level'

# Ensure pressure levels are sorted ascending (from top to bottom)
q = q.sortby('level')
u = u.sortby('level')
v = v.sortby('level')

# Compute vapor transport components
qu = q * u
qv = q * v

# Vertical integration over pressure levels using the trapezoidal rule
ivt_u = np.trapz(qu, p, axis=1) / g
ivt_v = np.trapz(qv, p, axis=1) / g

# Calculate IVT magnitude
ivt = np.sqrt(ivt_u**2 + ivt_v**2)

# Create new xarray Dataset
ivt_ds = xr.Dataset(
    {
        "ivt": (["time", "latitude", "longitude"], ivt),
        "ivt_u": (["time", "latitude", "longitude"], ivt_u),
        "ivt_v": (["time", "latitude", "longitude"], ivt_v),
    },
    coords={
        "time": ds.time,
        "latitude": ds.latitude,
        "longitude": ds.longitude,
    }
)

# Save the dataset to NetCDF
ivt_ds.to_netcdf("ivt_output.nc")

