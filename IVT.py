import numpy as np
import xarray as xr

import glob

files = sorted(glob.glob('ERA5_PL*.nc'))

for i in files:
    print(i)

    # Load ERA5 dataset (example file, adapt as needed)
    ds = xr.open_dataset(i)
    print(ds)
    # Extract required variables
    q = ds["q"]  # Specific humidity (kg/kg)
    u = ds["u"]  # Zonal wind (m/s)
    v = ds["v"]  # Meridional wind (m/s)
    p = ds["pressure_level"] * 100  # Convert hPa to Pa

    g = 9.81  # Gravity (m/s²)

    # Compute pressure layer thickness (Δp) along the vertical axis (level dimension)
    dp = np.abs(np.gradient(p))

    # Compute pressure layer thickness (Δp) along the vertical axis (level dimension)
    dp = np.abs(np.gradient(p))  # Compute pressure differences
    dp_3D = xr.DataArray(dp, dims=["pressure_level"], coords={"pressure_level": ds["pressure_level"]})  # Match dimensions

    # Compute IVT components using summation over the pressure levels
    IVT_x = (1 / g) * ((q * u * dp_3D).sum(dim="pressure_level"))
    IVT_y = (1 / g) * ((q * v * dp_3D).sum(dim="pressure_level"))

    # Compute IVT magnitude
    IVT = np.sqrt(IVT_x**2 + IVT_y**2)

    # Ensure dimensions are correct before saving
    IVT_x = IVT_x.squeeze()  # Remove any singleton dimensions
    IVT_y = IVT_y.squeeze()
    IVT = IVT.squeeze()
    print(IVT)


    # Save IVT as a new dataset
    # Create a new dataset ensuring the correct dimensions
    ds_ivt = xr.Dataset({
        "IVT": (["latitude", "longitude"], IVT.values),  
        "IVT_x": (["latitude", "longitude"], IVT_x.values),
        "IVT_y": (["latitude", "longitude"], IVT_y.values),
    }, coords={
        "latitude": ds["latitude"].values,  
        "longitude": ds["longitude"].values
    })

    print(ds_ivt)
    # Save to NetCDF
    ds_ivt.to_netcdf('IVT_'+i, format="NETCDF4_CLASSIC")

