import cdsapi

# Initialize API client
c = cdsapi.Client()

# Download ERA5 pressure level data
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'specific_humidity',
            'u_component_of_wind',
            'v_component_of_wind',
        ],
        'pressure_level': [
            '300', '400', '500', '600', '700', '850', '925', '1000'
        ],
        'year': '2024',
        'month': '01',
        'day': ['01'],  # You can add more days or automate for all month
        'time': [
            '00:00', '06:00', '12:00', '18:00'
        ],
        'area': [
            -75, -135, -20, -10,  # [south, west, north, east] in degrees (Patagonia example)
        ],
    },
    'era5_ivt_data.nc')

