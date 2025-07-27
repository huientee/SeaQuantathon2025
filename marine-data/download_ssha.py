import copernicusmarine
import numpy as np

# === Download data as NetCDF Dataset ===
ds = copernicusmarine.open_dataset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    variables=["err_sla"],
    minimum_longitude=102.11,
    maximum_longitude=122.28,
    minimum_latitude=-3.35,
    maximum_latitude=25.69,
    start_datetime="2016-01-01T00:00:00",
    end_datetime="2023-12-31T00:00:00",
)

print("Original dataset info:")
print(ds)
print("\nDataset dimensions:", dict(ds.dims))
print("Dataset coordinates:", list(ds.coords.keys()))
print("Dataset variables:", list(ds.data_vars.keys()))

# === Check if err_sla is available ===
if 'err_sla' in ds.data_vars:
    print(f"err_sla data shape: {ds['err_sla'].shape}")
    print(f"err_sla sample values:\n{ds['err_sla'].isel(time=0)}")

    # === Downsample from 0.125° to 0.25° ===
    input_grid_resolution = 1/8   # 0.125 degrees
    output_grid_resolution = 1/4  # 0.25 degrees
    weight = int(np.ceil(output_grid_resolution / input_grid_resolution))
    print(f"\nCoarsening factor: {weight}")

    # === Identify lat/lon coordinate names ===
    coord_names = list(ds.coords.keys())
    lon_coord = next((c for c in coord_names if 'lon' in c.lower()), None)
    lat_coord = next((c for c in coord_names if 'lat' in c.lower()), None)

    print(f"Using longitude coordinate: {lon_coord}")
    print(f"Using latitude coordinate: {lat_coord}")

    if lon_coord and lat_coord:
        # === Perform spatial coarsening ===
        coarsen_dict = {lat_coord: weight, lon_coord: weight}
        resample_request = ds.coarsen(coarsen_dict, boundary="pad").mean()

        print("\nResampled dataset info:")
        print(resample_request)
        print(f"New shape: {resample_request['err_sla'].shape}")

        # === Save to .nc file ===
        output_nc_file = "ssha_err_sla_resampled_025deg_20160101_20231231.nc"
        resample_request.to_netcdf(output_nc_file)
        print(f"\nSaved resampled data to: {output_nc_file}")

    else:
        print("Could not find longitude/latitude coordinates.")
        print("Available coordinates:", coord_names)

else:
    print("'err_sla' variable not found in dataset.")
    print("Available variables:", list(ds.data_vars.keys()))
