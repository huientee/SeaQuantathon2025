import copernicusmarine
import numpy as np

# === Download data as NetCDF Dataset ===
ds = copernicusmarine.open_dataset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    variables=["err_sla", "ugos", "vgos"],
    minimum_longitude=102.11,
    maximum_longitude=122.28,
    minimum_latitude=-3.35,
    maximum_latitude=25.69,
    start_datetime="2016-01-01T00:00:00",
    end_datetime="2024-11-19T00:00:00",
)

print("Original dataset info:")
print(ds)
print("\nDataset dimensions:", dict(ds.dims))
print("Dataset coordinates:", list(ds.coords.keys()))
print("Dataset variables:", list(ds.data_vars.keys()))

# === Identify lat/lon coordinate names ===
coord_names = list(ds.coords.keys())
lon_coord = next((c for c in coord_names if 'lon' in c.lower()), None)
lat_coord = next((c for c in coord_names if 'lat' in c.lower()), None)

if lon_coord and lat_coord:
    input_grid_resolution = 1/8   # 0.125 degrees
    output_grid_resolution = 1/4  # 0.25 degrees
    weight = int(np.ceil(output_grid_resolution / input_grid_resolution))
    print(f"\nUsing coordinates: {lat_coord}, {lon_coord}")
    print(f"Coarsening factor: {weight}")

    # === Perform spatial coarsening for all variables ===
    coarsen_dict = {lat_coord: weight, lon_coord: weight}
    resampled_ds = ds.coarsen(coarsen_dict, boundary="pad").mean()

    print("\nResampled dataset info:")
    print(resampled_ds)
    for var in ["err_sla", "ugos", "vgos"]:
        if var in resampled_ds.data_vars:
            print(f"{var} new shape: {resampled_ds[var].shape}")

    # === Save to .nc file ===
    output_nc_file = "ssha_errsla_ugos_vgos_20160101_20241119.nc"
    resampled_ds.to_netcdf(output_nc_file)
    print(f"\nSaved resampled data to: {output_nc_file}")

else:
    print("Could not find longitude/latitude coordinates.")
    print("Available coordinates:", coord_names)
