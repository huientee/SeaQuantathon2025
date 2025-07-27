import xarray as xr
import numpy as np
import pandas as pd

# === Step 0: Load merged dataset ===
file_path = "merged_sst_1991_2024.nc"
ds_merged = xr.open_dataset(file_path)

# === Step 1: Check for missing values and long gaps ===
n_missing = ds_merged['sst'].isnull().sum().item()
print(f"Total missing SST values: {n_missing}")

# Detect long missing sequences (>3 consecutive days)
sst_mean = ds_merged['sst'].mean(dim=['latitude', 'longitude'])
is_nan = sst_mean.isnull().values
dates = pd.to_datetime(ds_merged['time'].values)

gap_lengths = []
current_gap = 0
for missing in is_nan:
    if missing:
        current_gap += 1
    else:
        if current_gap > 0:
            gap_lengths.append(current_gap)
        current_gap = 0
if current_gap > 0:
    gap_lengths.append(current_gap)

long_gaps = [g for g in gap_lengths if g > 3]
if long_gaps:
    print(f"Found {len(long_gaps)} gaps longer than 3 days (maximum gap: {max(long_gaps)} days)")
else:
    print("No gaps longer than 3 consecutive days.")

# === Step 2: Interpolate short NaN gaps (limit 3 days) ===
if n_missing > 0:
    print("Interpolating short SST gaps (up to 3 days)...")
    ds_merged['sst'] = ds_merged['sst'].interpolate_na(dim='time', limit=3)
else:
    print("No missing SST values — skipping interpolation.")

# === Step 3: Compute daily climatology (1991–2020) ===
clim = ds_merged.sel(time=slice('1991-01-01', '2020-12-31')) \
                .groupby('time.dayofyear') \
                .mean(dim='time', skipna=True)

# === Step 4: Recompute SST anomalies (stored as 'ssta') ===
ds_merged['ssta'] = ds_merged['sst'].groupby('time.dayofyear') - clim['sst']

# === Step 5: Replace anomaly with corrected one ===
if 'anom' in ds_merged.variables:
    ds_merged = ds_merged.drop_vars('anom')
ds_merged = ds_merged.rename({'ssta': 'anom'})

# === Step 6: Subset final date range and save ===
subset = ds_merged.sel(time=slice('2016-01-01', '2023-12-31'))
output_file = "sst_processed_20160101_20231231.nc"
subset.to_netcdf(output_file)

print(f"Saved dataset with corrected anomalies from 2016-01-01 to 2023-12-31 to: {output_file}")
