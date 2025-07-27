#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import requests
import xarray as xr
import datetime
from tqdm import tqdm
import tempfile

# === CONFIGURATION ===
BASE_URL = "https://data.remss.com/ccmp/v03.1"
DEST_DIR = "ccmp_data"
MERGED_OUT = "ccmp_merged_20160101_20241119_scs.nc"

START_DATE = datetime.date(2016, 1, 1)
END_DATE = datetime.date(2024, 11, 19)

# Subsetting bounds for South China Sea
LAT_MIN = -3.35
LAT_MAX = 25.69
LON_MIN = 102.11
LON_MAX = 122.28

# === STEP 1: Generate daily file URLs ===
def generate_daily_urls(start_date, end_date):
    urls = []
    date = start_date
    while date <= end_date:
        datestr = date.strftime("%Y%m%d")
        filename = f"CCMP_Wind_Analysis_{datestr}_V03.1_L4.nc"
        url = f"{BASE_URL}/Y{date.year}/M{date.month:02d}/{filename}"
        urls.append((url, filename))
        date += datetime.timedelta(days=1)
    return urls

# === STEP 2: Search for existing files anywhere in DEST_DIR ===
def find_existing_files(dest_dir):
    existing_files = {}
    for root, _, files in os.walk(dest_dir):
        for file in files:
            if file.startswith("CCMP_Wind_Analysis_") and file.endswith(".nc"):
                existing_files[file] = os.path.join(root, file)
    return existing_files

# === STEP 3: Download missing files ===
def download_missing_files(urls, dest_dir, existing_files):
    os.makedirs(dest_dir, exist_ok=True)
    downloaded_paths = []

    print("Checking and downloading files (skipping existing)...")
    for url, filename in tqdm(urls):
        if filename in existing_files:
            downloaded_paths.append(existing_files[filename])
            continue

        year = filename[20:24]
        month = filename[24:26]
        out_dir = os.path.join(dest_dir, f"{year}_{month}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)

        try:
            r = requests.get(url, stream=True, timeout=20)
            if r.status_code != 200:
                print(f"Skipping missing file: {filename}")
                continue
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded_paths.append(out_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            continue

    return sorted(downloaded_paths)

# === STEP 4: Merge, subset, and remove 'nobs' variable ===
def merge_and_subset_netcdfs_in_batches(file_list, out_path, lat_min, lat_max, lon_min, lon_max, batch_size=10):
    if not file_list:
        print("No files to merge.")
        return

    tmp_dir = tempfile.mkdtemp(prefix="ccmp_tmp_")
    tmp_files = []

    print("Opening, subsetting, and batching datasets...")
    try:
        for i in range(0, len(file_list), batch_size):
            batch_files = file_list[i:i+batch_size]
            ds_list = []

            for f in batch_files:
                try:
                    ds = xr.open_dataset(f)
                    ds['latitude'] = ds['latitude'].astype('float64')
                    ds['longitude'] = ds['longitude'].astype('float64')
                    ds['latitude'].attrs = {}
                    ds['longitude'].attrs = {}

                    subset = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
                    subset.load()
                    ds_list.append(subset)
                    ds.close()
                except Exception as e:
                    print(f"Skipping corrupt file {f}: {e}")

            if ds_list:
                try:
                    if 'time' in ds_list[0].dims:
                        batch_ds = xr.concat(ds_list, dim='time', combine_attrs='override')
                    else:
                        batch_ds = xr.merge(ds_list, compat='override', combine_attrs='override')
                except Exception as e:
                    print(f"Concat/merge failed for batch: {e}")
                    batch_ds = ds_list[0]
                    for ds in ds_list[1:]:
                        try:
                            batch_ds = xr.combine_by_coords([batch_ds, ds], compat='no_conflicts', combine_attrs='drop_conflicts')
                        except:
                            print(f"Warning: Could not merge some datasets in batch {i//batch_size}")
                            break

                # Remove 'nobs' variable if it exists
                if "nobs" in batch_ds.variables:
                    batch_ds = batch_ds.drop_vars("nobs")

                tmp_path = os.path.join(tmp_dir, f"batch_{i//batch_size:04d}.nc")
                batch_ds.to_netcdf(tmp_path)
                tmp_files.append(tmp_path)
                print(f"Saved batch {i//batch_size + 1} to temporary file")

                for ds in ds_list:
                    ds.close()

        print("Merging all batches into final dataset...")
        final_ds_list = [xr.open_dataset(f) for f in tmp_files]

        if len(final_ds_list) == 1:
            final_merged = final_ds_list[0]
        else:
            try:
                if 'time' in final_ds_list[0].dims:
                    final_merged = xr.concat(final_ds_list, dim='time', combine_attrs='override')
                else:
                    final_merged = xr.merge(final_ds_list, compat='override', combine_attrs='override')
            except Exception as e:
                print(f"Concat/merge failed: {e}")
                final_merged = xr.combine_by_coords(final_ds_list, compat='no_conflicts', combine_attrs='drop_conflicts')

        # Sort by time if applicable
        if 'time' in final_merged.dims:
            final_merged = final_merged.sortby('time')

        # Final clean removal of 'nobs' if present
        if "nobs" in final_merged.variables:
            final_merged = final_merged.drop_vars("nobs")

        final_merged.to_netcdf(out_path)
        print(f"Final merged dataset saved to: {out_path}")

        for ds in final_ds_list:
            ds.close()

    except Exception as e:
        print(f"Merge failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("Cleaning up temporary files...")
        for f in tmp_files:
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

# === MAIN WORKFLOW ===
if __name__ == "__main__":
    urls = generate_daily_urls(START_DATE, END_DATE)
    existing_files = find_existing_files(DEST_DIR)
    files = download_missing_files(urls, DEST_DIR, existing_files)
    merge_and_subset_netcdfs_in_batches(files, MERGED_OUT, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, batch_size=5)
