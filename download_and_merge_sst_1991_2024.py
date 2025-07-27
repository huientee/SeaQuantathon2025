import os
import xarray as xr
import requests
from time import sleep

# === CONFIGURATION ===
years_1 = range(1991, 2016)
years_2 = range(2016, 2025)

dir1 = "sst_yearly_data_1991_2015"
dir2 = "sst_yearly_data_2016_2024"
os.makedirs(dir1, exist_ok=True)
os.makedirs(dir2, exist_ok=True)

file1 = "merged_sst_1991_2015.nc"
file2 = "merged_sst_2016_2024.nc"
final_output = "merged_sst_1991_2024.nc"

# === URL TEMPLATES ===
url_template_1 = (
    "http://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_afc8_9785_907e.nc"
    "?anom[({year}-01-01):1:({year}-12-31T00:00:00Z)][(-3.35):1:(25.69)][(102.11):1:(122.28)],"
    "err[({year}-01-01):1:({year}-12-31T00:00:00Z)][(-3.35):1:(25.69)][(102.11):1:(122.28)],"
    "sst[({year}-01-01):1:({year}-12-31T00:00:00Z)][(-3.35):1:(25.69)][(102.11):1:(122.28)]"
)

url_template_2 = (
    "http://apdrc.soest.hawaii.edu/erddap/griddap/hawaii_soest_330b_094e_ca45.nc"
    "?anom[({year}-01-01):1:({year}-12-31T00:00:00Z)][(0.0):1:(0.0)][(-3.35):1:(25.69)][(102.11):1:(122.28)],"
    "err[({year}-01-01):1:({year}-12-31T00:00:00Z)][(0.0):1:(0.0)][(-3.35):1:(25.69)][(102.11):1:(122.28)],"
    "sst[({year}-01-01):1:({year}-12-31T00:00:00Z)][(0.0):1:(0.0)][(-3.35):1:(25.69)][(102.11):1:(122.28)]"
)

# === DOWNLOAD AND MERGE FUNCTION ===
def download_and_merge(years, url_template, output_dir, output_file, squeeze=False):
    datasets = []

    for year in years:
        filename = os.path.join(output_dir, f"sst_{year}.nc")
        url = url_template.format(year=year)

        if not os.path.exists(filename):
            print(f"Downloading {year}...")
            try:
                response = requests.get(url, stream=True, timeout=60)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Saved {filename}")
                else:
                    print(f"Failed to download {year}: HTTP {response.status_code}")
                    continue
            except Exception as e:
                print(f"Error downloading {year}: {e}")
                continue
            sleep(1)
        else:
            print(f"{filename} already exists. Skipping.")

        try:
            ds = xr.open_dataset(filename)
            if squeeze:
                ds = ds.squeeze(drop=True)
            datasets.append(ds)
        except Exception as e:
            print(f"Error opening {filename}: {e}")

    if datasets:
        print(f"Merging {len(datasets)} datasets...")
        combined = xr.concat(datasets, dim="time")
        combined.to_netcdf(output_file)
        print(f"Saved merged file: {output_file}")
    else:
        print("No valid datasets to merge.")

# === PART 1: 1991–2015 ===
download_and_merge(years_1, url_template_1, dir1, file1, squeeze=False)

# === PART 2: 2016–2024 ===
download_and_merge(years_2, url_template_2, dir2, file2, squeeze=True)

# === FINAL MERGE ===
print("Merging both time periods...")
try:
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)
    combined = xr.concat([ds1, ds2], dim="time")
    combined.to_netcdf(final_output)
    print(f"Final merged file saved as: {final_output}")
except Exception as e:
    print(f"Final merge failed: {e}")
