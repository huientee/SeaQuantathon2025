# Marine Heatwave Forecasting – Data Preparation

This folder contains scripts for downloading and preprocessing key oceanographic datasets used in marine heatwave forecasting. The target domain is the South China Sea, with a 0.125° spatial buffer applied to ensure complete coverage of coastal grid cells.

## Included Datasets

- **Sea Surface Temperature (SST)**  
  Source: NOAA OISST v2.1 (Daily, Global)

- **Sea Surface Height Anomaly (SSHA)**  
  Source: Copernicus Marine – DUACS Global L4

- **Surface Wind (SSW)**  
  Source: RSS CCMP Wind Analysis v3.1

## Spatial Domain

Region based on [Marine Regions Gazetteer ID 4332](https://marineregions.org/gazetteer.php?p=details&id=4332):  
- **Latitude:** –3.35° to 25.69°  
- **Longitude:** 102.11° to 122.28°

A buffer of 0.125° is applied on all sides to fully encompass coastal grid cells.

## Grid Resolution

All datasets are spatially subset to the defined region and, where necessary, resampled to a common spatial resolution of **0.25°**.
