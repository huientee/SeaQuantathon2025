import xarray as xr
import numpy as np
import warnings
from typing import Tuple, Optional, Dict
from tqdm import tqdm

warnings.filterwarnings('ignore')

class SCSDataLoader:
    """
    South China Sea data loader for multiple NetCDF files with physics-informed features
    """
    
    def __init__(self, sst_path: str, ssha_path: str, wind_path: str):
        self.sst_path = sst_path
        self.ssha_path = ssha_path 
        self.wind_path = wind_path
        
        # Define SCS region bounds (0.25 degree resolution)
        self.lat_bounds = (-3.35, 25.69)
        self.lon_bounds = (102.11, 122.28)
        self.resolution = 0.25
        
    def load_individual_datasets(self) -> Tuple[Optional[xr.Dataset], Optional[xr.Dataset], Optional[xr.Dataset]]:
        """Load each dataset separately with consistent spatial/temporal bounds"""
        try:
            print("Loading individual datasets...")
            
            # Load SST data (sst, anom)
            print("- Loading SST dataset...")
            sst_ds = xr.open_dataset(self.sst_path)
            sst_ds = self._standardize_coords(sst_ds)
            sst_ds = self._crop_to_scs(sst_ds)
            
            # Load SSHA data (err_sla, ugos, vgos)
            print("- Loading SSHA/current dataset...")
            ssha_ds = xr.open_dataset(self.ssha_path)
            ssha_ds = self._standardize_coords(ssha_ds)
            ssha_ds = self._crop_to_scs(ssha_ds)
            
            # Load Wind data (ws, uwnd, vwind)  
            print("- Loading wind dataset...")
            wind_ds = xr.open_dataset(self.wind_path)
            wind_ds = self._standardize_coords(wind_ds)
            wind_ds = self._crop_to_scs(wind_ds)
            
            return sst_ds, ssha_ds, wind_ds
            
        except Exception as e:
            print(f"Error loading datasets: {str(e)}")
            return None, None, None
    
    def _standardize_coords(self, ds: xr.Dataset) -> xr.Dataset:
        """Standardize coordinate names and ensure consistent format"""
        coord_mapping = {
            'lat': 'latitude',
            'lon': 'longitude', 
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        }
        
        for old_name, new_name in coord_mapping.items():
            if old_name in ds.coords:
                ds = ds.rename({old_name: new_name})
        
        # Ensure longitude is in 0-360 format for SCS
        if 'longitude' in ds.coords:
            lon_vals = ds.longitude.values
            if np.any(lon_vals < 0):
                ds['longitude'] = ds.longitude % 360
                
        return ds
    
    def _crop_to_scs(self, ds: xr.Dataset) -> xr.Dataset:
        """Crop dataset to South China Sea region"""
        try:
            ds = ds.sel(
                latitude=slice(self.lat_bounds[0], self.lat_bounds[1]),
                longitude=slice(self.lon_bounds[0], self.lon_bounds[1])
            )
            return ds
        except Exception as e:
            print(f"Warning: Could not crop dataset to SCS bounds: {str(e)}")
            return ds
    
    def merge_datasets(self) -> Optional[xr.Dataset]:
        """Merge all datasets into single physics-informed dataset"""
        sst_ds, ssha_ds, wind_ds = self.load_individual_datasets()
        
        if any(ds is None for ds in [sst_ds, ssha_ds, wind_ds]):
            print("Failed to load one or more datasets")
            return None
        
        try:
            print("\nMerging datasets...")
            
            # Find common time period
            time_start = max(ds.time.min().values for ds in [sst_ds, ssha_ds, wind_ds])
            time_end = min(ds.time.max().values for ds in [sst_ds, ssha_ds, wind_ds])
            
            print(f"Common time period: {time_start} to {time_end}")
            
            # Crop all datasets to common time period
            datasets = []
            for ds in [sst_ds, ssha_ds, wind_ds]:
                ds_cropped = ds.sel(time=slice(time_start, time_end))
                datasets.append(ds_cropped)
            
            # Merge datasets
            merged_ds = xr.merge(datasets, compat='override')
            
            # Add computed variables
            merged_ds = self._add_computed_variables(merged_ds)
            
            # Validate merged dataset
            self._validate_merged_dataset(merged_ds)
            
            print(f"\nMerged dataset created with variables: {list(merged_ds.data_vars.keys())}")
            print(f"Spatial dimensions: {len(merged_ds.latitude)} x {len(merged_ds.longitude)}")
            print(f"Time steps: {len(merged_ds.time)}")
            
            return merged_ds
            
        except Exception as e:
            print(f"Error merging datasets: {str(e)}")
            return None
    
    def _add_computed_variables(self, ds: xr.Dataset) -> xr.Dataset:
        """Add physics-informed computed variables"""
        print("Computing derived variables...")
        
        # Current speed and direction from ugos, vgos
        if 'ugos' in ds and 'vgos' in ds:
            ds['current_speed'] = np.sqrt(ds.ugos**2 + ds.vgos**2)
            ds['current_dir'] = np.arctan2(ds.vgos, ds.ugos) * 180 / np.pi
            ds.current_dir.attrs = {'units': 'degrees', 'long_name': 'Current Direction'}
            ds.current_speed.attrs = {'units': 'm/s', 'long_name': 'Current Speed'}
        
        # Wind speed from components if not already present
        if 'ws' not in ds and 'uwnd' in ds and 'vwind' in ds:
            ds['ws'] = np.sqrt(ds.uwnd**2 + ds.vwind**2)
            ds.ws.attrs = {'units': 'm/s', 'long_name': 'Wind Speed'}
        
        # Wind direction
        if 'uwnd' in ds and 'vwind' in ds:
            ds['wind_dir'] = np.arctan2(ds.vwind, ds.uwnd) * 180 / np.pi
            ds.wind_dir.attrs = {'units': 'degrees', 'long_name': 'Wind Direction'}
        
        # Add day of year for seasonal features
        ds['dayofyear'] = ds.time.dt.dayofyear
        
        return ds
    
    def _validate_merged_dataset(self, ds: xr.Dataset):
        """Validate the merged dataset has all required variables"""
        expected_vars = ['sst', 'anom', 'err_sla', 'ugos', 'vgos', 'ws', 'uwnd', 'vwind']
        missing_vars = [var for var in expected_vars if var not in ds]
        
        if missing_vars:
            warnings.warn(f"Missing expected variables: {missing_vars}")
        
        # Check for NaN values and data quality
        for var in ds.data_vars:
            nan_count = ds[var].isnull().sum().compute()
            total_count = ds[var].size
            nan_percentage = (nan_count / total_count) * 100
            
            if nan_percentage > 50:
                warnings.warn(f"Variable {var} has {nan_percentage:.1f}% NaN values")
    
    def split_train_validation(self, ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        """Split dataset into training (2016-2020) and validation (2021-2023) periods"""
        try:
            train_ds = ds.sel(time=slice('2016-01-01', '2020-12-31'))
            val_ds = ds.sel(time=slice('2021-01-01', '2023-12-31'))
            
            print(f"\nData split:")
            print(f"Training: {train_ds.time[0].values} to {train_ds.time[-1].values} ({len(train_ds.time)} days)")
            print(f"Validation: {val_ds.time[0].values} to {val_ds.time[-1].values} ({len(val_ds.time)} days)")
            
            return train_ds, val_ds
            
        except Exception as e:
            print(f"Error splitting data: {str(e)}")
            return None, None
    
    def get_region_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate grid of 0.25 degree regions for SCS"""
        lats = np.arange(self.lat_bounds[0], self.lat_bounds[1] + self.resolution, self.resolution)
        lons = np.arange(self.lon_bounds[0], self.lon_bounds[1] + self.resolution, self.resolution)
        
        # Round to avoid floating point precision issues
        lats = np.round(lats, 2)
        lons = np.round(lons, 2)
        
        print(f"Region grid: {len(lats)} latitudes x {len(lons)} longitudes = {len(lats) * len(lons)} total regions")
        
        return lats, lons

def load_scs_data(sst_path: str, ssha_path: str, wind_path: str) -> Tuple[Optional[xr.Dataset], Optional[xr.Dataset]]:
    """
    Convenience function to load and split SCS data
    
    Returns:
        tuple: (train_ds, val_ds) or (None, None) if loading fails
    """
    loader = SCSDataLoader(sst_path, ssha_path, wind_path)
    
    # Load and merge data
    merged_ds = loader.merge_datasets()
    if merged_ds is None:
        return None, None
    
    # Split into train/validation
    train_ds, val_ds = loader.split_train_validation(merged_ds)
    
    return train_ds, val_ds