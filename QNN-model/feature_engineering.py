import numpy as np
import xarray as xr
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings

class PhysicsInformedFeatureEngineer:
    """
    Feature engineering for physics-informed marine heatwave prediction
    Focuses on extracting meaningful oceanographic and atmospheric features
    """
    
    def __init__(self, seq_length: int = 30, forecast_horizon: int = 7):
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        # Define physics-informed feature groups
        self.thermal_vars = ['sst', 'anom']
        self.dynamic_vars = ['ugos', 'vgos', 'current_speed', 'current_dir']  
        self.sea_level_vars = ['err_sla']
        self.atmospheric_vars = ['ws', 'uwnd', 'vwind', 'wind_dir']
        self.temporal_vars = ['dayofyear']
        
    def extract_regional_sequences(self, ds: xr.Dataset, lat: float, lon: float) -> List[Dict]:
        """
        Extract physics-informed feature sequences for a specific region
        
        Parameters:
            ds: Input dataset
            lat: Target latitude
            lon: Target longitude
            
        Returns:
            List of feature dictionaries
        """
        try:
            # Find nearest grid point
            lat_idx = np.abs(ds.latitude - lat).argmin()
            lon_idx = np.abs(ds.longitude - lon).argmin()
            
            # Extract regional data
            regional_data = ds.isel(latitude=lat_idx, longitude=lon_idx)
            
            # Create sequences
            sequences = self._create_sequences(regional_data)
            
            return sequences
            
        except Exception as e:
            warnings.warn(f"Failed to extract sequences for lat={lat}, lon={lon}: {str(e)}")
            return []
    
    def _create_sequences(self, regional_data: xr.Dataset) -> List[Dict]:
        """Create feature sequences from regional time series data"""
        sequences = []
        total_time = len(regional_data.time)
        max_start_idx = total_time - self.seq_length - self.forecast_horizon
        
        if max_start_idx <= 0:
            raise ValueError(f"Not enough time steps. Need at least {self.seq_length + self.forecast_horizon}, got {total_time}")
        
        for i in tqdm(range(max_start_idx), desc="Creating sequences", leave=False):
            try:
                sequence = self._extract_single_sequence(regional_data, i)
                if sequence is not None:
                    sequences.append(sequence)
            except Exception as e:
                warnings.warn(f"Skipping sequence {i}: {str(e)}")
                continue
        
        return sequences
    
    def _extract_single_sequence(self, data: xr.Dataset, start_idx: int) -> Optional[Dict]:
        """Extract a single feature sequence"""
        seq_end = start_idx + self.seq_length
        target_idx = seq_end + self.forecast_horizon
        
        try:
            # Extract sequence data
            seq_data = data.isel(time=slice(start_idx, seq_end))
            target_data = data.isel(time=target_idx)
            
            # Initialize feature dictionary
            features = {
                'time': data.time[seq_end].values,
                'target_time': data.time[target_idx].values,
                'lat': float(data.latitude.values),
                'lon': float(data.longitude.values)
            }
            
            # Extract physics-informed features
            self._extract_thermal_features(seq_data, features)
            self._extract_dynamic_features(seq_data, features)
            self._extract_sea_level_features(seq_data, features)
            self._extract_atmospheric_features(seq_data, features)
            self._extract_temporal_features(target_data, features)
            
            # Set target (SST anomaly for MHW prediction)
            if 'anom' in data:
                features['target'] = float(target_data['anom'].values)
            else:
                # Fallback to SST if anomaly not available
                features['target'] = float(target_data['sst'].values)
            
            # Validate sequence (no NaN values)
            if self._validate_sequence(features):
                return features
            else:
                return None
                
        except Exception as e:
            warnings.warn(f"Error extracting sequence at index {start_idx}: {str(e)}")
            return None
    
    def _extract_thermal_features(self, seq_data: xr.Dataset, features: Dict):
        """Extract thermal (temperature) features"""
        for var in self.thermal_vars:
            if var in seq_data:
                values = seq_data[var].values
                if not np.isnan(values).any():
                    features[f'{var}_seq'] = values
                    # Add statistical features
                    features[f'{var}_mean'] = np.mean(values)
                    features[f'{var}_std'] = np.std(values)
                    features[f'{var}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
                    features[f'{var}_max'] = np.max(values)
                    features[f'{var}_min'] = np.min(values)
    
    def _extract_dynamic_features(self, seq_data: xr.Dataset, features: Dict):
        """Extract ocean dynamics features (currents)"""
        for var in self.dynamic_vars:
            if var in seq_data:
                values = seq_data[var].values
                if not np.isnan(values).any():
                    features[f'{var}_seq'] = values
                    features[f'{var}_mean'] = np.mean(values)
                    features[f'{var}_std'] = np.std(values)
                    
        # Compute current divergence/convergence if components available
        if 'ugos' in seq_data and 'vgos' in seq_data:
            ugos_vals = seq_data['ugos'].values
            vgos_vals = seq_data['vgos'].values
            
            if not (np.isnan(ugos_vals).any() or np.isnan(vgos_vals).any()):
                # Current speed variability (proxy for mixing)
                speed_var = np.var(np.sqrt(ugos_vals**2 + vgos_vals**2))
                features['current_variability'] = speed_var
    
    def _extract_sea_level_features(self, seq_data: xr.Dataset, features: Dict):
        """Extract sea level anomaly features"""
        for var in self.sea_level_vars:
            if var in seq_data:
                values = seq_data[var].values
                if not np.isnan(values).any():
                    features[f'{var}_seq'] = values
                    features[f'{var}_mean'] = np.mean(values)
                    features[f'{var}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
    
    def _extract_atmospheric_features(self, seq_data: xr.Dataset, features: Dict):
        """Extract atmospheric forcing features"""
        for var in self.atmospheric_vars:
            if var in seq_data:
                values = seq_data[var].values
                if not np.isnan(values).any():
                    features[f'{var}_seq'] = values
                    features[f'{var}_mean'] = np.mean(values)
                    features[f'{var}_std'] = np.std(values)
        
        # Wind stress proxy (wind speed squared)
        if 'ws' in seq_data:
            ws_vals = seq_data['ws'].values
            if not np.isnan(ws_vals).any():
                features['wind_stress_proxy'] = np.mean(ws_vals**2)
    
    def _extract_temporal_features(self, target_data: xr.Dataset, features: Dict):
        """Extract temporal/seasonal features"""
        if 'dayofyear' in target_data:
            doy = int(target_data['dayofyear'].values)
            features['dayofyear'] = doy
            # Encode seasonality
            features['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
            features['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    
    def _validate_sequence(self, features: Dict) -> bool:
        """Validate that sequence has no NaN values and required features"""
        required_keys = ['target']
        
        # Check required keys exist
        for key in required_keys:
            if key not in features:
                return False
        
        # Check for NaN values in numeric features
        for key, value in features.items():
            if key in ['time', 'target_time', 'lat', 'lon']:
                continue
                
            if isinstance(value, (list, np.ndarray)):
                if np.isnan(value).any():
                    return False
            elif isinstance(value, (int, float)):
                if np.isnan(value):
                    return False
        
        return True
    
    def prepare_feature_matrix(self, sequences: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert sequences to feature matrix for ML training
        
        Returns:
            X: Feature matrix
            y: Target vector  
            feature_names: List of feature names
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Identify all available features (excluding metadata)
        sample_seq = sequences[0]
        feature_keys = []
        
        for key, value in sample_seq.items():
            if key in ['time', 'target_time', 'lat', 'lon', 'target']:
                continue
            feature_keys.append(key)
        
        # Build feature matrix
        X_list = []
        y_list = []
        feature_names = []
        
        for seq in sequences:
            row_features = []
            
            for key in feature_keys:
                if key in seq:
                    value = seq[key]
                    if isinstance(value, (list, np.ndarray)):
                        # Sequence data - flatten
                        row_features.extend(value.flatten())
                        if len(feature_names) < len(row_features):
                            # Add feature names for sequence elements
                            seq_len = len(value.flatten())
                            for i in range(seq_len):
                                feature_names.append(f"{key}_{i}")
                    else:
                        # Scalar feature
                        row_features.append(value)
                        if len(feature_names) < len(row_features):
                            feature_names.append(key)
            
            if len(row_features) > 0:
                X_list.append(row_features)
                y_list.append(seq['target'])
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        return X, y, feature_names

def create_regional_features(ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray, 
                           seq_length: int = 30, forecast_horizon: int = 7) -> Dict[Tuple[float, float], List[Dict]]:
    """
    Create physics-informed features for all regions
    
    Returns:
        Dictionary mapping (lat, lon) -> list of sequences
    """
    engineer = PhysicsInformedFeatureEngineer(seq_length, forecast_horizon)
    regional_features = {}
    
    total_regions = len(lats) * len(lons)
    
    with tqdm(total=total_regions, desc="Processing regions") as pbar:
        for lat in lats:
            for lon in lons:
                try:
                    sequences = engineer.extract_regional_sequences(ds, lat, lon)
                    if sequences:  # Only store if we got valid sequences
                        regional_features[(lat, lon)] = sequences
                except Exception as e:
                    warnings.warn(f"Failed to process region ({lat}, {lon}): {str(e)}")
                finally:
                    pbar.update(1)
    
    print(f"\nSuccessfully processed {len(regional_features)} out of {total_regions} regions")
    return regional_features