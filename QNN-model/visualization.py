import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from typing import Dict, Tuple, List, Optional
import torch
from datetime import datetime, timedelta
import warnings

class SCSHeatmapGenerator:
    """
    Generate heatmaps for South China Sea marine heatwave predictions
    """
    
    def __init__(self, lat_bounds: Tuple[float, float] = (-3.35, 25.69),
                 lon_bounds: Tuple[float, float] = (102.11, 122.28)):
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        
        # Set up the map projection
        self.projection = ccrs.PlateCarree()
        
    def create_prediction_grid(self, regional_predictions: Dict[Tuple[float, float], float],
                             resolution: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a regular grid from regional predictions
        
        Args:
            regional_predictions: Dictionary mapping (lat, lon) -> prediction value
            resolution: Grid resolution in degrees
            
        Returns:
            lats, lons, predictions as 2D grids
        """
        # Create coordinate arrays
        lats = np.arange(self.lat_bounds[0], self.lat_bounds[1] + resolution, resolution)
        lons = np.arange(self.lon_bounds[0], self.lon_bounds[1] + resolution, resolution)
        
        # Initialize prediction grid with NaN
        pred_grid = np.full((len(lats), len(lons)), np.nan)
        
        # Fill in predictions
        for (lat, lon), prediction in regional_predictions.items():
            try:
                lat_idx = np.argmin(np.abs(lats - lat))
                lon_idx = np.argmin(np.abs(lons - lon))
                pred_grid[lat_idx, lon_idx] = prediction
            except Exception as e:
                warnings.warn(f"Could not place prediction for ({lat}, {lon}): {str(e)}")
        
        # Create meshgrids
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        return lat_grid, lon_grid, pred_grid
    
    def plot_sst_anomaly_heatmap(self, regional_predictions: Dict[Tuple[float, float], float],
                                title: str = "SST Anomaly Prediction",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create SST anomaly heatmap
        
        Args:
            regional_predictions: Dictionary of regional predictions
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        lat_grid, lon_grid, pred_grid = self.create_prediction_grid(regional_predictions)
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=self.projection)
        
        # Plot the heatmap
        im = ax.pcolormesh(lon_grid, lat_grid, pred_grid, 
                          cmap='RdBu_r', 
                          vmin=-3, vmax=3,
                          transform=self.projection,
                          shading='auto')
        
        # Add geographical features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        
        # Set extent
        ax.set_extent([self.lon_bounds[0], self.lon_bounds[1], 
                      self.lat_bounds[0], self.lat_bounds[1]], 
                     crs=self.projection)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('SST Anomaly (°C)', fontsize=12)
        
        # Add title
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add annotation with prediction date
        ax.text(0.02, 0.98, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_mhw_probability_heatmap(self, regional_predictions: Dict[Tuple[float, float], float],
                                   threshold: float = 2.0,
                                   title: str = "Marine Heatwave Probability",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create marine heatwave probability heatmap
        
        Args:
            regional_predictions: Dictionary of regional predictions
            threshold: Temperature anomaly threshold for MHW (°C)
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        # Convert predictions to probabilities (simple threshold approach)
        mhw_probabilities = {}
        for coords, pred in regional_predictions.items():
            # Convert anomaly to probability using sigmoid function
            probability = 1 / (1 + np.exp(-(pred - threshold)))
            mhw_probabilities[coords] = probability
        
        lat_grid, lon_grid, prob_grid = self.create_prediction_grid(mhw_probabilities)
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=self.projection)
        
        # Plot the heatmap
        im = ax.pcolormesh(lon_grid, lat_grid, prob_grid, 
                          cmap='YlOrRd', 
                          vmin=0, vmax=1,
                          transform=self.projection,
                          shading='auto')
        
        # Add geographical features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        
        # Set extent
        ax.set_extent([self.lon_bounds[0], self.lon_bounds[1], 
                      self.lat_bounds[0], self.lat_bounds[1]], 
                     crs=self.projection)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label('MHW Probability', fontsize=12)
        
        # Add title
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add threshold information
        ax.text(0.02, 0.02, f'Threshold: {threshold}°C anomaly',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multi_panel_forecast(self, 
                                current_sst: Dict[Tuple[float, float], float],
                                predicted_anomaly: Dict[Tuple[float, float], float],
                                forecast_date: str,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create multi-panel forecast visualization
        
        Args:
            current_sst: Current SST values
            predicted_anomaly: Predicted SST anomalies
            forecast_date: Date of forecast
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(18, 6))
        
        # Panel 1: Current SST
        ax1 = fig.add_subplot(1, 3, 1, projection=self.projection)
        lat_grid, lon_grid, sst_grid = self.create_prediction_grid(current_sst)
        
        im1 = ax1.pcolormesh(lon_grid, lat_grid, sst_grid, 
                            cmap='inferno', vmin=20, vmax=35,
                            transform=self.projection, shading='auto')
        
        self._add_map_features(ax1)
        ax1.set_title('Current SST', fontsize=12)
        
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6)
        cbar1.set_label('SST (°C)', fontsize=10)
        
        # Panel 2: Predicted Anomaly
        ax2 = fig.add_subplot(1, 3, 2, projection=self.projection)
        lat_grid, lon_grid, anom_grid = self.create_prediction_grid(predicted_anomaly)
        
        im2 = ax2.pcolormesh(lon_grid, lat_grid, anom_grid, 
                            cmap='RdBu_r', vmin=-3, vmax=3,
                            transform=self.projection, shading='auto')
        
        self._add_map_features(ax2)
        ax2.set_title(f'Predicted Anomaly\n({forecast_date})', fontsize=12)
        
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
        cbar2.set_label('Anomaly (°C)', fontsize=10)
        
        # Panel 3: MHW Risk
        ax3 = fig.add_subplot(1, 3, 3, projection=self.projection)
        
        # Calculate MHW risk
        mhw_risk = {}
        for coords, anom in predicted_anomaly.items():
            risk = 1 / (1 + np.exp(-(anom - 1.5)))  # Risk based on 1.5°C threshold
            mhw_risk[coords] = risk
        
        lat_grid, lon_grid, risk_grid = self.create_prediction_grid(mhw_risk)
        
        im3 = ax3.pcolormesh(lon_grid, lat_grid, risk_grid, 
                            cmap='YlOrRd', vmin=0, vmax=1,
                            transform=self.projection, shading='auto')
        
        self._add_map_features(ax3)
        ax3.set_title('MHW Risk', fontsize=12)
        
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.6)
        cbar3.set_label('Risk Level', fontsize=10)
        
        plt.suptitle(f'South China Sea Marine Heatwave Forecast - {forecast_date}', 
                    fontsize=16, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _add_map_features(self, ax):
        """Add standard map features to axis"""
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        
        ax.set_extent([self.lon_bounds[0], self.lon_bounds[1], 
                      self.lat_bounds[0], self.lat_bounds[1]], 
                     crs=self.projection)
        
        gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
    
    def plot_validation_comparison(self, 
                                 predictions: Dict[Tuple[float, float], float],
                                 observations: Dict[Tuple[float, float], float],
                                 title: str = "Prediction vs Observation",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison plot between predictions and observations
        
        Args:
            predictions: Predicted values
            observations: Observed values
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Prediction map
        ax1 = fig.add_subplot(1, 3, 1, projection=self.projection)
        lat_grid, lon_grid, pred_grid = self.create_prediction_grid(predictions)
        
        im1 = ax1.pcolormesh(lon_grid, lat_grid, pred_grid, 
                            cmap='RdBu_r', vmin=-3, vmax=3,
                            transform=self.projection, shading='auto')
        
        self._add_map_features(ax1)
        ax1.set_title('Predictions', fontsize=12)
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.6)
        cbar1.set_label('Anomaly (°C)', fontsize=10)
        
        # Observation map
        ax2 = fig.add_subplot(1, 3, 2, projection=self.projection)
        lat_grid, lon_grid, obs_grid = self.create_prediction_grid(observations)
        
        im2 = ax2.pcolormesh(lon_grid, lat_grid, obs_grid, 
                            cmap='RdBu_r', vmin=-3, vmax=3,
                            transform=self.projection, shading='auto')
        
        self._add_map_features(ax2)
        ax2.set_title('Observations', fontsize=12)
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.6)
        cbar2.set_label('Anomaly (°C)', fontsize=10)
        
        # Difference map
        ax3 = fig.add_subplot(1, 3, 3, projection=self.projection)
        
        # Calculate differences
        differences = {}
        for coords in predictions.keys():
            if coords in observations:
                diff = predictions[coords] - observations[coords]
                differences[coords] = diff
        
        lat_grid, lon_grid, diff_grid = self.create_prediction_grid(differences)
        
        im3 = ax3.pcolormesh(lon_grid, lat_grid, diff_grid, 
                            cmap='RdBu_r', vmin=-2, vmax=2,
                            transform=self.projection, shading='auto')
        
        self._add_map_features(ax3)
        ax3.set_title('Prediction - Observation', fontsize=12)
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.6)
        cbar3.set_label('Difference (°C)', fontsize=10)
        
        plt.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig