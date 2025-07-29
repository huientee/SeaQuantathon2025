import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

class MHWValidationFramework:
    """
    Comprehensive validation framework for marine heatwave predictions
    Evaluates both short-term and long-term accuracy for scientific/policy applications
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_basic_metrics(self, predictions: np.ndarray, observations: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic regression metrics
        
        Args:
            predictions: Predicted values
            observations: Observed values
            
        Returns:
            Dictionary of metrics
        """
        # Remove any NaN pairs
        mask = ~(np.isnan(predictions) | np.isnan(observations))
        pred_clean = predictions[mask]
        obs_clean = observations[mask]
        
        if len(pred_clean) == 0:
            return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'bias': np.nan, 'n_samples': 0}
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(obs_clean, pred_clean)),
            'mae': mean_absolute_error(obs_clean, pred_clean),
            'r2': r2_score(obs_clean, pred_clean),
            'bias': np.mean(pred_clean - obs_clean),
            'n_samples': len(pred_clean)
        }
        
        return metrics
    
    def calculate_mhw_specific_metrics(self, predictions: np.ndarray, observations: np.ndarray,
                                    threshold: float = 1.5) -> Dict[str, float]:
        """
        Calculate marine heatwave specific metrics
        
        Args:
            predictions: Predicted SST anomalies
            observations: Observed SST anomalies
            threshold: MHW threshold in °C
            
        Returns:
            Dictionary of MHW-specific metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(predictions) | np.isnan(observations))
        pred_clean = predictions[mask]
        obs_clean = observations[mask]
        
        if len(pred_clean) == 0:
            return self._empty_mhw_metrics()
        
        # Binary classification: MHW vs no MHW
        pred_mhw = pred_clean > threshold
        obs_mhw = obs_clean > threshold
        
        # Confusion matrix elements
        tp = np.sum((pred_mhw == True) & (obs_mhw == True))  # True Positives
        tn = np.sum((pred_mhw == False) & (obs_mhw == False))  # True Negatives
        fp = np.sum((pred_mhw == True) & (obs_mhw == False))  # False Positives
        fn = np.sum((pred_mhw == False) & (obs_mhw == True))  # False Negatives
        
        # Calculate metrics with zero-division protection
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        # Critical Success Index (CSI) - important for extreme events
        csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        # False Alarm Rate
        far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Probability of Detection
        pod = recall
        
        metrics = {
            'mhw_precision': precision,
            'mhw_recall': recall,
            'mhw_specificity': specificity,
            'mhw_f1_score': f1_score,
            'mhw_accuracy': accuracy,
            'mhw_csi': csi,  # Critical for extreme events
            'mhw_far': far,  # False Alarm Rate
            'mhw_pod': pod,  # Probability of Detection
            'mhw_observed_freq': np.mean(obs_mhw),
            'mhw_predicted_freq': np.mean(pred_mhw),
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        }
        
        return metrics
    
    def _empty_mhw_metrics(self) -> Dict[str, float]:
        """Return empty MHW metrics dictionary"""
        return {
            'mhw_precision': np.nan, 'mhw_recall': np.nan, 'mhw_specificity': np.nan,
            'mhw_f1_score': np.nan, 'mhw_accuracy': np.nan, 'mhw_csi': np.nan,
            'mhw_far': np.nan, 'mhw_pod': np.nan, 'mhw_observed_freq': np.nan,
            'mhw_predicted_freq': np.nan, 'confusion_matrix': {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        }
    
    def evaluate_temporal_performance(self, 
                                    predictions: Dict[str, np.ndarray],
                                    observations: Dict[str, np.ndarray],
                                    forecast_horizons: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
        """
        Evaluate model performance across different forecast horizons
        
        Args:
            predictions: Dictionary with forecast horizon as key
            observations: Dictionary with forecast horizon as key
            forecast_horizons: List of forecast horizons to evaluate
            
        Returns:
            DataFrame with performance metrics by forecast horizon
        """
        results = []
        
        for horizon in forecast_horizons:
            if str(horizon) in predictions and str(horizon) in observations:
                pred = predictions[str(horizon)]
                obs = observations[str(horizon)]
                
                # Basic metrics
                basic_metrics = self.calculate_basic_metrics(pred, obs)
                
                # MHW-specific metrics
                mhw_metrics = self.calculate_mhw_specific_metrics(pred, obs)
                
                # Combine metrics
                all_metrics = {**basic_metrics, **mhw_metrics}
                all_metrics['forecast_horizon'] = horizon
                
                results.append(all_metrics)
        
        return pd.DataFrame(results)
    
    def evaluate_spatial_performance(self, 
                                   regional_predictions: Dict[Tuple[float, float], float],
                                   regional_observations: Dict[Tuple[float, float], float],
                                   region_type: str = 'all') -> Dict[str, float]:
        """
        Evaluate model performance across different spatial regions
        
        Args:
            regional_predictions: Dictionary mapping (lat, lon) -> prediction
            regional_observations: Dictionary mapping (lat, lon) -> observation
            region_type: Type of regional analysis ('all', 'coastal', 'deep_water')
            
        Returns:
            Dictionary of spatial performance metrics
        """
        # Extract common coordinates
        common_coords = set(regional_predictions.keys()) & set(regional_observations.keys())
        
        if not common_coords:
            warnings.warn("No common coordinates found between predictions and observations")
            return {}
        
        # Filter regions if needed
        if region_type != 'all':
            common_coords = self._filter_regions_by_type(common_coords, region_type)
        
        # Extract predictions and observations
        predictions = np.array([regional_predictions[coord] for coord in common_coords])
        observations = np.array([regional_observations[coord] for coord in common_coords])
        
        # Calculate metrics
        basic_metrics = self.calculate_basic_metrics(predictions, observations)
        mhw_metrics = self.calculate_mhw_specific_metrics(predictions, observations)
        
        # Add spatial information
        spatial_metrics = {
            **basic_metrics,
            **mhw_metrics,
            'region_type': region_type,
            'n_regions': len(common_coords)
        }
        
        return spatial_metrics
    
    def _filter_regions_by_type(self, coords: set, region_type: str) -> set:
        """Filter coordinates by region type"""
        if region_type == 'coastal':
            # Simple heuristic: closer to land boundaries
            filtered_coords = set()
            for lat, lon in coords:
                # Check if near coastal boundaries (simplified)
                if (lat < 10 and lon < 115) or (lat > 20) or (lon > 120):
                    filtered_coords.add((lat, lon))
            return filtered_coords
        elif region_type == 'deep_water':
            # Central SCS regions
            filtered_coords = set()
            for lat, lon in coords:
                if 10 <= lat <= 20 and 115 <= lon <= 120:
                    filtered_coords.add((lat, lon))
            return filtered_coords
        else:
            return coords
    
    def create_validation_report(self, 
                               temporal_results: pd.DataFrame,
                               spatial_results: Dict[str, Dict[str, float]],
                               model_name: str = "Quantum-Enhanced MHW Model") -> str:
        """
        Create comprehensive validation report
        
        Args:
            temporal_results: DataFrame with temporal validation results
            spatial_results: Dictionary with spatial validation results
            model_name: Name of the model being validated
            
        Returns:
            Formatted validation report as string
        """
        report = []
        report.append(f"="*60)
        report.append(f"MARINE HEATWAVE MODEL VALIDATION REPORT")
        report.append(f"Model: {model_name}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"="*60)
        
        # Temporal Performance Summary
        report.append(f"\n1. TEMPORAL PERFORMANCE ANALYSIS")
        report.append(f"-"*40)
        
        if not temporal_results.empty:
            # Short-term performance (1-7 days)
            short_term = temporal_results[temporal_results['forecast_horizon'] <= 7]
            if not short_term.empty:
                report.append(f"\nShort-term Forecast Performance (1-7 days):")
                report.append(f"  Average RMSE: {short_term['rmse'].mean():.3f} °C")
                report.append(f"  Average MAE:  {short_term['mae'].mean():.3f} °C")
                report.append(f"  Average R²:   {short_term['r2'].mean():.3f}")
                report.append(f"  MHW Detection F1: {short_term['mhw_f1_score'].mean():.3f}")
                report.append(f"  Critical Success Index: {short_term['mhw_csi'].mean():.3f}")
            
            # Long-term performance (14-30 days)
            long_term = temporal_results[temporal_results['forecast_horizon'] >= 14]
            if not long_term.empty:
                report.append(f"\nLong-term Forecast Performance (14-30 days):")
                report.append(f"  Average RMSE: {long_term['rmse'].mean():.3f} °C")
                report.append(f"  Average MAE:  {long_term['mae'].mean():.3f} °C")
                report.append(f"  Average R²:   {long_term['r2'].mean():.3f}")
                report.append(f"  MHW Detection F1: {long_term['mhw_f1_score'].mean():.3f}")
                report.append(f"  Critical Success Index: {long_term['mhw_csi'].mean():.3f}")
        
        # Spatial Performance Summary
        report.append(f"\n2. SPATIAL PERFORMANCE ANALYSIS")
        report.append(f"-"*40)
        
        for region_name, metrics in spatial_results.items():
            if metrics:
                report.append(f"\n{region_name.title()} Regions:")
                report.append(f"  RMSE: {metrics.get('rmse', 'N/A'):.3f} °C")
                report.append(f"  MAE:  {metrics.get('mae', 'N/A'):.3f} °C")
                report.append(f"  R²:   {metrics.get('r2', 'N/A'):.3f}")
                report.append(f"  MHW F1: {metrics.get('mhw_f1_score', 'N/A'):.3f}")
                report.append(f"  Regions: {metrics.get('n_regions', 'N/A')}")
        
        # Scientific Relevance Assessment
        report.append(f"\n3. SCIENTIFIC & POLICY RELEVANCE")
        report.append(f"-"*40)
        
        # Extract key metrics for assessment
        if not temporal_results.empty:
            short_f1 = temporal_results[temporal_results['forecast_horizon'] <= 7]['mhw_f1_score'].mean()
            short_csi = temporal_results[temporal_results['forecast_horizon'] <= 7]['mhw_csi'].mean()
            
            report.append(f"\nMarine Heatwave Detection Capability:")
            if short_f1 >= 0.7:
                report.append(f"  ✓ EXCELLENT: F1-Score = {short_f1:.3f} (≥0.7)")
            elif short_f1 >= 0.5:
                report.append(f"  ○ GOOD: F1-Score = {short_f1:.3f} (0.5-0.7)")
            else:
                report.append(f"  ✗ NEEDS IMPROVEMENT: F1-Score = {short_f1:.3f} (<0.5)")
            
            report.append(f"\nEarly Warning Utility:")
            if short_csi >= 0.5:
                report.append(f"  ✓ HIGH UTILITY: CSI = {short_csi:.3f} (≥0.5)")
            elif short_csi >= 0.3:
                report.append(f"  ○ MODERATE UTILITY: CSI = {short_csi:.3f} (0.3-0.5)")
            else:
                report.append(f"  ✗ LIMITED UTILITY: CSI = {short_csi:.3f} (<0.3)")
        
        # Recommendations
        report.append(f"\n4. RECOMMENDATIONS")
        report.append(f"-"*40)
        
        if not temporal_results.empty:
            avg_rmse = temporal_results['rmse'].mean()
            avg_f1 = temporal_results['mhw_f1_score'].mean()
            
            if avg_rmse < 1.0 and avg_f1 > 0.6:
                report.append(f"✓ Model shows strong performance for operational use")
                report.append(f"✓ Suitable for both scientific research and policy applications")
            elif avg_rmse < 1.5 and avg_f1 > 0.4:
                report.append(f"○ Model shows moderate performance")
                report.append(f"○ Recommend additional validation before operational deployment")
            else:
                report.append(f"✗ Model requires further development")
                report.append(f"✗ Consider ensemble methods or additional training data")
        
        report.append(f"\n" + "="*60)
        
        return "\n".join(report)
    
    def plot_validation_summary(self, temporal_results: pd.DataFrame, 
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create validation summary plots
        
        Args:
            temporal_results: DataFrame with temporal validation results
            save_path: Optional path to save figure
            
        Returns:
            matplotlib Figure object
        """
        if temporal_results.empty:
            print("No temporal results to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: RMSE vs Forecast Horizon
        axes[0,0].plot(temporal_results['forecast_horizon'], temporal_results['rmse'], 'o-')
        axes[0,0].set_xlabel('Forecast Horizon (days)')
        axes[0,0].set_ylabel('RMSE (°C)')
        axes[0,0].set_title('Prediction Error vs Forecast Horizon')
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: R² vs Forecast Horizon
        axes[0,1].plot(temporal_results['forecast_horizon'], temporal_results['r2'], 'o-', color='green')
        axes[0,1].set_xlabel('Forecast Horizon (days)')
        axes[0,1].set_ylabel('R² Score')
        axes[0,1].set_title('Correlation vs Forecast Horizon')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: MHW Detection Performance
        axes[1,0].plot(temporal_results['forecast_horizon'], temporal_results['mhw_f1_score'], 'o-', color='red', label='F1-Score')
        axes[1,0].plot(temporal_results['forecast_horizon'], temporal_results['mhw_csi'], 's-', color='orange', label='CSI')
        axes[1,0].set_xlabel('Forecast Horizon (days)')
        axes[1,0].set_ylabel('Score')
        axes[1,0].set_title('MHW Detection Performance')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Detection vs False Alarm
        axes[1,1].scatter(temporal_results['mhw_far'], temporal_results['mhw_pod'], 
                         c=temporal_results['forecast_horizon'], cmap='viridis')
        axes[1,1].set_xlabel('False Alarm Rate')
        axes[1,1].set_ylabel('Probability of Detection')
        axes[1,1].set_title('Detection vs False Alarm Trade-off')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add colorbar for the scatter plot
        cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
        cbar.set_label('Forecast Horizon (days)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig