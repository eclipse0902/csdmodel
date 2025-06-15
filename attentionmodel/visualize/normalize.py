#!/usr/bin/env python3
# csd_normalization_heatmap_comparison.py - Heatmap-focused CSD Normalization Comparison

import os
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
import pandas as pd
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')

# Set font for better display
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

class CSDHeatmapComparison:
    """Heatmap-focused CSD Normalization Comparison Visualizer"""
    
    def __init__(self):
        # Frequency band definitions
        self.freq_bands = {
            'Delta': [0, 1, 2],
            'Theta': [3, 4, 5], 
            'Alpha': [6, 7, 8, 9],
            'Beta': [10, 11, 12, 13],
            'Gamma': [14]
        }
        
        # Enhanced color schemes
        self.color_schemes = {
            'magnitude': 'viridis',
            'log_magnitude': 'plasma',
            'difference': 'RdBu_r',
            'ratio': 'coolwarm',
            'phase': 'hsv'
        }
        
        # Electrode positions (19 electrodes)
        self.electrode_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4', 
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2'
        ]
        
        # Analysis parameters
        self.sample_size = 5  # Number of samples to analyze
        
    def load_file(self, filepath):
        """Safe file loading with error handling"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                return data
        except Exception as e:
            print(f"? Failed to load {Path(filepath).name}: {str(e)[:50]}...")
            return None
    
    def extract_csd(self, data):
        """Extract and validate CSD data"""
        if isinstance(data, dict) and 'csd' in data:
            csd = data['csd']
        elif isinstance(data, np.ndarray):
            csd = data
        else:
            return None
        
        # Validate shape and finite values
        if not isinstance(csd, np.ndarray) or csd.shape != (15, 19, 19, 2):
            return None
        
        # Check if data has finite values
        if not np.isfinite(csd).any():
            return None
            
        return csd
    
    def compute_magnitude_phase(self, csd):
        """Compute magnitude and phase from complex CSD"""
        # Handle both 3D and 4D arrays
        if csd.ndim == 4:
            # Full CSD array (freq, elec, elec, real/imag)
            complex_csd = csd[:, :, :, 0] + 1j * csd[:, :, :, 1]
        elif csd.ndim == 3:
            # Already averaged or single frequency (elec, elec, real/imag)
            complex_csd = csd[:, :, 0] + 1j * csd[:, :, 1]
        else:
            raise ValueError(f"Unexpected CSD dimensions: {csd.shape}")
        
        magnitude = np.abs(complex_csd)
        phase = np.angle(complex_csd)
        return magnitude, phase
    
    def safe_log_transform(self, data, epsilon=1e-12):
        """Safe logarithmic transformation"""
        # Handle negative values and zeros
        abs_data = np.abs(data)
        abs_data = np.where(abs_data < epsilon, epsilon, abs_data)
        return np.log10(abs_data)
    
    def compute_relative_change(self, before, after, method='ratio'):
        """Compute relative change between before and after"""
        if method == 'ratio':
            # Avoid division by zero
            ratio = np.where(np.abs(before) > 1e-12, 
                           after / before, 
                           np.ones_like(after))
            return ratio
        elif method == 'log_ratio':
            # Log ratio for better visualization of multiplicative changes
            return self.safe_log_transform(after) - self.safe_log_transform(before)
        elif method == 'percent_change':
            # Percentage change
            return np.where(np.abs(before) > 1e-12,
                          100 * (after - before) / before,
                          np.zeros_like(after))
    
    def create_frequency_heatmaps(self, before_files, after_files, output_dir):
        """Create comprehensive frequency-wise heatmaps"""
        
        print("?? Creating frequency-wise heatmaps...")
        
        # Load sample data
        before_samples = []
        after_samples = []
        
        for i in range(min(self.sample_size, len(before_files), len(after_files))):
            before_data = self.load_file(before_files[i])
            after_data = self.load_file(after_files[i])
            
            if before_data and after_data:
                before_csd = self.extract_csd(before_data)
                after_csd = self.extract_csd(after_data)
                
                if before_csd is not None and after_csd is not None:
                    before_samples.append(before_csd)
                    after_samples.append(after_csd)
        
        if not before_samples:
            print("? No valid samples found")
            return
        
        # Average across samples
        before_avg = np.mean(before_samples, axis=0)
        after_avg = np.mean(after_samples, axis=0)
        
        # Compute magnitudes
        before_mag, before_phase = self.compute_magnitude_phase(before_avg)
        after_mag, after_phase = self.compute_magnitude_phase(after_avg)
        
        # Create comprehensive heatmap comparison
        fig = plt.figure(figsize=(24, 20))
        gs = GridSpec(5, 6, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('CSD Normalization: Frequency-wise Heatmap Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # For each frequency band
        band_names = list(self.freq_bands.keys())
        
        for band_idx, (band_name, freq_indices) in enumerate(self.freq_bands.items()):
            row = band_idx
            
            # Average magnitude for this band
            before_band_mag = np.mean(before_mag[freq_indices], axis=0)
            after_band_mag = np.mean(after_mag[freq_indices], axis=0)
            
            # 1. Before normalization (log scale)
            ax1 = fig.add_subplot(gs[row, 0])
            before_log = self.safe_log_transform(before_band_mag)
            im1 = ax1.imshow(before_log, cmap='viridis', aspect='equal')
            ax1.set_title(f'{band_name}\nBefore (log10)', fontweight='bold')
            self._add_electrode_labels(ax1)
            self._add_colorbar(fig, ax1, im1)
            
            # 2. After normalization (log scale)  
            ax2 = fig.add_subplot(gs[row, 1])
            after_log = self.safe_log_transform(after_band_mag)
            im2 = ax2.imshow(after_log, cmap='viridis', aspect='equal')
            ax2.set_title(f'{band_name}\nAfter (log10)', fontweight='bold')
            self._add_electrode_labels(ax2)
            self._add_colorbar(fig, ax2, im2)
            
            # 3. Log ratio (after/before)
            ax3 = fig.add_subplot(gs[row, 2])
            log_ratio = self.compute_relative_change(before_band_mag, after_band_mag, 'log_ratio')
            # Center around 0 for better visualization
            vmax = np.nanpercentile(np.abs(log_ratio), 95)
            im3 = ax3.imshow(log_ratio, cmap='RdBu_r', aspect='equal', 
                           vmin=-vmax, vmax=vmax)
            ax3.set_title(f'{band_name}\nLog Ratio\n(After/Before)', fontweight='bold')
            self._add_electrode_labels(ax3)
            self._add_colorbar(fig, ax3, im3)
            
            # 4. Absolute magnitude ratio
            ax4 = fig.add_subplot(gs[row, 3])
            ratio = self.compute_relative_change(before_band_mag, after_band_mag, 'ratio')
            # Use log scale for ratio visualization
            ratio_safe = np.where(ratio > 0, ratio, 1e-6)
            im4 = ax4.imshow(ratio_safe, cmap='coolwarm', aspect='equal', 
                           norm=LogNorm(vmin=np.nanpercentile(ratio_safe, 5),
                                      vmax=np.nanpercentile(ratio_safe, 95)))
            ax4.set_title(f'{band_name}\nMagnitude Ratio\n(After/Before)', fontweight='bold')
            self._add_electrode_labels(ax4)
            self._add_colorbar(fig, ax4, im4)
            
            # 5. Statistical summary
            ax5 = fig.add_subplot(gs[row, 4])
            self._create_stats_heatmap(ax5, before_band_mag, after_band_mag, band_name)
            
            # 6. Connectivity strength comparison
            ax6 = fig.add_subplot(gs[row, 5])
            self._create_connectivity_comparison(ax6, before_band_mag, after_band_mag, band_name)
        
        plt.savefig(f"{output_dir}/frequency_heatmaps_comprehensive.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("? Frequency heatmaps completed")
    
    def create_sample_comparison_heatmaps(self, before_files, after_files, output_dir):
        """Create individual sample comparison heatmaps"""
        
        print("?? Creating individual sample heatmaps...")
        
        n_samples = min(3, len(before_files), len(after_files))
        
        fig = plt.figure(figsize=(24, 8*n_samples))
        gs = GridSpec(n_samples, 8, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Individual Sample Comparisons: CSD Heatmaps', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        for sample_idx in range(n_samples):
            before_data = self.load_file(before_files[sample_idx])
            after_data = self.load_file(after_files[sample_idx])
            
            if not (before_data and after_data):
                continue
                
            before_csd = self.extract_csd(before_data)
            after_csd = self.extract_csd(after_data)
            
            if before_csd is None or after_csd is None:
                continue
            
            # Focus on Alpha band for detailed comparison
            alpha_indices = self.freq_bands['Alpha']
            before_alpha = np.mean(before_csd[alpha_indices], axis=0)
            after_alpha = np.mean(after_csd[alpha_indices], axis=0)
            
            before_mag, before_phase = self.compute_magnitude_phase(before_alpha)
            after_mag, after_phase = self.compute_magnitude_phase(after_alpha)
            
            row = sample_idx
            
            # 1. Before magnitude (log)
            ax = fig.add_subplot(gs[row, 0])
            before_log = self.safe_log_transform(before_mag)
            im = ax.imshow(before_log, cmap='viridis', aspect='equal')
            ax.set_title(f'Sample {sample_idx+1}\nBefore Mag (log)', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im, size="5%")
            
            # 2. After magnitude (log)
            ax = fig.add_subplot(gs[row, 1])
            after_log = self.safe_log_transform(after_mag)
            im = ax.imshow(after_log, cmap='viridis', aspect='equal')
            ax.set_title(f'Sample {sample_idx+1}\nAfter Mag (log)', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im, size="5%")
            
            # 3. Magnitude difference (log scale)
            ax = fig.add_subplot(gs[row, 2])
            log_diff = after_log - before_log
            vmax = np.nanpercentile(np.abs(log_diff), 95)
            im = ax.imshow(log_diff, cmap='RdBu_r', aspect='equal',
                         vmin=-vmax, vmax=vmax)
            ax.set_title(f'Sample {sample_idx+1}\nLog Difference', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im, size="5%")
            
            # 4. Before phase
            ax = fig.add_subplot(gs[row, 3])
            im = ax.imshow(before_phase, cmap='hsv', aspect='equal',
                         vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'Sample {sample_idx+1}\nBefore Phase', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im, size="5%")
            
            # 5. After phase  
            ax = fig.add_subplot(gs[row, 4])
            im = ax.imshow(after_phase, cmap='hsv', aspect='equal',
                         vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'Sample {sample_idx+1}\nAfter Phase', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im, size="5%")
            
            # 6. Phase difference
            ax = fig.add_subplot(gs[row, 5])
            phase_diff = np.angle(np.exp(1j * (after_phase - before_phase)))
            im = ax.imshow(phase_diff, cmap='RdBu_r', aspect='equal',
                         vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'Sample {sample_idx+1}\nPhase Difference', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im, size="5%")
            
            # 7. Coherence before
            ax = fig.add_subplot(gs[row, 6])
            coherence_before = self._compute_coherence_matrix(before_alpha)
            im = ax.imshow(coherence_before, cmap='hot', aspect='equal', vmin=0, vmax=1)
            ax.set_title(f'Sample {sample_idx+1}\nCoherence Before', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im, size="5%")
            
            # 8. Coherence after
            ax = fig.add_subplot(gs[row, 7])
            coherence_after = self._compute_coherence_matrix(after_alpha)
            im = ax.imshow(coherence_after, cmap='hot', aspect='equal', vmin=0, vmax=1)
            ax.set_title(f'Sample {sample_idx+1}\nCoherence After', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im, size="5%")
        
        plt.savefig(f"{output_dir}/sample_heatmaps_detailed.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("? Sample comparison heatmaps completed")
    
    def create_statistical_summary_heatmaps(self, before_files, after_files, output_dir):
        """Create statistical summary heatmaps"""
        
        print("?? Creating statistical summary heatmaps...")
        
        # Collect statistics across multiple samples
        n_samples = min(10, len(before_files), len(after_files))
        
        stats_before = {'mean': [], 'std': [], 'max': [], 'min': []}
        stats_after = {'mean': [], 'std': [], 'max': [], 'min': []}
        
        for i in range(n_samples):
            before_data = self.load_file(before_files[i])
            after_data = self.load_file(after_files[i])
            
            if before_data and after_data:
                before_csd = self.extract_csd(before_data)
                after_csd = self.extract_csd(after_data)
                
                if before_csd is not None and after_csd is not None:
                    # Compute magnitudes
                    before_mag, _ = self.compute_magnitude_phase(before_csd)
                    after_mag, _ = self.compute_magnitude_phase(after_csd)
                    
                    # Average across frequencies for each electrode pair
                    before_avg = np.mean(before_mag, axis=0)
                    after_avg = np.mean(after_mag, axis=0)
                    
                    stats_before['mean'].append(before_avg)
                    stats_before['std'].append(np.std(before_mag, axis=0))
                    stats_before['max'].append(np.max(before_mag, axis=0))
                    stats_before['min'].append(np.min(before_mag, axis=0))
                    
                    stats_after['mean'].append(after_avg)
                    stats_after['std'].append(np.std(after_mag, axis=0))
                    stats_after['max'].append(np.max(after_mag, axis=0))
                    stats_after['min'].append(np.min(after_mag, axis=0))
        
        if not stats_before['mean']:
            print("? No valid samples for statistical analysis")
            return
        
        # Compute ensemble statistics
        ensemble_before = {}
        ensemble_after = {}
        
        for stat_name in ['mean', 'std', 'max', 'min']:
            ensemble_before[stat_name] = np.mean(stats_before[stat_name], axis=0)
            ensemble_after[stat_name] = np.mean(stats_after[stat_name], axis=0)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Statistical Summary: Ensemble Analysis Across Samples', 
                    fontsize=16, fontweight='bold')
        
        stat_names = ['mean', 'std', 'max', 'min']
        
        for col, stat_name in enumerate(stat_names):
            # Before statistics (log scale)
            ax = axes[0, col]
            before_stat_log = self.safe_log_transform(ensemble_before[stat_name])
            im = ax.imshow(before_stat_log, cmap='viridis', aspect='equal')
            ax.set_title(f'Before: {stat_name.title()} (log)', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im)
            
            # Relative change
            ax = axes[1, col]
            ratio = self.compute_relative_change(ensemble_before[stat_name], 
                                               ensemble_after[stat_name], 'log_ratio')
            vmax = np.nanpercentile(np.abs(ratio), 95)
            im = ax.imshow(ratio, cmap='RdBu_r', aspect='equal',
                         vmin=-vmax, vmax=vmax)
            ax.set_title(f'Change: {stat_name.title()}\n(log after/before)', fontweight='bold')
            self._add_electrode_labels(ax, fontsize=8)
            self._add_colorbar(fig, ax, im)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/statistical_summary_heatmaps.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("? Statistical summary heatmaps completed")
    
    def _add_electrode_labels(self, ax, fontsize=10):
        """Add electrode labels to heatmap"""
        if len(self.electrode_names) == 19:
            ax.set_xticks(range(19))
            ax.set_yticks(range(19))
            ax.set_xticklabels(self.electrode_names, rotation=45, fontsize=fontsize)
            ax.set_yticklabels(self.electrode_names, fontsize=fontsize)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    
    def _add_colorbar(self, fig, ax, im, size="3%"):
        """Add colorbar to axes"""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=size, pad=0.05)
        fig.colorbar(im, cax=cax)
    
    def _create_stats_heatmap(self, ax, before, after, band_name):
        """Create statistics comparison heatmap"""
        # Compute various statistics
        stats = np.zeros((4, 4))  # 4x4 grid for different statistics
        
        # Row 0: Basic statistics before
        stats[0, 0] = np.nanmean(self.safe_log_transform(before))
        stats[0, 1] = np.nanstd(self.safe_log_transform(before))
        stats[0, 2] = np.nanmax(self.safe_log_transform(before))
        stats[0, 3] = np.nanmin(self.safe_log_transform(before))
        
        # Row 1: Basic statistics after
        stats[1, 0] = np.nanmean(self.safe_log_transform(after))
        stats[1, 1] = np.nanstd(self.safe_log_transform(after))
        stats[1, 2] = np.nanmax(self.safe_log_transform(after))
        stats[1, 3] = np.nanmin(self.safe_log_transform(after))
        
        # Row 2: Ratios
        stats[2, 0] = stats[1, 0] - stats[0, 0]  # Mean difference (log scale)
        stats[2, 1] = stats[1, 1] / (stats[0, 1] + 1e-12)  # Std ratio
        stats[2, 2] = stats[1, 2] - stats[0, 2]  # Max difference
        stats[2, 3] = stats[1, 3] - stats[0, 3]  # Min difference
        
        # Row 3: Additional metrics
        stats[3, 0] = np.nanpercentile(after/np.maximum(before, 1e-12), 50)  # Median ratio
        stats[3, 1] = np.nanpercentile(after/np.maximum(before, 1e-12), 90)  # 90th percentile
        stats[3, 2] = np.sum(np.isfinite(before)) / before.size  # Finite ratio before
        stats[3, 3] = np.sum(np.isfinite(after)) / after.size    # Finite ratio after
        
        im = ax.imshow(stats, cmap='RdBu_r', aspect='equal')
        ax.set_title(f'{band_name}\nStatistics Grid', fontweight='bold')
        
        # Add labels
        row_labels = ['Before', 'After', 'Change', 'Metrics']
        col_labels = ['Mean', 'Std', 'Max', 'Min']
        
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticklabels(row_labels, fontsize=8)
        
        # Add values as text
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{stats[i,j]:.2f}', ha='center', va='center',
                       color='white' if abs(stats[i,j]) > np.nanmax(np.abs(stats))/2 else 'black',
                       fontsize=8, fontweight='bold')
        
        self._add_colorbar(plt.gcf(), ax, im, size="3%")
    
    def _create_connectivity_comparison(self, ax, before, after, band_name):
        """Create connectivity strength comparison"""
        # Compute connectivity strength (sum of connections for each electrode)
        before_strength = np.sum(before, axis=1) + np.sum(before, axis=0)
        after_strength = np.sum(after, axis=1) + np.sum(after, axis=0)
        
        # Create comparison heatmap
        comparison = np.vstack([before_strength, after_strength, 
                              after_strength - before_strength,
                              after_strength / np.maximum(before_strength, 1e-12)])
        
        im = ax.imshow(comparison, cmap='viridis', aspect='auto')
        ax.set_title(f'{band_name}\nConnectivity\nStrength', fontweight='bold')
        
        # Labels
        if len(self.electrode_names) == 19:
            ax.set_xticks(range(19))
            ax.set_xticklabels(self.electrode_names, rotation=90, fontsize=6)
        
        ax.set_yticks(range(4))
        ax.set_yticklabels(['Before', 'After', 'Diff', 'Ratio'], fontsize=8)
        
        self._add_colorbar(plt.gcf(), ax, im, size="3%")
    
    def _compute_coherence_matrix(self, csd_matrix):
        """Compute coherence matrix from CSD"""
        # Handle both 3D and 2D complex matrices
        if csd_matrix.ndim == 3:
            # (elec, elec, real/imag)
            magnitude = np.abs(csd_matrix[:, :, 0] + 1j * csd_matrix[:, :, 1])
        elif csd_matrix.ndim == 2:
            # Already complex or magnitude only
            if np.iscomplexobj(csd_matrix):
                magnitude = np.abs(csd_matrix)
            else:
                magnitude = csd_matrix
        else:
            raise ValueError(f"Unexpected CSD matrix dimensions: {csd_matrix.shape}")
        
        # Normalize by diagonal elements to get coherence-like measure
        diag_vals = np.diag(magnitude)
        coherence = magnitude / np.sqrt(np.outer(diag_vals, diag_vals) + 1e-12)
        
        return np.clip(coherence, 0, 1)
    
    def run_heatmap_analysis(self, before_pattern, after_pattern, output_dir="./heatmap_comparison"):
        """Run comprehensive heatmap analysis"""
        
        print("?? Starting Heatmap-focused CSD Normalization Analysis")
        print("=" * 60)
        
        # Find files
        before_files = glob.glob(before_pattern)
        after_files = glob.glob(after_pattern)
        
        if not before_files:
            print(f"? Before files not found: {before_pattern}")
            return
        
        if not after_files:
            print(f"? After files not found: {after_pattern}")
            return
        
        print(f"?? Before files: {len(before_files)}")
        print(f"?? After files: {len(after_files)}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Match files by name
        before_names = {Path(f).name: f for f in before_files}
        after_names = {Path(f).name: f for f in after_files}
        common_names = set(before_names.keys()) & set(after_names.keys())
        
        if not common_names:
            print("?? No matching files. Using order-based matching.")
            matched_before = before_files[:self.sample_size]
            matched_after = after_files[:self.sample_size]
        else:
            print(f"? {len(common_names)} files matched")
            matched_before = [before_names[name] for name in sorted(list(common_names))[:self.sample_size]]
            matched_after = [after_names[name] for name in sorted(list(common_names))[:self.sample_size]]
        
        try:
            # 1. Frequency-wise comprehensive heatmaps
            self.create_frequency_heatmaps(matched_before, matched_after, output_dir)
            
            # 2. Individual sample comparisons
            self.create_sample_comparison_heatmaps(matched_before, matched_after, output_dir)
            
            # 3. Statistical summary heatmaps
            self.create_statistical_summary_heatmaps(matched_before, matched_after, output_dir)
            
            print(f"\n?? Heatmap analysis completed!")
            print(f"?? Results saved to: {output_dir}")
            print("\nGenerated files:")
            print("  ?? frequency_heatmaps_comprehensive.png - Detailed frequency analysis")
            print("  ?? sample_heatmaps_detailed.png - Individual sample comparisons")
            print("  ?? statistical_summary_heatmaps.png - Statistical ensemble analysis")
            
        except Exception as e:
            print(f"? Error during analysis: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main execution function"""
    
    print("?? CSD Normalization Heatmap Comparison Tool")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = CSDHeatmapComparison()
    
    # Data path configuration
    base_path = "/remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf"
    
    # Before normalization (original)
    before_pattern = os.path.join(base_path, "var_scaled/train/*.pkl")
    
    # After normalization
    after_pattern = os.path.join(base_path, "normalized_fq/train/*.pkl")
    
    # Check paths and try alternatives
    if not glob.glob(before_pattern):
        print(f"?? Original files not found: {before_pattern}")
        
        # Alternative paths
        alternative_before = [
            "/remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/1000sample/train/*.pkl",
            "./original_data/train/*.pkl",
            "./data/train/*.pkl"
        ]
        
        for alt in alternative_before:
            if glob.glob(alt):
                before_pattern = alt
                print(f"? Using alternative path: {alt}")
                break
        else:
            print("? Original data not found.")
            print("\nUsage:")
            print("python csd_normalization_heatmap_comparison.py")
            print("\nPlease check data paths and modify paths in the script.")
            return
    
    if not glob.glob(after_pattern):
        print(f"?? Normalized files not found: {after_pattern}")
        
        # Alternative paths
        alternative_after = [
            "./normalized_clean/train/*.pkl",
            "./normalized_gpu_clean/train/*.pkl",
            "./normalized_fq/train/*.pkl",
            "./output/train/*.pkl"
        ]
        
        for alt in alternative_after:
            if glob.glob(alt):
                after_pattern = alt
                print(f"? Using alternative path: {alt}")
                break
        else:
            print("? Normalized data not found.")
            print("Please run normalization first.")
            return
    
    # Output directory
    output_dir = "./heatmap_comparison_results"
    
    print(f"\n?? Before normalization: {before_pattern}")
    print(f"?? After normalization: {after_pattern}")
    print(f"?? Results will be saved to: {output_dir}")
    
    # Check sample count
    before_count = len(glob.glob(before_pattern))
    after_count = len(glob.glob(after_pattern))
    print(f"?? Available samples - Before: {before_count}, After: {after_count}")
    
    # Run heatmap analysis
    try:
        visualizer.run_heatmap_analysis(
            before_pattern=before_pattern,
            after_pattern=after_pattern,
            output_dir=output_dir
        )
        
        print(f"\n?? Heatmap analysis completed! Check results at: {output_dir}")
        print("\n?? Key Features of This Analysis:")
        print("  ? Log-scale visualization for better small value representation")
        print("  ? Relative change analysis (ratios and differences)")
        print("  ? Frequency band-specific comparisons")
        print("  ? Phase and magnitude separated analysis")
        print("  ? Coherence matrix comparisons")
        print("  ? Statistical ensemble analysis across multiple samples")
        print("\n?? Generated Files:")
        print("  ?? frequency_heatmaps_comprehensive.png")
        print("     - Complete frequency band analysis with log scaling")
        print("     - Before/after magnitude and phase comparisons")
        print("     - Ratio and difference visualizations")
        print("  ?? sample_heatmaps_detailed.png")
        print("     - Individual sample detailed comparisons")
        print("     - Alpha band focus with magnitude/phase analysis")
        print("     - Coherence matrix before/after")
        print("  ?? statistical_summary_heatmaps.png")
        print("     - Ensemble statistics across multiple samples")
        print("     - Mean, std, max, min comparisons")
        print("     - Relative change analysis")
        
        print(f"\n?? Tips for Interpretation:")
        print("  ? Red areas in difference plots = increased values")
        print("  ? Blue areas in difference plots = decreased values")
        print("  ? Log scale helps visualize small magnitude changes")
        print("  ? Ratio plots show multiplicative changes")
        print("  ? Coherence plots show functional connectivity changes")
        
    except KeyboardInterrupt:
        print("\n?? Interrupted by user")
    except Exception as e:
        print(f"\n? Error occurred: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n?? Troubleshooting:")
        print("  1. Check if data files exist and are readable")
        print("  2. Ensure sufficient memory for large heatmaps")
        print("  3. Verify file format is correct (15, 19, 19, 2) shape")
        print("  4. Check for NaN/infinite values in data")


if __name__ == "__main__":
    main()