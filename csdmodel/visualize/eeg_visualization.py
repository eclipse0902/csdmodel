"""
EEG Connectivity Analysis - Enhanced Reconstruction Quality Visualizer

15°³ ÁÖÆÄ¼ö ´ë¿ª ±â¹Ý ¸ðµ¨ÀÇ ¸¶½ºÅ· º¹¿ø Ç°ÁúÀ» ´Ù°¢µµ·Î ½Ã°¢È­
Æ¯È÷ ¸¶½ºÅ·µÈ ¿µ¿ªÀÇ º¹¿ø Ç°Áú¿¡ ÁýÁßÇÏ¿© ºÐ¼® (Magnitude + Phase + Masking overlay)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Tuple, Optional, List
import json
from datetime import datetime

class EEGReconstructionVisualizer:
    """
    Enhanced EEG º¹¿ø Ç°Áú ½Ã°¢È­ Å¬·¡½º
    
    ÇÙ½É ±â´É:
    1. ¸¶½ºÅ·µÈ ¿µ¿ªÀÇ º¹¿ø ¼º´É¿¡ ÁýÁß
    2. Magnitude + Phase µ¿½Ã ½Ã°¢È­
    3. ¸¶½ºÅ· ¿µ¿ªÀ» ½Ã°¢ÀûÀ¸·Î ±¸ºÐ (Èò»ö °æ°è)
    4. º¹¿ø Ç°Áú »ó¼¼ Åë°è ºÐ¼®
    """
    
    def __init__(self, model=None, config=None, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        # 15°³ ÁÖÆÄ¼ö ¼³Á¤
        self.num_frequencies = 15
        self.num_electrodes = 19
        self.frequency_hz = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        
        # ÁÖÆÄ¼ö ´ë¿ª Á¤ÀÇ
        self.frequency_bands = {
            'delta': [0, 1],              # 1, 3Hz
            'theta': [2, 3],              # 5, 7Hz  
            'alpha': [4, 5, 6],           # 9, 11, 13Hz
            'beta': [7, 8, 9, 10, 11, 12, 13, 14]  # 15-29Hz
        }
        
        # Àü±Ø ÀÌ¸§
        self.electrode_names = [
            'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
        ]
        
        # ³ú ¿µ¿ª Á¤ÀÇ
        self.brain_regions = {
            'frontal': [0, 1, 2, 3, 10, 11, 16],
            'central': [4, 5, 17],
            'parietal': [6, 7, 18],
            'temporal': [12, 13, 14, 15],
            'occipital': [8, 9]
        }
        
        print(f"?? Enhanced EEG Reconstruction Visualizer initialized")
        print(f"   Focus: Masked region reconstruction (Magnitude + Phase)")
        print(f"   Features: Masking overlay + Statistical analysis")
    
    def compute_reconstruction_metrics(self, original: torch.Tensor, 
                                     reconstructed: torch.Tensor, 
                                     mask: torch.Tensor) -> Dict:
        """º¹¿ø Ç°Áú ÁöÇ¥ °è»ê (¸¶½ºÅ· ¿µ¿ª ÁýÁß)"""
        
        # Convert to numpy if needed
        if isinstance(original, torch.Tensor):
            original = original.cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # ¸¶½ºÅ· Á¤º¸
        mask_binary = mask[..., 0] == 0  # ¸¶½ºÅ·µÈ À§Ä¡ (True = masked)
        unmask_binary = mask[..., 0] == 1  # ºñ¸¶½ºÅ·µÈ À§Ä¡
        
        # Magnitude and Phase °è»ê
        orig_mag = np.sqrt(original[..., 0]**2 + original[..., 1]**2 + 1e-8)
        recon_mag = np.sqrt(reconstructed[..., 0]**2 + reconstructed[..., 1]**2 + 1e-8)
        orig_phase = np.arctan2(original[..., 1], original[..., 0] + 1e-8)
        recon_phase = np.arctan2(reconstructed[..., 1], reconstructed[..., 0] + 1e-8)
        
        # ÀüÃ¼ ¼º´É
        overall_mag_error = np.mean(np.abs(orig_mag - recon_mag) / (orig_mag + 1e-8)) * 100
        phase_diff = orig_phase - recon_phase
        overall_phase_error = np.sqrt(np.mean(phase_diff**2)) * 180.0 / np.pi
        
        if np.std(orig_mag) > 1e-6 and np.std(recon_mag) > 1e-6:
            overall_correlation = np.corrcoef(orig_mag.flatten(), recon_mag.flatten())[0, 1]
        else:
            overall_correlation = 0.0
        
        # ¸¶½ºÅ·µÈ ¿µ¿ª ¼º´É (ÇÙ½É!)
        if mask_binary.any():
            masked_orig_mag = orig_mag[mask_binary]
            masked_recon_mag = recon_mag[mask_binary]
            masked_orig_phase = orig_phase[mask_binary]
            masked_recon_phase = recon_phase[mask_binary]
            
            masked_mag_error = np.mean(np.abs(masked_orig_mag - masked_recon_mag) / (masked_orig_mag + 1e-8)) * 100
            masked_phase_diff = masked_orig_phase - masked_recon_phase
            masked_phase_error = np.sqrt(np.mean(masked_phase_diff**2)) * 180.0 / np.pi
            
            if len(masked_orig_mag) > 1:
                masked_correlation = np.corrcoef(masked_orig_mag, masked_recon_mag)[0, 1]
            else:
                masked_correlation = 0.0
        else:
            masked_mag_error = 0.0
            masked_phase_error = 0.0
            masked_correlation = 0.0
        
        # ºñ¸¶½ºÅ·µÈ ¿µ¿ª ¼º´É (ºñ±³¿ë)
        if unmask_binary.any():
            unmask_orig_mag = orig_mag[unmask_binary]
            unmask_recon_mag = recon_mag[unmask_binary]
            unmask_mag_error = np.mean(np.abs(unmask_orig_mag - unmask_recon_mag) / (unmask_orig_mag + 1e-8)) * 100
        else:
            unmask_mag_error = 0.0
        
        # SNR °è»ê
        signal_power = np.mean(original**2)
        noise_power = np.mean((reconstructed - original)**2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-8))
        
        return {
            # ÀüÃ¼ ¼º´É
            'magnitude_relative_error': overall_mag_error,
            'phase_error_degrees': overall_phase_error,
            'correlation': overall_correlation,
            'snr_db': snr_db,
            
            # ?? ¸¶½ºÅ·µÈ ¿µ¿ª ¼º´É (ÇÙ½É!)
            'masked_magnitude_error': masked_mag_error,
            'masked_phase_error_degrees': masked_phase_error,
            'masked_correlation': masked_correlation,
            
            # ºñ±³¿ë
            'unmasked_magnitude_error': unmask_mag_error,
            
            # ¸¶½ºÅ· Á¤º¸
            'mask_ratio': np.mean(mask_binary.astype(float)),
            'masked_positions': np.sum(mask_binary),
            'total_positions': mask_binary.size
        }
    
    def visualize_magnitude_phase_overview(self, original: torch.Tensor, 
                                         reconstructed: torch.Tensor, 
                                         mask: torch.Tensor,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """Magnitude + Phase + Masking ÀüÃ¼ °³¿ä"""
        
        # Convert to numpy
        if isinstance(original, torch.Tensor):
            original = original.cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Compute magnitude and phase
        orig_mag = np.sqrt(original[..., 0]**2 + original[..., 1]**2)
        recon_mag = np.sqrt(reconstructed[..., 0]**2 + reconstructed[..., 1]**2)
        orig_phase = np.arctan2(original[..., 1], original[..., 0])
        recon_phase = np.arctan2(reconstructed[..., 1], reconstructed[..., 0])
        
        # Average across frequencies
        orig_mag_avg = np.mean(orig_mag, axis=0)
        recon_mag_avg = np.mean(recon_mag, axis=0)
        orig_phase_avg = np.mean(orig_phase, axis=0)
        recon_phase_avg = np.mean(recon_phase, axis=0)
        
        # Masking info
        mask_binary = mask[..., 0] == 0
        mask_avg = np.mean(mask_binary.astype(float), axis=0)
        
        # Errors
        mag_error = np.abs(orig_mag_avg - recon_mag_avg)
        phase_error = np.abs(orig_phase_avg - recon_phase_avg)
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('?? Magnitude + Phase + Masking Overview', fontsize=16, fontweight='bold')
        
        # Row 1: Magnitude Analysis
        mag_vmax = max(orig_mag_avg.max(), recon_mag_avg.max())
        
        # Original Magnitude
        im1 = axes[0, 0].imshow(orig_mag_avg, cmap='viridis', vmin=0, vmax=mag_vmax)
        axes[0, 0].set_title('Original\nMagnitude')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        # Reconstructed Magnitude
        im2 = axes[0, 1].imshow(recon_mag_avg, cmap='viridis', vmin=0, vmax=mag_vmax)
        axes[0, 1].set_title('Reconstructed\nMagnitude')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # Magnitude Error + Masking overlay
        im3 = axes[0, 2].imshow(mag_error, cmap='Reds', alpha=0.8)
        axes[0, 2].contour(mask_avg, levels=[0.5], colors='white', linewidths=2)
        axes[0, 2].set_title('Magnitude Error\n(White: Masked)')
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        
        # Mask Pattern
        im4 = axes[0, 3].imshow(mask_avg, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0, 3].set_title('Mask Pattern\n(Red=Masked)')
        plt.colorbar(im4, ax=axes[0, 3], shrink=0.8)
        
        # Row 2: Phase Analysis
        phase_vmin, phase_vmax = -np.pi, np.pi
        
        # Original Phase
        im5 = axes[1, 0].imshow(orig_phase_avg, cmap='hsv', vmin=phase_vmin, vmax=phase_vmax)
        axes[1, 0].set_title('Original\nPhase')
        plt.colorbar(im5, ax=axes[1, 0], shrink=0.8)
        
        # Reconstructed Phase
        im6 = axes[1, 1].imshow(recon_phase_avg, cmap='hsv', vmin=phase_vmin, vmax=phase_vmax)
        axes[1, 1].set_title('Reconstructed\nPhase')
        plt.colorbar(im6, ax=axes[1, 1], shrink=0.8)
        
        # Phase Error + Masking overlay
        im7 = axes[1, 2].imshow(phase_error, cmap='plasma', alpha=0.8)
        axes[1, 2].contour(mask_avg, levels=[0.5], colors='white', linewidths=2)
        axes[1, 2].set_title('Phase Error\n(White: Masked)')
        plt.colorbar(im7, ax=axes[1, 2], shrink=0.8)
        
        # ¸¶½ºÅ·µÈ ¿µ¿ª¸¸ÀÇ º¹¿ø Ç°Áú
        quality_score = 1 - (mag_error / (orig_mag_avg + 1e-8))
        masked_quality = np.where(mask_avg > 0.5, quality_score, np.nan)
        im8 = axes[1, 3].imshow(masked_quality, cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, 3].set_title('Masked Region\nQuality (Green=Good)')
        plt.colorbar(im8, ax=axes[1, 3], shrink=0.8)
        
        # Add electrode labels to key plots
        for ax in [axes[0, 0], axes[0, 2], axes[1, 0], axes[1, 2]]:
            self._add_electrode_labels(ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"?? Magnitude+Phase overview saved: {save_path}")
        
        return fig
    
    def visualize_frequency_bands_masked(self, original: torch.Tensor, 
                                       reconstructed: torch.Tensor, 
                                       mask: torch.Tensor,
                                       save_path: Optional[str] = None) -> plt.Figure:
        """ÁÖÆÄ¼ö ´ë¿ªº° ¸¶½ºÅ· º¹¿ø ºÐ¼®"""
        
        # Convert to numpy
        if isinstance(original, torch.Tensor):
            original = original.cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        mask_binary = mask[..., 0] == 0
        
        # Create figure
        fig, axes = plt.subplots(4, 3, figsize=(12, 16))
        fig.suptitle('?? Frequency Band Analysis (Magnitude + Phase + Masking)', fontsize=16, fontweight='bold')
        
        for idx, (band_name, freq_indices) in enumerate(self.frequency_bands.items()):
            row = idx
            
            # Extract frequency band
            band_orig = original[freq_indices]
            band_recon = reconstructed[freq_indices]
            band_mask = mask_binary[freq_indices]
            
            # Compute magnitude and phase
            orig_mag = np.sqrt(band_orig[..., 0]**2 + band_orig[..., 1]**2)
            recon_mag = np.sqrt(band_recon[..., 0]**2 + band_recon[..., 1]**2)
            orig_phase = np.arctan2(band_orig[..., 1], band_orig[..., 0])
            recon_phase = np.arctan2(band_recon[..., 1], band_recon[..., 0])
            
            # Average across frequencies in this band
            orig_mag_avg = np.mean(orig_mag, axis=0)
            recon_mag_avg = np.mean(recon_mag, axis=0)
            orig_phase_avg = np.mean(orig_phase, axis=0)
            recon_phase_avg = np.mean(recon_phase, axis=0)
            mask_avg = np.mean(band_mask.astype(float), axis=0)
            
            # Magnitude difference
            mag_diff = np.abs(orig_mag_avg - recon_mag_avg)
            masked_mag_diff = np.where(mask_avg > 0.5, mag_diff, np.nan)
            
            # Phase difference
            phase_diff = np.abs(orig_phase_avg - recon_phase_avg)
            masked_phase_diff = np.where(mask_avg > 0.5, phase_diff, np.nan)
            
            # Plot magnitude error in masked regions
            im1 = axes[row, 0].imshow(masked_mag_diff, cmap='Reds')
            axes[row, 0].set_title(f'{band_name.capitalize()}\nMagnitude Error (Masked)')
            if band_name == 'delta':
                axes[row, 0].set_ylabel('Magnitude\nError', fontsize=12)
            
            # Plot phase error in masked regions
            im2 = axes[row, 1].imshow(masked_phase_diff, cmap='plasma')
            axes[row, 1].set_title(f'{band_name.capitalize()}\nPhase Error (Masked)')
            if band_name == 'delta':
                axes[row, 1].set_ylabel('Phase\nError', fontsize=12)
            
            # Plot mask pattern for this band
            im3 = axes[row, 2].imshow(mask_avg, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[row, 2].set_title(f'{band_name.capitalize()}\nMask Pattern')
            if band_name == 'delta':
                axes[row, 2].set_ylabel('Mask\nPattern', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"?? Frequency band masked analysis saved: {save_path}")
        
        return fig
    
    def visualize_masked_region_statistics(self, original: torch.Tensor, 
                                         reconstructed: torch.Tensor, 
                                         mask: torch.Tensor,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """¸¶½ºÅ·µÈ ¿µ¿ª Åë°è ºÐ¼®"""
        
        # Convert to numpy
        if isinstance(original, torch.Tensor):
            original = original.cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Compute metrics
        orig_mag = np.sqrt(original[..., 0]**2 + original[..., 1]**2)
        recon_mag = np.sqrt(reconstructed[..., 0]**2 + reconstructed[..., 1]**2)
        orig_phase = np.arctan2(original[..., 1], original[..., 0])
        recon_phase = np.arctan2(reconstructed[..., 1], reconstructed[..., 0])
        
        # Masking info
        mask_binary = mask[..., 0] == 0
        
        # Extract masked and unmasked data
        masked_orig_mag = orig_mag[mask_binary]
        masked_recon_mag = recon_mag[mask_binary]
        masked_orig_phase = orig_phase[mask_binary]
        masked_recon_phase = recon_phase[mask_binary]
        
        unmasked_orig_mag = orig_mag[~mask_binary]
        unmasked_recon_mag = recon_mag[~mask_binary]
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('?? Masked vs Unmasked Statistical Analysis', fontsize=16, fontweight='bold')
        
        # 1. Magnitude scatter: Masked vs Unmasked
        axes[0, 0].scatter(masked_orig_mag, masked_recon_mag, alpha=0.6, s=2, 
                          color='red', label=f'Masked (n={len(masked_orig_mag)})')
        axes[0, 0].scatter(unmasked_orig_mag, unmasked_recon_mag, alpha=0.3, s=1, 
                          color='blue', label=f'Unmasked (n={len(unmasked_orig_mag)})')
        max_val = max(np.max(masked_orig_mag), np.max(masked_recon_mag))
        axes[0, 0].plot([0, max_val], [0, max_val], 'k--', alpha=0.8)
        axes[0, 0].set_xlabel('Original Magnitude')
        axes[0, 0].set_ylabel('Reconstructed Magnitude')
        axes[0, 0].set_title('Magnitude Correlation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Error distribution
        masked_mag_error = np.abs(masked_orig_mag - masked_recon_mag)
        unmasked_mag_error = np.abs(unmasked_orig_mag - unmasked_recon_mag)
        
        axes[0, 1].hist(masked_mag_error, bins=30, alpha=0.7, color='red', 
                       label='Masked', density=True)
        axes[0, 1].hist(unmasked_mag_error, bins=30, alpha=0.7, color='blue', 
                       label='Unmasked', density=True)
        axes[0, 1].set_xlabel('Magnitude Error')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Phase error (circular)
        masked_phase_diff = masked_orig_phase - masked_recon_phase
        masked_phase_diff_wrapped = np.arctan2(np.sin(masked_phase_diff), np.cos(masked_phase_diff))
        
        axes[0, 2].hist(masked_phase_diff_wrapped * 180 / np.pi, bins=30, alpha=0.7, 
                       color='green', density=True)
        axes[0, 2].set_xlabel('Phase Error (degrees)')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Phase Error (Masked)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Box plot comparison
        error_data = [masked_mag_error, unmasked_mag_error]
        box_plot = axes[0, 3].boxplot(error_data, labels=['Masked', 'Unmasked'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('red')
        box_plot['boxes'][0].set_alpha(0.7)
        box_plot['boxes'][1].set_facecolor('blue')
        box_plot['boxes'][1].set_alpha(0.7)
        axes[0, 3].set_ylabel('Magnitude Error')
        axes[0, 3].set_title('Error Comparison')
        axes[0, 3].grid(True, alpha=0.3)
        
        # 5. Frequency-wise masked error
        freq_errors = []
        freq_labels = []
        colors = []
        
        for freq_idx in range(self.num_frequencies):
            freq_mask = mask_binary[freq_idx]
            if freq_mask.any():
                freq_orig = orig_mag[freq_idx][freq_mask]
                freq_recon = recon_mag[freq_idx][freq_mask]
                freq_error = np.mean(np.abs(freq_orig - freq_recon))
                freq_errors.append(freq_error)
                freq_labels.append(f'{self.frequency_hz[freq_idx]}Hz')
                colors.append(self._get_freq_color(self.frequency_hz[freq_idx]))
        
        if freq_errors:
            axes[1, 0].bar(range(len(freq_errors)), freq_errors, color=colors)
            axes[1, 0].set_xlabel('Frequency')
            axes[1, 0].set_ylabel('Mean Error (Masked)')
            axes[1, 0].set_title('Masked Error by Frequency')
            axes[1, 0].set_xticks(range(len(freq_labels)))
            axes[1, 0].set_xticklabels(freq_labels, rotation=45)
        
        # 6. Performance statistics text
        masked_stats = {
            'mean_error': np.mean(masked_mag_error),
            'std_error': np.std(masked_mag_error),
            'correlation': np.corrcoef(masked_orig_mag, masked_recon_mag)[0, 1] if len(masked_orig_mag) > 1 else 0
        }
        
        unmasked_stats = {
            'mean_error': np.mean(unmasked_mag_error),
            'std_error': np.std(unmasked_mag_error),
            'correlation': np.corrcoef(unmasked_orig_mag, unmasked_recon_mag)[0, 1] if len(unmasked_orig_mag) > 1 else 0
        }
        
        stats_text = f"""?? MASKED REGION STATS:
Mean Error: {masked_stats['mean_error']:.4f}
Std Error: {masked_stats['std_error']:.4f}
Correlation: {masked_stats['correlation']:.3f}
Count: {len(masked_orig_mag):,}

?? UNMASKED REFERENCE:
Mean Error: {unmasked_stats['mean_error']:.4f}
Std Error: {unmasked_stats['std_error']:.4f}
Correlation: {unmasked_stats['correlation']:.3f}
Count: {len(unmasked_orig_mag):,}

?? PERFORMANCE RATIO:
Error Ratio: {masked_stats['mean_error']/unmasked_stats['mean_error']:.2f}x
Quality: {'Good' if masked_stats['mean_error']/unmasked_stats['mean_error'] < 2.0 else 'Poor'}"""
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=9, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Statistics Summary')
        
        # 7. Alpha band focus (¸¶½ºÅ·µÈ alpha¸¸)
        alpha_indices = self.frequency_bands['alpha']
        alpha_orig = orig_mag[alpha_indices]
        alpha_recon = recon_mag[alpha_indices]
        alpha_mask = mask_binary[alpha_indices]
        
        if alpha_mask.any():
            alpha_masked_orig = alpha_orig[alpha_mask]
            alpha_masked_recon = alpha_recon[alpha_mask]
            
            axes[1, 2].scatter(alpha_masked_orig, alpha_masked_recon, alpha=0.7, s=3, color='red')
            max_alpha = max(alpha_masked_orig.max(), alpha_masked_recon.max())
            axes[1, 2].plot([0, max_alpha], [0, max_alpha], 'k--', alpha=0.8)
            axes[1, 2].set_xlabel('Original Alpha Magnitude')
            axes[1, 2].set_ylabel('Reconstructed Alpha Magnitude')
            axes[1, 2].set_title(f'Alpha Band (Masked Only)')
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'No Masked\nAlpha Data', 
                           transform=axes[1, 2].transAxes, ha='center', va='center')
            axes[1, 2].set_title('Alpha Band')
        
        # 8. Brain regions summary
        region_errors = []
        region_names = []
        
        avg_orig_mag = np.mean(orig_mag, axis=0)
        avg_recon_mag = np.mean(recon_mag, axis=0)
        avg_mask = np.mean(mask_binary.astype(float), axis=0)
        
        for region_name, electrode_indices in self.brain_regions.items():
            region_orig = avg_orig_mag[np.ix_(electrode_indices, electrode_indices)]
            region_recon = avg_recon_mag[np.ix_(electrode_indices, electrode_indices)]
            region_mask = avg_mask[np.ix_(electrode_indices, electrode_indices)]
            
            if region_mask.any():
                masked_region_orig = region_orig[region_mask > 0.5]
                masked_region_recon = region_recon[region_mask > 0.5]
                if len(masked_region_orig) > 0:
                    region_error = np.mean(np.abs(masked_region_orig - masked_region_recon))
                    region_errors.append(region_error)
                    region_names.append(region_name[:4])  # Ãà¾à
        
        if region_errors:
            axes[1, 3].bar(range(len(region_errors)), region_errors, 
                          color=['#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c'][:len(region_errors)])
            axes[1, 3].set_xlabel('Brain Regions')
            axes[1, 3].set_ylabel('Mean Error (Masked)')
            axes[1, 3].set_title('Masked Error by Region')
            axes[1, 3].set_xticklabels(region_names, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"?? Masked region statistics saved: {save_path}")
        
        return fig
    
    def plot_training_progress(self, log_file_path: str, 
                             save_path: Optional[str] = None) -> plt.Figure:
        """ÈÆ·Ã °úÁ¤ ½Ã°¢È­"""
        
        # Load training log
        training_data = []
        try:
            with open(log_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            training_data.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            print(f"? Log file not found: {log_file_path}")
            return None
        
        if len(training_data) < 2:
            print(f"? Insufficient training data in log file")
            return None
        
        # Extract metrics
        epochs = [entry['epoch'] for entry in training_data]
        train_losses = [entry['train_metrics']['total_loss'] for entry in training_data]
        phase_errors = [entry['train_metrics']['phase_error_degrees'] for entry in training_data]
        alpha_mag_errors = [entry['train_metrics']['alpha_magnitude_error'] * 100 for entry in training_data]
        correlations = [entry['train_metrics']['correlation'] for entry in training_data]
        learning_rates = [entry['train_metrics']['learning_rate'] for entry in training_data]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('?? Training Progress Analysis', fontsize=16, fontweight='bold')
        
        # Training loss
        axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Phase error
        axes[0, 1].plot(epochs, phase_errors, 'r-', linewidth=2, label='Phase Error')
        axes[0, 1].axhline(y=25, color='r', linestyle='--', alpha=0.7, label='Target (<25¡Æ)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Phase Error (degrees)')
        axes[0, 1].set_title('Phase Error Progress')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Alpha magnitude error
        axes[0, 2].plot(epochs, alpha_mag_errors, 'g-', linewidth=2, label='Alpha Mag Error')
        axes[0, 2].axhline(y=8, color='r', linestyle='--', alpha=0.7, label='Target (<8%)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Alpha Magnitude Error (%)')
        axes[0, 2].set_title('Alpha Magnitude Error')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
        
        # Correlation
        axes[1, 0].plot(epochs, correlations, 'purple', linewidth=2, label='Correlation')
        axes[1, 0].axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Target (>0.8)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Correlation')
        axes[1, 0].set_title('Reconstruction Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Learning rate
        axes[1, 1].plot(epochs, learning_rates, 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Performance summary
        final_metrics = training_data[-1]['train_metrics']
        best_loss = min(train_losses)
        best_phase = min(phase_errors)
        best_alpha_mag = min(alpha_mag_errors)
        best_correlation = max(correlations)
        
        summary_text = f"""Training Summary:

Final Metrics:
? Loss: {final_metrics['total_loss']:.6f}
? Phase Error: {final_metrics['phase_error_degrees']:.1f}¡Æ
? Alpha Mag Error: {final_metrics['alpha_magnitude_error']*100:.1f}%
? Correlation: {final_metrics['correlation']:.3f}

Best Achieved:
? Best Loss: {best_loss:.6f}
? Best Phase: {best_phase:.1f}¡Æ
? Best Alpha Mag: {best_alpha_mag:.1f}%
? Best Correlation: {best_correlation:.3f}

Training Info:
? Total Epochs: {len(epochs)}
? Final LR: {learning_rates[-1]:.2e}
"""
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Training Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"?? Training progress saved: {save_path}")
        
        return fig
    
    def generate_full_report(self, original: torch.Tensor, 
                           reconstructed: torch.Tensor, 
                           mask: torch.Tensor,
                           output_dir: str = "./enhanced_visualization_report",
                           log_file_path: Optional[str] = None) -> Dict[str, str]:
        """Á¾ÇÕ ½Ã°¢È­ ¸®Æ÷Æ® »ý¼º (¸¶½ºÅ· ÁýÁß)"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_files = {}
        
        print(f"?? Generating masked reconstruction analysis report...")
        print(f"   Focus: Magnitude + Phase + Masking overlay")
        print(f"   Output directory: {output_dir}")
        
        # 1. Magnitude + Phase + Masking °³¿ä
        fig1 = self.visualize_magnitude_phase_overview(
            original, reconstructed, mask,
            save_path=os.path.join(output_dir, f"01_mag_phase_overview_{timestamp}.png")
        )
        report_files['overview'] = f"01_mag_phase_overview_{timestamp}.png"
        plt.close(fig1)
        
        # 2. ÁÖÆÄ¼ö ´ë¿ªº° ¸¶½ºÅ· ºÐ¼®
        fig2 = self.visualize_frequency_bands_masked(
            original, reconstructed, mask,
            save_path=os.path.join(output_dir, f"02_frequency_masked_{timestamp}.png")
        )
        report_files['frequency_masked'] = f"02_frequency_masked_{timestamp}.png"
        plt.close(fig2)
        
        # 3. ¸¶½ºÅ· ¿µ¿ª Åë°è ºÐ¼®
        fig3 = self.visualize_masked_region_statistics(
            original, reconstructed, mask,
            save_path=os.path.join(output_dir, f"03_masked_statistics_{timestamp}.png")
        )
        report_files['masked_statistics'] = f"03_masked_statistics_{timestamp}.png"
        plt.close(fig3)
        
        # 4. ÈÆ·Ã °úÁ¤ (¼±ÅÃÀû)
        if log_file_path and os.path.exists(log_file_path):
            fig4 = self.plot_training_progress(
                log_file_path,
                save_path=os.path.join(output_dir, f"04_training_progress_{timestamp}.png")
            )
            if fig4:
                report_files['training_progress'] = f"04_training_progress_{timestamp}.png"
                plt.close(fig4)
        
        # 5. ¸ÞÆ®¸¯ °è»ê ¹× ÀúÀå
        metrics = self.compute_reconstruction_metrics(original, reconstructed, mask)
        
        # JSON serializable·Î º¯È¯
        serializable_metrics = self._convert_to_serializable(metrics)
        
        metrics_file = os.path.join(output_dir, f"metrics_summary_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'metrics': serializable_metrics,
                'configuration': {
                    'num_frequencies': self.num_frequencies,
                    'num_electrodes': self.num_electrodes,
                    'frequency_bands': self.frequency_bands,
                    'brain_regions': {k: len(v) for k, v in self.brain_regions.items()}
                }
            }, f, indent=2)
        
        report_files['metrics'] = f"metrics_summary_{timestamp}.json"
        
        # 6. HTML ¸®Æ÷Æ® »ý¼º
        html_report = self._generate_html_report(report_files, serializable_metrics, timestamp)
        html_file = os.path.join(output_dir, f"masked_reconstruction_report_{timestamp}.html")
        with open(html_file, 'w') as f:
            f.write(html_report)
        
        report_files['html_report'] = f"masked_reconstruction_report_{timestamp}.html"
        
        print(f"? Enhanced report generation completed!")
        print(f"   Generated files: {len(report_files)}")
        print(f"   Main report: {html_file}")
        print(f"   ?? Focus: Masked region reconstruction quality")
        
        return report_files
    
    def _generate_html_report(self, report_files: Dict[str, str], 
                            metrics: Dict, timestamp: str) -> str:
        """HTML ¸®Æ÷Æ® »ý¼º"""
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>?? Masked EEG Reconstruction Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; }}
        .section {{ background-color: white; margin: 20px 0; padding: 25px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric-card {{ padding: 20px; border-radius: 8px; text-align: center; color: white; }}
        .metric-card.good {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .metric-card.warning {{ background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333; }}
        .metric-card.poor {{ background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); }}
        .metric-value {{ font-size: 28px; font-weight: bold; margin-bottom: 5px; }}
        .metric-label {{ font-size: 14px; opacity: 0.9; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 25px; }}
        .image-card {{ text-align: center; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }}
        .image-card img {{ max-width: 100%; border-radius: 8px; }}
        .highlight {{ background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); padding: 15px; border-radius: 8px; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>?? Masked EEG Reconstruction Analysis</h1>
        <p><strong>Focus:</strong> How well does the model reconstruct HIDDEN connectivity patterns?</p>
        <p>Generated on: {timestamp} | 15-Frequency Band Model</p>
    </div>
    
    <div class="section">
        <h2>?? Masked Region Performance</h2>
        <div class="highlight">
            <strong>Key Question:</strong> Can the model accurately fill in missing EEG connectivity information?
            <br><strong>Analysis:</strong> Red metrics show performance on masked (hidden) regions vs blue (visible) regions.
        </div>
        <div class="metrics">
            <div class="metric-card {'good' if metrics.get('masked_magnitude_error', 100) < 15 else 'warning' if metrics.get('masked_magnitude_error', 100) < 25 else 'poor'}">
                <div class="metric-value">{metrics.get('masked_magnitude_error', 0):.1f}%</div>
                <div class="metric-label">?? Masked Magnitude Error<br>(Target: &lt;15%)</div>
            </div>
            <div class="metric-card {'good' if metrics.get('masked_phase_error_degrees', 100) < 30 else 'warning' if metrics.get('masked_phase_error_degrees', 100) < 45 else 'poor'}">
                <div class="metric-value">{metrics.get('masked_phase_error_degrees', 0):.1f}¡Æ</div>
                <div class="metric-label">?? Masked Phase Error<br>(Target: &lt;30¡Æ)</div>
            </div>
            <div class="metric-card {'good' if metrics.get('masked_correlation', 0) > 0.7 else 'warning' if metrics.get('masked_correlation', 0) > 0.5 else 'poor'}">
                <div class="metric-value">{metrics.get('masked_correlation', 0):.3f}</div>
                <div class="metric-label">?? Masked Correlation<br>(Target: &gt;0.7)</div>
            </div>
            <div class="metric-card good">
                <div class="metric-value">{metrics.get('mask_ratio', 0)*100:.1f}%</div>
                <div class="metric-label">?? Masked Ratio<br>({metrics.get('masked_positions', 0):,} positions)</div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>?? Masked vs Unmasked Comparison</h2>
        <p><strong>Masked Error:</strong> {metrics.get('masked_magnitude_error', 0):.1f}% | 
           <strong>Unmasked Error:</strong> {metrics.get('unmasked_magnitude_error', 0):.1f}% | 
           <strong>Ratio:</strong> {metrics.get('masked_magnitude_error', 1) / max(metrics.get('unmasked_magnitude_error', 1), 0.001):.2f}x</p>
        <p>{'? Good - Model reconstructs hidden regions well' if metrics.get('masked_magnitude_error', 1) / max(metrics.get('unmasked_magnitude_error', 1), 0.001) < 2.0 else '?? Poor - Model struggles with hidden regions'}</p>
    </div>
    
    <div class="section">
        <h2>?? Visualization Analysis</h2>
        <div class="image-grid">
"""
        
        # Add images
        image_descriptions = {
            'overview': '?? Magnitude + Phase + Masking overview with white boundaries showing masked regions',
            'frequency_masked': '?? Frequency band analysis focusing on masked region reconstruction quality',
            'masked_statistics': '?? Statistical analysis comparing masked vs unmasked region performance',
            'training_progress': '?? Training progress showing model learning to reconstruct patterns'
        }
        
        for key, filename in report_files.items():
            if filename.endswith('.png'):
                description = image_descriptions.get(key, f'{key.replace("_", " ").title()} analysis')
                html_content += f"""
            <div class="image-card">
                <h3>{key.replace('_', ' ').title()}</h3>
                <img src="{filename}" alt="{description}">
                <p style="color: #636e72; font-size: 14px; margin-top: 10px;">{description}</p>
            </div>
"""
        
        html_content += """
        </div>
    </div>
    
    <div class="section">
        <h2>?? Technical Details</h2>
        <ul>
            <li><strong>?? Core Question:</strong> How well can the model reconstruct missing EEG connectivity data?</li>
            <li><strong>Visualization Innovation:</strong> Magnitude + Phase heatmaps with masking overlay (white boundaries)</li>
            <li><strong>Analysis Focus:</strong> Separate performance metrics for masked vs unmasked regions</li>
            <li><strong>Model Architecture:</strong> 15-frequency structured feature extraction with global attention</li>
            <li><strong>Key Features:</strong> White boundaries show masked regions, color intensity shows reconstruction quality</li>
        </ul>
    </div>
    
    <footer style="text-align: center; color: #74b9ff; margin-top: 40px;">
        <p><strong>?? Enhanced Masked Reconstruction Analyzer</strong></p>
        <p>Specialized for evaluating model performance on hidden connectivity patterns</p>
    </footer>
</body>
</html>
"""
        
        return html_content
    
    def _convert_to_serializable(self, obj):
        """numpy/torch typesÀ» JSON serializable·Î º¯È¯"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # torch tensor scalar
            return obj.item()
        else:
            return obj
    
    def _add_electrode_labels(self, ax, step=4):
        """Àü±Ø ¶óº§ Ãß°¡"""
        indices = list(range(0, self.num_electrodes, step))
        ax.set_xticks(indices)
        ax.set_yticks(indices)
        ax.set_xticklabels([self.electrode_names[i] for i in indices], fontsize=8, rotation=45)
        ax.set_yticklabels([self.electrode_names[i] for i in indices], fontsize=8)
    
    def _get_freq_color(self, freq_hz):
        """ÁÖÆÄ¼öº° »ö»ó ¹ÝÈ¯"""
        if freq_hz in [1, 3]:
            return '#9467bd'  # delta
        elif freq_hz in [5, 7]:
            return '#2ca02c'  # theta
        elif freq_hz in [9, 11, 13]:
            return '#d62728'  # alpha
        else:
            return '#1f77b4'  # beta


# Demo function
def create_enhanced_demo():
    """Çâ»óµÈ µ¥¸ð »ý¼º"""
    print("?? Creating Enhanced Masked Reconstruction Demo...")
    
    # Çö½ÇÀûÀÎ EEG µ¥ÀÌÅÍ »ý¼º
    original = torch.zeros(15, 19, 19, 2)
    
    for freq in range(15):
        freq_hz = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29][freq]
        
        for i in range(19):
            for j in range(19):
                if i == j:
                    # ´ë°¢¼± (ÀÚ±â ¿¬°á)
                    original[freq, i, j, 0] = 0.8 + torch.randn(1) * 0.1
                    original[freq, i, j, 1] = torch.randn(1) * 0.05
                else:
                    # °Å¸® ¹× ÁÖÆÄ¼ö ±â¹Ý ¿¬°á¼º
                    distance = abs(i - j)
                    freq_factor = 1.2 if freq_hz in [9, 11, 13] else 0.8  # Alpha °­Á¶
                    strength = freq_factor * 0.4 / (1 + distance * 0.15)
                    
                    original[freq, i, j, 0] = strength * (1 + torch.randn(1) * 0.2)
                    original[freq, i, j, 1] = strength * torch.randn(1) * 0.3
    
    # 50% ¸¶½ºÅ·
    mask = torch.ones_like(original)
    num_pairs = 19 * 19
    num_to_mask = int(num_pairs * 0.5)
    
    masked_positions = torch.randperm(num_pairs)[:num_to_mask]
    for pos in masked_positions:
        i, j = pos // 19, pos % 19
        mask[:, i, j, :] = 0
    
    # º¹¿ø µ¥ÀÌÅÍ (¸¶½ºÅ·µÈ ºÎºÐ¿¡ ³ëÀÌÁî)
    reconstructed = original.clone()
    mask_binary = mask[..., 0] == 0
    
    # Çö½ÇÀûÀÎ º¹¿ø ¿ÀÂ÷
    noise_factor = 0.25
    reconstructed[mask_binary] = original[mask_binary] + torch.randn_like(original[mask_binary]) * noise_factor
    
    # ½Ã°¢È­
    visualizer = EEGReconstructionVisualizer()
    
    report_files = visualizer.generate_full_report(
        original, reconstructed, mask,
        output_dir="./masked_reconstruction_demo"
    )
    
    print(f"? Enhanced demo completed!")
    print(f"   ?? Focus: Masked region reconstruction (Magnitude + Phase)")
    print(f"   ?? Output: ./masked_reconstruction_demo/")
    print(f"   ?? Report: {report_files.get('html_report', 'report.html')}")
    
    return visualizer, original, reconstructed, mask, report_files


if __name__ == "__main__":
    print("="*80)
    print("?? ENHANCED MASKED EEG RECONSTRUCTION VISUALIZER")
    print("="*80)
    print("Key Features:")
    print("? ?? Masked region reconstruction focus")
    print("? ?? Magnitude + Phase analysis")
    print("? ?? Masking overlay (white boundaries)")
    print("? ?? Statistical comparison (masked vs unmasked)")
    print("="*80)
    
    # Run demo
    visualizer, original, reconstructed, mask, report_files = create_enhanced_demo()
    
    # Show sample metrics
    metrics = visualizer.compute_reconstruction_metrics(original, reconstructed, mask)
    
    print(f"\n?? SAMPLE RESULTS:")
    print(f"   Masked Magnitude Error: {metrics['masked_magnitude_error']:.1f}%")
    print(f"   Masked Phase Error: {metrics['masked_phase_error_degrees']:.1f}¡Æ")
    print(f"   Masked Correlation: {metrics['masked_correlation']:.3f}")
    print(f"   Performance Ratio: {metrics['masked_magnitude_error']/max(metrics['unmasked_magnitude_error'], 0.001):.2f}x")
    print(f"   Mask Coverage: {metrics['mask_ratio']*100:.1f}%")
    
    print("\n" + "="*80)
    print("? DEMO COMPLETED - Check ./masked_reconstruction_demo/")
    print("?? Open HTML report for full analysis")
    print("="*80)