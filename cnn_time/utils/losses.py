"""
EEG Connectivity Analysis - Simplified Loss Functions

ÇÙ½É ¼³°è ¿øÄ¢:
1. 4°³ ÇÙ½É loss¸¸ »ç¿ë (º¹Àâ¼º Á¦°Å)
2. ¹°¸®Àû ÀÇ¹Ì Áß½É (magnitude, phase)
3. ¼öÄ¡Àû ¾ÈÁ¤¼º º¸Àå
4. Config ±â¹Ý °¡ÁßÄ¡ Á¶Á¤
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EEGConfig

class EEGLossCalculator:
    """
    EEG Connectivity¸¦ À§ÇÑ ÅëÇÕ Loss Calculator
    
    4°³ ÇÙ½É Loss:
    1. MSE Loss - ±âº» º¹¿ø ¿ÀÂ÷
    2. Magnitude Loss - º¹¼Ò¼ö Å©±â º¸Á¸
    3. Phase Loss - º¹¼Ò¼ö À§»ó º¸Á¸  
    4. Coherence Loss - º¹¼Ò¼ö ÀÏ°ü¼º
    """
    
    def __init__(self, config: EEGConfig = None):
        if config is None:
            config = EEGConfig()
            
        self.config = config
        self.loss_config = config.LOSS_CONFIG
        self.frequency_bands = config.FREQUENCY_BANDS
        
        # Loss weights
        self.loss_weights = self.loss_config['loss_weights']
        
        # Loss-specific configurations
        self.magnitude_config = self.loss_config.get('magnitude_loss_config', {})
        self.phase_config = self.loss_config.get('phase_loss_config', {})
        self.coherence_config = self.loss_config.get('coherence_loss_config', {})
        
        # Frequency weights for magnitude loss
        self.frequency_weights = self.magnitude_config.get('frequency_weights', {
            'delta': 0.5, 'theta': 1.0, 'alpha': 2.0, 'beta': 1.5
        })
        
        print(f"? EEG Loss Calculator:")
        print(f"   Loss weights: {self.loss_weights}")
        print(f"   Magnitude type: {self.magnitude_config.get('loss_type', 'l2')}")
        print(f"   Phase type: {self.phase_config.get('loss_type', 'cosine')}")
        print(f"   Frequency weights: {self.frequency_weights}")
    
    def compute_total_loss(self, 
                          reconstructed: torch.Tensor, 
                          original: torch.Tensor, 
                          mask: torch.Tensor,
                          return_breakdown: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        ÅëÇÕ loss °è»ê
        
        Args:
            reconstructed: (batch, 15, 19, 19, 2) or (batch, 361, 15, 2)
            original: (batch, 15, 19, 19, 2)
            mask: (batch, 15, 19, 19, 2)
            
        Returns:
            total_loss: scalar tensor
            loss_breakdown: detailed components
        """
        
        # =============== SHAPE NORMALIZATION ===============
        reconstructed_norm, original_norm, mask_norm = self._normalize_shapes(
            reconstructed, original, mask
        )
        
        # =============== APPLY MASKING ===============
        masked_recon = reconstructed_norm * mask_norm
        masked_orig = original_norm * mask_norm
        
        # =============== COMPUTE INDIVIDUAL LOSSES ===============
        
        # 1. MSE Loss
        mse_loss = self.compute_mse_loss(masked_recon, masked_orig)
        
        # 2. Magnitude Loss
        magnitude_loss, magnitude_metrics = self.compute_magnitude_loss(
            masked_recon, masked_orig, return_metrics=True
        )
        
        # 3. Phase Loss
        phase_loss, phase_metrics = self.compute_phase_loss(
            masked_recon, masked_orig, return_metrics=True
        )
        
        # 4. Coherence Loss
        coherence_loss, coherence_metrics = self.compute_coherence_loss(
            masked_recon, masked_orig, return_metrics=True
        )
        
        # =============== WEIGHTED TOTAL LOSS ===============
        total_loss = (
            self.loss_weights['mse'] * mse_loss +
            self.loss_weights['magnitude'] * magnitude_loss +
            self.loss_weights['phase'] * phase_loss +
            self.loss_weights['coherence'] * coherence_loss
        )
        
        # =============== LOSS BREAKDOWN ===============
        if return_breakdown:
            loss_breakdown = {
                'total_loss': total_loss,
                'mse_loss': mse_loss,
                'magnitude_loss': magnitude_loss,
                'phase_loss': phase_loss,
                'coherence_loss': coherence_loss,
                
                # Weighted components
                'weighted_mse': self.loss_weights['mse'] * mse_loss,
                'weighted_magnitude': self.loss_weights['magnitude'] * magnitude_loss,
                'weighted_phase': self.loss_weights['phase'] * phase_loss,
                'weighted_coherence': self.loss_weights['coherence'] * coherence_loss,
                
                # Detailed metrics
                **magnitude_metrics,
                **phase_metrics,
                **coherence_metrics,
                
                # Additional metrics
                'mask_ratio': mask_norm.mean().item(),
                'loss_weights': self.loss_weights
            }
            
            return total_loss, loss_breakdown
        
        return total_loss
    
    def compute_mse_loss(self, recon: torch.Tensor, orig: torch.Tensor) -> torch.Tensor:
        """±âº» MSE loss"""
        return F.mse_loss(recon, orig)
    
    def compute_magnitude_loss(self, 
                              recon: torch.Tensor, 
                              orig: torch.Tensor,
                              return_metrics: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Magnitude loss with frequency weighting
        """
        # Extract magnitude
        recon_mag = torch.sqrt(recon[..., 0]**2 + recon[..., 1]**2 + 1e-8)
        orig_mag = torch.sqrt(orig[..., 0]**2 + orig[..., 1]**2 + 1e-8)
        
        # Basic magnitude loss
        loss_type = self.magnitude_config.get('loss_type', 'l2')
        if loss_type == 'l1':
            basic_loss = F.l1_loss(recon_mag, orig_mag)
        elif loss_type == 'huber':
            basic_loss = F.huber_loss(recon_mag, orig_mag)
        else:  # l2
            basic_loss = F.mse_loss(recon_mag, orig_mag)
        
        # Relative magnitude error
        relative_weight = self.magnitude_config.get('relative_weight', 0.5)
        relative_error = torch.mean(torch.abs(recon_mag - orig_mag) / (orig_mag + 1e-8))
        
        # Frequency-weighted magnitude loss
        freq_weighted_loss = self._compute_frequency_weighted_magnitude_loss(recon_mag, orig_mag)
        
        # Combined magnitude loss
        magnitude_loss = (
            (1 - relative_weight) * basic_loss +
            relative_weight * relative_error +
            0.3 * freq_weighted_loss
        )
        
        if return_metrics:
            # Additional metrics
            alpha_mag_error = self._compute_alpha_magnitude_error(recon_mag, orig_mag)
            
            metrics = {
                'magnitude_relative_error': relative_error,
                'magnitude_basic_loss': basic_loss,
                'magnitude_freq_weighted': freq_weighted_loss,
                'alpha_magnitude_error': alpha_mag_error,
                'magnitude_mean_recon': recon_mag.mean(),
                'magnitude_mean_orig': orig_mag.mean()
            }
            return magnitude_loss, metrics
        
        return magnitude_loss
    
    def compute_phase_loss(self, 
                          recon: torch.Tensor, 
                          orig: torch.Tensor,
                          return_metrics: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Phase loss with circular-aware computation
        """
        # Extract phase
        recon_phase = torch.atan2(recon[..., 1], recon[..., 0] + 1e-8)
        orig_phase = torch.atan2(orig[..., 1], orig[..., 0] + 1e-8)
        
        # Phase difference with wraparound
        phase_diff = recon_phase - orig_phase
        
        # Choose phase loss type
        loss_type = self.phase_config.get('loss_type', 'cosine')
        
        if loss_type == 'cosine':
            # Cosine phase loss (circular-aware)
            phase_loss = torch.mean(1 - torch.cos(phase_diff))
            
        elif loss_type == 'von_mises':
            # Von Mises phase loss (more sophisticated circular)
            kappa = 2.0  # concentration parameter
            cos_diff = torch.cos(phase_diff)
            cos_diff = torch.clamp(cos_diff, -1.0, 1.0)
            phase_loss = torch.mean(1 - torch.exp(kappa * (cos_diff - 1)))
            
        else:  # mse
            # MSE phase loss with wrapping
            if self.phase_config.get('wrap_around', True):
                phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
                phase_loss = torch.mean(phase_diff_wrapped**2) / (math.pi**2)
            else:
                phase_loss = torch.mean(phase_diff**2) / (math.pi**2)
        
        # Frequency emphasis (Alpha band)
        if self.phase_config.get('frequency_emphasis') == 'alpha':
            alpha_phase_loss = self._compute_alpha_phase_loss(recon_phase, orig_phase)
            phase_loss = 0.7 * phase_loss + 0.3 * alpha_phase_loss
        
        if return_metrics:
            # Phase metrics
            phase_error_rad = torch.sqrt(torch.mean(phase_diff**2))
            phase_error_degrees = phase_error_rad * 180.0 / math.pi
            alpha_phase_error = self._compute_alpha_phase_error(recon_phase, orig_phase)
            
            metrics = {
                'phase_error_degrees': phase_error_degrees,
                'phase_error_radians': phase_error_rad,
                'alpha_phase_error_degrees': alpha_phase_error,
                'phase_loss_type': loss_type,
                'phase_mean_diff': torch.mean(torch.abs(phase_diff))
            }
            return phase_loss, metrics
        
        return phase_loss
    
    def compute_coherence_loss(self, 
                              recon: torch.Tensor, 
                              orig: torch.Tensor,
                              return_metrics: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Coherence loss for complex number consistency
        """
        # Convert to complex numbers
        recon_complex = recon[..., 0] + 1j * recon[..., 1]
        orig_complex = orig[..., 0] + 1j * orig[..., 1]
        
        # Power coherence
        recon_power = torch.abs(recon_complex)**2
        orig_power = torch.abs(orig_complex)**2
        power_coherence = F.mse_loss(recon_power, orig_power)
        
        # Cross coherence (simplified)
        coherence_type = self.coherence_config.get('coherence_type', 'magnitude_consistency')
        
        if coherence_type == 'magnitude_consistency':
            coherence_loss = power_coherence
            
        elif coherence_type == 'complex_consistency':
            # Complex number consistency
            complex_diff = recon_complex - orig_complex
            coherence_loss = torch.mean(torch.abs(complex_diff)**2)
            
        else:  # power_consistency
            coherence_loss = power_coherence
        
        # Spatial and temporal coherence weights
        spatial_weight = self.coherence_config.get('spatial_coherence_weight', 0.3)
        temporal_weight = self.coherence_config.get('temporal_coherence_weight', 0.7)
        
        # Simple spatial coherence (neighboring pairs)
        spatial_coherence = self._compute_spatial_coherence(recon_complex, orig_complex)
        
        # Simple temporal coherence (frequency consistency)
        temporal_coherence = self._compute_temporal_coherence(recon_complex, orig_complex)
        
        # Combined coherence
        combined_coherence = (
            coherence_loss + 
            spatial_weight * spatial_coherence + 
            temporal_weight * temporal_coherence
        )
        
        if return_metrics:
            metrics = {
                'power_coherence': power_coherence,
                'spatial_coherence': spatial_coherence,
                'temporal_coherence': temporal_coherence,
                'coherence_type': coherence_type
            }
            return combined_coherence, metrics
        
        return combined_coherence
    
    def _normalize_shapes(self, reconstructed, original, mask):
        """Shape normalization to (batch, 15, 19, 19, 2)"""
        
        # Handle reconstructed shape
        if reconstructed.dim() == 4 and reconstructed.shape[1] == 361:
            # (batch, 361, 15, 2) ¡æ (batch, 15, 19, 19, 2)
            batch_size = reconstructed.shape[0]
            reconstructed = reconstructed.reshape(batch_size, 19, 19, 20, 2)
            reconstructed = reconstructed.permute(0, 3, 1, 2, 4)
        
        return reconstructed, original, mask
    
    def _compute_frequency_weighted_magnitude_loss(self, recon_mag, orig_mag):
        """ÁÖÆÄ¼ö ´ë¿ªº° °¡ÁßÄ¡ Àû¿ëÇÑ magnitude loss"""
        total_loss = 0.0
        total_weight = 0.0
        
        for band_name, freq_indices in self.frequency_bands.items():
            if band_name in self.frequency_weights:
                weight = self.frequency_weights[band_name]
                
                # Extract frequency band
                band_recon = recon_mag[:, freq_indices]
                band_orig = orig_mag[:, freq_indices]
                
                # Compute loss for this band
                band_loss = F.mse_loss(band_recon, band_orig)
                
                total_loss += weight * band_loss
                total_weight += weight
        
        return total_loss / (total_weight + 1e-8)
    
    def _compute_alpha_magnitude_error(self, recon_mag, orig_mag):
        """Alpha ´ë¿ª magnitude error"""
        alpha_indices = self.frequency_bands.get('alpha', [8, 9])
        
        alpha_recon = recon_mag[:, alpha_indices]
        alpha_orig = orig_mag[:, alpha_indices]
        
        alpha_error = torch.mean(torch.abs(alpha_recon - alpha_orig) / (alpha_orig + 1e-8))
        return alpha_error
    
    def _compute_alpha_phase_loss(self, recon_phase, orig_phase):
        """Alpha ´ë¿ª phase loss"""
        alpha_indices = self.frequency_bands.get('alpha', [8, 9])
        
        alpha_recon = recon_phase[:, alpha_indices]
        alpha_orig = orig_phase[:, alpha_indices]
        
        phase_diff = alpha_recon - alpha_orig
        alpha_loss = torch.mean(1 - torch.cos(phase_diff))
        return alpha_loss
    
    def _compute_alpha_phase_error(self, recon_phase, orig_phase):
        """Alpha ´ë¿ª phase error in degrees"""
        alpha_indices = self.frequency_bands.get('alpha', [8, 9])
        
        alpha_recon = recon_phase[:, alpha_indices]
        alpha_orig = orig_phase[:, alpha_indices]
        
        phase_diff = alpha_recon - alpha_orig
        alpha_error = torch.sqrt(torch.mean(phase_diff**2)) * 180.0 / math.pi
        return alpha_error
    
    def _compute_spatial_coherence(self, recon_complex, orig_complex):
        """°£´ÜÇÑ spatial coherence (neighboring electrode pairs)"""
        # Simple implementation: compare adjacent spatial positions
        batch, freq, height, width = recon_complex.shape
        
        spatial_diff_recon = torch.abs(recon_complex[:, :, 1:, :] - recon_complex[:, :, :-1, :])
        spatial_diff_orig = torch.abs(orig_complex[:, :, 1:, :] - orig_complex[:, :, :-1, :])
        
        spatial_coherence = F.mse_loss(spatial_diff_recon, spatial_diff_orig)
        return spatial_coherence
    
    def _compute_temporal_coherence(self, recon_complex, orig_complex):
        """°£´ÜÇÑ temporal coherence (frequency consistency)"""
        # Simple implementation: compare adjacent frequency bands
        batch, freq, height, width = recon_complex.shape
        
        if freq > 1:
            temporal_diff_recon = torch.abs(recon_complex[:, 1:, :, :] - recon_complex[:, :-1, :, :])
            temporal_diff_orig = torch.abs(orig_complex[:, 1:, :, :] - orig_complex[:, :-1, :, :])
            
            temporal_coherence = F.mse_loss(temporal_diff_recon, temporal_diff_orig)
        else:
            temporal_coherence = torch.tensor(0.0, device=recon_complex.device)
        
        return temporal_coherence

class EEGMetricsCalculator:
    """
    EEG ¼º´É ÁöÇ¥ °è»ê±â
    Training°ú evaluation¿¡¼­ »ç¿ëÇÒ »ó¼¼ ÁöÇ¥µé
    """
    
    def __init__(self, config: EEGConfig = None):
        if config is None:
            config = EEGConfig()
            
        self.config = config
        self.frequency_bands = config.FREQUENCY_BANDS
    
    def compute_signal_quality_metrics(self, 
                                     reconstructed: torch.Tensor, 
                                     original: torch.Tensor,
                                     mask: torch.Tensor) -> Dict:
        """½ÅÈ£ Ç°Áú ÁöÇ¥ °è»ê"""
        
        # Apply masking
        masked_recon = reconstructed * mask
        masked_orig = original * mask
        
        metrics = {}
        
        # 1. SNR (Signal-to-Noise Ratio)
        metrics['snr_db'] = self._compute_snr(masked_recon, masked_orig)
        
        # 2. Correlation
        metrics['correlation'] = self._compute_correlation(masked_recon, masked_orig)
        
        # 3. Magnitude-related metrics
        mag_metrics = self._compute_magnitude_metrics(masked_recon, masked_orig)
        metrics.update(mag_metrics)
        
        # 4. Phase-related metrics
        phase_metrics = self._compute_phase_metrics(masked_recon, masked_orig)
        metrics.update(phase_metrics)
        
        # 5. Frequency band-specific metrics
        band_metrics = self._compute_frequency_band_metrics(masked_recon, masked_orig)
        metrics.update(band_metrics)
        
        return metrics
    
    def _compute_snr(self, recon, orig):
        """Signal-to-Noise Ratio in dB"""
        signal_power = torch.mean(orig**2)
        noise_power = torch.mean((recon - orig)**2)
        snr_linear = signal_power / (noise_power + 1e-8)
        snr_db = 10 * torch.log10(snr_linear + 1e-8)
        return snr_db
    
    def _compute_correlation(self, recon, orig):
        """Pearson correlation coefficient"""
        recon_flat = recon.reshape(-1)
        orig_flat = orig.reshape(-1)
        
        recon_mean = torch.mean(recon_flat)
        orig_mean = torch.mean(orig_flat)
        
        numerator = torch.mean((recon_flat - recon_mean) * (orig_flat - orig_mean))
        recon_std = torch.sqrt(torch.mean((recon_flat - recon_mean)**2) + 1e-8)
        orig_std = torch.sqrt(torch.mean((orig_flat - orig_mean)**2) + 1e-8)
        
        correlation = numerator / (recon_std * orig_std + 1e-8)
        return correlation
    
    def _compute_magnitude_metrics(self, recon, orig):
        """Magnitude-related metrics"""
        recon_mag = torch.sqrt(recon[..., 0]**2 + recon[..., 1]**2 + 1e-8)
        orig_mag = torch.sqrt(orig[..., 0]**2 + orig[..., 1]**2 + 1e-8)
        
        metrics = {
            'magnitude_mse': F.mse_loss(recon_mag, orig_mag),
            'magnitude_mae': F.l1_loss(recon_mag, orig_mag),
            'magnitude_relative_error': torch.mean(torch.abs(recon_mag - orig_mag) / (orig_mag + 1e-8)),
            'magnitude_correlation': self._compute_correlation(recon_mag.unsqueeze(-1), orig_mag.unsqueeze(-1))
        }
        
        return metrics
    
    def _compute_phase_metrics(self, recon, orig):
        """Phase-related metrics"""
        recon_phase = torch.atan2(recon[..., 1], recon[..., 0] + 1e-8)
        orig_phase = torch.atan2(orig[..., 1], orig[..., 0] + 1e-8)
        
        phase_diff = recon_phase - orig_phase
        phase_diff_wrapped = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        
        metrics = {
            'phase_mse': torch.mean(phase_diff_wrapped**2) / (math.pi**2),
            'phase_mae': torch.mean(torch.abs(phase_diff_wrapped)) / math.pi,
            'phase_error_degrees': torch.sqrt(torch.mean(phase_diff**2)) * 180.0 / math.pi,
            'phase_cosine_similarity': torch.mean(torch.cos(phase_diff))
        }
        
        return metrics
    
    def _compute_frequency_band_metrics(self, recon, orig):
        """ÁÖÆÄ¼ö ´ë¿ªº° »ó¼¼ ÁöÇ¥"""
        metrics = {}
        
        for band_name, freq_indices in self.frequency_bands.items():
            band_recon = recon[:, freq_indices]
            band_orig = orig[:, freq_indices]
            
            # Magnitude metrics for this band
            band_recon_mag = torch.sqrt(band_recon[..., 0]**2 + band_recon[..., 1]**2 + 1e-8)
            band_orig_mag = torch.sqrt(band_orig[..., 0]**2 + band_orig[..., 1]**2 + 1e-8)
            
            # Phase metrics for this band  
            band_recon_phase = torch.atan2(band_recon[..., 1], band_recon[..., 0] + 1e-8)
            band_orig_phase = torch.atan2(band_orig[..., 1], band_orig[..., 0] + 1e-8)
            
            band_phase_diff = band_recon_phase - band_orig_phase
            
            metrics[f'{band_name}_magnitude_error'] = F.mse_loss(band_recon_mag, band_orig_mag)
            metrics[f'{band_name}_magnitude_relative'] = torch.mean(
                torch.abs(band_recon_mag - band_orig_mag) / (band_orig_mag + 1e-8)
            )
            metrics[f'{band_name}_phase_error_degrees'] = torch.sqrt(
                torch.mean(band_phase_diff**2)
            ) * 180.0 / math.pi
            metrics[f'{band_name}_snr_db'] = self._compute_snr(band_recon, band_orig)
            
        return metrics

# Factory functions for easy use

def create_loss_calculator(config: EEGConfig = None) -> EEGLossCalculator:
    """Loss calculator »ý¼º"""
    return EEGLossCalculator(config)

def create_metrics_calculator(config: EEGConfig = None) -> EEGMetricsCalculator:
    """Metrics calculator »ý¼º"""
    return EEGMetricsCalculator(config)

# Backward compatibility
def compute_eeg_loss(reconstructed, original, mask, config=None):
    """ÀÌÀü ¹öÀü È£È¯¼º"""
    calculator = EEGLossCalculator(config)
    return calculator.compute_total_loss(reconstructed, original, mask)

if __name__ == "__main__":
    print("="*80)
    print("?? SIMPLIFIED EEG LOSS FUNCTIONS")
    print("="*80)
    
    config = EEGConfig()
    
    # Loss calculator Å×½ºÆ®
    loss_calc = create_loss_calculator(config)
    metrics_calc = create_metrics_calculator(config)
    
    # Sample data
    batch_size = 4
    reconstructed = torch.randn(batch_size, 20, 19, 19, 2)
    original = torch.randn(batch_size, 20, 19, 19, 2)
    mask = torch.ones_like(original)
    mask = mask * (torch.rand_like(mask) > 0.5).float()  # 50% masking
    
    # Loss °è»ê
    total_loss, loss_breakdown = loss_calc.compute_total_loss(
        reconstructed, original, mask, return_breakdown=True
    )
    
    print(f"? Loss Calculation Test:")
    print(f"   Total Loss: {total_loss.item():.6f}")
    print(f"   MSE: {loss_breakdown['mse_loss'].item():.6f}")
    print(f"   Magnitude: {loss_breakdown['magnitude_loss'].item():.6f}")
    print(f"   Phase: {loss_breakdown['phase_loss'].item():.6f}")
    print(f"   Coherence: {loss_breakdown['coherence_loss'].item():.6f}")
    
    print(f"\n?? Detailed Metrics:")
    print(f"   Phase Error: {loss_breakdown['phase_error_degrees'].item():.1f}¡Æ")
    print(f"   Alpha Phase: {loss_breakdown['alpha_phase_error_degrees'].item():.1f}¡Æ")
    print(f"   Magnitude Relative: {loss_breakdown['magnitude_relative_error'].item()*100:.1f}%")
    print(f"   Alpha Magnitude: {loss_breakdown['alpha_magnitude_error'].item()*100:.1f}%")
    
    # Signal quality metrics
    quality_metrics = metrics_calc.compute_signal_quality_metrics(
        reconstructed, original, mask
    )
    
    print(f"\n?? Signal Quality:")
    print(f"   SNR: {quality_metrics['snr_db'].item():.1f} dB")
    print(f"   Correlation: {quality_metrics['correlation'].item():.3f}")
    print(f"   Magnitude MSE: {quality_metrics['magnitude_mse'].item():.6f}")
    print(f"   Phase MSE: {quality_metrics['phase_mse'].item():.6f}")
    
    # Frequency band metrics
    print(f"\n?? Frequency Band Analysis:")
    for band in ['delta', 'theta', 'alpha', 'beta']:
        if f'{band}_magnitude_error' in quality_metrics:
            mag_err = quality_metrics[f'{band}_magnitude_error'].item()
            phase_err = quality_metrics[f'{band}_phase_error_degrees'].item()
            snr = quality_metrics[f'{band}_snr_db'].item()
            print(f"   {band.capitalize()}: Mag={mag_err:.4f}, Phase={phase_err:.1f}¡Æ, SNR={snr:.1f}dB")
    
    print("="*80)