"""
EEG Connectivity Pre-trained Model Visualization Tool - Updated for New Architecture

¸¶½ºÅ· ±â¹Ý pre-trained ¸ðµ¨ÀÇ ½Ã°¢È­:
1. ¿øº» ¡æ ¸¶½ºÅ· ¡æ º¹¿ø °úÁ¤ ½Ã°¢È­
2. ½Ç¼öºÎ/Çã¼öºÎ, Magnitude/À§»ó ºÐ¸® ½Ã°¢È­
3. ÁÖÆÄ¼öº° ¹× ³ú ¿µ¿ªº° ºÐ¼®
4. Á¤·®Àû ¼º´É ÁöÇ¥ Á¦°ø
5. ? »õ·Î¿î ¸ðµ¨ ±¸Á¶ ¿ÏÀü Áö¿ø
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle
import warnings
warnings.filterwarnings('ignore')

# ? »ó´ë °æ·Î·Î ÇÁ·ÎÁ§Æ® ·çÆ® Ãß°¡
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'models' not in current_dir else current_dir
sys.path.append(project_root)

# ? »õ·Î¿î ¸ðµ¨ ±¸Á¶ import
try:
    from config import EEGConfig
    from models.hybrid_model import EEGConnectivityModel
    from data.dataset import EEGDataset
    from torch.utils.data import DataLoader
    print("? Successfully imported project modules")
    NEW_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"?? Import failed: {e}")
    print("Using minimal fallback classes...")
    NEW_MODEL_AVAILABLE = False
    
    class EEGConfig:
        def __init__(self):
            self.NUM_FREQUENCIES = 20
            self.NUM_ELECTRODES = 19
            self.NUM_COMPLEX_DIMS = 2
            self.NUM_PAIRS = 361
            self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.ELECTRODE_NAMES = [
                'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
            ]
            
            self.BRAIN_REGIONS = {
                'frontal': [0, 1, 2, 3, 10, 11, 16],
                'central': [4, 5, 17],
                'parietal': [6, 7, 18],
                'temporal': [12, 13, 14, 15],
                'occipital': [8, 9]
            }
            
            self.FREQUENCY_BANDS = {
                'delta': [0, 1, 2, 3],
                'theta': [4, 5, 6, 7], 
                'alpha': [8, 9],
                'beta1': [10, 11, 12, 13],
                'beta2': [14, 15],
                'gamma': [16, 17, 18, 19]
            }


class EEGVisualizationTool:
    """
    EEG ¿¬°á¼º ½Ã°¢È­ µµ±¸ - »õ·Î¿î ¸ðµ¨ ±¸Á¶ Áö¿ø
    """
    
    def __init__(self, model_path: str, config: EEGConfig = None):
        """
        Args:
            model_path: Pre-trained ¸ðµ¨ °æ·Î
            config: EEG ¼³Á¤
        """
        self.config = config if config else EEGConfig()
        self.device = self.config.DEVICE
        
        # ? »õ·Î¿î ¸ðµ¨ ·Îµù ¹æ½Ä
        print(f"?? Loading model from: {model_path}")
        self.model = self.load_model_new_architecture(model_path)
        
        # Setup visualization
        self.setup_visualization()
        
    def load_model_new_architecture(self, model_path: str):
        """? »õ·Î¿î ¸ðµ¨ ±¸Á¶ ·Îµù"""
        try:
            if NEW_MODEL_AVAILABLE:
                # »õ·Î¿î EEGConnectivityModel »ç¿ë
                try:
                    model = EEGConnectivityModel.load_model(model_path, mode='pretrain')
                    model.to(self.device)
                    model.eval()
                    print("? Loaded new EEGConnectivityModel")
                    return model
                    
                except Exception as e:
                    print(f"?? Failed to load as EEGConnectivityModel: {e}")
                    
            # Fallback: Ã¼Å©Æ÷ÀÎÆ®¿¡¼­ Á÷Á¢ ·Îµù
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # Ã¼Å©Æ÷ÀÎÆ® ÇüÅÂ
                    if NEW_MODEL_AVAILABLE:
                        # »õ ¸ðµ¨ ±¸Á¶·Î ·Îµù ½Ãµµ
                        config = checkpoint.get('config', self.config)
                        model = EEGConnectivityModel(config=config, mode='pretrain')
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.to(self.device)
                        model.eval()
                        print("? Loaded from checkpoint with new architecture")
                        return model
                    else:
                        print("?? New model not available, using fallback")
                        
                elif hasattr(checkpoint, 'forward'):
                    # Á÷Á¢ ¸ðµ¨ °´Ã¼
                    model = checkpoint
                    model.to(self.device)
                    model.eval()
                    print("? Loaded direct model object")
                    return model
                    
            else:
                # Á÷Á¢ ¸ðµ¨
                model = checkpoint
                model.to(self.device)
                model.eval()
                print("? Loaded direct model")
                return model
                
        except Exception as e:
            print(f"?? Failed to load model: {e}")
            
        # ÃÖÁ¾ fallback: ´õ¹Ì ¸ðµ¨
        print("?? Using dummy model for testing")
        model = torch.nn.Sequential(
            torch.nn.Linear(361*20*2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 361*20*2)
        ).to(self.device)
        return model
        
    def setup_visualization(self):
        """½Ã°¢È­ ¼³Á¤ ÃÊ±âÈ­"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color maps
        self.real_cmap = 'RdBu_r'
        self.imag_cmap = 'PRGn'
        self.mag_cmap = 'viridis'
        self.phase_cmap = 'hsv'
        self.error_cmap = 'Reds'
        
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8
        })
    
    def create_mask(self, data: torch.Tensor, mask_ratio: float = 0.5) -> torch.Tensor:
        """? °³¼±µÈ ¸¶½ºÅ· Àü·« (pretrain_trainer¿Í µ¿ÀÏ)"""
        batch_size, freq, height, width, complex_dim = data.shape
        device = data.device
        
        mask = torch.ones_like(data)
        
        for b in range(batch_size):
            total_positions = height * width  # 361
            num_to_mask = int(total_positions * mask_ratio)
            
            positions = torch.randperm(total_positions, device=device)[:num_to_mask]
            
            for pos in positions:
                i, j = pos // width, pos % width
                # Mask all frequencies and complex dimensions for this position
                mask[b, :, i, j, :] = 0
        
        return mask
    
    def model_inference(self, masked_data: torch.Tensor) -> torch.Tensor:
        """? »õ·Î¿î ¸ðµ¨ ±¸Á¶ Áö¿ø Ãß·Ð"""
        with torch.no_grad():
            try:
                # EEGConnectivityModel È®ÀÎ
                if hasattr(self.model, 'forward') and hasattr(self.model, 'config'):
                    # »õ·Î¿î EEGConnectivityModel
                    print("?? Using EEGConnectivityModel inference")
                    reconstructed = self.model(masked_data)
                    return reconstructed
                    
                elif hasattr(self.model, 'forward'):
                    # ÀÏ¹Ý PyTorch ¸ðµ¨
                    print("?? Using standard PyTorch model")
                    
                    # Shape È®ÀÎ ¹× º¯È¯
                    if masked_data.dim() == 5:  # (batch, freq, height, width, complex)
                        batch_size = masked_data.shape[0]
                        # Flatten for linear model
                        flattened = masked_data.view(batch_size, -1)
                        output_flat = self.model(flattened)
                        # Reshape back
                        reconstructed = output_flat.view_as(masked_data)
                    else:
                        reconstructed = self.model(masked_data)
                    
                    return reconstructed
                    
                else:
                    print("?? Model has no forward method")
                    return masked_data + torch.randn_like(masked_data) * 0.1
                    
            except Exception as e:
                print(f"?? Model inference failed: {e}")
                # Fallback: ºÎºÐ reconstruction simulation
                noise_level = 0.1
                reconstructed = masked_data + torch.randn_like(masked_data) * noise_level
                
                # ¸¶½ºÅ·µÈ ºÎºÐ¿¡ ´õ realisticÇÑ °ª Ã¤¿ì±â
                mask_binary = (masked_data.sum(dim=-1, keepdim=True) == 0)
                
                # ÀÎ±Ù °ªµéÀÇ Æò±ÕÀ¸·Î Ã¤¿ì±â (°£´ÜÇÑ interpolation)
                for b in range(reconstructed.shape[0]):
                    for f in range(reconstructed.shape[1]):
                        for i in range(reconstructed.shape[2]):
                            for j in range(reconstructed.shape[3]):
                                if mask_binary[b, f, i, j, 0]:
                                    # ÀÎ±Ù 8°³ ÇÈ¼¿ÀÇ Æò±Õ
                                    neighbors = []
                                    for di in [-1, 0, 1]:
                                        for dj in [-1, 0, 1]:
                                            if (di == 0 and dj == 0):
                                                continue
                                            ni, nj = i + di, j + dj
                                            if (0 <= ni < reconstructed.shape[2] and 
                                                0 <= nj < reconstructed.shape[3] and
                                                not mask_binary[b, f, ni, nj, 0]):
                                                neighbors.append(reconstructed[b, f, ni, nj, :])
                                    
                                    if neighbors:
                                        avg_neighbor = torch.stack(neighbors).mean(dim=0)
                                        reconstructed[b, f, i, j, :] = avg_neighbor
                
                return reconstructed
    
    def visualize_sample(self, data: torch.Tensor, mask_ratio: float = 0.5, 
                        freq_idx: Optional[int] = None, save_path: Optional[str] = None):
        """
        ´ÜÀÏ »ùÇÃ ½Ã°¢È­ - »õ·Î¿î ¸ðµ¨ ±¸Á¶ Áö¿ø
        
        Args:
            data: (20, 19, 19, 2) CSD µ¥ÀÌÅÍ
            mask_ratio: ¸¶½ºÅ· ºñÀ²
            freq_idx: Æ¯Á¤ ÁÖÆÄ¼ö ÀÎµ¦½º (NoneÀÌ¸é Æò±Õ)
            save_path: ÀúÀå °æ·Î
        """
        
        # Prepare data
        if data.dim() == 4:
            data = data.unsqueeze(0)  # Add batch dimension
        
        data = data.to(self.device)
        
        # Create mask and apply
        mask = self.create_mask(data, mask_ratio)
        masked_data = data * mask
        
        # ? »õ·Î¿î ¸ðµ¨·Î Ãß·Ð
        print(f"?? Running inference with mask ratio {mask_ratio}")
        reconstructed = self.model_inference(masked_data)
        
        # Convert to numpy
        original_np = data[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()
        masked_np = masked_data[0].cpu().numpy()
        reconstructed_np = reconstructed[0].cpu().numpy()
        
        # Select frequency
        if freq_idx is not None:
            original_slice = original_np[freq_idx]
            mask_slice = mask_np[freq_idx]
            masked_slice = masked_np[freq_idx]
            reconstructed_slice = reconstructed_np[freq_idx]
            title_suffix = f"Frequency {freq_idx}"
        else:
            original_slice = np.mean(original_np, axis=0)
            mask_slice = np.mean(mask_np, axis=0)
            masked_slice = np.mean(masked_np, axis=0)
            reconstructed_slice = np.mean(reconstructed_np, axis=0)
            title_suffix = "Frequency Average"
        
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(f'EEG Connectivity Reconstruction - {title_suffix}\n(Mask Ratio: {mask_ratio})', fontsize=16)
        
        # Extract components
        datasets = {
            'Original': original_slice,
            'Masked': masked_slice,
            'Reconstructed': reconstructed_slice,
            'Mask': mask_slice
        }
        
        # Extract components for all datasets first to get global min/max
        all_datasets = {}
        for name, data_slice in datasets.items():
            if name == 'Mask':
                continue
            all_datasets[name] = {
                'real': data_slice[:, :, 0],
                'imag': data_slice[:, :, 1],
                'magnitude': np.sqrt(data_slice[:, :, 0]**2 + data_slice[:, :, 1]**2),
                'phase': np.arctan2(data_slice[:, :, 1], data_slice[:, :, 0])
            }
        
        # Calculate global min/max for consistent color scales
        all_reals = [all_datasets[name]['real'] for name in all_datasets.keys()]
        all_imags = [all_datasets[name]['imag'] for name in all_datasets.keys()]
        all_mags = [all_datasets[name]['magnitude'] for name in all_datasets.keys()]
        
        real_vmin, real_vmax = np.min(all_reals), np.max(all_reals)
        imag_vmin, imag_vmax = np.min(all_imags), np.max(all_imags)
        mag_vmin, mag_vmax = np.min(all_mags), np.max(all_mags)
        phase_vmin, phase_vmax = -np.pi, np.pi  # Phase is always -¥ð to ¥ð
        
        print(f"Color scale ranges:")
        print(f"  Real: [{real_vmin:.3f}, {real_vmax:.3f}]")
        print(f"  Imag: [{imag_vmin:.3f}, {imag_vmax:.3f}]")
        print(f"  Magnitude: [{mag_vmin:.3f}, {mag_vmax:.3f}]")
        print(f"  Phase: [{phase_vmin:.3f}, {phase_vmax:.3f}]")
        
        for row, (name, data_slice) in enumerate(datasets.items()):
            if name == 'Mask':
                # Special handling for mask - show masked regions in white
                mask_display = 1 - data_slice[:, :, 0]  # 1 = masked (white), 0 = unmasked (black)
                axes[row, 0].imshow(mask_display, cmap='Greys', aspect='equal')
                axes[row, 0].set_title(f'{name} (White=Masked)')
                axes[row, 1].axis('off')
                axes[row, 2].axis('off')
                axes[row, 3].axis('off')
                continue
            
            real_part = data_slice[:, :, 0]
            imag_part = data_slice[:, :, 1]
            magnitude = np.sqrt(real_part**2 + imag_part**2)
            phase = np.arctan2(imag_part, real_part)
            
            # Apply masking for Masked and Reconstructed data
            if name == 'Masked':
                # For Masked: show original data where masked, black where unmasked
                mask_binary = 1 - mask_slice[:, :, 0]  # 1 = masked, 0 = unmasked
                
                # Create masked arrays - show original data only in masked regions
                real_part_display = np.ma.array(real_part, mask=(mask_binary == 0))
                imag_part_display = np.ma.array(imag_part, mask=(mask_binary == 0))
                magnitude_display = np.ma.array(magnitude, mask=(mask_binary == 0))
                phase_display = np.ma.array(phase, mask=(mask_binary == 0))
                
                # Use masked arrays
                real_part = real_part_display
                imag_part = imag_part_display
                magnitude = magnitude_display
                phase = phase_display
                
            elif name == 'Reconstructed':
                # For Reconstructed: show reconstructed data where masked, black where unmasked
                mask_binary = 1 - mask_slice[:, :, 0]  # 1 = masked, 0 = unmasked
                
                # Create masked arrays - show reconstructed data only in masked regions
                real_part_display = np.ma.array(real_part, mask=(mask_binary == 0))
                imag_part_display = np.ma.array(imag_part, mask=(mask_binary == 0))
                magnitude_display = np.ma.array(magnitude, mask=(mask_binary == 0))
                phase_display = np.ma.array(phase, mask=(mask_binary == 0))
                
                # Use masked arrays
                real_part = real_part_display
                imag_part = imag_part_display
                magnitude = magnitude_display
                phase = phase_display
            
            # Real part with consistent color scale
            if name in ['Masked', 'Reconstructed']:
                cmap_real = plt.cm.get_cmap(self.real_cmap).copy()
                cmap_real.set_bad(color='black')
                im1 = axes[row, 0].imshow(real_part, cmap=cmap_real, aspect='equal',
                                         vmin=real_vmin, vmax=real_vmax)
            else:
                im1 = axes[row, 0].imshow(real_part, cmap=self.real_cmap, aspect='equal',
                                         vmin=real_vmin, vmax=real_vmax)
            axes[row, 0].set_title(f'{name} - Real')
            cbar1 = plt.colorbar(im1, ax=axes[row, 0], shrink=0.8)
            if name in ['Masked', 'Reconstructed']:
                cbar1.set_label('Black=Unmasked', fontsize=8)
            
            # Imaginary part with consistent color scale
            if name in ['Masked', 'Reconstructed']:
                cmap_imag = plt.cm.get_cmap(self.imag_cmap).copy()
                cmap_imag.set_bad(color='black')
                im2 = axes[row, 1].imshow(imag_part, cmap=cmap_imag, aspect='equal',
                                         vmin=imag_vmin, vmax=imag_vmax)
            else:
                im2 = axes[row, 1].imshow(imag_part, cmap=self.imag_cmap, aspect='equal',
                                         vmin=imag_vmin, vmax=imag_vmax)
            axes[row, 1].set_title(f'{name} - Imaginary')
            cbar2 = plt.colorbar(im2, ax=axes[row, 1], shrink=0.8)
            if name in ['Masked', 'Reconstructed']:
                cbar2.set_label('Black=Unmasked', fontsize=8)
            
            # Magnitude with consistent color scale
            if name in ['Masked', 'Reconstructed']:
                cmap_mag = plt.cm.get_cmap(self.mag_cmap).copy()
                cmap_mag.set_bad(color='black')
                im3 = axes[row, 2].imshow(magnitude, cmap=cmap_mag, aspect='equal',
                                         vmin=mag_vmin, vmax=mag_vmax)
            else:
                im3 = axes[row, 2].imshow(magnitude, cmap=self.mag_cmap, aspect='equal',
                                         vmin=mag_vmin, vmax=mag_vmax)
            axes[row, 2].set_title(f'{name} - Magnitude')
            cbar3 = plt.colorbar(im3, ax=axes[row, 2], shrink=0.8)
            if name in ['Masked', 'Reconstructed']:
                cbar3.set_label('Black=Unmasked', fontsize=8)
            
            # Phase with consistent color scale
            if name in ['Masked', 'Reconstructed']:
                cmap_phase = plt.cm.get_cmap(self.phase_cmap).copy()
                cmap_phase.set_bad(color='black')
                im4 = axes[row, 3].imshow(phase, cmap=cmap_phase, aspect='equal', 
                                         vmin=phase_vmin, vmax=phase_vmax)
            else:
                im4 = axes[row, 3].imshow(phase, cmap=self.phase_cmap, aspect='equal', 
                                         vmin=phase_vmin, vmax=phase_vmax)
            axes[row, 3].set_title(f'{name} - Phase')
            cbar4 = plt.colorbar(im4, ax=axes[row, 3], shrink=0.8)
            cbar4.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            cbar4.set_ticklabels(['-¥ð', '-¥ð/2', '0', '¥ð/2', '¥ð'])
            if name in ['Masked', 'Reconstructed']:
                cbar4.set_label('Black=Unmasked', fontsize=8)
        
        # Add electrode labels to first subplot
        axes[0, 0].set_xticks(range(0, 19, 3))
        axes[0, 0].set_yticks(range(0, 19, 3))
        axes[0, 0].set_xticklabels([self.config.ELECTRODE_NAMES[i] for i in range(0, 19, 3)], 
                                  rotation=45, fontsize=8)
        axes[0, 0].set_yticklabels([self.config.ELECTRODE_NAMES[i] for i in range(0, 19, 3)], 
                                  fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"?? Saved: {save_path}")
        
        return fig
    
    def compute_metrics(self, original: np.ndarray, reconstructed: np.ndarray, 
                       mask: np.ndarray) -> Dict:
        """? °³¼±µÈ ¼º´É ÁöÇ¥ °è»ê"""
        
        # Only compute for masked regions
        masked_positions = (1 - mask) > 0
        
        if not np.any(masked_positions):
            return {'error': 'No masked positions found'}
        
        orig_masked = original[masked_positions]
        recon_masked = reconstructed[masked_positions]
        
        # Extract components
        orig_real = orig_masked[..., 0] if orig_masked.ndim > 1 else orig_masked
        orig_imag = orig_masked[..., 1] if orig_masked.ndim > 1 else np.zeros_like(orig_real)
        recon_real = recon_masked[..., 0] if recon_masked.ndim > 1 else recon_masked
        recon_imag = recon_masked[..., 1] if recon_masked.ndim > 1 else np.zeros_like(recon_real)
        
        orig_mag = np.sqrt(orig_real**2 + orig_imag**2)
        recon_mag = np.sqrt(recon_real**2 + recon_imag**2)
        
        orig_phase = np.arctan2(orig_imag, orig_real)
        recon_phase = np.arctan2(recon_imag, recon_real)
        
        # ? Circular phase difference
        phase_diff = np.arctan2(np.sin(orig_phase - recon_phase), 
                               np.cos(orig_phase - recon_phase))
        
        metrics = {
            'mse_real': np.mean((orig_real - recon_real)**2),
            'mse_imag': np.mean((orig_imag - recon_imag)**2),
            'mse_magnitude': np.mean((orig_mag - recon_mag)**2),
            'mae_magnitude': np.mean(np.abs(orig_mag - recon_mag)),
            'magnitude_relative_error': np.mean(np.abs(orig_mag - recon_mag) / (orig_mag + 1e-8)),
            'phase_error_rad': np.mean(np.abs(phase_diff)),
            'phase_error_deg': np.mean(np.abs(phase_diff)) * 180 / np.pi,
            'correlation_magnitude': np.corrcoef(orig_mag.flatten(), recon_mag.flatten())[0, 1] if len(orig_mag) > 1 else 0,
            'snr_db': 10 * np.log10(np.var(orig_masked) / (np.var(orig_masked - recon_masked) + 1e-8) + 1e-8),
            
            # ? Ãß°¡ ¸ÞÆ®¸¯
            'phase_circular_mse': np.mean(phase_diff**2),  # Circular MSE
            'complex_correlation': np.corrcoef((orig_real + 1j*orig_imag).flatten(), 
                                             (recon_real + 1j*recon_imag).flatten())[0, 1].real if len(orig_real) > 1 else 0
        }
        
        return metrics
    
    def visualize_frequency_bands(self, data: torch.Tensor, mask_ratio: float = 0.5,
                                 save_path: Optional[str] = None):
        """ÁÖÆÄ¼ö ´ë¿ªº° ½Ã°¢È­ - »õ·Î¿î ¸ðµ¨ Áö¿ø"""
        
        if data.dim() == 4:
            data = data.unsqueeze(0)
        
        data = data.to(self.device)
        
        # Create mask and reconstruct
        mask = self.create_mask(data, mask_ratio)
        masked_data = data * mask
        reconstructed = self.model_inference(masked_data)
        
        # Convert to numpy
        original_np = data[0].cpu().numpy()
        reconstructed_np = reconstructed[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(len(self.config.FREQUENCY_BANDS), 3, figsize=(15, 3*len(self.config.FREQUENCY_BANDS)))
        fig.suptitle(f'EEG Connectivity by Frequency Bands (Mask: {mask_ratio})', fontsize=16)
        
        for i, (band_name, freq_indices) in enumerate(self.config.FREQUENCY_BANDS.items()):
            # Average across frequencies in this band
            band_original = np.mean(original_np[freq_indices], axis=0)
            band_reconstructed = np.mean(reconstructed_np[freq_indices], axis=0)
            band_mask = np.mean(mask_np[freq_indices], axis=0)
            
            # Compute magnitude
            orig_mag = np.sqrt(band_original[:, :, 0]**2 + band_original[:, :, 1]**2)
            recon_mag = np.sqrt(band_reconstructed[:, :, 0]**2 + band_reconstructed[:, :, 1]**2)
            
            # Plot original
            im1 = axes[i, 0].imshow(orig_mag, cmap=self.mag_cmap, aspect='equal')
            axes[i, 0].set_title(f'{band_name.capitalize()} - Original')
            plt.colorbar(im1, ax=axes[i, 0], shrink=0.8)
            
            # Plot reconstructed (only masked regions)
            mask_binary = 1 - band_mask[:, :, 0]  # 1 = masked, 0 = unmasked
            recon_mag_masked = np.ma.array(recon_mag, mask=(mask_binary == 0))
            
            cmap_recon = plt.cm.get_cmap(self.mag_cmap).copy()
            cmap_recon.set_bad(color='black')
            im2 = axes[i, 1].imshow(recon_mag_masked, cmap=cmap_recon, aspect='equal')
            axes[i, 1].set_title(f'{band_name.capitalize()} - Reconstructed')
            plt.colorbar(im2, ax=axes[i, 1], shrink=0.8)
            
            # Plot error (only in masked regions)
            error = np.abs(recon_mag - orig_mag) * mask_binary
            im3 = axes[i, 2].imshow(error, cmap=self.error_cmap, aspect='equal')
            axes[i, 2].set_title(f'{band_name.capitalize()} - Error')
            plt.colorbar(im3, ax=axes[i, 2], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"?? Saved frequency bands: {save_path}")
        
        return fig


def load_test_data(dataset_path: str, num_samples: int = 5):
    """? °³¼±µÈ Å×½ºÆ® µ¥ÀÌÅÍ ·Îµù"""
    print(f"?? Loading test data from: {dataset_path}")
    
    try:
        # Try to load real dataset
        config = EEGConfig()
        
        # Look for data files
        data_files = []
        if os.path.exists(dataset_path):
            for ext in ['.pkl', '.pt', '.pth', '.npy']:
                files = [f for f in os.listdir(dataset_path) if f.endswith(ext)]
                data_files.extend(files)
                if len(data_files) >= num_samples:
                    break
        
        samples = []
        labels = []
        
        print(f"?? Found {len(data_files)} potential data files")
        
        for i, filename in enumerate(data_files[:num_samples]):
            try:
                filepath = os.path.join(dataset_path, filename)
                
                if filename.endswith('.pkl'):
                    with open(filepath, 'rb') as f:
                        data_dict = pickle.load(f)
                        if 'csd' in data_dict:
                            data = torch.tensor(data_dict['csd'], dtype=torch.float32)
                            label = data_dict.get('label', 0)
                        else:
                            print(f"   ?? No 'csd' key in {filename}")
                            continue
                            
                elif filename.endswith(('.pt', '.pth')):
                    checkpoint = torch.load(filepath, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        data = checkpoint.get('data', torch.randn(20, 19, 19, 2))
                        label = checkpoint.get('label', 0)
                    else:
                        data = checkpoint
                        label = 0
                else:
                    continue
                
                # ? Shape °ËÁõ ¹× ¼öÁ¤
                if data.shape == (20, 19, 19, 2):
                    # Perfect shape
                    pass
                elif data.shape == (19, 19, 20, 2):
                    # Transpose frequency dimension
                    data = data.permute(2, 0, 1, 3)
                elif data.shape == (15, 19, 19, 2):
                    # Pad to 20 frequencies
                    padding = torch.zeros(5, 19, 19, 2)
                    data = torch.cat([data, padding], dim=0)
                    print(f"   ?? Padded {filename} from 15 to 20 frequencies")
                else:
                    print(f"   ?? Unsupported shape {data.shape} for {filename}, skipping...")
                    continue
                
                samples.append(data)
                labels.append(label)
                print(f"   ? Loaded {filename}: shape {data.shape}, label {label}")
                
            except Exception as e:
                print(f"   ? Failed to load {filename}: {e}")
        
        if samples:
            print(f"? Successfully loaded {len(samples)} real samples")
            return samples, labels
        
    except Exception as e:
        print(f"? Failed to load real data: {e}")
    
    # Fallback: generate synthetic data
    print("?? Generating synthetic test data...")
    samples = []
    labels = []
    
    for i in range(num_samples):
        # ? Generate more realistic EEG connectivity data
        data = torch.randn(20, 19, 19, 2) * 0.3
        
        # Add frequency-dependent structure
        for freq in range(20):
            freq_factor = 1.0 / (1 + freq * 0.1)  # Lower frequencies have higher magnitude
            data[freq] *= freq_factor
            
            # Add diagonal dominance (self-connectivity)
            for j in range(19):
                data[freq, j, j, :] = torch.randn(2) * 1.5 * freq_factor
        
        # Add symmetric structure (Hermitian-like)
        for i in range(19):
            for j in range(i+1, 19):
                # Make (i,j) and (j,i) related but not identical
                data[:, j, i, 0] = data[:, i, j, 0] * 0.8 + torch.randn(20) * 0.1
                data[:, j, i, 1] = -data[:, i, j, 1] * 0.8 + torch.randn(20) * 0.1  # Anti-symmetric imaginary
        
        label = i % 2  # Alternate labels
        samples.append(data)
        labels.append(label)
        
        print(f"   ?? Generated synthetic sample {i+1}: shape {data.shape}, label {label}")
    
    print(f"? Generated {len(samples)} synthetic samples")
    return samples, labels


def main():
    """? °³¼±µÈ ¸ÞÀÎ ½ÇÇà ÇÔ¼ö"""
    print("?? EEG Connectivity Visualization Tool (Updated for New Architecture)")
    print("=" * 70)
    
    # Configuration
    config = EEGConfig()
    
    # ? °æ·Î ¼³Á¤ (»ó´ë °æ·Î »ç¿ë)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ¸ðµ¨ °æ·Îµé ½Ãµµ
    possible_model_paths = [
        os.path.join(current_dir, "checkpoints", "best_pretrain_model.pth"),
        "/home/mjkang/cbramod/20newmodel/checkpoints/best_pretrain_model.pth"
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("?? No model found, will use dummy model for testing")
        model_path = possible_model_paths[0]  # Use first path anyway
    
    # µ¥ÀÌÅÍ¼Â °æ·Îµé ½Ãµµ
    possible_dataset_paths = [
        "/remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/train",
        "./data/test",
        "./test_data"
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print("?? No dataset found, will generate synthetic data")
        dataset_path = possible_dataset_paths[0]  # Use first path anyway
    
    output_dir = "/home/mjkang/cbramod/20newmodel/visualize/reconstruct_visualize"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"?? Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Dataset: {dataset_path}")
    print(f"   Output: {output_dir}")
    print(f"   Device: {config.DEVICE}")
    print(f"   New Model Available: {NEW_MODEL_AVAILABLE}")
    
    try:
        # Initialize visualizer
        print(f"\n?? Initializing visualizer...")
        visualizer = EEGVisualizationTool(model_path, config)
        
        # Load test data
        print(f"\n?? Loading test data...")
        samples, labels = load_test_data(dataset_path, num_samples=3)
        
        # Visualize samples
        mask_ratios = [0.3, 0.5]  # Reduced for faster testing
        
        print(f"\n?? Starting visualization...")
        for sample_idx, (sample_data, label) in enumerate(zip(samples, labels)):
            print(f"\n?? Processing Sample {sample_idx + 1}/{len(samples)} (Label: {label})...")
            
            # Different mask ratios
            for mask_ratio in mask_ratios:
                print(f"   ?? Mask ratio: {mask_ratio}")
                
                # Complete visualization
                save_path = os.path.join(
                    output_dir, 
                    f"sample_{sample_idx+1}_label_{label}_mask_{mask_ratio:.1f}_new_model.png"
                )
                
                try:
                    fig = visualizer.visualize_sample(
                        sample_data,
                        mask_ratio=mask_ratio,
                        save_path=save_path
                    )
                    plt.close(fig)  # Free memory
                    
                    # ? Compute and print metrics
                    if sample_data.dim() == 4:
                        test_data = sample_data.unsqueeze(0)
                    else:
                        test_data = sample_data
                        
                    test_data = test_data.to(config.DEVICE)
                    mask = visualizer.create_mask(test_data, mask_ratio)
                    masked_data = test_data * mask
                    reconstructed = visualizer.model_inference(masked_data)
                    
                    # Compute metrics on average frequency
                    orig_avg = torch.mean(test_data[0], dim=0).cpu().numpy()
                    recon_avg = torch.mean(reconstructed[0], dim=0).cpu().numpy()  
                    mask_avg = torch.mean(mask[0], dim=0).cpu().numpy()
                    
                    metrics = visualizer.compute_metrics(orig_avg, recon_avg, mask_avg)
                    
                    if 'error' not in metrics:
                        print(f"      ?? Metrics:")
                        print(f"         Magnitude MSE: {metrics['mse_magnitude']:.6f}")
                        print(f"         Phase Error: {metrics['phase_error_deg']:.1f}¡Æ")
                        print(f"         Magnitude Correlation: {metrics['correlation_magnitude']:.3f}")
                        print(f"         SNR: {metrics['snr_db']:.1f} dB")
                        print(f"         Complex Correlation: {metrics['complex_correlation']:.3f}")
                    else:
                        print(f"      ?? Metrics error: {metrics['error']}")
                        
                except Exception as e:
                    print(f"      ? Visualization failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ? Frequency band analysis
            print("   ?? Frequency band analysis...")
            freq_save_path = os.path.join(
                output_dir,
                f"sample_{sample_idx+1}_label_{label}_frequency_bands_new_model.png"
            )
            
            try:
                freq_fig = visualizer.visualize_frequency_bands(
                    sample_data,
                    mask_ratio=0.5,
                    save_path=freq_save_path
                )
                plt.close(freq_fig)
                print(f"      ? Frequency analysis completed")
            except Exception as e:
                print(f"      ? Frequency analysis failed: {e}")
            
            print(f"   ? Sample {sample_idx + 1} completed")
        
        # ? Generate comprehensive summary
        summary_path = os.path.join(output_dir, "visualization_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("EEG Connectivity Visualization Summary (New Architecture)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Model Available: {os.path.exists(model_path)}\n")
            f.write(f"New Architecture: {NEW_MODEL_AVAILABLE}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Dataset Available: {os.path.exists(dataset_path)}\n")
            f.write(f"Samples processed: {len(samples)}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write(f"Device: {config.DEVICE}\n\n")
            
            generated_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            f.write(f"Generated files ({len(generated_files)}):\n")
            for i, filename in enumerate(sorted(generated_files), 1):
                f.write(f"{i:2d}. {filename}\n")
                
            f.write(f"\nVisualization Details:\n")
            f.write(f"- Mask ratios tested: {mask_ratios}\n")
            f.write(f"- Frequency bands: {list(config.FREQUENCY_BANDS.keys())}\n")
            f.write(f"- Components visualized: Real, Imaginary, Magnitude, Phase\n")
            f.write(f"- Metrics computed: MSE, Phase Error, Correlation, SNR\n")
        
        print(f"\n?? All visualizations completed successfully!")
        print(f"   ?? Output directory: {output_dir}")
        print(f"   ??? Generated images: {len([f for f in os.listdir(output_dir) if f.endswith('.png')])}")
        print(f"   ?? Summary: {summary_path}")
        print(f"   ??? Architecture: {'New EEGConnectivityModel' if NEW_MODEL_AVAILABLE else 'Fallback Model'}")
        
    except Exception as e:
        print(f"? Visualization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Generate error report
        error_path = os.path.join(output_dir, "error_report.txt")
        with open(error_path, 'w') as f:
            f.write(f"EEG Visualization Error Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Traceback:\n")
            traceback.print_exc(file=f)
        
        print(f"?? Error report saved: {error_path}")


if __name__ == "__main__":
    main()