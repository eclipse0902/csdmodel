"""
EEG Connectivity Analysis - Fixed Dataset Module

ÇÙ½É ¼³°è ¿øÄ¢:
1. ´Ü¼øÇÏ°í È¿°úÀûÀÎ µ¥ÀÌÅÍ ·Îµù
2. º¹ÀâÇÑ ¸¶½ºÅ· Á¦°Å (trainer¿¡¼­ Ã³¸®)
3. °­·ÂÇÑ µ¥ÀÌÅÍ °ËÁõ
4. Config ±â¹Ý ¼³Á¤
"""

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import glob
import os
from typing import List, Tuple, Dict, Optional, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_gnn import EEGConfig

class EEGDataset(Dataset):
    """
    ´Ü¼øÈ­µÈ EEG Dataset
    
    ÇÙ½É Æ¯Â¡:
    1. .pkl ÆÄÀÏ¿¡¼­ CSD µ¥ÀÌÅÍ ·Îµù
    2. µ¥ÀÌÅÍ °ËÁõ ¹× Á¤±ÔÈ­
    3. Hermitian ´ëÄª¼º È®ÀÎ (°­Á¦ÇÏÁö ¾ÊÀ½)
    4. ¸Þ¸ð¸® È¿À²Àû ·Îµù
    """
    
    def __init__(self, 
                 data_path: str, 
                 config: EEGConfig = None,
                 apply_masking: bool = False,
                 normalize_data: bool = True,
                 validate_hermitian: bool = True):
        """
        Args:
            data_path: .pkl ÆÄÀÏµéÀÌ ÀÖ´Â µð·ºÅä¸® °æ·Î
            config: EEG configuration
            apply_masking: µ¥ÀÌÅÍ¼Â¿¡¼­ ¸¶½ºÅ· Àû¿ë ¿©ºÎ (º¸Åë False, trainer¿¡¼­ Ã³¸®)
            normalize_data: µ¥ÀÌÅÍ Á¤±ÔÈ­ ¿©ºÎ
            validate_hermitian: Hermitian ´ëÄª¼º °ËÁõ ¿©ºÎ
        """
        
        if config is None:
            config = EEGConfig()
            
        self.config = config
        self.data_path = data_path
        self.apply_masking = apply_masking
        self.normalize_data = normalize_data
        self.validate_hermitian = validate_hermitian
        
        # =============== FILE DISCOVERY ===============
        self.file_paths = self._discover_files()
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No .pkl files found in {data_path}")
        
        # =============== DATA KEY SETUP ===============
        self.data_key = config.DATA_CONFIG['data_key']
        
        # =============== DATASET STATISTICS (FIRST) ===============
        # Åë°è¸¦ ¸ÕÀú °è»êÇØ¾ß normalize_sample¿¡¼­ »ç¿ë °¡´É
        self.dataset_stats = self._compute_dataset_statistics()
        
        # =============== DATA VALIDATION (AFTER STATS) ===============
        # Åë°è °è»ê ÈÄ¿¡ °ËÁõ ¼öÇà
        self._validate_data_format()
        
        print(f"? EEG Dataset Initialized:")
        print(f"   Path: {data_path}")
        print(f"   Files: {len(self.file_paths)}")
        print(f"   Data key: '{self.data_key}'")
        print(f"   Apply masking: {apply_masking}")
        print(f"   Normalize: {normalize_data}")
        print(f"   Validate Hermitian: {validate_hermitian}")
        print(f"   Class distribution: {self.dataset_stats['class_distribution']}")
    
    def _discover_files(self) -> List[str]:
        """ÆÄÀÏ Å½»ö"""
        patterns = [
            os.path.join(self.data_path, "*.pkl"),
            os.path.join(self.data_path, "**/*.pkl")  # ÇÏÀ§ µð·ºÅä¸® Æ÷ÇÔ
        ]
        
        file_paths = []
        for pattern in patterns:
            file_paths.extend(glob.glob(pattern, recursive=True))
        
        # Áßº¹ Á¦°Å ¹× Á¤·Ä
        file_paths = sorted(list(set(file_paths)))
        
        print(f"?? File Discovery:")
        print(f"   Search patterns: {patterns}")
        print(f"   Found files: {len(file_paths)}")
        
        return file_paths
    
    def _load_sample_raw(self, idx: int) -> Tuple[torch.Tensor, int]:
        """´ÜÀÏ »ùÇÃ ·Îµù (Á¤±ÔÈ­ ¾øÀÌ raw µ¥ÀÌÅÍ)"""
        file_path = self.file_paths[idx]
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # CSD µ¥ÀÌÅÍ ÃßÃâ
            if self.data_key not in data:
                available_keys = list(data.keys())
                raise KeyError(f"Key '{self.data_key}' not found. Available keys: {available_keys}")
            
            csd_data = data[self.data_key]
            label = int(data.get('label', 0))
            
            # Tensor º¯È¯
            if not isinstance(csd_data, torch.Tensor):
                csd_data = torch.from_numpy(csd_data) if isinstance(csd_data, np.ndarray) else torch.tensor(csd_data)
            
            # Float32·Î º¯È¯
            csd_data = csd_data.float()
            
            return csd_data, label
            
        except Exception as e:
            raise ValueError(f"Error loading {file_path}: {str(e)}")
    
    def _compute_dataset_statistics(self) -> Dict:
        """µ¥ÀÌÅÍ¼Â Åë°è °è»ê (Á¤±ÔÈ­ Àü¿¡ ¼öÇà)"""
        print(f"?? Computing dataset statistics...")
        
        labels = []
        magnitude_stats = []
        
        # ÀüÃ¼ µ¥ÀÌÅÍ ¼øÈ¸ (Å« µ¥ÀÌÅÍ¼ÂÀÇ °æ¿ì »ùÇÃ¸µ)
        sample_size = min(100, len(self.file_paths))  # ÃÖ´ë 100°³ »ùÇÃ
        sample_indices = random.sample(range(len(self.file_paths)), sample_size)
        
        for idx in sample_indices:
            try:
                csd_data, label = self._load_sample_raw(idx)
                labels.append(label)
                
                # Magnitude Åë°è
                magnitude = torch.sqrt(csd_data[..., 0]**2 + csd_data[..., 1]**2)
                magnitude_stats.append({
                    'mean': magnitude.mean().item(),
                    'std': magnitude.std().item(),
                    'max': magnitude.max().item(),
                    'min': magnitude.min().item()
                })
                
            except Exception as e:
                print(f"Warning: Failed to process file {idx}: {e}")
                continue
        
        if len(labels) == 0:
            # ±âº» Åë°è°ª
            return {
                'total_samples': len(self.file_paths),
                'sampled_for_stats': 0,
                'class_distribution': {0: 0, 1: 0},
                'class_balance': 0.5,
                'magnitude_stats': {'mean': 1.0, 'std': 1.0, 'min': 0.0, 'max': 10.0}
            }
        
        labels = np.array(labels)
        
        # Class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_distribution = {int(label): int(count) for label, count in zip(unique_labels, counts)}
        
        # Magnitude statistics
        if magnitude_stats:
            magnitude_means = [s['mean'] for s in magnitude_stats]
            magnitude_stds = [s['std'] for s in magnitude_stats]
            
            overall_magnitude_stats = {
                'mean': np.mean(magnitude_means),
                'std': np.mean(magnitude_stds),
                'min': min(s['min'] for s in magnitude_stats),
                'max': max(s['max'] for s in magnitude_stats)
            }
        else:
            overall_magnitude_stats = {'mean': 1.0, 'std': 1.0, 'min': 0.0, 'max': 10.0}
        
        stats = {
            'total_samples': len(self.file_paths),
            'sampled_for_stats': len(labels),
            'class_distribution': class_distribution,
            'class_balance': len(labels[labels == 1]) / len(labels) if len(labels) > 0 else 0.5,
            'magnitude_stats': overall_magnitude_stats
        }
        
        print(f"?? Dataset Statistics:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Class 0 (normal): {stats['class_distribution'].get(0, 0)}")
        print(f"   Class 1 (abnormal): {stats['class_distribution'].get(1, 0)}")
        print(f"   Class balance: {stats['class_balance']:.3f}")
        print(f"   Magnitude - Mean: {stats['magnitude_stats']['mean']:.3f}, Std: {stats['magnitude_stats']['std']:.3f}")
        
        return stats
    
    def _validate_data_format(self):
        """µ¥ÀÌÅÍ Çü½Ä °ËÁõ"""
        print(f"?? Validating data format...")
        
        # »ùÇÃ ÆÄÀÏµé °ËÁõ
        num_samples_to_check = min(5, len(self.file_paths))
        
        for i in range(num_samples_to_check):
            try:
                csd_data, label = self._load_sample_raw(i)  # raw µ¥ÀÌÅÍ ·Îµù
                
                # 1. Shape °ËÁõ
                expected_shape = (self.config.NUM_FREQUENCIES, 
                                self.config.NUM_ELECTRODES, 
                                self.config.NUM_ELECTRODES, 
                                self.config.NUM_COMPLEX_DIMS)
                
                if csd_data.shape != expected_shape:
                    raise ValueError(f"Shape mismatch in {self.file_paths[i]}: "
                                   f"expected {expected_shape}, got {csd_data.shape}")
                
                # 2. Label °ËÁõ
                if label not in [0, 1]:
                    raise ValueError(f"Invalid label in {self.file_paths[i]}: {label}")
                
                # 3. Data type °ËÁõ
                if not torch.is_floating_point(csd_data):
                    print(f"Warning: Converting data to float32 in {self.file_paths[i]}")
                
                # 4. NaN/Inf °ËÁõ
                if torch.isnan(csd_data).any() or torch.isinf(csd_data).any():
                    print(f"Warning: Found NaN/Inf values in {self.file_paths[i]}")
                
                # 5. Hermitian °ËÁõ (¼±ÅÃÀû)
                if self.validate_hermitian:
                    self._check_hermitian_properties(csd_data, i)
                
            except Exception as e:
                raise ValueError(f"Data validation failed for {self.file_paths[i]}: {str(e)}")
        
        print(f"? Data format validation passed ({num_samples_to_check} samples checked)")
    
    def _check_hermitian_properties(self, csd_data: torch.Tensor, file_idx: int):
        """Hermitian ´ëÄª¼º È®ÀÎ (°­Á¦ÇÏÁö ¾ÊÀ½)"""
        real_part = csd_data[:, :, :, 0]  # (15, 19, 19)
        imag_part = csd_data[:, :, :, 1]  # (15, 19, 19)
        
        # Real part ´ëÄª¼º È®ÀÎ
        real_symmetric_error = torch.mean(torch.abs(real_part - real_part.transpose(-1, -2)))
        
        # Imaginary part ¹Ý´ëÄª¼º È®ÀÎ
        imag_antisymmetric_error = torch.mean(torch.abs(imag_part + imag_part.transpose(-1, -2)))
        
        # Diagonal imaginary È®ÀÎ
        diag_imag_error = torch.mean(torch.abs(torch.diagonal(imag_part, dim1=-2, dim2=-1)))
        
        # Çã¿ë ¿ÀÂ÷
        tolerance = 1e-6
        
        hermitian_quality = {
            'real_symmetric': real_symmetric_error.item() < tolerance,
            'imag_antisymmetric': imag_antisymmetric_error.item() < tolerance,
            'diag_imag_zero': diag_imag_error.item() < tolerance
        }
        
        # °æ°í¸¸ Ãâ·Â (°­Á¦ÇÏÁö ¾ÊÀ½)
        if not all(hermitian_quality.values()):
            print(f"Info: File {file_idx} has approximate Hermitian properties:")
            print(f"  Real symmetric: {hermitian_quality['real_symmetric']} (error: {real_symmetric_error.item():.2e})")
            print(f"  Imag antisymmetric: {hermitian_quality['imag_antisymmetric']} (error: {imag_antisymmetric_error.item():.2e})")
            print(f"  Diag imag zero: {hermitian_quality['diag_imag_zero']} (error: {diag_imag_error.item():.2e})")
    
    def _load_sample(self, idx: int) -> Tuple[torch.Tensor, int]:
        """´ÜÀÏ »ùÇÃ ·Îµù (Á¤±ÔÈ­ Æ÷ÇÔ)"""
        # Raw µ¥ÀÌÅÍ ·Îµù
        csd_data, label = self._load_sample_raw(idx)
        
        # Á¤±ÔÈ­ (¼±ÅÃÀû)
        if self.normalize_data:
            csd_data = self._normalize_sample(csd_data)
        
        return csd_data, label
    
    def _normalize_sample(self, csd_data: torch.Tensor) -> torch.Tensor:
        """»ùÇÃº° Á¤±ÔÈ­"""
        # Magnitude ±â¹Ý Á¤±ÔÈ­
        magnitude = torch.sqrt(csd_data[..., 0]**2 + csd_data[..., 1]**2 + 1e-8)
        
        # Global magnitude statistics »ç¿ë
        global_mean = self.dataset_stats['magnitude_stats']['mean']
        global_std = self.dataset_stats['magnitude_stats']['std']
        
        # Z-score normalization (magnitude ±âÁØ)
        mean_magnitude = magnitude.mean()
        normalized_magnitude = (mean_magnitude - global_mean) / (global_std + 1e-8)
        
        # °úµµÇÑ Á¤±ÔÈ­ ¹æÁö
        scale_factor = torch.clamp(torch.exp(normalized_magnitude), 0.1, 10.0)
        
        normalized_csd = csd_data / scale_factor
        
        return normalized_csd
    
    def __len__(self) -> int:
        """µ¥ÀÌÅÍ¼Â Å©±â"""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ´ÜÀÏ ¾ÆÀÌÅÛ ¹ÝÈ¯
        
        Args:
            idx: »ùÇÃ ÀÎµ¦½º
            
        Returns:
            csd_data: (15, 19, 19, 2) CSD connectivity data
            label: (1,) Label tensor
        """
        # µ¥ÀÌÅÍ ·Îµù
        csd_data, label = self._load_sample(idx)
        
        # ¸¶½ºÅ· Àû¿ë (º¸Åë »ç¿ëÇÏÁö ¾ÊÀ½, trainer¿¡¼­ Ã³¸®)
        if self.apply_masking:
            csd_data = self._apply_simple_masking(csd_data)
        
        # LabelÀ» tensor·Î º¯È¯
        label_tensor = torch.LongTensor([label])
        
        return csd_data, label_tensor
    
    def _apply_simple_masking(self, data: torch.Tensor, mask_ratio: float = 0.3) -> torch.Tensor:
        """°£´ÜÇÑ ¸¶½ºÅ· (º¸Åë »ç¿ëÇÏÁö ¾ÊÀ½)"""
        if random.random() > mask_ratio:
            return data
        
        masked_data = data.clone()
        num_positions = 19 * 19
        num_to_mask = int(num_positions * 0.1)  # 10% ¸¶½ºÅ·
        
        positions = random.sample(range(num_positions), num_to_mask)
        
        for pos in positions:
            i, j = pos // 19, pos % 19
            masked_data[:, i, j, :] = 0
        
        return masked_data
    
    def get_sample_by_class(self, class_label: int, num_samples: int = 5) -> List[Tuple[torch.Tensor, int]]:
        """Æ¯Á¤ Å¬·¡½ºÀÇ »ùÇÃµé ¹ÝÈ¯"""
        samples = []
        count = 0
        
        for idx in range(len(self.file_paths)):
            if count >= num_samples:
                break
                
            try:
                csd_data, label = self._load_sample(idx)
                if label == class_label:
                    samples.append((csd_data, label))
                    count += 1
            except:
                continue
        
        return samples
    
    def analyze_data_quality(self) -> Dict:
        """µ¥ÀÌÅÍ Ç°Áú ºÐ¼®"""
        print(f"?? Analyzing data quality...")
        
        quality_metrics = {
            'corrupt_files': 0,
            'nan_inf_files': 0,
            'shape_errors': 0,
            'label_errors': 0,
            'hermitian_violations': 0,
            'magnitude_outliers': 0
        }
        
        sample_size = min(50, len(self.file_paths))
        sample_indices = random.sample(range(len(self.file_paths)), sample_size)
        
        for idx in sample_indices:
            try:
                csd_data, label = self._load_sample_raw(idx)  # raw µ¥ÀÌÅÍ »ç¿ë
                
                # Shape È®ÀÎ
                expected_shape = (20, 19, 19, 2)
                if csd_data.shape != expected_shape:
                    quality_metrics['shape_errors'] += 1
                
                # Label È®ÀÎ
                if label not in [0, 1]:
                    quality_metrics['label_errors'] += 1
                
                # NaN/Inf È®ÀÎ
                if torch.isnan(csd_data).any() or torch.isinf(csd_data).any():
                    quality_metrics['nan_inf_files'] += 1
                
                # Magnitude ÀÌ»óÄ¡ È®ÀÎ
                magnitude = torch.sqrt(csd_data[..., 0]**2 + csd_data[..., 1]**2)
                if magnitude.max() > 100 or magnitude.min() < 0:
                    quality_metrics['magnitude_outliers'] += 1
                
                # Hermitian È®ÀÎ (¾ö°ÝÇÏÁö ¾ÊÀ½)
                real_part = csd_data[:, :, :, 0]
                real_sym_error = torch.mean(torch.abs(real_part - real_part.transpose(-1, -2)))
                if real_sym_error > 0.1:  # °ü´ëÇÑ ÀÓ°è°ª
                    quality_metrics['hermitian_violations'] += 1
                
            except Exception as e:
                quality_metrics['corrupt_files'] += 1
        
        # Ç°Áú Á¡¼ö °è»ê
        total_checks = len(sample_indices)
        quality_score = 1.0 - sum(quality_metrics.values()) / (total_checks * len(quality_metrics))
        
        quality_report = {
            **quality_metrics,
            'total_checked': total_checks,
            'quality_score': max(0.0, quality_score),
            'recommendations': []
        }
        
        # ±ÇÀå»çÇ×
        if quality_metrics['corrupt_files'] > 0:
            quality_report['recommendations'].append("Check for corrupt .pkl files")
        if quality_metrics['nan_inf_files'] > 0:
            quality_report['recommendations'].append("Handle NaN/Inf values in preprocessing")
        if quality_metrics['magnitude_outliers'] > 0:
            quality_report['recommendations'].append("Consider magnitude normalization")
        
        print(f"?? Data Quality Report:")
        print(f"   Quality Score: {quality_score:.3f}")
        print(f"   Corrupt files: {quality_metrics['corrupt_files']}/{total_checks}")
        print(f"   NaN/Inf files: {quality_metrics['nan_inf_files']}/{total_checks}")
        print(f"   Shape errors: {quality_metrics['shape_errors']}/{total_checks}")
        
        return quality_report

def create_data_loaders(config: EEGConfig, 
                       train_data_path: Optional[str] = None,
                       val_data_path: Optional[str] = None,
                       test_data_path: Optional[str] = None) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    µ¥ÀÌÅÍ ·Î´õ »ý¼º
    
    Args:
        config: EEG configuration
        train_data_path: ÈÆ·Ã µ¥ÀÌÅÍ °æ·Î (NoneÀÌ¸é config¿¡¼­ °¡Á®¿È)
        val_data_path: °ËÁõ µ¥ÀÌÅÍ °æ·Î
        test_data_path: Å×½ºÆ® µ¥ÀÌÅÍ °æ·Î
        
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # °æ·Î ¼³Á¤
    if train_data_path is None:
        train_data_path = config.DATA_CONFIG['train_data_path']
    if val_data_path is None:
        val_data_path = config.DATA_CONFIG.get('val_data_path')
    if test_data_path is None:
        test_data_path = config.DATA_CONFIG.get('test_data_path')
    
    print(f"?? Creating data loaders:")
    print(f"   Train: {train_data_path}")
    print(f"   Val: {val_data_path}")
    print(f"   Test: {test_data_path}")
    
    # µ¥ÀÌÅÍ¼Â »ý¼º
    train_dataset = EEGDataset(
        data_path=train_data_path,
        config=config,
        apply_masking=False,  # Trainer¿¡¼­ Ã³¸®
        normalize_data=True
    )
    
    val_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = EEGDataset(
            data_path=val_data_path,
            config=config,
            apply_masking=False,
            normalize_data=True
        )
    
    test_dataset = None
    if test_data_path and os.path.exists(test_data_path):
        test_dataset = EEGDataset(
            data_path=test_data_path,
            config=config,
            apply_masking=False,
            normalize_data=True
        )
    
    # ¹èÄ¡ Å©±â ¼³Á¤
    batch_size = config.TRAINING_CONFIG['batch_size']
    
    # µ¥ÀÌÅÍ ·Î´õ »ý¼º
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    # ¿ä¾à Ãâ·Â
    print(f"? Data loaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    if val_loader:
        print(f"   Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    if test_loader:
        print(f"   Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"   Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader

# Utility functions

def validate_data_directory(data_path: str, config: EEGConfig = None) -> Dict:
    """µ¥ÀÌÅÍ µð·ºÅä¸® °ËÁõ"""
    if not os.path.exists(data_path):
        raise ValueError(f"Data path does not exist: {data_path}")
    
    # ÀÓ½Ã µ¥ÀÌÅÍ¼ÂÀ¸·Î °ËÁõ
    try:
        temp_dataset = EEGDataset(data_path, config, apply_masking=False)
        quality_report = temp_dataset.analyze_data_quality()
        
        validation_result = {
            'valid': True,
            'num_files': len(temp_dataset),
            'quality_score': quality_report['quality_score'],
            'class_distribution': temp_dataset.dataset_stats['class_distribution'],
            'recommendations': quality_report['recommendations']
        }
        
    except Exception as e:
        validation_result = {
            'valid': False,
            'error': str(e),
            'num_files': 0,
            'quality_score': 0.0
        }
    
    return validation_result

def preview_dataset_samples(data_path: str, num_samples: int = 3, config: EEGConfig = None) -> Dict:
    """µ¥ÀÌÅÍ¼Â »ùÇÃ ¹Ì¸®º¸±â"""
    dataset = EEGDataset(data_path, config, apply_masking=False)
    
    preview = {
        'dataset_info': {
            'total_samples': len(dataset),
            'data_shape': (20, 19, 19, 2),
            'class_distribution': dataset.dataset_stats['class_distribution']
        },
        'samples': []
    }
    
    for i in range(min(num_samples, len(dataset))):
        csd_data, label = dataset[i]
        
        magnitude = torch.sqrt(csd_data[..., 0]**2 + csd_data[..., 1]**2)
        phase = torch.atan2(csd_data[..., 1], csd_data[..., 0])
        
        sample_info = {
            'index': i,
            'label': label.item(),
            'data_shape': list(csd_data.shape),
            'magnitude_stats': {
                'mean': magnitude.mean().item(),
                'std': magnitude.std().item(),
                'min': magnitude.min().item(),
                'max': magnitude.max().item()
            },
            'phase_stats': {
                'mean': phase.mean().item(),
                'std': phase.std().item()
            }
        }
        
        preview['samples'].append(sample_info)
    
    return preview

if __name__ == "__main__":
    print("="*80)
    print("? FIXED EEG DATASET")
    print("="*80)
    
    # Å×½ºÆ® ¼³Á¤ (½ÇÁ¦ µ¥ÀÌÅÍ °æ·Î·Î º¯°æ ÇÊ¿ä)
    test_data_path = "/path/to/test/data"  # ½ÇÁ¦ °æ·Î·Î º¯°æ
    
    config = EEGConfig()
    
    # Mock dataset for testing
    if not os.path.exists(test_data_path):
        print("?? Real data path not found, running with mock data...")
        
        # Create mock data for testing
        mock_data_dir = "/tmp/mock_eeg_data"
        os.makedirs(mock_data_dir, exist_ok=True)
        
        # Create mock .pkl files
        for i in range(10):
            mock_csd = np.random.randn(20, 19, 19, 2).astype(np.float32)
            mock_label = random.randint(0, 1)
            
            mock_data = {
                'csd': mock_csd,
                'label': mock_label,
                'frequency': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50])
            }
            
            with open(os.path.join(mock_data_dir, f"sample_{i:03d}.pkl"), 'wb') as f:
                pickle.dump(mock_data, f)
        
        test_data_path = mock_data_dir
        print(f"? Created mock data: {test_data_path}")
    
    # µ¥ÀÌÅÍ¼Â Å×½ºÆ®
    try:
        print(f"\n?? Testing EEG Dataset:")
        dataset = EEGDataset(
            data_path=test_data_path,
            config=config,
            apply_masking=False,
            normalize_data=True
        )
        
        print(f"? Dataset created successfully:")
        print(f"   Samples: {len(dataset)}")
        
        # »ùÇÃ Å×½ºÆ®
        sample_csd, sample_label = dataset[0]
        print(f"   Sample shape: {sample_csd.shape}")
        print(f"   Sample label: {sample_label}")
        
        # µ¥ÀÌÅÍ Ç°Áú ºÐ¼®
        quality_report = dataset.analyze_data_quality()
        print(f"\n?? Data Quality Analysis:")
        print(f"   Quality Score: {quality_report['quality_score']:.3f}")
        print(f"   Recommendations: {quality_report['recommendations']}")
        
        # µ¥ÀÌÅÍ ·Î´õ Å×½ºÆ®
        print(f"\n?? Testing Data Loader:")
        train_loader, val_loader, test_loader = create_data_loaders(
            config, train_data_path=test_data_path
        )
        
        # Ã¹ ¹øÂ° ¹èÄ¡ Å×½ºÆ®
        for batch_csd, batch_labels in train_loader:
            print(f"   Batch CSD shape: {batch_csd.shape}")
            print(f"   Batch labels shape: {batch_labels.shape}")
            print(f"   Labels: {batch_labels.flatten().tolist()}")
            break
        
        # µ¥ÀÌÅÍ ¹Ì¸®º¸±â
        print(f"\n?? Dataset Preview:")
        preview = preview_dataset_samples(test_data_path, num_samples=3, config=config)
        
        for sample in preview['samples']:
            print(f"   Sample {sample['index']}: Label={sample['label']}, "
                  f"Mag mean={sample['magnitude_stats']['mean']:.3f}")
        
    except Exception as e:
        print(f"? Dataset test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("? FIXED DATASET TESTING COMPLETED")
    print("="*80)