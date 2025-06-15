#!/usr/bin/env python3
# clean_gpu_normalizer.py - ±ò²ûÇÑ GPU ÃÖÀûÈ­ CSD Á¤±ÔÈ­±â

import os
import pickle
import glob
import time
import psutil
import gc
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import logging
import warnings

warnings.filterwarnings('ignore')

# ·Î±ë ¼³Á¤
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUCSDNormalizer:
    """
    GPU ÃÖÀûÈ­ CSD Á¤±ÔÈ­±â
    - CPU: 90% ÀÌÇÏ »ç¿ë
    - GPU: ÃÖ´ëÇÑ È°¿ë
    - °ËÁõµÈ frequency_aware_preserve ¹æ¹ý
    """
    
    def __init__(self):
        # GPU ¼³Á¤
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            # GPU ¸Þ¸ð¸® ÃÖ´ë È°¿ë
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
            
            # GPU ¸Þ¸ð¸® ±â¹Ý ¹èÄ¡ Å©±â ÀÚµ¿ ¼³Á¤
            if gpu_memory >= 40:
                self.batch_size = 128
            elif gpu_memory >= 20:
                self.batch_size = 64
            else:
                self.batch_size = 32
            
            logger.info(f"?? GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"? Batch size: {self.batch_size}")
        else:
            self.batch_size = 8
            logger.info("?? Using CPU")
        
        # ÁÖÆÄ¼ö Áß¿äµµ (°ËÁõµÈ °ª)
        freq_weights = [0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.0, 1.8, 2.2, 2.5, 2.2, 1.8, 1.5, 1.2, 1.0]
        self.freq_importance = torch.tensor(freq_weights, device=self.device, dtype=torch.float32)
        
        # Á¤±ÔÈ­ ÆÄ¶ó¹ÌÅÍ
        self.norm_params = None
    
    def check_cpu_memory(self):
        """CPU ¸Þ¸ð¸® 90% Á¦ÇÑ"""
        cpu_usage = psutil.virtual_memory().percent
        if cpu_usage > 90:
            logger.warning(f"?? CPU memory: {cpu_usage:.1f}%")
            gc.collect()
            time.sleep(0.5)
            return False
        return True
    
    def load_file(self, filepath):
        """ÆÄÀÏ ¾ÈÀü ·Îµå"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    
    def extract_csd(self, data):
        """CSD µ¥ÀÌÅÍ ÃßÃâ"""
        if isinstance(data, dict) and 'csd' in data:
            csd = data['csd']
        elif isinstance(data, np.ndarray):
            csd = data
        else:
            return None
        
        # ÇüÅÂ °ËÁõ
        if not isinstance(csd, np.ndarray) or csd.shape != (15, 19, 19, 2):
            return None
        
        # À¯ÇÑ°ª Ã¼Å©
        if not np.isfinite(csd).any():
            return None
        
        return csd
    
    def normalize_csd_gpu(self, csd_tensor):
        """GPU¿¡¼­ CSD Á¤±ÔÈ­ (°ËÁõµÈ ¹æ¹ý)"""
        
        # ¹èÄ¡ Â÷¿ø Ãß°¡
        if csd_tensor.dim() == 4:
            csd_tensor = csd_tensor.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        batch_size = csd_tensor.shape[0]
        result = csd_tensor.clone()
        
        # ÁÖÆÄ¼öº° Á¤±ÔÈ­
        for freq_idx in range(15):
            importance = self.freq_importance[freq_idx].item()
            
            # ÇØ´ç ÁÖÆÄ¼ö µ¥ÀÌÅÍ ÃßÃâ
            freq_data = csd_tensor[:, freq_idx, :, :, :]
            flat_data = freq_data.reshape(-1)
            finite_mask = torch.isfinite(flat_data)
            
            if not finite_mask.any():
                continue
            
            finite_data = flat_data[finite_mask]
            if len(finite_data) < 10:
                continue
            
            # Áß¿äµµº° ºÐÀ§¼ö ¼³Á¤
            if importance > 1.5:  # Alpha, Theta
                p_low, p_high = 0.005, 0.995
                scale = 4.0 * importance
            else:  # Delta, Beta
                p_low, p_high = 0.02, 0.98
                scale = 3.0
            
            # ºÐÀ§¼ö °è»ê
            try:
                q_low = torch.quantile(finite_data, p_low)
                q_high = torch.quantile(finite_data, p_high)
            except:
                q_low = torch.min(finite_data)
                q_high = torch.max(finite_data)
            
            # ½ºÄÉÀÏ¸µ
            data_range = q_high - q_low
            if data_range > 1e-10:
                scale_factor = scale / (data_range / 2.0)
                scale_factor = torch.clamp(scale_factor, 1e-6, 10.0)
                result[:, freq_idx, :, :, :] = freq_data * scale_factor
        
        # ÃÖÁ¾ ¾ÈÀüÀåÄ¡
        result_flat = result.reshape(-1)
        finite_result = result_flat[torch.isfinite(result_flat)]
        
        if len(finite_result) > 0:
            final_range = torch.max(finite_result) - torch.min(finite_result)
            if final_range > 15:
                result = result * (12.0 / final_range)
        
        # ¿ø·¡ ÇüÅÂ·Î º¹¿ø
        if single_sample:
            result = result.squeeze(0)
        
        return result
    
    def apply_hermitian(self, csd_tensor):
        """¿¡¸£¹ÌÆ® ´ëÄª¼º º¹¿ø"""
        
        if csd_tensor.dim() == 4:  # ´ÜÀÏ »ùÇÃ
            freq_dim, h, w, _ = csd_tensor.shape
            result = csd_tensor.clone()
            
            for freq_idx in range(freq_dim):
                real_part = csd_tensor[freq_idx, :, :, 0]
                imag_part = csd_tensor[freq_idx, :, :, 1]
                
                complex_matrix = torch.complex(real_part, imag_part)
                hermitian = (complex_matrix + complex_matrix.conj().T) / 2
                
                result[freq_idx, :, :, 0] = hermitian.real
                result[freq_idx, :, :, 1] = hermitian.imag
            
            return result
        
        elif csd_tensor.dim() == 5:  # ¹èÄ¡
            batch_size, freq_dim, h, w, _ = csd_tensor.shape
            result = csd_tensor.clone()
            
            for b in range(batch_size):
                for freq_idx in range(freq_dim):
                    real_part = csd_tensor[b, freq_idx, :, :, 0]
                    imag_part = csd_tensor[b, freq_idx, :, :, 1]
                    
                    complex_matrix = torch.complex(real_part, imag_part)
                    hermitian = (complex_matrix + complex_matrix.conj().T) / 2
                    
                    result[b, freq_idx, :, :, 0] = hermitian.real
                    result[b, freq_idx, :, :, 1] = hermitian.imag
            
            return result
        
        return csd_tensor
    
    def calculate_stats(self, train_files, max_files=1000):
        """Á¤±ÔÈ­ Åë°è °è»ê"""
        
        logger.info(f"?? Calculating statistics from {len(train_files)} files...")
        
        # ÆÄÀÏ »ùÇÃ¸µ
        sample_files = train_files[:max_files] if len(train_files) > max_files else train_files
        
        all_samples = []
        freq_samples = [[] for _ in range(15)]
        processed = 0
        
        for filepath in tqdm(sample_files, desc="Sampling"):
            if not self.check_cpu_memory():
                break
            
            data = self.load_file(filepath)
            if data is None:
                continue
            
            csd = self.extract_csd(data)
            if csd is None:
                continue
            
            processed += 1
            
            # ÀüÃ¼ »ùÇÃ¸µ
            flat_csd = csd.flatten()
            finite_csd = flat_csd[np.isfinite(flat_csd)]
            
            if len(finite_csd) > 100:
                sample_size = min(300, len(finite_csd))
                indices = np.random.choice(len(finite_csd), sample_size, replace=False)
                all_samples.extend(finite_csd[indices].tolist())
            
            # ÁÖÆÄ¼öº° »ùÇÃ¸µ
            for freq_idx in range(15):
                freq_data = csd[freq_idx].flatten()
                finite_freq = freq_data[np.isfinite(freq_data)]
                
                if len(finite_freq) > 50:
                    importance = self.freq_importance[freq_idx].item()
                    sample_size = min(100 if importance > 1.5 else 50, len(finite_freq))
                    indices = np.random.choice(len(finite_freq), sample_size, replace=False)
                    freq_samples[freq_idx].extend(finite_freq[indices].tolist())
        
        logger.info(f"?? Processed {processed} files, collected {len(all_samples)} samples")
        
        if len(all_samples) < 1000:
            logger.error(f"? Insufficient samples: {len(all_samples)}")
            return False
        
        # GPU¿¡¼­ Åë°è °è»ê
        logger.info("?? Computing statistics on GPU...")
        
        all_tensor = torch.tensor(all_samples, device=self.device, dtype=torch.float32)
        global_stats = {
            'mean': float(torch.mean(all_tensor)),
            'std': float(torch.std(all_tensor)),
            'min': float(torch.min(all_tensor)),
            'max': float(torch.max(all_tensor)),
            'p01': float(torch.quantile(all_tensor, 0.01)),
            'p99': float(torch.quantile(all_tensor, 0.99))
        }
        
        # ÁÖÆÄ¼öº° Åë°è
        freq_stats = {}
        for freq_idx in range(15):
            if len(freq_samples[freq_idx]) > 20:
                freq_tensor = torch.tensor(freq_samples[freq_idx], device=self.device)
                importance = self.freq_importance[freq_idx].item()
                
                if importance > 1.5:
                    p_low, p_high = 0.005, 0.995
                else:
                    p_low, p_high = 0.02, 0.98
                
                freq_stats[freq_idx] = {
                    'importance': importance,
                    'p_low': float(torch.quantile(freq_tensor, p_low)),
                    'p_high': float(torch.quantile(freq_tensor, p_high)),
                    'samples': len(freq_samples[freq_idx])
                }
            else:
                freq_stats[freq_idx] = {
                    'importance': self.freq_importance[freq_idx].item(),
                    'p_low': global_stats['p01'],
                    'p_high': global_stats['p99'],
                    'samples': 0
                }
        
        self.norm_params = {
            'global': global_stats,
            'frequency': freq_stats,
            'method': 'frequency_aware_preserve',
            'version': 'clean_gpu',
            'created': datetime.now().isoformat()
        }
        
        # ÀúÀå
        with open('normalization_params_clean.json', 'w') as f:
            json.dump(self.norm_params, f, indent=2)
        
        logger.info(f"? Statistics saved")
        logger.info(f"   Range: [{global_stats['min']:.2f}, {global_stats['max']:.2f}]")
        
        return True
    
    def process_batch(self, files, output_dir):
        """¹èÄ¡ ÆÄÀÏ Ã³¸®"""
        
        stats = {'success': 0, 'failed': 0}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # ¹èÄ¡ µ¥ÀÌÅÍ ÁØºñ
        batch_data = []
        batch_originals = []
        batch_names = []
        
        for filepath in files:
            if not self.check_cpu_memory():
                break
            
            data = self.load_file(filepath)
            if data is None:
                stats['failed'] += 1
                continue
            
            csd = self.extract_csd(data)
            if csd is None:
                stats['failed'] += 1
                continue
            
            batch_data.append(csd)
            batch_originals.append(data)
            batch_names.append(Path(filepath).name)
        
        if not batch_data:
            return stats
        
        try:
            # GPU·Î ÀÌµ¿
            batch_tensor = torch.tensor(np.stack(batch_data), device=self.device, dtype=torch.float32)
            
            # Á¤±ÔÈ­
            normalized = self.normalize_csd_gpu(batch_tensor)
            normalized = self.apply_hermitian(normalized)
            
            # CPU·Î ÀÌµ¿
            normalized_cpu = normalized.cpu().numpy()
            
            # ÀúÀå
            for i, (original, normalized_csd, filename) in enumerate(zip(batch_originals, normalized_cpu, batch_names)):
                try:
                    if isinstance(original, dict):
                        result = original.copy()
                        result['csd'] = normalized_csd
                        result['normalization_info'] = {
                            'method': 'frequency_aware_preserve_clean',
                            'applied_at': datetime.now().isoformat(),
                            'gpu_processed': True
                        }
                    else:
                        result = normalized_csd
                    
                    output_file = output_path / filename
                    with open(output_file, 'wb') as f:
                        pickle.dump(result, f)
                    
                    stats['success'] += 1
                except Exception as e:
                    logger.debug(f"Save error: {e}")
                    stats['failed'] += 1
        
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            stats['failed'] += len(batch_data)
        
        finally:
            # GPU ¸Þ¸ð¸® Á¤¸®
            if 'batch_tensor' in locals():
                del batch_tensor
            if 'normalized' in locals():
                del normalized
            torch.cuda.empty_cache()
        
        return stats
    
    def process_files(self, files, output_dir, desc):
        """ÆÄÀÏµé Ã³¸®"""
        
        total_stats = {'success': 0, 'failed': 0}
        
        logger.info(f"?? Processing {len(files)} files with batch size {self.batch_size}")
        
        for i in tqdm(range(0, len(files), self.batch_size), desc=desc):
            batch_files = files[i:i + self.batch_size]
            batch_stats = self.process_batch(batch_files, output_dir)
            
            total_stats['success'] += batch_stats['success']
            total_stats['failed'] += batch_stats['failed']
        
        return total_stats
    
    def find_files(self, pattern):
        """È£È¯ ÆÄÀÏ Ã£±â"""
        
        all_files = glob.glob(pattern)
        if not all_files:
            return []
        
        logger.info(f"?? Found {len(all_files)} files, checking compatibility...")
        
        # »ùÇÃ Ã¼Å©
        sample_files = all_files[:20] if len(all_files) > 20 else all_files
        compatible = 0
        
        for filepath in sample_files:
            data = self.load_file(filepath)
            if data is not None and self.extract_csd(data) is not None:
                compatible += 1
        
        rate = compatible / len(sample_files)
        logger.info(f"? Compatibility: {rate*100:.1f}%")
        
        if rate < 0.5:
            logger.error(f"? Low compatibility: {rate*100:.1f}%")
            return []
        
        return all_files
    
    def verify_output(self, output_dir):
        """°á°ú °ËÁõ"""
        
        train_files = glob.glob(os.path.join(output_dir, "train", "*.pkl"))
        if not train_files:
            logger.warning("No output files for verification")
            return
        
        logger.info("?? Verifying results...")
        
        ranges = []
        for filepath in train_files[:3]:
            try:
                data = self.load_file(filepath)
                if data is not None:
                    csd = self.extract_csd(data)
                    if csd is not None:
                        finite_data = csd[np.isfinite(csd)]
                        if len(finite_data) > 0:
                            data_range = np.ptp(finite_data)
                            ranges.append(data_range)
                            logger.info(f"   ?? {Path(filepath).name}: range={data_range:.2f}")
            except Exception as e:
                logger.debug(f"Verification error: {e}")
        
        if ranges:
            avg_range = np.mean(ranges)
            logger.info(f"?? Average range: {avg_range:.2f}")
            
            if avg_range < 15:
                logger.info("   ? Safety: EXCELLENT")
            else:
                logger.warning("   ?? Safety: CHECK NEEDED")
    
    def normalize_dataset(self, train_pattern, val_pattern=None, output_dir="./normalized_clean"):
        """ÀüÃ¼ µ¥ÀÌÅÍ¼Â Á¤±ÔÈ­"""
        
        logger.info("?? CLEAN GPU CSD NORMALIZATION")
        logger.info("=" * 50)
        
        # 1. Train ÆÄÀÏ Ã£±â
        logger.info("\n?? Step 1: Find train files")
        train_files = self.find_files(train_pattern)
        if not train_files:
            logger.error("? No train files found")
            return
        
        logger.info(f"? Found {len(train_files)} train files")
        
        # 2. Åë°è °è»ê
        logger.info("\n?? Step 2: Calculate statistics")
        if not self.calculate_stats(train_files):
            logger.error("? Failed to calculate statistics")
            return
        
        # 3. Train Ã³¸®
        logger.info("\n?? Step 3: Process train files")
        train_stats = self.process_files(
            train_files, 
            os.path.join(output_dir, "train"), 
            "Train"
        )
        logger.info(f"Train: {train_stats}")
        
        # 4. Val Ã³¸® (¼±ÅÃÀû)
        if val_pattern:
            logger.info("\n?? Step 4: Process val files")
            val_files = self.find_files(val_pattern)
            if val_files:
                val_stats = self.process_files(
                    val_files,
                    os.path.join(output_dir, "val"),
                    "Val"
                )
                logger.info(f"Val: {val_stats}")
        
        # 5. °ËÁõ
        self.verify_output(output_dir)
        
        logger.info("\n? NORMALIZATION COMPLETED!")

def main():
    """¸ÞÀÎ ÇÔ¼ö"""
    
    print("?? Clean GPU CSD Normalizer")
    print("=" * 40)
    
    # Á¤±ÔÈ­±â ÃÊ±âÈ­
    normalizer = GPUCSDNormalizer()
    
    
    # µ¥ÀÌÅÍ °æ·Î
    base_path = "/remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/var_scaled"
    
    train_pattern = os.path.join(base_path, "train", "*.pkl")
    val_pattern = os.path.join(base_path, "val", "*.pkl")
    
    # °æ·Î È®ÀÎ
    if not glob.glob(train_pattern):
        logger.error(f"? No files: {train_pattern}")
        
        # ´ë¾È ½Ãµµ
        alt_patterns = [
            "/remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/1000sample/train/*.pkl",
        ]
        
        for alt in alt_patterns:
            if glob.glob(alt):
                train_pattern = alt
                val_pattern = alt.replace("/train/", "/val/")
                logger.info(f"? Using: {alt}")
                break
        else:
            logger.error("? No data found!")
            return
    
    # Á¤±ÔÈ­ ½ÇÇà
    normalizer.normalize_dataset(
        train_pattern=train_pattern,
        val_pattern=val_pattern,
        output_dir=" /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_gpu_clean"
    )
    
    print("\n?? Clean GPU normalization completed!")

if __name__ == "__main__":
    main()