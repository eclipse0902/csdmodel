""" # ½ÇÁ¦ Ã³¸® ¾øÀÌ ½Ã¹Ä·¹ÀÌ¼Ç¸¸
python /home/mjkang/cbramod/preprocess_var_20fq_scaled.py --dry_run"""
import os
import numpy as np
import mne
import pandas as pd
import glob
import json
import pickle
from scipy import signal as scipy_signal
from sklearn.model_selection import train_test_split
from statsmodels.tsa.vector_ar.var_model import VAR
from tqdm import tqdm
import logging
import warnings
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time
import signal
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
import gc
import psutil
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import traceback
from contextlib import contextmanager

# °æ°í ¸Þ½ÃÁö ¾ïÁ¦
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in exp")
warnings.filterwarnings("ignore", category=UserWarning, message="The unit has changed from")

@dataclass
class Config:
    """¼³Á¤ Å¬·¡½º - ´õ Ã¼°èÀûÀÎ ±¸Á¶·Î °³¼±"""
    
    # EEG Ã¤³Î
    TARGET_CHANNELS: List[str] = None
    
    # Ã³¸® ¼³Á¤
    SEGMENT_DURATION: float = 4.0
    VAR_ORDER: int = 4
    TIMEOUT: int = 30
    
    # ÁÖÆÄ¼ö ¼³Á¤ - ´õ ¼¼¹ÐÇÑ ÁÖÆÄ¼ö ¹êµå
    FREQS: np.ndarray = None
    
    # ¸Þ¸ð¸® ¹× ¸®¼Ò½º ¼³Á¤
    MAX_CPU_MEMORY_USAGE: float = 85.0  # 90%¿¡¼­ 85%·Î Á¶Á¤
    MAX_CPU_USAGE: float = 70.0  # 60%¿¡¼­ 70%·Î Á¶Á¤
    CPU_CHECK_INTERVAL: int = 5  # 10ÃÊ¿¡¼­ 5ÃÊ·Î ´ÜÃà
    MEMORY_CHECK_INTERVAL: int = 3  # 5ÃÊ¿¡¼­ 3ÃÊ·Î ´ÜÃà
    
    # GPU ¼³Á¤
    GPU_MEMORY_FRACTION: float = 0.7  # 80%¿¡¼­ 70%·Î Á¶Á¤ (¾ÈÁ¤¼º)
    
    # Ã³¸® ¼³Á¤
    WORKER_THREADS: int = 1
    FILE_BATCH_SIZE: int = 2  # 3¿¡¼­ 2·Î ÁÙÀÓ (¸Þ¸ð¸® ¾ÈÁ¤¼º)
    
    # ´ÜÀ§ º¯È¯
    MICROVOLTS_TO_VOLTS_SQUARED_SCALE: float = 1e12
    
    # Ç°Áú °ü¸® ¼³Á¤ (»õ·Î Ãß°¡)
    MIN_SEGMENT_QUALITY_SCORE: float = 0.7
    MAX_ARTIFACT_THRESHOLD: float = 100.0  # ¥ìV
    
    def __post_init__(self):
        if self.TARGET_CHANNELS is None:
            self.TARGET_CHANNELS = [
                'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
                'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
                'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
                'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
            ]
        
        if self.FREQS is None:
            # ´õ ¼¼¹ÐÇÑ ÁÖÆÄ¼ö ºÐÇØ´ÉÀ¸·Î °³¼±
            self.FREQS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  # µ¨Å¸-¾ËÆÄ
                                  13, 15, 17, 20, 22, 25,                    # º£Å¸1
                                  28, 30, 32, 35,                           # º£Å¸2
                                  38, 40, 42, 45, 48, 50])                  # °¨¸¶

class DataQualityChecker:
    """µ¥ÀÌÅÍ Ç°Áú °Ë»ç Å¬·¡½º (»õ·Î Ãß°¡)"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def check_segment_quality(self, data: np.ndarray, sfreq: float) -> Dict[str, Any]:
        """¼¼±×¸ÕÆ® Ç°Áú °Ë»ç"""
        n_times, n_channels = data.shape
        
        # ±âº» Åë°è
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        max_vals = np.max(np.abs(data), axis=0)
        
        # ¾ÆÆ¼ÆÑÆ® °ËÃâ
        artifact_mask = max_vals > self.config.MAX_ARTIFACT_THRESHOLD
        artifact_ratio = np.sum(artifact_mask) / n_channels
        
        # ½ÅÈ£ ´ë ÀâÀ½ºñ ÃßÁ¤
        signal_power = np.mean(std_vals**2)
        noise_estimate = np.median(std_vals)  # ´õ robustÇÑ ÃßÁ¤
        snr = signal_power / (noise_estimate**2 + 1e-8)
        
        # Ã¤³Î °£ »ó°ü°ü°è È®ÀÎ
        correlation_matrix = np.corrcoef(data.T)
        mean_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        
        # ÁÖÆÄ¼ö µµ¸ÞÀÎ ºÐ¼®
        freqs, psd = scipy_signal.welch(data, fs=sfreq, axis=0, nperseg=min(256, n_times//4))
        
        # ¾ËÆÄ ÆÄ¿ö ºñÀ² (8-12Hz)
        alpha_idx = (freqs >= 8) & (freqs <= 12)
        total_power = np.sum(psd, axis=0)
        alpha_power = np.sum(psd[alpha_idx], axis=0)
        alpha_ratio = np.mean(alpha_power / (total_power + 1e-8))
        
        # Ç°Áú Á¡¼ö °è»ê
        quality_score = self._calculate_quality_score(
            artifact_ratio, snr, mean_correlation, alpha_ratio
        )
        
        return {
            "quality_score": quality_score,
            "artifact_ratio": artifact_ratio,
            "snr": float(snr),
            "mean_correlation": float(mean_correlation),
            "alpha_ratio": float(alpha_ratio),
            "max_amplitude": float(np.max(max_vals)),
            "passes_quality": quality_score >= self.config.MIN_SEGMENT_QUALITY_SCORE
        }
    
    def _calculate_quality_score(self, artifact_ratio: float, snr: float, 
                                mean_correlation: float, alpha_ratio: float) -> float:
        """Ç°Áú Á¡¼ö °è»ê"""
        # °¢ ÁöÇ¥¸¦ 0-1 ¹üÀ§·Î Á¤±ÔÈ­
        artifact_score = max(0, 1 - artifact_ratio * 2)  # ¾ÆÆ¼ÆÑÆ® ºñÀ²ÀÌ ³·À»¼ö·Ï ÁÁÀ½
        snr_score = min(1, snr / 10.0)  # SNRÀÌ 10 ÀÌ»óÀÌ¸é ¸¸Á¡
        correlation_score = min(1, max(0, mean_correlation * 2))  # ÀûÀýÇÑ »ó°ü°ü°è
        alpha_score = min(1, alpha_ratio * 5)  # ¾ËÆÄ ÆÄ¿ö Á¸Àç
        
        # °¡Áß Æò±Õ
        weights = [0.4, 0.3, 0.2, 0.1]  # ¾ÆÆ¼ÆÑÆ® > SNR > »ó°ü°ü°è > ¾ËÆÄ
        scores = [artifact_score, snr_score, correlation_score, alpha_score]
        
        return sum(w * s for w, s in zip(weights, scores))

class MemoryManager:
    """°³¼±µÈ ¸Þ¸ð¸® °ü¸®"""
    
    def __init__(self, max_memory_percent: float = None):
        self.max_memory_percent = max_memory_percent or Config().MAX_CPU_MEMORY_USAGE
        self.last_check_time = 0
        self.check_interval = Config().MEMORY_CHECK_INTERVAL
        self.cleanup_count = 0
        self.warning_threshold = self.max_memory_percent - 10  # °æ°í ÀÓ°è°ª
        
    def check_memory_usage(self, force_check: bool = False) -> bool:
        """°³¼±µÈ ¸Þ¸ð¸® »ç¿ë·® È®ÀÎ"""
        current_time = time.time()
        
        if not force_check and current_time - self.last_check_time < self.check_interval:
            return True
            
        self.last_check_time = current_time
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        # °æ°í ´Ü°è
        if memory_percent > self.warning_threshold:
            logging.warning(f"Memory usage approaching limit: {memory_percent:.1f}%")
        
        # ÀÓ°è°ª ÃÊ°ú ½Ã Á¤¸®
        if memory_percent > self.max_memory_percent:
            logging.error(f"Critical memory usage: {memory_percent:.1f}%")
            success = self._emergency_cleanup()
            
            if not success:
                logging.critical("Emergency memory cleanup failed!")
                return False
                
        return True
    
    def _emergency_cleanup(self) -> bool:
        """ÀÀ±Þ ¸Þ¸ð¸® Á¤¸®"""
        self.cleanup_count += 1
        initial_memory = psutil.virtual_memory().percent
        
        # ´Ü°èÀû Á¤¸®
        gc.collect()
        time.sleep(1)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        time.sleep(2)
        gc.collect()  # ÇÑ ¹ø ´õ
        
        final_memory = psutil.virtual_memory().percent
        freed_memory = initial_memory - final_memory
        
        logging.info(f"Memory cleanup: {initial_memory:.1f}% -> {final_memory:.1f}% "
                    f"(freed {freed_memory:.1f}%)")
        
        return final_memory <= self.max_memory_percent
    
    @contextmanager
    def memory_context(self):
        """¸Þ¸ð¸® ÄÁÅØ½ºÆ® ¸Å´ÏÀú"""
        initial_memory = psutil.virtual_memory().percent
        try:
            yield
        finally:
            final_memory = psutil.virtual_memory().percent
            if final_memory > initial_memory + 5:  # 5% ÀÌ»ó Áõ°¡½Ã Á¤¸®
                self._emergency_cleanup()

class CPUMonitor:
    """°³¼±µÈ CPU ¸ð´ÏÅÍ¸µ"""
    
    def __init__(self, max_cpu_usage: float = None):
        self.max_cpu_usage = max_cpu_usage or Config().MAX_CPU_USAGE
        self.check_interval = Config().CPU_CHECK_INTERVAL
        self.last_check_time = 0
        self.overload_count = 0
        self.process = psutil.Process()
        self.cpu_history = []
        
    def check_cpu_usage(self, force_check: bool = False) -> bool:
        """°³¼±µÈ CPU »ç¿ë·® È®ÀÎ"""
        current_time = time.time()
        
        if not force_check and current_time - self.last_check_time < self.check_interval:
            return True
            
        self.last_check_time = current_time
        
        # ¿©·¯ ¹ø ÃøÁ¤ÇÏ¿© Æò±Õ °è»ê
        cpu_readings = []
        for _ in range(3):
            cpu_readings.append(psutil.cpu_percent(interval=0.1))
        
        cpu_usage = np.mean(cpu_readings)
        self.cpu_history.append(cpu_usage)
        
        # ÃÖ±Ù 5°³ ÃøÁ¤°ª À¯Áö
        if len(self.cpu_history) > 5:
            self.cpu_history.pop(0)
        
        # Áö¼ÓÀûÀÎ °í»ç¿ë·® È®ÀÎ
        if len(self.cpu_history) >= 3:
            recent_avg = np.mean(self.cpu_history[-3:])
            if recent_avg > self.max_cpu_usage:
                return self._handle_cpu_overload(recent_avg)
        
        return True
    
    def _handle_cpu_overload(self, cpu_usage: float) -> bool:
        """CPU °úºÎÇÏ Ã³¸®"""
        self.overload_count += 1
        logging.warning(f"Sustained high CPU usage: {cpu_usage:.1f}%")
        
        try:
            # ÇÁ·Î¼¼½º ¿ì¼±¼øÀ§ ³·Ãã
            current_nice = self.process.nice()
            if current_nice < 10:
                self.process.nice(min(19, current_nice + 5))
                logging.info(f"Adjusted process priority: {current_nice} -> {self.process.nice()}")
        except:
            pass
        
        # ÀûÀÀÀû ´ë±â ½Ã°£
        wait_time = min(30, 5 + self.overload_count * 2)
        logging.info(f"Waiting {wait_time}s for CPU cooldown...")
        time.sleep(wait_time)
        
        return psutil.cpu_percent(interval=1.0) <= self.max_cpu_usage

class GPUManager:
    """°³¼±µÈ GPU °ü¸®"""
    
    def __init__(self, memory_fraction: float = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_gpu = torch.cuda.is_available()
        self.memory_fraction = memory_fraction or Config().GPU_MEMORY_FRACTION
        self.stream = None
        
        if self.use_gpu:
            self._setup_gpu()
            self.batch_size = self._calculate_adaptive_batch_size()
            self._warmup_gpu()
        else:
            self.batch_size = 1
            logging.info("GPU not available, using CPU only")
            
    def _setup_gpu(self):
        """°³¼±µÈ GPU ¼³Á¤"""
        try:
            # CUDA ÃÖÀûÈ­
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # ¸Þ¸ð¸® ºÐÇÒ ÃÖÀûÈ­
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # ½ºÆ®¸² »ý¼º
            self.stream = torch.cuda.Stream()
            
            logging.info(f"GPU setup complete: {torch.cuda.get_device_name(0)}")
            
        except Exception as e:
            logging.error(f"GPU setup failed: {e}")
            self.use_gpu = False
    
    def _calculate_adaptive_batch_size(self) -> int:
        """ÀûÀÀÀû ¹èÄ¡ Å©±â °è»ê"""
        if not self.use_gpu:
            return 1
        
        try:
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / 1e9
            available_memory = total_memory * self.memory_fraction
            
            # ÇöÀç »ç¿ë ÁßÀÎ ¸Þ¸ð¸® °í·Á
            allocated_memory = torch.cuda.memory_allocated() / 1e9
            free_memory = available_memory - allocated_memory
            
            # µ¿Àû ¹èÄ¡ Å©±â °è»ê
            if free_memory >= 15:
                batch_size = 64
            elif free_memory >= 10:
                batch_size = 48
            elif free_memory >= 6:
                batch_size = 32
            elif free_memory >= 4:
                batch_size = 24
            elif free_memory >= 2:
                batch_size = 16
            else:
                batch_size = 8
            
            logging.info(f"Calculated batch size: {batch_size} "
                        f"(Free GPU memory: {free_memory:.1f}GB)")
            
            return batch_size
            
        except Exception as e:
            logging.warning(f"Batch size calculation failed: {e}")
            return 16
    
    def _warmup_gpu(self):
        """°³¼±µÈ GPU ¿ö¹Ö¾÷"""
        if not self.use_gpu:
            return
            
        try:
            with torch.cuda.stream(self.stream):
                # ½ÇÁ¦ ÀÛ¾÷ Å©±â¿Í À¯»çÇÑ ¿ö¹Ö¾÷
                for size in [50, 100, 150]:
                    dummy_data = torch.randn(size, 19, 
                                           device=self.device, 
                                           dtype=torch.complex64)
                    
                    # ½ÇÁ¦ CSD °è»ê°ú À¯»çÇÑ ¿¬»ê
                    result = torch.matmul(dummy_data, dummy_data.conj().T)
                    inv_result = torch.linalg.inv(result + 
                                               torch.eye(19, device=self.device) * 0.1)
                    del dummy_data, result, inv_result
                
                torch.cuda.synchronize()
            
            logging.info("GPU warmup completed successfully")
            
        except Exception as e:
            logging.warning(f"GPU warmup failed: {e}")
    
    def monitor_memory(self) -> Dict[str, float]:
        """GPU ¸Þ¸ð¸® ¸ð´ÏÅÍ¸µ"""
        if not self.use_gpu:
            return {"available": False}
        
        try:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1e9
            
            return {
                "available": True,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "usage_percent": (allocated / total) * 100,
                "free_gb": total - allocated
            }
        except Exception as e:
            logging.error(f"GPU memory monitoring failed: {e}")
            return {"available": False, "error": str(e)}

class VARModelProcessor:
    """°³¼±µÈ VAR ¸ðµ¨ Ã³¸®"""
    
    @staticmethod
    def check_stability(coefs: np.ndarray, max_order: int) -> bool:
        """°³¼±µÈ VAR ¸ðµ¨ ¾ÈÁ¤¼º È®ÀÎ"""
        try:
            n_channels = coefs.shape[1]
            companion_size = max_order * n_channels
            
            # Companion matrix È¿À²Àû ±¸¼º
            companion = np.zeros((companion_size, companion_size), dtype=np.float64)
            
            # »óÀ§ ºí·Ï Ã¤¿ì±â
            for i in range(max_order):
                start_col = i * n_channels
                end_col = (i + 1) * n_channels
                companion[0:n_channels, start_col:end_col] = coefs[i]
            
            # Ç×µî ºí·Ïµé È¿À²Àû ¹èÄ¡
            for i in range(1, max_order):
                start_row = i * n_channels
                end_row = (i + 1) * n_channels
                start_col = (i - 1) * n_channels
                end_col = i * n_channels
                companion[start_row:end_row, start_col:end_col] = np.eye(n_channels)
            
            # °íÀ¯°ª °è»ê - ´õ ¾ÈÁ¤ÀûÀÎ ¹æ¹ý »ç¿ë
            eigenvals = np.linalg.eigvals(companion)
            max_eigenval = np.max(np.abs(eigenvals))
            
            # ¾ÈÁ¤¼º ±âÁØÀ» ¾à°£ ¿ÏÈ­
            return max_eigenval < 0.99
            
        except Exception as e:
            logging.debug(f"Stability check failed: {e}")
            return False
    
    @staticmethod
    def fit_robust(data: np.ndarray, max_order: int = None) -> Tuple[Optional[np.ndarray], 
                                                                   Optional[np.ndarray], 
                                                                   int]:
        """±âÁ¸ ¹æ½ÄÀ¸·Î º¹¿øµÈ VAR ¸ðµ¨ ÀûÇÕ"""
        if max_order is None:
            max_order = Config().VAR_ORDER
            
        n_times, n_channels = data.shape
        
        # ±âº» Á¤±ÔÈ­¸¸ (±âÁ¸ ¹æ½Ä)
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_std = np.where(data_std < 1e-8, 1.0, data_std)
        data_normalized = (data - data_mean) / data_std
        
        # Â÷¼ö¸¦ ÁÙ¿©°¡¸ç ÀûÇÕ (±âÁ¸ ¹æ½Ä°ú À¯»ç)
        for order in range(max_order, 0, -1):
            try:
                # ±âº» ÃÖ¼Ò »ùÇÃ Á¶°Ç (´õ °ü´ëÇÏ°Ô)
                min_samples = order * n_channels + 10
                if n_times < min_samples:
                    continue
                
                var_model = VAR(data_normalized)
                var_results = var_model.fit(maxlags=order, verbose=False)
                
                coefs = var_results.coefs
                sigma_u = var_results.sigma_u
                
                # ±âº» °ËÁõ¸¸ (¾ÈÁ¤¼º °Ë»ç Á¦°Å)
                if coefs is not None and sigma_u is not None:
                    return coefs, sigma_u, order
                    
            except Exception:
                continue
        
        # Æú¹é ¸ðµ¨ (±âÁ¸°ú µ¿ÀÏ)
        logging.warning("All VAR orders failed, using fallback AR(1) model")
        return VARModelProcessor._create_fallback_model(data_normalized)
    
    @staticmethod
    def _create_fallback_model(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """´ë¾È AR(1) ¸ðµ¨ »ý¼º"""
        n_times, n_channels = data.shape
        
        # ´Ü¼ø AR(1) °è¼ö ÃßÁ¤
        coefs = np.zeros((1, n_channels, n_channels))
        sigma_u = np.eye(n_channels)
        
        for i in range(n_channels):
            if n_times > 10:
                x = data[:-1, i]
                y = data[1:, i]
                if np.std(x) > 1e-8:
                    coef = np.corrcoef(x, y)[0, 1] * 0.5  # ¾ÈÁ¤¼ºÀ» À§ÇØ °¨¼è
                    coefs[0, i, i] = coef
                    
                    # ÀÜÂ÷ ºÐ»ê ÃßÁ¤
                    y_pred = coef * x
                    residuals = y - y_pred
                    sigma_u[i, i] = np.var(residuals)
        
        return coefs, sigma_u, 1

# CSDComputer Å¬·¡½ºµµ À¯»çÇÏ°Ô °³¼±... (°è¼Ó)

class CSDComputer:
    """°³¼±µÈ CSD °è»ê - ±âÁ¸ ¹æ½ÄÀ¸·Î º¹¿ø"""
    
    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.computation_stats = {
            "gpu_computations": 0,
            "cpu_computations": 0,
            "failed_computations": 0
        }
    
    def compute_var_csd_batch(self, data_batch: List[np.ndarray], 
                             sfreq: float, freqs: np.ndarray) -> List[Optional[Dict]]:
        """¹èÄ¡ CSD °è»ê - Ç°Áú °Ë»ç ¾øÀÌ ±âÁ¸ ¹æ½Ä´ë·Î"""
        if not data_batch:
            return []
        
        results = []
        
        try:
            if self.gpu_manager.use_gpu and len(data_batch) > 1:
                # GPU ¹èÄ¡ Ã³¸®
                batch_results = self._process_gpu_batch(data_batch, sfreq, freqs)
                self.computation_stats["gpu_computations"] += len(batch_results)
            else:
                # CPU Ã³¸®
                batch_results = []
                for data in data_batch:
                    result = self._compute_single_csd_cpu(data, sfreq, freqs)
                    batch_results.append(result)
                self.computation_stats["cpu_computations"] += len(batch_results)
            
            return batch_results
            
        except Exception as e:
            logging.error(f"Batch CSD computation failed: {e}")
            self.computation_stats["failed_computations"] += len(data_batch)
            
            # CPU Æú¹é
            fallback_results = []
            for data in data_batch:
                result = self._compute_single_csd_cpu(data, sfreq, freqs)
                fallback_results.append(result)
            
            return fallback_results
    
    def _process_gpu_batch(self, data_batch: List[np.ndarray], 
                          sfreq: float, freqs: np.ndarray) -> List[Optional[Dict]]:
        """°³¼±µÈ GPU ¹èÄ¡ Ã³¸®"""
        results = []
        
        # ¸Þ¸ð¸® ¸ð´ÏÅÍ¸µ
        memory_info = self.gpu_manager.monitor_memory()
        if memory_info.get("usage_percent", 100) > 90:
            logging.warning("GPU memory usage high, reducing batch size")
            # ¹ÝÀ¸·Î ³ª´©¾î Ã³¸®
            mid = len(data_batch) // 2
            results.extend(self._process_gpu_batch(data_batch[:mid], sfreq, freqs))
            results.extend(self._process_gpu_batch(data_batch[mid:], sfreq, freqs))
            return results
        
        try:
            with torch.cuda.stream(self.gpu_manager.stream):
                for data in data_batch:
                    result = self._compute_single_csd_gpu(data, sfreq, freqs)
                    results.append(result)
            
            torch.cuda.synchronize()
            return results
            
        except torch.cuda.OutOfMemoryError:
            logging.warning("GPU OOM, falling back to CPU")
            torch.cuda.empty_cache()
            return [self._compute_single_csd_cpu(data, sfreq, freqs) for data in data_batch]
    
    def _compute_single_csd_gpu(self, data: np.ndarray, 
                               sfreq: float, freqs: np.ndarray) -> Optional[Dict]:
        """°³¼±µÈ GPU CSD °è»ê - ÁÖÆÄ¼öº° ÀÏ°üµÈ ½ºÄÉÀÏ¸µ"""
        try:
            # VAR ¸ðµ¨ ÀûÇÕ (CPU¿¡¼­)
            coefs, sigma_u, fitted_order = VARModelProcessor.fit_robust(data)
            
            if coefs is None:
                return None
            
            # ?? ½ºÄÉÀÏ¸µ Á¤±ÔÈ­ - ¿øº» µ¥ÀÌÅÍÀÇ ºÐ»êÀ» º¸Á¸
            original_variance = np.var(data, axis=0)
            scaling_factor = np.sqrt(np.mean(original_variance))
            
            # GPU ÅÙ¼­·Î º¯È¯ (¸Þ¸ð¸® È¿À²Àû)
            device = self.gpu_manager.device
            
            with torch.cuda.device(device):
                coefs_gpu = torch.tensor(coefs, dtype=torch.complex64, device=device)
                sigma_u_gpu = torch.tensor(sigma_u, dtype=torch.complex64, device=device)
                freqs_gpu = torch.tensor(freqs, dtype=torch.float32, device=device)
                
                n_freqs = len(freqs)
                n_channels = data.shape[1]
                
                # °á°ú ÅÙ¼­ »çÀü ÇÒ´ç
                csd_result = torch.zeros((n_freqs, n_channels, n_channels, 2), 
                                       dtype=torch.float32, device=device)
                
                # º¤ÅÍÈ­µÈ ÁÖÆÄ¼ö °è»ê
                omega = 2 * torch.pi * freqs_gpu / sfreq
                
                # ¹èÄ¡ Ã³¸®·Î ¸ðµç ÁÖÆÄ¼ö ÇÑ¹ø¿¡ °è»ê
                for f_idx, w in enumerate(omega):
                    # Àü´ÞÇÔ¼ö °è»ê
                    h_f = torch.eye(n_channels, dtype=torch.complex64, device=device)
                    
                    for p in range(fitted_order):
                        angle = -w * (p + 1)
                        exp_term = torch.complex(torch.cos(angle), torch.sin(angle))
                        h_f = h_f - coefs_gpu[p] * exp_term
                    
                    # ¾ÈÁ¤ÀûÀÎ ¿ªÇà·Ä °è»ê
                    try:
                        # Á¤±ÔÈ­ Ãß°¡
                        regularizer = torch.eye(n_channels, device=device) * 1e-6
                        h_f_reg = h_f + regularizer
                        h_f_inv = torch.linalg.inv(h_f_reg)
                    except:
                        h_f_inv = torch.linalg.pinv(h_f)
                    
                    # CSD °è»ê
                    s_f = torch.matmul(torch.matmul(h_f_inv, sigma_u_gpu), h_f_inv.conj().T)
                    
                    # ?? ÁÖÆÄ¼öº° ÀÏ°üµÈ ½ºÄÉÀÏ¸µ Àû¿ë
                    # ¸ðµç ÁÖÆÄ¼ö¿¡¼­ µ¿ÀÏÇÑ ¹°¸®Àû ´ÜÀ§ À¯Áö
                    s_f_scaled = s_f * (scaling_factor ** 2)
                    
                    csd_result[f_idx, :, :, 0] = torch.real(s_f_scaled)
                    csd_result[f_idx, :, :, 1] = torch.imag(s_f_scaled)
                
                # CPU·Î º¹»ç Àü¿¡ µ¿±âÈ­
                torch.cuda.synchronize()
                result_cpu = csd_result.cpu().numpy()
                
                # GPU ¸Þ¸ð¸® Áï½Ã ÇØÁ¦
                del coefs_gpu, sigma_u_gpu, freqs_gpu, csd_result, h_f, h_f_inv, s_f, s_f_scaled
                torch.cuda.empty_cache()
                
                return {
                    "csd": result_cpu.astype(np.float32),
                    "fitted_order": fitted_order,
                    "computation_method": "gpu",
                    "scaling_factor": float(scaling_factor),
                    "original_variance": original_variance.astype(np.float32)
                }
            
        except Exception as e:
            logging.warning(f"GPU CSD computation failed: {e}")
            torch.cuda.empty_cache()
            return self._compute_single_csd_cpu(data, sfreq, freqs)
    
    def _compute_single_csd_cpu(self, data: np.ndarray, 
                               sfreq: float, freqs: np.ndarray) -> Optional[Dict]:
        """°³¼±µÈ CPU CSD °è»ê - ÁÖÆÄ¼öº° ÀÏ°üµÈ ½ºÄÉÀÏ¸µ"""
        try:
            # VAR ¸ðµ¨ ÀûÇÕ
            coefs, sigma_u, fitted_order = VARModelProcessor.fit_robust(data)
            
            if coefs is None:
                return None
            
            # ?? ½ºÄÉÀÏ¸µ Á¤±ÔÈ­ - ¿øº» µ¥ÀÌÅÍÀÇ ºÐ»êÀ» º¸Á¸
            original_variance = np.var(data, axis=0)
            scaling_factor = np.sqrt(np.mean(original_variance))
            
            n_freqs = len(freqs)
            n_channels = data.shape[1]
            
            csd_result = np.zeros((n_freqs, n_channels, n_channels, 2), dtype=np.float32)
            omega = 2 * np.pi * freqs / sfreq
            
            # º¤ÅÍÈ­µÈ °è»êÀ¸·Î ÃÖÀûÈ­
            for f_idx, w in enumerate(omega):
                h_f = np.eye(n_channels, dtype=np.complex128)  # ´õ ³ôÀº Á¤¹Ðµµ
                
                for p in range(fitted_order):
                    angle = -w * (p + 1)
                    exp_term = np.exp(1j * angle)
                    h_f = h_f - coefs[p] * exp_term
                
                try:
                    # Á¶°Ç¼ö È®ÀÎ ¹× Á¤±ÔÈ­
                    cond_num = np.linalg.cond(h_f)
                    if cond_num > 1e12:
                        regularizer = np.eye(n_channels) * 1e-6
                        h_f = h_f + regularizer
                    
                    h_f_inv = np.linalg.inv(h_f)
                except:
                    h_f_inv = np.linalg.pinv(h_f)
                
                s_f = h_f_inv @ sigma_u @ h_f_inv.conj().T
                
                # ?? ÁÖÆÄ¼öº° ÀÏ°üµÈ ½ºÄÉÀÏ¸µ Àû¿ë
                # ¸ðµç ÁÖÆÄ¼ö¿¡¼­ µ¿ÀÏÇÑ ¹°¸®Àû ´ÜÀ§ À¯Áö
                s_f_scaled = s_f * (scaling_factor ** 2)
                
                csd_result[f_idx, :, :, 0] = np.real(s_f_scaled).astype(np.float32)
                csd_result[f_idx, :, :, 1] = np.imag(s_f_scaled).astype(np.float32)
            
            return {
                "csd": csd_result,
                "fitted_order": fitted_order,
                "computation_method": "cpu",
                "scaling_factor": float(scaling_factor),
                "original_variance": original_variance.astype(np.float32)
            }
            
        except Exception as e:
            logging.error(f"CPU CSD computation failed: {e}")
            return None
    
    def get_computation_stats(self) -> Dict[str, int]:
        """°è»ê Åë°è ¹ÝÈ¯"""
        return self.computation_stats.copy()

class ResourceManager:
    """°³¼±µÈ ÅëÇÕ ¸®¼Ò½º °ü¸®"""
    
    def __init__(self, gpu_memory_fraction: float = None):
        self.memory_manager = MemoryManager()
        self.cpu_monitor = CPUMonitor()
        self.gpu_manager = GPUManager(gpu_memory_fraction)
        
        # ¼º´É ¸ð´ÏÅÍ¸µ
        self.performance_stats = {
            "resource_checks": 0,
            "memory_warnings": 0,
            "cpu_warnings": 0,
            "gpu_errors": 0
        }
    
    def check_resources(self, force_check: bool = False) -> bool:
        """°³¼±µÈ ¸®¼Ò½º »óÅÂ È®ÀÎ"""
        self.performance_stats["resource_checks"] += 1
        
        # ¸Þ¸ð¸® Ã¼Å©
        memory_ok = self.memory_manager.check_memory_usage(force_check)
        if not memory_ok:
            self.performance_stats["memory_warnings"] += 1
        
        # CPU Ã¼Å©
        cpu_ok = self.cpu_monitor.check_cpu_usage(force_check)
        if not cpu_ok:
            self.performance_stats["cpu_warnings"] += 1
        
        # GPU ¸Þ¸ð¸® Ã¼Å©
        gpu_ok = True
        if self.gpu_manager.use_gpu:
            gpu_memory = self.gpu_manager.monitor_memory()
            if gpu_memory.get("usage_percent", 0) > 95:
                logging.warning("GPU memory critically high")
                self.gpu_manager.cleanup()
                gpu_ok = False
                self.performance_stats["gpu_errors"] += 1
        
        # ÀüÃ¼ÀûÀÎ Á¤¸®°¡ ÇÊ¿äÇÑ °æ¿ì
        if not (memory_ok and cpu_ok):
            self._emergency_cleanup()
        
        return memory_ok and cpu_ok and gpu_ok
    
    def _emergency_cleanup(self):
        """ÀÀ±Þ ½Ã½ºÅÛ Á¤¸®"""
        logging.info("Performing emergency system cleanup...")
        
        # GPU Á¤¸®
        if self.gpu_manager.use_gpu:
            self.gpu_manager.cleanup()
        
        # ¸Þ¸ð¸® Á¤¸®
        gc.collect()
        time.sleep(2)
        gc.collect()
        
        # CPU ¿ì¼±¼øÀ§ Á¶Á¤
        try:
            process = psutil.Process()
            process.nice(15)
        except:
            pass
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Á¾ÇÕ ¸®¼Ò½º Åë°è"""
        try:
            # GPU Åë°è
            try:
                gpu_stats = self.gpu_manager.monitor_memory()
            except:
                gpu_stats = {"available": False, "error": "GPU stats unavailable"}
            
            # ¸Þ¸ð¸® Åë°è
            try:
                memory_stats = self.memory_manager.get_stats()
            except:
                memory_stats = {"current_memory_percent": 0, "error": "Memory stats unavailable"}
            
            # CPU Åë°è
            try:
                cpu_stats = self.cpu_monitor.get_stats()
            except:
                cpu_stats = {"current_cpu_usage": 0, "error": "CPU stats unavailable"}
            
            return {
                "gpu": gpu_stats,
                "memory": memory_stats,
                "cpu": cpu_stats,
                "performance": self.performance_stats
            }
        except Exception as e:
            logging.error(f"Failed to get comprehensive stats: {e}")
            return {
                "gpu": {"available": False, "error": str(e)},
                "memory": {"current_memory_percent": 0, "error": str(e)},
                "cpu": {"current_cpu_usage": 0, "error": str(e)},
                "performance": self.performance_stats
            }

def setup_logging(output_dir: Path) -> logging.Logger:
    """°³¼±µÈ ·Î±ë ¼³Á¤"""
    log_file = output_dir / f"tuab_var_csd_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # ·Î±× Æ÷¸ËÅÍ °³¼±
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # ÆÄÀÏ ÇÚµé·¯
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # ÄÜ¼Ö ÇÚµé·¯
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # ·çÆ® ·Î°Å ¼³Á¤
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)

def process_file_enhanced(file_path: Path, label: int, output_dir: Path, 
                         resource_manager: ResourceManager, 
                         csd_computer: CSDComputer) -> Dict[str, Any]:
    """°³¼±µÈ ÆÄÀÏ Ã³¸® ÇÔ¼ö"""
    basename = file_path.stem
    config = Config()
    
    result = {
        "file": basename,
        "label": label,
        "success": False,
        "segments_processed": 0,
        "segments_failed": 0,
        "quality_filtered": 0,
        "processing_time": 0,
        "error": None,
        "file_stats": {}
    }
    
    start_time = time.time()
    
    try:
        # ¸®¼Ò½º Ã¼Å©
        if not resource_manager.check_resources(force_check=True):
            result["error"] = "Resource limits exceeded before processing"
            return result
        
        logging.info(f"Processing {basename}")
        
        # ¸Þ¸ð¸® ÄÁÅØ½ºÆ® »ç¿ë
        with resource_manager.memory_manager.memory_context():
            # EEG ÆÄÀÏ ·Îµå (´õ ¾ÈÀüÇÑ ¹æ½Ä)
            try:
                raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
            except Exception as e:
                result["error"] = f"File loading failed: {str(e)}"
                return result
            
            sfreq = raw.info['sfreq']
            
            # Ã¤³Î È®ÀÎ ¹× ¸ÅÇÎ
            available_channels = raw.ch_names
            missing_channels = [ch for ch in config.TARGET_CHANNELS if ch not in available_channels]
            
            if len(missing_channels) > len(config.TARGET_CHANNELS) * 0.2:  # 20% ÀÌ»ó ´©¶ô½Ã ½ºÅµ
                result["error"] = f"Too many missing channels: {len(missing_channels)}/{len(config.TARGET_CHANNELS)}"
                return result
            
            # »ç¿ë °¡´ÉÇÑ Ã¤³Î¸¸ ¼±ÅÃ
            valid_channels = [ch for ch in config.TARGET_CHANNELS if ch in available_channels]
            raw.pick(valid_channels)
            
            # ÆÄÀÏ ±âº» Á¤º¸
            duration = raw.times[-1]
            result["file_stats"] = {
                "duration": duration,
                "sfreq": sfreq,
                "n_channels": len(valid_channels),
                "missing_channels": len(missing_channels)
            }
            
            if duration < config.SEGMENT_DURATION:
                result["error"] = f"File too short: {duration:.2f}s < {config.SEGMENT_DURATION}s"
                return result
            
            # µ¥ÀÌÅÍ ·Îµå ¹× ÀüÃ³¸®
            raw.load_data()
            
            # ±âº»ÀûÀÎ ¾ÆÆ¼ÆÑÆ® Á¦°Å
            raw.filter(l_freq=0.5, h_freq=50, verbose=False)  # ±âº» ´ë¿ªÅë°ú ÇÊÅÍ
            
            data = raw.get_data().T
            
            # ¼¼±×¸ÕÆ® ÃßÃâ - ±âº» À¯È¿¼º °Ë»ç¸¸
            n_segments = int(np.floor(duration / config.SEGMENT_DURATION))
            valid_segments = []
            
            for seg_idx in range(n_segments):
                start_sample = int(seg_idx * config.SEGMENT_DURATION * sfreq)
                end_sample = int((seg_idx + 1) * config.SEGMENT_DURATION * sfreq)
                segment_data = data[start_sample:end_sample, :]
                
                # ±âº» À¯È¿¼º °Ë»ç¸¸ (NaN, Inf Ã¼Å©)
                if not np.any(np.isnan(segment_data)) and not np.any(np.isinf(segment_data)):
                    valid_segments.append((seg_idx, segment_data))
            
            if not valid_segments:
                result["error"] = "No valid segments found"
                return result
            
            # ÀûÀÀÀû ¹èÄ¡ Å©±â
            batch_size = min(resource_manager.gpu_manager.batch_size, len(valid_segments))
            processed = 0
            failed = 0
            quality_filtered = 0
            
            # ¹èÄ¡ Ã³¸®
            for i in range(0, len(valid_segments), batch_size):
                # ¸®¼Ò½º Ã¼Å©
                if not resource_manager.check_resources():
                    logging.warning(f"Resource limit reached at segment {i}")
                    break
                
                batch_end = min(i + batch_size, len(valid_segments))
                batch_segments = valid_segments[i:batch_end]
                
                # ¼¼±×¸ÕÆ® µ¥ÀÌÅÍ¸¸ ÃßÃâ
                segment_data_list = [seg_data for _, seg_data in batch_segments]
                
                # CSD °è»ê
                batch_results = csd_computer.compute_var_csd_batch(
                    segment_data_list, sfreq, config.FREQS
                )
                
                # °á°ú ÀúÀå - ±âÁ¸ ¹æ½Ä
                for (seg_idx, _), csd_result in zip(batch_segments, batch_results):
                    if csd_result is not None:
                        segment_info = {
                            "csd": csd_result["csd"],
                            "label": label,
                            "frequency": config.FREQS,
                            "start_time": seg_idx * config.SEGMENT_DURATION,
                            "sfreq": sfreq,
                            "fitted_order": csd_result.get("fitted_order", 1),
                            "computation_method": csd_result.get("computation_method", "unknown"),
                            "scaling_info": {  # ½ºÄÉÀÏ¸µ Á¤º¸ À¯Áö
                                "scaling_factor": csd_result.get("scaling_factor", 1.0),
                                "original_variance": csd_result.get("original_variance", None),
                                "units": "¥ìV©÷ (variance-normalized)",
                                "scaling_method": "consistent_across_frequencies"
                            },
                            "file_info": {
                                "original_file": basename,
                                "segment_index": seg_idx,
                                "channels_used": valid_channels
                            },
                            "units": "¥ìV©÷ (variance-normalized)",
                            "scaling_factor": Config.MICROVOLTS_TO_VOLTS_SQUARED_SCALE
                        }
                        
                        output_path = output_dir / f"{basename}_seg{seg_idx:03d}.pkl"
                        with open(output_path, 'wb') as f:
                            pickle.dump(segment_info, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
                        processed += 1
                    else:
                        failed += 1
                
                # ÁÖ±âÀû Á¤¸®
                if i % (batch_size * 3) == 0:
                    gc.collect()
        
        result["success"] = True
        result["segments_processed"] = processed
        result["segments_failed"] = failed
        
    except Exception as e:
        logging.error(f"Error processing {basename}: {str(e)}")
        logging.error(traceback.format_exc())
        result["error"] = str(e)
    
    finally:
        # ¸Þ¸ð¸® Á¤¸®
        locals_to_delete = ['data', 'raw', 'segment_data', 'batch_results']
        for var_name in locals_to_delete:
            if var_name in locals():
                del locals()[var_name]
        
        gc.collect()
        #if resource_manager.gpu_manager.use_gpu:
           # resource_manager.gpu_manager.cleanup()
        
        result["processing_time"] = time.time() - start_time
    
    return result

def parse_arguments():
    """°³¼±µÈ ¸í·ÉÇà ÀÎ¼ö ÆÄ½Ì"""
    parser = argparse.ArgumentParser(
        description='Enhanced EEG VAR-CSD Preprocessing with Quality Control',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_root', type=str, 
                       default="/remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/",
                       help='Data root directory')
    parser.add_argument('--output_root', type=str,
                       default="/remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/var_scaled",
                       help='Output directory')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.7,
                       help='GPU memory fraction to use (0.1-1.0)')
    parser.add_argument('--file_batch_size', type=int, default=2,
                       help='Number of files to process in each batch')
    parser.add_argument('--max_cpu_usage', type=float, default=70.0,
                       help='Maximum CPU usage percentage')
    parser.add_argument('--max_memory_usage', type=float, default=85.0,
                       help='Maximum memory usage percentage')
    parser.add_argument('--quality_threshold', type=float, default=0.7,
                       help='Minimum quality score for segments')
    parser.add_argument('--resume', action='store_true',
                       help='Resume processing from existing outputs')
    parser.add_argument('--dry_run', action='store_true',
                       help='Perform dry run without actual processing')
    
    return parser.parse_args()

def main():
    """°³¼±µÈ ¸ÞÀÎ ½ÇÇà ÇÔ¼ö"""
    args = parse_arguments()
    
    # ÀÎ¼ö °ËÁõ
    if not 0.1 <= args.gpu_memory_fraction <= 1.0:
        raise ValueError("GPU memory fraction must be between 0.1 and 1.0")
    
    if not 10 <= args.max_cpu_usage <= 100:
        raise ValueError("Max CPU usage must be between 10 and 100")
    
    if not 50 <= args.max_memory_usage <= 95:
        raise ValueError("Max memory usage must be between 50 and 95")
    
    # ¼³Á¤ ¾÷µ¥ÀÌÆ®
    config = Config()
    config.MAX_CPU_USAGE = args.max_cpu_usage
    config.MAX_CPU_MEMORY_USAGE = args.max_memory_usage
    config.MIN_SEGMENT_QUALITY_SCORE = args.quality_threshold
    config.GPU_MEMORY_FRACTION = args.gpu_memory_fraction
    config.FILE_BATCH_SIZE = args.file_batch_size
    
    # Ãâ·Â µð·ºÅä¸® »ý¼º
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_root / split).mkdir(exist_ok=True)
    
    # ·Î±ë ¼³Á¤
    logger = setup_logging(output_root)
    
    # Ã³¸® ½ÃÀÛ Á¤º¸
    logger.info("=== Enhanced EEG VAR-CSD Processing Started ===")
    logger.info(f"Configuration: {vars(args)}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual processing will be performed")
    
    # ¸®¼Ò½º °ü¸®ÀÚ ÃÊ±âÈ­ - Ç°Áú °Ë»ç Á¦°Å
    resource_manager = ResourceManager(args.gpu_memory_fraction)
    csd_computer = CSDComputer(resource_manager.gpu_manager)
    
    # ÃÊ±â ½Ã½ºÅÛ »óÅÂ
    initial_stats = resource_manager.get_comprehensive_stats()
    logger.info("=== Initial System Resources ===")
    logger.info(f"GPU: {initial_stats['gpu']}")
    logger.info(f"CPU Memory: {initial_stats['memory']['current_memory_percent']:.1f}%")
    logger.info(f"CPU Usage: {initial_stats['cpu']['current_cpu_usage']:.1f}%")
    logger.info(f"Frequency bands: {len(config.FREQS)} bins ({config.FREQS[0]}-{config.FREQS[-1]} Hz)")
    
    # ÆÄÀÏ °Ë»ö
    data_root = Path(args.data_root)
    file_patterns = {
        "train_normal": "train/normal/**/*.edf",
        "train_abnormal": "train/abnormal/**/*.edf",
        "eval_normal": "eval/normal/**/*.edf",
        "eval_abnormal": "eval/abnormal/**/*.edf"
    }
    
    file_lists = {}
    for key, pattern in file_patterns.items():
        files = list(data_root.glob(pattern))
        file_lists[key] = files
        logger.info(f"Found {len(files)} files for {key}")
    
    # ±âÁ¸ Ãâ·Â ÆÄÀÏ È®ÀÎ (resume ±â´É)
    existing_files = set()
    if args.resume:
        for split_dir in output_root.iterdir():
            if split_dir.is_dir():
                existing_files.update(f.stem.split('_seg')[0] for f in split_dir.glob("*.pkl"))
        logger.info(f"Found {len(existing_files)} already processed files")
    
    # Train/Val ºÐÇÒ
    train_normal_split, val_normal_files = train_test_split(
        file_lists["train_normal"], test_size=0.2, random_state=42
    )
    train_abnormal_split, val_abnormal_files = train_test_split(
        file_lists["train_abnormal"], test_size=0.2, random_state=42
    )
    
    # Ã³¸®ÇÒ ÆÄÀÏ ¹èÄ¡ Á¤ÀÇ
    processing_batches = [
        (train_normal_split, 0, output_root / "train", "Train Normal"),
        (train_abnormal_split, 1, output_root / "train", "Train Abnormal"),
        (val_normal_files, 0, output_root / "val", "Val Normal"),
        (val_abnormal_files, 1, output_root / "val", "Val Abnormal"),
        (file_lists["eval_normal"], 0, output_root / "test", "Test Normal"),
        (file_lists["eval_abnormal"], 1, output_root / "test", "Test Abnormal")
    ]
    
    # ÀüÃ¼ Åë°è
    total_stats = {
        "processed_files": 0,
        "failed_files": 0,
        "skipped_files": 0,
        "total_segments": 0,
        "processing_times": [],
        "batch_stats": []
    }
    
    try:
        # °¢ ¹èÄ¡ Ã³¸®
        for batch_idx, (file_list, label, output_dir, desc) in enumerate(processing_batches):
            if not file_list:
                continue
            
            # Resume ±â´É Àû¿ë
            if args.resume:
                file_list = [f for f in file_list if f.stem not in existing_files]
            
            if not file_list:
                logger.info(f"No new files to process for {desc}")
                continue
            
            logger.info(f"Processing {desc}: {len(file_list)} files")
            
            if args.dry_run:
                logger.info(f"DRY RUN: Would process {len(file_list)} files for {desc}")
                continue
            
            batch_start_time = time.time()
            batch_stats = {
                "batch_name": desc,
                "total_files": len(file_list),
                "processed": 0,
                "failed": 0,
                "skipped": 0
            }
            
            # ÆÄÀÏÀ» ÀÛÀº ¹èÄ¡·Î ³ª´©¾î Ã³¸®
            file_batch_size = args.file_batch_size
            
            for i in range(0, len(file_list), file_batch_size):
                sub_batch_files = file_list[i:i+file_batch_size]
                
                logger.info(f"Processing sub-batch {i//file_batch_size + 1}/"
                           f"{(len(file_list)+file_batch_size-1)//file_batch_size} of {desc}")
                
                # ¹èÄ¡ Ã³¸® Àü ¸®¼Ò½º »óÅÂ È®ÀÎ
                if not resource_manager.check_resources(force_check=True):
                    logger.error("Resource limits exceeded, stopping processing")
                    break
                
                # ¼øÂ÷ Ã³¸® (¾ÈÁ¤¼ºÀ» À§ÇØ)
                for file_path in tqdm(sub_batch_files, desc=f"Sub-batch {i//file_batch_size + 1}"):
                    
                    # °¢ ÆÄÀÏ Ã³¸® Àü ¸®¼Ò½º Ã¼Å©
                    if not resource_manager.check_resources():
                        logger.warning(f"Resource limits reached, skipping remaining files")
                        total_stats["skipped_files"] += len(sub_batch_files) - sub_batch_files.index(file_path)
                        break
                    
                    # ÆÄÀÏ Ã³¸®
                    result = process_file_enhanced(
                        file_path, label, output_dir, resource_manager, csd_computer
                    )
                    
                    # °á°ú Ã³¸®
                    if result["success"]:
                        total_stats["processed_files"] += 1
                        total_stats["total_segments"] += result["segments_processed"]
                        batch_stats["processed"] += 1
                    else:
                        total_stats["failed_files"] += 1
                        batch_stats["failed"] += 1
                        logger.warning(f"Failed to process {file_path.name}: {result['error']}")
                    
                    total_stats["processing_times"].append(result["processing_time"])
                    
                    # ÁÖ±âÀû »óÅÂ ·Î±ë
                    if total_stats["processed_files"] % 10 == 0:
                        current_stats = resource_manager.get_comprehensive_stats()
                        comp_stats = csd_computer.get_computation_stats()
                        
                        logger.info(f"Progress: {total_stats['processed_files']} processed, "
                                   f"{total_stats['total_segments']} segments created")
                        logger.info(f"Computation: {comp_stats['gpu_computations']} GPU, "
                                   f"{comp_stats['cpu_computations']} CPU")
                        
                        if current_stats["gpu"]["available"]:
                            logger.info(f"GPU Memory: {current_stats['gpu']['allocated_gb']:.2f}GB used")
                
                # ¼­ºê ¹èÄ¡ ¿Ï·á ÈÄ Á¤¸®
                gc.collect()
                #resource_manager.gpu_manager.cleanup()
                time.sleep(1)  # ½Ã½ºÅÛ ¾ÈÁ¤È­
            
            # ¹èÄ¡ ¿Ï·á
            batch_stats["processing_time"] = time.time() - batch_start_time
            total_stats["batch_stats"].append(batch_stats)
            
            logger.info(f"Completed {desc}: {batch_stats['processed']} processed, "
                       f"{batch_stats['failed']} failed in {batch_stats['processing_time']:.1f}s")
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main processing: {e}")
        logger.error(traceback.format_exc())
    
    finally:
        # ÃÖÁ¾ Åë°è ¹× Á¤¸®
        final_stats = resource_manager.get_comprehensive_stats()
        computation_stats = csd_computer.get_computation_stats()
        
        # Ã³¸® °á°ú ¿ä¾à
        processing_summary = {
            "processing_completed_at": datetime.now().isoformat(),
            "total_processed_files": total_stats["processed_files"],
            "total_failed_files": total_stats["failed_files"],
            "total_skipped_files": total_stats["skipped_files"],
            "total_segments_created": total_stats["total_segments"],
            "average_processing_time": np.mean(total_stats["processing_times"]) if total_stats["processing_times"] else 0,
            "total_processing_time": sum(total_stats["processing_times"]),
            "computation_stats": computation_stats,
            "final_resource_stats": final_stats,
            "batch_details": total_stats["batch_stats"],
            "configuration": {
                "gpu_memory_fraction": args.gpu_memory_fraction,
                "max_cpu_memory_usage": config.MAX_CPU_MEMORY_USAGE,
                "max_cpu_usage": config.MAX_CPU_USAGE,
                "quality_threshold": config.MIN_SEGMENT_QUALITY_SCORE,
                "file_batch_size": config.FILE_BATCH_SIZE,
                "frequencies": config.FREQS.tolist(),
                "segment_duration": config.SEGMENT_DURATION,
                "var_order": config.VAR_ORDER,
                "target_channels": config.TARGET_CHANNELS
            }
        }
        
        # Åë°è ÆÄÀÏ ÀúÀå
        stats_file = output_root / "processing_summary_enhanced.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(processing_summary, f, indent=2, default=str, ensure_ascii=False)
        
        # ÃÖÁ¾ ·Î±×
        logger.info("=== Processing Complete ===")
        logger.info(f"Files processed: {total_stats['processed_files']}")
        logger.info(f"Files failed: {total_stats['failed_files']}")
        logger.info(f"Files skipped: {total_stats['skipped_files']}")
        logger.info(f"Total segments: {total_stats['total_segments']}")
        logger.info(f"Average processing time: {processing_summary['average_processing_time']:.2f}s per file")
        logger.info(f"Total processing time: {processing_summary['total_processing_time']:.1f}s")
        
        # °è»ê ¹æ¹ý Åë°è
        logger.info(f"GPU computations: {computation_stats['gpu_computations']}")
        logger.info(f"CPU computations: {computation_stats['cpu_computations']}")
        logger.info(f"Failed computations: {computation_stats['failed_computations']}")
        
        # ¸®¼Ò½º »ç¿ë Åë°è
        if final_stats["gpu"]["available"]:
            logger.info(f"Final GPU memory: {final_stats['gpu']['allocated_gb']:.2f}GB")
            logger.info(f"GPU memory fraction used: {args.gpu_memory_fraction}")
        
        logger.info(f"Final CPU memory: {final_stats['memory']['current_memory_percent']:.1f}%")
        logger.info(f"CPU overload events: {final_stats['cpu']['overload_events']}")
        logger.info(f"Memory cleanup events: {final_stats['memory']['cleanup_events']}")
        logger.info(f"Resource check events: {final_stats['performance']['resource_checks']}")
        
        # ¼º°ø·ü °è»ê
        total_attempted = total_stats['processed_files'] + total_stats['failed_files']
        if total_attempted > 0:
            success_rate = (total_stats['processed_files'] / total_attempted) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("Enhanced EEG processing completed!")
        
        # ÃßÃµ»çÇ× Ãâ·Â
        if total_stats['failed_files'] > total_stats['processed_files'] * 0.1:
            logger.warning("High failure rate detected. Consider:")
            logger.warning("- Reducing batch size")
            logger.warning("- Lowering GPU memory fraction")
            logger.warning("- Checking data quality")

def create_dataset_info(output_root: Path):
    """µ¥ÀÌÅÍ¼Â Á¤º¸ ÆÄÀÏ »ý¼º"""
    config = Config()
    
    dataset_info = {
        "dataset_name": "TUH EEG Abnormal - VAR-CSD Enhanced",
        "created_at": datetime.now().isoformat(),
        "preprocessing_method": "VAR-based Cross-Spectral Density with Quality Control",
        "data_format": {
            "file_format": "pickle (.pkl)",
            "data_structure": {
                "csd": "Complex cross-spectral density matrix [freq, channel, channel, real/imag]",
                "label": "Binary classification (0=normal, 1=abnormal)",
                "frequency": "Frequency bins in Hz",
                "start_time": "Segment start time in seconds",
                "sfreq": "Original sampling frequency",
                "fitted_order": "VAR model order used",
                "computation_method": "gpu or cpu",
                "quality_metrics": "Data quality assessment",
                "file_info": "Original file and segment information"
            }
        },
        "parameters": {
            "segment_duration": config.SEGMENT_DURATION,
            "var_order": config.VAR_ORDER,
            "frequency_bands": config.FREQS.tolist(),
            "target_channels": config.TARGET_CHANNELS,
            "quality_threshold": config.MIN_SEGMENT_QUALITY_SCORE,
            "artifact_threshold": config.MAX_ARTIFACT_THRESHOLD
        },
        "splits": {
            "train": "Training set",
            "val": "Validation set (20% of original train)",
            "test": "Test set (original eval)"
        },
        "usage_notes": [
            "Each .pkl file contains one 4-second EEG segment",
            "CSD data is in microvolts squared with 1e12 scaling factor",
            "Quality metrics should be checked before use",
            "GPU-computed results may have slightly different precision than CPU"
        ]
    }
    
    info_file = output_root / "dataset_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

def validate_output_data(output_root: Path, logger: logging.Logger):
    """Ãâ·Â µ¥ÀÌÅÍ °ËÁõ"""
    logger.info("Validating output data...")
    
    validation_stats = {
        "total_files": 0,
        "corrupted_files": 0,
        "valid_files": 0,
        "splits": {}
    }
    
    for split in ["train", "val", "test"]:
        split_dir = output_root / split
        if not split_dir.exists():
            continue
            
        pkl_files = list(split_dir.glob("*.pkl"))
        split_stats = {
            "total": len(pkl_files),
            "valid": 0,
            "corrupted": 0,
            "sample_data": None
        }
        
        for pkl_file in pkl_files[:min(10, len(pkl_files))]:  # »ùÇÃ °ËÁõ
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # ±âº» ±¸Á¶ È®ÀÎ
                required_keys = ["csd", "label", "frequency", "quality_metrics"]
                if all(key in data for key in required_keys):
                    if split_stats["sample_data"] is None:
                        split_stats["sample_data"] = {
                            "csd_shape": data["csd"].shape,
                            "n_frequencies": len(data["frequency"]),
                            "quality_score": data["quality_metrics"].get("quality_score", "N/A")
                        }
                    split_stats["valid"] += 1
                else:
                    split_stats["corrupted"] += 1
                    
            except Exception as e:
                split_stats["corrupted"] += 1
                logger.warning(f"Corrupted file {pkl_file}: {e}")
        
        validation_stats["splits"][split] = split_stats
        validation_stats["total_files"] += split_stats["total"]
        validation_stats["valid_files"] += split_stats["valid"]
        validation_stats["corrupted_files"] += split_stats["corrupted"]
    
    # °ËÁõ °á°ú ÀúÀå
    validation_file = output_root / "validation_report.json"
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation_stats, f, indent=2, default=str)
    
    logger.info(f"Validation complete: {validation_stats['valid_files']}/{validation_stats['total_files']} files valid")
    
    return validation_stats

if __name__ == "__main__":
    try:
        # ¸ÞÀÎ Ã³¸® ½ÇÇà
        main()
        
        # Ãß°¡ À¯Æ¿¸®Æ¼ ½ÇÇà
        args = parse_arguments()
        output_root = Path(args.output_root)
        
        if not args.dry_run and output_root.exists():
            logger = logging.getLogger(__name__)
            
            # µ¥ÀÌÅÍ¼Â Á¤º¸ »ý¼º
            create_dataset_info(output_root)
            logger.info("Dataset info file created")
            
            # µ¥ÀÌÅÍ °ËÁõ
            validation_stats = validate_output_data(output_root, logger)
            
            if validation_stats["corrupted_files"] > 0:
                logger.warning(f"Found {validation_stats['corrupted_files']} corrupted files")
            else:
                logger.info("All output files validated successfully")
    
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}")
        logging.error(traceback.format_exc())
        raise