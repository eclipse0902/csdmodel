"""
EEG Connectivity Analysis - Updated Main Fine-tuning Execution Script

ÇöÀç ¸ðµ¨ ±¸Á¶¿¡ ¸Â°Ô ¾÷µ¥ÀÌÆ®:
1. EEGConnectivityModel »ç¿ë
2. Physics-aware Loss Áö¿ø
3. Enhanced Masking Strategy È£È¯
4. Mixed Precision ¹× ¸Þ¸ð¸® ÃÖÀûÈ­
5. Magnitude/Phase ±â¹Ý Ã³¸®

»ç¿ë¹ý:
python main_finetune.py --mode finetune \
    --pretrain_model ./checkpoints/best_pretrain_model.pth \
    --train_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/train \
    --val_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/val \
    --test_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/test \
    --output_dir ./results \
    --freeze_encoder
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EEGConfig
from data.dataset import EEGDataset, validate_data_directory, preview_dataset_samples
from models.hybrid_model import EEGConnectivityModel, create_finetune_model
from utils.layers import get_memory_info, clear_memory, count_parameters

class FineTuneTrainer:
    """Fine-tuning Trainer for EEG Classification"""
    
    def __init__(self, 
                 model: EEGConnectivityModel,
                 train_loader: DataLoader,
                 config: EEGConfig,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE
        
        # Training configuration
        self.train_config = config.TRAINING_CONFIG
        self.memory_config = getattr(config, 'MEMORY_CONFIG', {})
        self.num_epochs = self.train_config.get('num_epochs', 50)
        self.learning_rate = self.train_config.get('learning_rate', 1e-4)
        self.batch_size = self.train_config['batch_size']
        
        # Model to device
        self.model.to(self.device)
        
        # Memory optimization
        self.use_mixed_precision = self.memory_config.get('mixed_precision', True)
        self.gradient_checkpointing = self.memory_config.get('gradient_checkpointing', False)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.train_config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_val_auc = 0.0
        self.best_val_f1_macro = 0.0
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_f1_macro': [],
            'learning_rates': [],
            'epoch_times': [],
            'memory_usage_gb': []
        }
        
        # Directories
        self.checkpoint_dir = config.DATA_CONFIG['checkpoint_path']
        self.log_dir = config.DATA_CONFIG['log_path']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Session info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_count = sum(p.numel() for p in model.parameters())
        param_suffix = f"{param_count//1000000}M" if param_count >= 1000000 else f"{param_count//1000}k"
        self.session_id = f"{timestamp}_finetune_{param_suffix}"
        self.log_file = os.path.join(self.log_dir, f"finetune_log_{self.session_id}.json")
        
        print(f"? Fine-tune Trainer initialized:")
        print(f"   Parameters: {param_count:,}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Session ID: {self.session_id}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Single epoch training"""
        self.model.train()
        
        epoch_metrics = {
            'train_loss': 0.0,
            'train_acc': 0.0,
            'memory_peak_gb': 0.0
        }
        
        num_batches = 0
        epoch_start_time = time.time()
        initial_memory = get_memory_info()
        
        for batch_idx, (csd_data, labels) in enumerate(self.train_loader):
            csd_data = csd_data.to(self.device, non_blocking=True)
            labels = labels.squeeze().to(self.device, non_blocking=True)
            
            # Memory monitoring
            batch_start_memory = get_memory_info()
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_mixed_precision:
                with autocast():
                    # Forward pass
                    logits = self.model.forward_classification(csd_data)
                    loss = nn.functional.cross_entropy(logits, labels)
                
                # Mixed precision backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.TRAINING_CONFIG['gradient_clip_norm']
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # Regular precision
                logits = self.model.forward_classification(csd_data)
                loss = nn.functional.cross_entropy(logits, labels)
                
                # Regular backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.TRAINING_CONFIG['gradient_clip_norm']
                )
                
                self.optimizer.step()
            
            # Memory monitoring
            batch_peak_memory = get_memory_info()
            
            # Compute accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()
            
            # Accumulate metrics
            epoch_metrics['train_loss'] += loss.item()
            epoch_metrics['train_acc'] += accuracy.item()
            epoch_metrics['memory_peak_gb'] = max(epoch_metrics['memory_peak_gb'], batch_peak_memory['allocated'])
            
            num_batches += 1
            
            # Progress logging
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                memory_usage = batch_peak_memory['allocated'] - batch_start_memory['allocated']
                
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}")
                print(f"  Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}, LR: {current_lr:.2e}")
                print(f"  Memory: {memory_usage:.3f} GB")
            
            # Memory cleanup
            if batch_idx % 50 == 0:
                clear_memory()
        
        # Average metrics
        for key in epoch_metrics.keys():
            if key != 'memory_peak_gb':
                epoch_metrics[key] /= num_batches
        
        epoch_metrics['epoch_time'] = time.time() - epoch_start_time
        epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        return epoch_metrics
    
    def validate_epoch(self) -> Optional[Dict[str, float]]:
        """Validation epoch"""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        
        val_metrics = {
            'val_loss': 0.0,
            'val_acc': 0.0
        }
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        num_batches = 0
        
        with torch.no_grad():
            for csd_data, labels in self.val_loader:
                csd_data = csd_data.to(self.device, non_blocking=True)
                labels = labels.squeeze().to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model.forward_classification(csd_data)
                        loss = nn.functional.cross_entropy(logits, labels)
                else:
                    logits = self.model.forward_classification(csd_data)
                    loss = nn.functional.cross_entropy(logits, labels)
                
                # Predictions and probabilities
                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()
                
                # Accumulate metrics
                val_metrics['val_loss'] += loss.item()
                val_metrics['val_acc'] += accuracy.item()
                
                # Collect for detailed metrics
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                num_batches += 1
        
        # Average metrics
        for key in val_metrics.keys():
            val_metrics[key] /= num_batches
        
        # Compute detailed metrics
        try:
            from sklearn.metrics import roc_auc_score, f1_score
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            all_probabilities = np.array(all_probabilities)
            
            # AUC
            if all_probabilities.shape[1] == 2:
                auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
            else:
                auc_score = 0.0
            
            # F1 scores
            f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            
            val_metrics['val_auc'] = auc_score
            val_metrics['val_f1_macro'] = f1_macro
            val_metrics['val_f1_weighted'] = f1_weighted
            
        except ImportError:
            print("Warning: sklearn not available for detailed metrics")
            val_metrics['val_auc'] = 0.0
            val_metrics['val_f1_macro'] = 0.0
            val_metrics['val_f1_weighted'] = 0.0
        
        return val_metrics
    
    def test_epoch(self) -> Optional[Dict[str, float]]:
        """Test epoch"""
        if self.test_loader is None:
            return None
            
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        test_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for csd_data, labels in self.test_loader:
                csd_data = csd_data.to(self.device, non_blocking=True)
                labels = labels.squeeze().to(self.device, non_blocking=True)
                
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model.forward_classification(csd_data)
                        loss = nn.functional.cross_entropy(logits, labels)
                else:
                    logits = self.model.forward_classification(csd_data)
                    loss = nn.functional.cross_entropy(logits, labels)
                
                # Predictions and probabilities
                predictions = torch.argmax(logits, dim=1)
                probabilities = torch.softmax(logits, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                test_loss += loss.item()
                
                num_batches += 1
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = (all_predictions == all_labels).mean()
        avg_loss = test_loss / num_batches
        
        test_metrics = {
            'test_loss': avg_loss,
            'accuracy': accuracy
        }
        
        # Detailed metrics
        try:
            from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
            
            # AUC
            if all_probabilities.shape[1] == 2:
                auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
            else:
                auc_score = 0.0
            
            # F1 scores
            f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            
            # Classification report
            report = classification_report(
                all_labels, all_predictions,
                target_names=['Normal', 'Abnormal'],
                output_dict=True,
                zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_predictions)
            
            test_metrics.update({
                'auc': auc_score,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'classification_report': report,
                'confusion_matrix': cm.toList() if hasattr(cm, 'toList') else cm.tolist(),
                'predictions': all_predictions.tolist(),
                'labels': all_labels.tolist(),
                'probabilities': all_probabilities.tolist()
            })
            
        except ImportError:
            print("Warning: sklearn not available for detailed test metrics")
        
        return test_metrics
    
    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None,
                       is_best: bool = False, checkpoint_type: str = "regular"):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': self.config,
            'session_id': self.session_id,
            'model_info': count_parameters(self.model),
            'best_metrics': {
                'best_val_accuracy': self.best_val_accuracy,
                'best_val_auc': self.best_val_auc,
                'best_val_f1_macro': self.best_val_f1_macro
            }
        }
        
        # Add scaler state if using mixed precision
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if checkpoint_type == "regular":
            checkpoint_path = os.path.join(self.checkpoint_dir, f"finetune_checkpoint_epoch_{epoch:03d}.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"best_finetune_{checkpoint_type}.pth")
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            # Also save as best overall
            best_path = os.path.join(self.checkpoint_dir, "best_finetune_model.pth")
            torch.save(checkpoint, best_path)
            print(f"?? New best model saved: {checkpoint_type}")
    
    def log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """Log epoch results"""
        
        # Update history
        for key, value in train_metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)
        
        if val_metrics:
            for key, value in val_metrics.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
        
        # Console logging
        print(f"\n{'='*60}")
        print(f"?? EPOCH {epoch + 1}/{self.num_epochs} RESULTS")
        print(f"{'='*60}")
        print(f"?? Training Metrics:")
        print(f"   Loss: {train_metrics['train_loss']:.6f}")
        print(f"   Accuracy: {train_metrics['train_acc']:.4f}")
        
        if val_metrics:
            print(f"?? Validation Metrics:")
            print(f"   Loss: {val_metrics['val_loss']:.6f}")
            print(f"   Accuracy: {val_metrics['val_acc']:.4f}")
            print(f"   AUC: {val_metrics['val_auc']:.4f}")
            print(f"   F1 (Macro): {val_metrics['val_f1_macro']:.4f}")
        
        print(f"?? Training Info:")
        print(f"   Learning Rate: {train_metrics['learning_rate']:.2e}")
        print(f"   Epoch Time: {train_metrics['epoch_time']:.1f}s")
        print(f"   Memory Peak: {train_metrics['memory_peak_gb']:.3f} GB")
        print(f"   Best Val Acc: {self.best_val_accuracy:.4f}")
        
        # JSON logging
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': self._convert_to_serializable(train_metrics),
            'val_metrics': self._convert_to_serializable(val_metrics) if val_metrics else None,
            'best_metrics': {
                'best_val_accuracy': float(self.best_val_accuracy),
                'best_val_auc': float(self.best_val_auc),
                'best_val_f1_macro': float(self.best_val_f1_macro)
            },
            'session_id': self.session_id,
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'memory_estimate_mb': count_parameters(self.model)['memory_mb']
            }
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write log: {e}")
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/torch types to JSON serializable"""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def train(self) -> Dict:
        """Complete training loop"""
        
        print(f"?? STARTING FINE-TUNING")
        print(f"{'='*60}")
        print(f"?? Target: Binary Classification (Normal vs Abnormal)")
        print(f"??? Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"?? Optimizations:")
        print(f"   Mixed Precision: {self.use_mixed_precision}")
        print(f"   Gradient Checkpointing: {self.gradient_checkpointing}")
        print(f"{'='*60}")
        
        training_start_time = time.time()
        early_stopping_counter = 0
        patience = 10
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n?? Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Check for improvements
            improved = False
            
            if val_metrics:
                # Best accuracy
                if val_metrics['val_acc'] > self.best_val_accuracy:
                    self.best_val_accuracy = val_metrics['val_acc']
                    self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True, checkpoint_type="accuracy")
                    improved = True
                
                # Best AUC
                if val_metrics['val_auc'] > self.best_val_auc:
                    self.best_val_auc = val_metrics['val_auc']
                    self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True, checkpoint_type="auc")
                    improved = True
                
                # Best F1
                if val_metrics['val_f1_macro'] > self.best_val_f1_macro:
                    self.best_val_f1_macro = val_metrics['val_f1_macro']
                    self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True, checkpoint_type="f1")
                    improved = True
            
            # Early stopping
            if improved:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, train_metrics, val_metrics, checkpoint_type="regular")
            
            # Log results
            self.log_epoch_results(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if early_stopping_counter >= patience:
                print(f"?? Early stopping: No improvement for {patience} epochs")
                break
            
            # Memory cleanup
            clear_memory()
        
        total_time = time.time() - training_start_time
        
        # Test evaluation
        test_metrics = self.test_epoch()
        
        # Final results
        final_results = {
            'total_epochs_trained': self.current_epoch + 1,
            'total_training_time_hours': total_time / 3600,
            'best_metrics': {
                'best_val_accuracy': float(self.best_val_accuracy),
                'best_val_auc': float(self.best_val_auc),
                'best_val_f1_macro': float(self.best_val_f1_macro)
            },
            'test_metrics': test_metrics,
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'model_size_mb': count_parameters(self.model)['memory_mb'],
                'architecture': 'EEG Connectivity + Classification Head'
            },
            'training_history': self.training_history,
            'session_id': self.session_id
        }
        
        # Final summary
        print(f"\n?? FINE-TUNING COMPLETED!")
        print(f"{'='*60}")
        print(f"?? Training Summary:")
        print(f"   Total Time: {total_time/3600:.2f} hours")
        print(f"   Epochs: {final_results['total_epochs_trained']}")
        print(f"   Session ID: {final_results['session_id']}")
        
        print(f"\n?? Best Validation Results:")
        print(f"   Best Accuracy: {final_results['best_metrics']['best_val_accuracy']:.4f}")
        print(f"   Best AUC: {final_results['best_metrics']['best_val_auc']:.4f}")
        print(f"   Best F1: {final_results['best_metrics']['best_val_f1_macro']:.4f}")
        
        if test_metrics:
            print(f"\n?? Test Results:")
            print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Test AUC: {test_metrics.get('auc', 0.0):.4f}")
            print(f"   Test F1: {test_metrics.get('f1_macro', 0.0):.4f}")
        
        print(f"\n?? Saved Models:")
        print(f"   Best Accuracy: best_finetune_accuracy.pth")
        print(f"   Best AUC: best_finetune_auc.pth")
        print(f"   Best F1: best_finetune_f1.pth")
        print(f"   Training Log: {self.log_file}")
        print(f"{'='*60}")
        
        return final_results

def setup_finetune_training(train_data_path: str,
                           config: EEGConfig,
                           pretrain_checkpoint: str,
                           val_data_path: Optional[str] = None,
                           test_data_path: Optional[str] = None,
                           freeze_encoder: bool = False) -> Tuple[EEGConnectivityModel, DataLoader, Optional[DataLoader], Optional[DataLoader], FineTuneTrainer]:
    """Setup fine-tuning training"""
    
    print(f"?? Setting up fine-tuning...")
    print(f"   Training data: {train_data_path}")
    print(f"   Validation data: {val_data_path}")
    print(f"   Test data: {test_data_path}")
    print(f"   Pre-trained model: {pretrain_checkpoint}")
    print(f"   Freeze encoder: {freeze_encoder}")
    
    # Create fine-tuning model
    model = create_finetune_model(config, pretrain_checkpoint)
    
    if freeze_encoder:
        model.freeze_encoder()
        print("?? Encoder frozen for fine-tuning")
    
    # Create datasets
    train_dataset = EEGDataset(train_data_path, config, apply_masking=False)
    
    val_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        val_dataset = EEGDataset(val_data_path, config, apply_masking=False)
    
    test_dataset = None
    if test_data_path and os.path.exists(test_data_path):
        test_dataset = EEGDataset(test_data_path, config, apply_masking=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=config.MEMORY_CONFIG.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=config.MEMORY_CONFIG.get('num_workers', 4),
            pin_memory=True
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=config.MEMORY_CONFIG.get('num_workers', 4),
            pin_memory=True
        )
    
    # Create trainer
    trainer = FineTuneTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    print(f"? Fine-tuning setup completed!")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset) if val_dataset else 0}")
    print(f"   Test samples: {len(test_dataset) if test_dataset else 0}")
    
    return model, train_loader, val_loader, test_loader, trainer

def analyze_finetune_results(log_file_path: str) -> Dict:
    """Analyze fine-tuning results"""
    print(f"?? Analyzing fine-tuning results: {log_file_path}")
    
    try:
        training_data = []
        with open(log_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        log_entry = json.loads(line.strip())
                        training_data.append(log_entry)
                    except json.JSONDecodeError:
                        continue
        
        if not training_data:
            print("? No valid training data found")
            return {}
        
        print(f"? Loaded {len(training_data)} training epochs")
        
        # Extract metrics
        final_entry = training_data[-1]
        final_train_metrics = final_entry.get('train_metrics', {})
        final_val_metrics = final_entry.get('val_metrics', {})
        best_metrics = final_entry.get('best_metrics', {})
        model_info = final_entry.get('model_info', {})
        
        # Analysis
        analysis = {
            'training_summary': {
                'total_epochs': len(training_data),
                'session_id': final_entry.get('session_id', 'unknown'),
                'final_epoch': final_entry.get('epoch', 0),
                'model_parameters': model_info.get('parameters', 0),
                'model_size_mb': model_info.get('memory_estimate_mb', 0),
                'attention_type': 'Enhanced EEG Connectivity'
            },
            'final_performance': {
                'final_train_loss': final_train_metrics.get('train_loss', 0),
                'final_train_acc': final_train_metrics.get('train_acc', 0),
                'final_val_loss': final_val_metrics.get('val_loss', 0) if final_val_metrics else 0,
                'final_val_acc': final_val_metrics.get('val_acc', 0) if final_val_metrics else 0,
                'final_val_auc': final_val_metrics.get('val_auc', 0) if final_val_metrics else 0,
                'final_val_f1': final_val_metrics.get('val_f1_macro', 0) if final_val_metrics else 0
            },
            'best_performance': {
                'best_accuracy': best_metrics.get('best_val_accuracy', 0),
                'best_auc': best_metrics.get('best_val_auc', 0),
                'best_f1': best_metrics.get('best_val_f1_macro', 0)
            },
            'training_stability': {
                'converged': len(training_data) > 5,
                'early_stopped': len(training_data) < 30
            }
        }
        
        print(f"\n?? Analysis Complete:")
        print(f"   Epochs: {analysis['training_summary']['total_epochs']}")
        print(f"   Model: {analysis['training_summary']['model_parameters']:,} parameters")
        print(f"   Best Accuracy: {analysis['best_performance']['best_accuracy']:.4f}")
        print(f"   Best AUC: {analysis['best_performance']['best_auc']:.4f}")
        print(f"   Best F1: {analysis['best_performance']['best_f1']:.4f}")
        
        return analysis
        
    except Exception as e:
        print(f"? Error analyzing results: {str(e)}")
        return {}

def plot_finetune_curves(log_file: str, output_dir: Optional[str] = None):
    """Plot fine-tuning curves"""
    
    # Load training data
    training_data = []
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    training_data.append(json.loads(line.strip()))
                except:
                    continue
    
    if len(training_data) < 2:
        return
    
    # Extract metrics
    epochs = [entry['epoch'] for entry in training_data]
    train_losses = [entry['train_metrics']['train_loss'] for entry in training_data]
    train_accs = [entry['train_metrics']['train_acc'] for entry in training_data]
    
    val_losses = []
    val_accs = []
    val_aucs = []
    val_f1s = []
    
    for entry in training_data:
        val_metrics = entry.get('val_metrics')
        if val_metrics:
            val_losses.append(val_metrics['val_loss'])
            val_accs.append(val_metrics['val_acc'])
            val_aucs.append(val_metrics.get('val_auc', 0))
            val_f1s.append(val_metrics.get('val_f1_macro', 0))
        else:
            val_losses.append(0)
            val_accs.append(0)
            val_aucs.append(0)
            val_f1s.append(0)
    
    # Create plots
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('EEG Fine-tuning Progress', fontsize=16)
        
        # Loss curves
        axes[0,0].plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        if any(val_losses):
            axes[0,0].plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
        axes[0,0].set_title('Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0,1].plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
        if any(val_accs):
            axes[0,1].plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
        axes[0,1].set_title('Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # AUC curve
        if any(val_aucs):
            axes[1,0].plot(epochs, val_aucs, 'g-', linewidth=2)
            axes[1,0].set_title('Validation AUC')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('AUC')
            axes[1,0].grid(True, alpha=0.3)
        
        # F1 curve
        if any(val_f1s):
            axes[1,1].plot(epochs, val_f1s, 'm-', linewidth=2)
            axes[1,1].set_title('Validation F1 (Macro)')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('F1 Score')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if output_dir:
            plot_path = os.path.join(output_dir, "finetune_curves.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"?? Training curves saved to: {plot_path}")
        
        plt.close()
        
    except ImportError:
        print("?? matplotlib not available for plotting")

def finetune_mode(args):
    """Fine-tuning mode"""
    print("="*80)
    print("?? EEG CONNECTIVITY FINE-TUNING")
    print("="*80)
    
    # Configuration
    config = EEGConfig()
    if args.config_overrides:
        print(f"?? Applying config overrides: {args.config_overrides}")
    
    print(f"?? Task: Binary Classification (Normal vs Abnormal)")
    
    # Pre-trained model validation
    if not args.pretrain_model or not os.path.exists(args.pretrain_model):
        print(f"? Pre-trained model not found: {args.pretrain_model}")
        print(f"?? Tip: Run pre-training first or provide valid checkpoint path")
        return
    
    print(f"? Pre-trained model found: {args.pretrain_model}")
    
    # Try to load and analyze pre-trained model
    try:
        checkpoint = torch.load(args.pretrain_model, map_location='cpu', weights_only=False)
        if 'session_id' in checkpoint:
            print(f"   Source session: {checkpoint['session_id']}")
        if 'best_metrics' in checkpoint:
            best = checkpoint['best_metrics']
            print(f"   Pre-training performance:")
            print(f"     Best loss: {best.get('best_train_loss', 'N/A')}")
            print(f"     Best phase error: {best.get('best_phase_error_degrees', 'N/A')}¡Æ")
            print(f"     Best correlation: {best.get('best_correlation', 'N/A')}")
    except Exception as e:
        print(f"?? Warning: Could not analyze pre-trained model: {e}")
    
    # Data validation
    print(f"\n?? Validating datasets...")
    
    # Training data validation
    train_validation = validate_data_directory(args.train_data, config)
    if not train_validation['valid']:
        print(f"? Training data validation failed: {train_validation['error']}")
        return
    
    print(f"? Training data validation passed:")
    print(f"   Files: {train_validation['num_files']}")
    print(f"   Quality score: {train_validation['quality_score']:.3f}")
    print(f"   Class distribution: {train_validation['class_distribution']}")
    
    # Validation data validation (optional)
    if args.val_data and os.path.exists(args.val_data):
        val_validation = validate_data_directory(args.val_data, config)
        if val_validation['valid']:
            print(f"? Validation data: {val_validation['num_files']} files")
        else:
            print(f"?? Validation data issues: {val_validation['error']}")
    
    # Test data validation (optional)
    if args.test_data and os.path.exists(args.test_data):
        test_validation = validate_data_directory(args.test_data, config)
        if test_validation['valid']:
            print(f"? Test data: {test_validation['num_files']} files")
        else:
            print(f"?? Test data issues: {test_validation['error']}")
    
    # Data preview
    if args.preview_data:
        print(f"\n?? Data preview:")
        preview = preview_dataset_samples(args.train_data, num_samples=3, config=config)
        for sample in preview['samples']:
            label_name = "Abnormal" if sample['label'] == 1 else "Normal"
            print(f"   Sample {sample['index']}: {label_name}, "
                  f"Magnitude={sample['magnitude_stats']['mean']:.3f}¡¾{sample['magnitude_stats']['std']:.3f}")
    
    # Setup fine-tuning
    print(f"\n?? Setting up fine-tuning...")
    
    model, train_loader, val_loader, test_loader, trainer = setup_finetune_training(
        train_data_path=args.train_data,
        config=config,
        pretrain_checkpoint=args.pretrain_model,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        freeze_encoder=args.freeze_encoder
    )
    
    # Model summary
    model_info = model.get_model_analysis(torch.randn(1, 20, 19, 19, 2))
    print(f"\n??? Model Information:")
    print(f"   Architecture: Enhanced EEG Connectivity + Classification Head")
    print(f"   Total Parameters: {model_info['model_info']['total_parameters']:,}")
    print(f"   Memory Estimate: ~{model_info['model_info']['memory_mb']:.1f} MB")
    print(f"   Encoder Frozen: {args.freeze_encoder}")
    
    # Start fine-tuning
    print(f"\n?? Starting fine-tuning...")
    
    training_results = trainer.train()
    
    # Results summary
    print(f"\n?? Fine-tuning Results Summary:")
    print(f"   Total epochs: {training_results['total_epochs_trained']}")
    print(f"   Training time: {training_results['total_training_time_hours']:.2f} hours")
    
    best_metrics = training_results['best_metrics']
    print(f"   ?? Best validation performance:")
    print(f"     Accuracy: {best_metrics['best_val_accuracy']:.4f}")
    print(f"     AUC: {best_metrics['best_val_auc']:.4f}")
    print(f"     F1 Score: {best_metrics['best_val_f1_macro']:.4f}")
    
    if 'test_metrics' in training_results and training_results['test_metrics']:
        test_metrics = training_results['test_metrics']
        print(f"   ?? Test performance:")
        print(f"     Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"     AUC: {test_metrics.get('auc', 0.0):.4f}")
        print(f"     F1 Score: {test_metrics.get('f1_macro', 0.0):.4f}")
    
    print(f"   Session ID: {training_results['session_id']}")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, f"finetune_results_{training_results['session_id']}.json")
        
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        print(f"?? Results saved to: {results_path}")
        
        # Generate training curves
        try:
            plot_finetune_curves(trainer.log_file, args.output_dir)
        except Exception as e:
            print(f"?? Could not generate plots: {e}")
    
    # Performance analysis
    print(f"\n?? Performance Analysis:")
    
    best_acc = best_metrics['best_val_accuracy']
    best_auc = best_metrics['best_val_auc']
    
    if best_acc > 0.85:
        print(f"   ?? Excellent performance! (Accuracy > 85%)")
    elif best_acc > 0.75:
        print(f"   ? Good performance (Accuracy > 75%)")
    elif best_acc > 0.65:
        print(f"   ?? Decent performance (Accuracy > 65%)")
    else:
        print(f"   ?? Consider hyperparameter tuning or more training data")
    
    if best_auc > 0.90:
        print(f"   ?? Outstanding AUC! (> 90%)")
    elif best_auc > 0.80:
        print(f"   ? Good AUC (> 80%)")
    else:
        print(f"   ?? AUC could be improved")
    
    print(f"\n?? Fine-tuning completed successfully!")
    print(f"   ?? Best models saved in: {trainer.checkpoint_dir}")
    print(f"   ?? Training log: {trainer.log_file}")

def evaluate_mode(args):
    """Evaluation mode - Fine-tuned model evaluation"""
    print("="*80)
    print("?? EEG CONNECTIVITY MODEL EVALUATION")
    print("="*80)
    
    if not args.model_path or not os.path.exists(args.model_path):
        print(f"? Model not found: {args.model_path}")
        return
    
    config = EEGConfig()
    
    # Load model
    print(f"?? Loading fine-tuned model from: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=config.DEVICE, weights_only=False)
        
        # Model info
        if 'session_id' in checkpoint:
            print(f"   Session: {checkpoint['session_id']}")
        
        # Create model and load weights
        model = EEGConnectivityModel.load_model(args.model_path, mode='inference')
        model.eval()
        
    except Exception as e:
        print(f"? Failed to load model: {str(e)}")
        return
    
    # Data loading
    if not args.eval_data or not os.path.exists(args.eval_data):
        print(f"? Evaluation data not found: {args.eval_data}")
        return
    
    # Validate evaluation data
    eval_validation = validate_data_directory(args.eval_data, config)
    if not eval_validation['valid']:
        print(f"? Evaluation data validation failed: {eval_validation['error']}")
        return
    
    print(f"? Evaluation data validated:")
    print(f"   Files: {eval_validation['num_files']}")
    print(f"   Class distribution: {eval_validation['class_distribution']}")
    
    # Create data loader
    eval_dataset = EEGDataset(args.eval_data, config, apply_masking=False)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    print(f"?? Evaluating on {len(eval_dataset)} samples...")
    
    # Evaluation
    device = config.DEVICE
    model.to(device)
    
    # Classification evaluation
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_losses = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"?? Running evaluation...")
    
    with torch.no_grad():
        for batch_idx, (csd_data, labels) in enumerate(eval_loader):
            csd_data = csd_data.to(device)
            labels = labels.squeeze().to(device)
            
            # Forward pass
            logits = model.forward_classification(csd_data)
            loss = criterion(logits, labels)
            
            # Predictions
            predictions = torch.argmax(logits, dim=1)
            probabilities = torch.softmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_losses.append(loss.item())
            
            if batch_idx % 10 == 0:
                print(f"   Processed {batch_idx}/{len(eval_loader)} batches")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Compute detailed metrics
    try:
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        avg_loss = np.mean(all_losses)
        accuracy = (all_predictions == all_labels).mean()
        
        # Detailed classification metrics
        report = classification_report(
            all_labels, all_predictions,
            target_names=['Normal', 'Abnormal'],
            output_dict=True,
            zero_division=0
        )
        
        # AUC
        if all_probabilities.shape[1] == 2:
            auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            auc_score = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Class-wise accuracy
        class_accuracies = {}
        for class_idx in [0, 1]:
            class_mask = (all_labels == class_idx)
            if class_mask.sum() > 0:
                class_acc = (all_predictions[class_mask] == all_labels[class_mask]).mean()
                class_name = 'Normal' if class_idx == 0 else 'Abnormal'
                class_accuracies[class_name] = class_acc
        
        # Results summary
        evaluation_results = {
            'evaluation_type': 'classification',
            'total_samples': len(all_predictions),
            'average_loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc_score,
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'class_accuracies': class_accuracies,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist(),
            'probabilities': all_probabilities.tolist()
        }
        
        print(f"\n?? Evaluation Results:")
        print(f"   ?? Overall Performance:")
        print(f"     Total samples: {len(all_predictions)}")
        print(f"     Average loss: {avg_loss:.4f}")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     AUC: {auc_score:.4f}")
        print(f"     F1 (Macro): {report['macro avg']['f1-score']:.4f}")
        print(f"     F1 (Weighted): {report['weighted avg']['f1-score']:.4f}")
        
        print(f"\n   ?? Class-wise Performance:")
        for class_name in ['Normal', 'Abnormal']:
            if class_name in report:
                metrics = report[class_name]
                print(f"     {class_name}:")
                print(f"       Precision: {metrics['precision']:.4f}")
                print(f"       Recall: {metrics['recall']:.4f}")
                print(f"       F1-Score: {metrics['f1-score']:.4f}")
                print(f"       Support: {metrics['support']}")
        
        print(f"\n   ?? Confusion Matrix:")
        print(f"        Predicted")
        print(f"        Normal  Abnormal")
        print(f"Normal   {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"Abnormal {cm[1,0]:4d}    {cm[1,1]:4d}")
        
        # Performance interpretation
        print(f"\n?? Performance Analysis:")
        if accuracy > 0.85:
            print(f"   ?? Excellent accuracy (> 85%)")
        elif accuracy > 0.75:
            print(f"   ? Good accuracy (> 75%)")
        else:
            print(f"   ?? Accuracy could be improved")
        
        if auc_score > 0.90:
            print(f"   ?? Outstanding AUC (> 90%)")
        elif auc_score > 0.80:
            print(f"   ? Good AUC (> 80%)")
        else:
            print(f"   ?? AUC could be improved")
        
        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            results_path = os.path.join(args.output_dir, "evaluation_results.json")
            
            with open(results_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            print(f"\n?? Results saved to: {results_path}")
        
        print(f"\n?? Evaluation completed successfully!")
        
    except ImportError:
        print("? sklearn not available for detailed metrics")

def analyze_mode(args):
    """Analysis mode - training results analysis"""
    print("="*80)
    print("?? EEG FINE-TUNING ANALYSIS")
    print("="*80)
    
    if not args.log_file or not os.path.exists(args.log_file):
        print(f"? Log file not found: {args.log_file}")
        return
    
    print(f"?? Analyzing fine-tuning log: {args.log_file}")
    
    # Analyze training results
    analysis_results = analyze_finetune_results(args.log_file)
    
    if not analysis_results:
        print(f"? Failed to analyze fine-tuning results")
        return
    
    # Print analysis
    print(f"\n?? Fine-tuning Analysis Results:")
    print(f"   Session: {analysis_results['training_summary']['session_id']}")
    print(f"   Total epochs: {analysis_results['training_summary']['total_epochs']}")
    print(f"   Attention: {analysis_results['training_summary']['attention_type']}")
    print(f"   Converged: {analysis_results['training_stability']['converged']}")
    print(f"   Early stopped: {analysis_results['training_stability']['early_stopped']}")
    
    print(f"\n?? Best Performance:")
    print(f"   Best Accuracy: {analysis_results['best_performance']['best_accuracy']:.4f}")
    print(f"   Best AUC: {analysis_results['best_performance']['best_auc']:.4f}")
    print(f"   Best F1: {analysis_results['best_performance']['best_f1']:.4f}")
    
    # Save analysis
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        analysis_path = os.path.join(args.output_dir, "finetune_analysis.json")
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"?? Analysis saved to: {analysis_path}")
    
    # Generate training curves
    try:
        plot_finetune_curves(args.log_file, args.output_dir)
        print(f"?? Training curves generated")
    except Exception as e:
        print(f"?? Could not generate plots: {e}")
    
    print(f"\n? Analysis completed successfully!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="EEG Connectivity Fine-tuning with Enhanced Model Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tuning
  python main_finetune.py --mode finetune \
      --pretrain_model ./checkpoints/best_pretrain_model.pth \
      --train_data /path/to/train \
      --val_data /path/to/val \
      --output_dir ./results
  
  # Evaluation
  python main_finetune.py --mode evaluate \
      --model_path ./checkpoints/best_finetune_accuracy.pth \
      --eval_data /path/to/test \
      --output_dir ./results
  
  # Analysis
  python main_finetune.py --mode analyze \
      --log_file ./logs/finetune_log_*.json \
      --output_dir ./analysis
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['finetune', 'evaluate', 'analyze'], 
                       required=True, 
                       help='Operation mode')
    
    # Data paths
    parser.add_argument('--train_data', type=str, help='Training data directory')
    parser.add_argument('--val_data', type=str, help='Validation data directory')
    parser.add_argument('--test_data', type=str, help='Test data directory')
    parser.add_argument('--eval_data', type=str, help='Evaluation data directory')
    
    # Model paths
    parser.add_argument('--model_path', type=str, help='Fine-tuned model path for evaluation')
    parser.add_argument('--pretrain_model', type=str, help='Pre-trained model checkpoint path')
    
    # Fine-tuning options
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder during fine-tuning')
    parser.add_argument('--preview_data', action='store_true',
                       help='Preview dataset samples before training')
    
    # Analysis options
    parser.add_argument('--log_file', type=str, help='Training log file for analysis')
    
    # Output options
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    
    # Configuration
    parser.add_argument('--config_overrides', type=str, 
                       help='JSON string with config overrides')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device and print system info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"??? System Information:")
    print(f"   Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute based on mode
    try:
        if args.mode == 'finetune':
            if not args.pretrain_model:
                print("? --pretrain_model required for fine-tuning")
                return
            if not args.train_data:
                print("? --train_data required for fine-tuning")
                return
            finetune_mode(args)
            
        elif args.mode == 'evaluate':
            if not args.model_path:
                print("? --model_path required for evaluation")
                return
            if not args.eval_data:
                print("? --eval_data required for evaluation")
                return
            evaluate_mode(args)
            
        elif args.mode == 'analyze':
            if not args.log_file:
                print("? --log_file required for analysis")
                return
            analyze_mode(args)
            
    except KeyboardInterrupt:
        print(f"\n?? Process interrupted by user")
    except Exception as e:
        print(f"? Error in {args.mode} mode: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()