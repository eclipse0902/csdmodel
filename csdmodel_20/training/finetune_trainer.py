"""
EEG Connectivity Analysis - Enhanced Fine-tuning Trainer

ÇÙ½É Æ¯Â¡:
1. Pre-trained ¸ðµ¨ ·Îµù ¹× Classification Head Ãß°¡
2. Cross-Attention & Self-Attention ÀÚµ¿ È£È¯
3. TUH Abnormal Dataset Áö¿ø
4. ¸Þ¸ð¸® ÃÖÀûÈ­ ¹× ¼º´É ¸ð´ÏÅÍ¸µ
5. »ó¼¼ÇÑ Å¬·¡½ºº° ºÐ¼®
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EEGConfig
from data.dataset import EEGDataset, create_data_loaders
from models.hybrid_model import EEGConnectivityModel, create_finetune_model
from utils.losses import EEGLossCalculator, EEGMetricsCalculator
from utils.layers import get_memory_info, clear_memory, count_parameters

class EEGFinetuneTrainer:
    """
    Enhanced EEG Fine-tuning Trainer
    
    Cross-Attention & Self-Attention ¿ÏÀü È£È¯
    TUH Abnormal Dataset ÃÖÀûÈ­
    """
    
    def __init__(self, 
                 model: EEGConnectivityModel,
                 train_loader: DataLoader,
                 config: EEGConfig = None,
                 val_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None,
                 pretrain_checkpoint: Optional[str] = None):
        
        if config is None:
            config = EEGConfig()
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE
        
        # Fine-tuning configuration
        self.train_config = config.TRAINING_CONFIG
        self.memory_config = getattr(config, 'MEMORY_CONFIG', {})
        self.num_epochs = self.train_config['num_epochs']
        self.learning_rate = self.train_config['learning_rate']
        self.batch_size = self.train_config['batch_size']
        
        # Model to device
        self.model.to(self.device)
        
        # ?? Pre-trained weights ·Îµù
        if pretrain_checkpoint:
            success = self.load_pretrained_weights(pretrain_checkpoint)
            if not success:
                print("?? Warning: Failed to load pre-trained weights, training from scratch")
        
        # ?? Classification-specific setup
        self.num_classes = config.NUM_CLASSES
        self.class_names = ['Normal', 'Abnormal']
        
        # Enhanced memory optimization
        self.use_mixed_precision = self.memory_config.get('mixed_precision', True)
        self.gradient_checkpointing = self.memory_config.get('gradient_checkpointing', False)
        
        # =============== OPTIMIZER ===============
        # ´Ù¸¥ learning rate for encoder vs classifier
        encoder_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'classification_head' in name:
                classifier_params.append(param)
            else:
                encoder_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': self.learning_rate * 0.1},  # Encoder: ³·Àº LR
            {'params': classifier_params, 'lr': self.learning_rate}       # Classifier: ³ôÀº LR
        ], weight_decay=self.train_config['weight_decay'])
        
        # =============== SCHEDULER ===============
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.num_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        # =============== MIXED PRECISION ===============
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # =============== LOSS & METRICS ===============
        # Classification loss with class balancing
        self.criterion = nn.CrossEntropyLoss()
        
        # Class weights for imbalanced data
        if hasattr(train_loader.dataset, 'dataset_stats'):
            class_dist = train_loader.dataset.dataset_stats['class_distribution']
            total = sum(class_dist.values())
            class_weights = torch.tensor([
                total / (len(class_dist) * class_dist.get(0, 1)),
                total / (len(class_dist) * class_dist.get(1, 1))
            ]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"?? Class weights applied: {class_weights.cpu().numpy()}")
        
        # =============== TRAINING STATE ===============
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_val_f1 = 0.0
        
        # Enhanced training history
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': [],
            'learning_rates': [], 'epoch_times': [], 'memory_usage_gb': [],
            'class_metrics': {'precision': [], 'recall': [], 'f1': []},
            'attention_type': 'cross_attention' if config.USE_CROSS_ATTENTION else 'self_attention'
        }
        
        # =============== DIRECTORIES ===============
        self.checkpoint_dir = config.DATA_CONFIG['checkpoint_path']
        self.log_dir = config.DATA_CONFIG['log_path']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Session info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attention_type = "cross" if config.USE_CROSS_ATTENTION else "self"
        self.session_id = f"{timestamp}_finetune_{attention_type}"
        self.log_file = os.path.join(self.log_dir, f"finetune_log_{self.session_id}.json")
        
        # Model analysis
        model_analysis = count_parameters(self.model)
        
        print(f"?? Enhanced EEG Fine-tuning Trainer:")
        print(f"   Model: {model.__class__.__name__}")
        print(f"   ?? Attention: {'Cross-Attention' if config.USE_CROSS_ATTENTION else 'Self-Attention'}")
        print(f"   Parameters: {model_analysis['total_parameters']:,}")
        print(f"   Classes: {self.num_classes} ({self.class_names})")
        print(f"   Train samples: {len(train_loader.dataset)}")
        print(f"   Val samples: {len(val_loader.dataset) if val_loader else 0}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Encoder LR: {self.learning_rate * 0.1:.2e}")
        print(f"   Classifier LR: {self.learning_rate:.2e}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Session ID: {self.session_id}")
    
    def load_pretrained_weights(self, checkpoint_path: str) -> bool:
        """?? Pre-trained weights ·Îµù (Cross/Self-Attention È£È¯)"""
        try:
            print(f"?? Loading pre-trained weights from: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                pretrained_state = checkpoint['model_state_dict']
            else:
                pretrained_state = checkpoint
            
            # ?? Encoder¸¸ ·Îµù (Classification head Á¦¿Ü)
            encoder_state = {}
            for key, value in pretrained_state.items():
                if not key.startswith('classification_head'):
                    encoder_state[key] = value
            
            # ¸ðµ¨¿¡ ·Îµù
            missing_keys, unexpected_keys = self.model.load_state_dict(encoder_state, strict=False)
            
            # ¿¹»ó missing keys (classification head)
            expected_missing = [k for k in missing_keys if 'classification_head' in k]
            unexpected_missing = [k for k in missing_keys if k not in expected_missing]
            
            if unexpected_missing:
                print(f"?? Unexpected missing keys: {unexpected_missing}")
            
            if unexpected_keys:
                print(f"?? Unexpected keys in checkpoint: {unexpected_keys}")
            
            print(f"? Pre-trained encoder loaded successfully!")
            print(f"   Loaded parameters: {len(encoder_state)}")
            print(f"   Expected missing (classifier): {len(expected_missing)}")
            
            # Pre-training Á¤º¸ Ãâ·Â
            if 'session_id' in checkpoint:
                print(f"   Source session: {checkpoint['session_id']}")
            if 'best_metrics' in checkpoint:
                best = checkpoint['best_metrics']
                print(f"   Pre-train performance:")
                print(f"     Best loss: {best.get('best_train_loss', 'N/A')}")
                print(f"     Best phase error: {best.get('best_phase_error_degrees', 'N/A')}¡Æ")
            
            return True
            
        except Exception as e:
            print(f"? Failed to load pre-trained weights: {str(e)}")
            return False
    
    def train_epoch(self) -> Dict[str, float]:
        """Enhanced single epoch training"""
        self.model.train()
        
        epoch_metrics = {
            'total_loss': 0.0,
            'accuracy': 0.0,
            'predictions': [],
            'labels': [],
            'class_correct': [0] * self.num_classes,
            'class_total': [0] * self.num_classes,
            'memory_peak_gb': 0.0
        }
        
        num_batches = 0
        epoch_start_time = time.time()
        initial_memory = get_memory_info()
        
        for batch_idx, (csd_data, labels) in enumerate(self.train_loader):
            csd_data = csd_data.to(self.device, non_blocking=True)
            labels = labels.squeeze().to(self.device, non_blocking=True)
            
            batch_start_memory = get_memory_info()
            
            # =============== FORWARD PASS ===============
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_mixed_precision:
                with autocast():
                    logits = self.model.forward_classification(csd_data)
                    loss = self.criterion(logits, labels)
                
                # Mixed precision backward
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_config['gradient_clip_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                logits = self.model.forward_classification(csd_data)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_config['gradient_clip_norm']
                )
                self.optimizer.step()
            
            # =============== METRICS ===============
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()
                
                # Store for epoch-level metrics
                epoch_metrics['predictions'].extend(predictions.cpu().numpy())
                epoch_metrics['labels'].extend(labels.cpu().numpy())
                
                # Class-wise accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = predictions[i].item()
                    epoch_metrics['class_total'][label] += 1
                    if pred == label:
                        epoch_metrics['class_correct'][label] += 1
                
                # Accumulate metrics
                epoch_metrics['total_loss'] += loss.item()
                epoch_metrics['accuracy'] += accuracy.item()
            
            # Memory monitoring
            batch_peak_memory = get_memory_info()
            epoch_metrics['memory_peak_gb'] = max(
                epoch_metrics['memory_peak_gb'], 
                batch_peak_memory['allocated']
            )
            
            num_batches += 1
            
            # Progress logging
            if batch_idx % 25 == 0:
                current_lr_encoder = self.optimizer.param_groups[0]['lr']
                current_lr_classifier = self.optimizer.param_groups[1]['lr']
                memory_usage = batch_peak_memory['allocated'] - batch_start_memory['allocated']
                
                print(f"Epoch {self.current_epoch+1}, Batch {batch_idx}/{len(self.train_loader)}")
                print(f"  Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}")
                print(f"  LR (Enc/Cls): {current_lr_encoder:.2e}/{current_lr_classifier:.2e}")
                print(f"  Memory: +{memory_usage:.3f} GB")
            
            # Memory cleanup
            if batch_idx % 100 == 0:
                clear_memory()
        
        # Average metrics
        epoch_metrics['total_loss'] /= num_batches
        epoch_metrics['accuracy'] /= num_batches
        epoch_metrics['epoch_time'] = time.time() - epoch_start_time
        epoch_metrics['learning_rates'] = {
            'encoder': self.optimizer.param_groups[0]['lr'],
            'classifier': self.optimizer.param_groups[1]['lr']
        }
        
        # Class-wise accuracies
        epoch_metrics['class_accuracies'] = {}
        for i in range(self.num_classes):
            if epoch_metrics['class_total'][i] > 0:
                acc = epoch_metrics['class_correct'][i] / epoch_metrics['class_total'][i]
                epoch_metrics['class_accuracies'][self.class_names[i]] = acc
        
        return epoch_metrics
    
    def validate_epoch(self) -> Optional[Dict[str, float]]:
        """Enhanced validation with detailed metrics"""
        if self.val_loader is None:
            return None
            
        self.model.eval()
        
        val_metrics = {
            'total_loss': 0.0,
            'accuracy': 0.0,
            'predictions': [],
            'labels': [],
            'probabilities': []
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for csd_data, labels in self.val_loader:
                csd_data = csd_data.to(self.device, non_blocking=True)
                labels = labels.squeeze().to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model.forward_classification(csd_data)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model.forward_classification(csd_data)
                    loss = self.criterion(logits, labels)
                
                # Metrics
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()
                probabilities = torch.softmax(logits, dim=1)
                
                # Store for detailed analysis
                val_metrics['predictions'].extend(predictions.cpu().numpy())
                val_metrics['labels'].extend(labels.cpu().numpy())
                val_metrics['probabilities'].extend(probabilities.cpu().numpy())
                
                # Accumulate
                val_metrics['total_loss'] += loss.item()
                val_metrics['accuracy'] += accuracy.item()
                num_batches += 1
        
        # Average metrics
        val_metrics['total_loss'] /= num_batches
        val_metrics['accuracy'] /= num_batches
        
        # ?? Detailed classification metrics
        val_metrics = self._compute_detailed_metrics(val_metrics)
        
        return val_metrics
    
    def _compute_detailed_metrics(self, val_metrics: Dict) -> Dict:
        """»ó¼¼ÇÑ ºÐ·ù ¼º´É ÁöÇ¥ °è»ê"""
        y_true = np.array(val_metrics['labels'])
        y_pred = np.array(val_metrics['predictions'])
        y_prob = np.array(val_metrics['probabilities'])
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        # Additional metrics
        try:
            if y_prob.shape[1] == 2:  # Binary classification
                auc_score = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc_score = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            auc_score = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Update metrics
        val_metrics.update({
            'auc': auc_score,
            'f1_macro': report['macro avg']['f1-score'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_metrics': {
                self.class_names[i]: {
                    'precision': report[self.class_names[i]]['precision'],
                    'recall': report[self.class_names[i]]['recall'],
                    'f1-score': report[self.class_names[i]]['f1-score'],
                    'support': report[self.class_names[i]]['support']
                } for i in range(self.num_classes) if self.class_names[i] in report
            }
        })
        
        return val_metrics
    
    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None,
                       is_best: str = None):
        """Enhanced checkpoint saving - FIXED METRIC KEYS"""
        
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
                'best_val_acc': self.best_val_acc,
                'best_val_auc': self.best_val_auc,
                'best_val_f1': self.best_val_f1
            },
            'attention_type': 'cross_attention' if self.config.USE_CROSS_ATTENTION else 'self_attention'
        }
        
        # Mixed precision scaler
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best and val_metrics:
            # ?? FIX: Use correct metric keys
            metric_mapping = {
                'accuracy': 'accuracy',
                'auc': 'auc', 
                'f1': 'f1_macro'  # This is the fix!
            }
            
            metric_key = metric_mapping.get(is_best, is_best)
            if metric_key in val_metrics:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"best_finetune_{is_best}.pth")
                print(f"?? New best {is_best}: {val_metrics[metric_key]:.4f}")
            else:
                print(f"?? Warning: Metric '{metric_key}' not found in val_metrics. Available keys: {list(val_metrics.keys())}")
                checkpoint_path = os.path.join(self.checkpoint_dir, f"finetune_epoch_{epoch:03d}.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"finetune_epoch_{epoch:03d}.pth")
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = os.path.join(self.checkpoint_dir, "latest_finetune.pth")
        torch.save(checkpoint, latest_path)
    
    def log_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Optional[Dict] = None):
        """Enhanced epoch logging with detailed classification metrics"""
        
        # Update history
        self.training_history['train_loss'].append(train_metrics['total_loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['learning_rates'].append(train_metrics['learning_rates'])
        self.training_history['epoch_times'].append(train_metrics['epoch_time'])
        self.training_history['memory_usage_gb'].append(train_metrics['memory_peak_gb'])
        
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_auc'].append(val_metrics['auc'])
            self.training_history['val_f1'].append(val_metrics['f1_macro'])
        
        # Console logging
        print(f"\n{'='*60}")
        print(f"?? EPOCH {epoch + 1}/{self.num_epochs} FINE-TUNING RESULTS")
        print(f"{'='*60}")
        print(f"?? Training Metrics:")
        print(f"   Loss: {train_metrics['total_loss']:.6f}")
        print(f"   Accuracy: {train_metrics['accuracy']:.4f}")
        
        if 'class_accuracies' in train_metrics:
            print(f"   Class Accuracies:")
            for class_name, acc in train_metrics['class_accuracies'].items():
                print(f"     {class_name}: {acc:.4f}")
        
        if val_metrics:
            print(f"?? Validation Metrics:")
            print(f"   Loss: {val_metrics['total_loss']:.6f}")
            print(f"   Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"   AUC: {val_metrics['auc']:.4f}")
            print(f"   F1 (Macro): {val_metrics['f1_macro']:.4f}")
            print(f"   F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            
            print(f"?? Class-wise Performance:")
            for class_name, metrics in val_metrics['class_metrics'].items():
                print(f"   {class_name}:")
                print(f"     Precision: {metrics['precision']:.4f}")
                print(f"     Recall: {metrics['recall']:.4f}")
                print(f"     F1-Score: {metrics['f1-score']:.4f}")
        
        print(f"?? Training Info:")
        print(f"   Encoder LR: {train_metrics['learning_rates']['encoder']:.2e}")
        print(f"   Classifier LR: {train_metrics['learning_rates']['classifier']:.2e}")
        print(f"   Epoch Time: {train_metrics['epoch_time']:.1f}s")
        print(f"   Memory Peak: {train_metrics['memory_peak_gb']:.3f} GB")
        print(f"   ?? Attention: {'Cross' if self.config.USE_CROSS_ATTENTION else 'Self'}")
        
        # JSON logging
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train_metrics': self._convert_to_serializable(train_metrics),
            'val_metrics': self._convert_to_serializable(val_metrics) if val_metrics else None,
            'best_metrics': {
                'best_val_acc': float(self.best_val_acc),
                'best_val_auc': float(self.best_val_auc),
                'best_val_f1': float(self.best_val_f1)
            },
            'session_id': self.session_id,
            'attention_type': self.training_history['attention_type']
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
        elif hasattr(obj, 'item') and hasattr(obj, 'size'):
            # Handle tensor/numpy scalars
            if obj.size == 1:
                return obj.item()
            else:
                return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        else:
            # Try to convert to string as fallback
            try:
                return str(obj)
            except:
                return None
    
    def train(self) -> Dict:
        """?? Complete fine-tuning process"""
        
        print(f"?? STARTING ENHANCED FINE-TUNING")
        print(f"{'='*60}")
        print(f"?? Attention Type: {'Cross-Attention' if self.config.USE_CROSS_ATTENTION else 'Self-Attention'}")
        print(f"?? Dataset: TUH Abnormal EEG")
        print(f"?? Task: Binary Classification (Normal vs Abnormal)")
        print(f"?? Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"? Optimization: Mixed Precision={self.use_mixed_precision}")
        print(f"{'='*60}")
        
        training_start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n?? Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Check for improvements
            improved = []
            
            if val_metrics:
                # Best accuracy
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint(epoch, train_metrics, val_metrics, 'accuracy')
                    improved.append(f"Accuracy: {val_metrics['accuracy']:.4f}")
                
                # Best AUC
                if val_metrics['auc'] > self.best_val_auc:
                    self.best_val_auc = val_metrics['auc']
                    self.save_checkpoint(epoch, train_metrics, val_metrics, 'auc')
                    improved.append(f"AUC: {val_metrics['auc']:.4f}")
                
                # Best F1 - ?? FIXED: Use correct key
                if val_metrics['f1_macro'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1_macro']
                    self.save_checkpoint(epoch, train_metrics, val_metrics, 'f1')
                    improved.append(f"F1: {val_metrics['f1_macro']:.4f}")
            
            if improved:
                print(f"?? Improvements: {', '.join(improved)}")
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Log results
            self.log_epoch_results(epoch, train_metrics, val_metrics)
            
            # Memory cleanup
            clear_memory()
        
        total_time = time.time() - training_start_time
        
        # Final results
        final_results = {
            'total_epochs_trained': self.num_epochs,
            'total_training_time_hours': total_time / 3600,
            'best_metrics': {
                'best_val_accuracy': float(self.best_val_acc),
                'best_val_auc': float(self.best_val_auc),
                'best_val_f1_macro': float(self.best_val_f1)  # ?? FIXED: Use consistent naming
            },
            'final_metrics': {
                'final_train_acc': self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0,
                'final_val_acc': self.training_history['val_acc'][-1] if self.training_history['val_acc'] else 0
            },
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'attention_type': self.training_history['attention_type']
            },
            'training_history': self.training_history,
            'session_id': self.session_id
        }
        
        # Test evaluation if available
        if self.test_loader:
            test_results = self.evaluate_test_set()
            final_results['test_metrics'] = test_results
        
        self._print_final_summary(final_results, total_time)
        
        return final_results
    
    def evaluate_test_set(self) -> Dict:
        """Test set evaluation with detailed analysis"""
        print(f"\n?? Evaluating on Test Set...")
        
        self.model.eval()
        
        test_metrics = {
            'total_loss': 0.0,
            'accuracy': 0.0,
            'predictions': [],
            'labels': [],
            'probabilities': []
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for csd_data, labels in self.test_loader:
                csd_data = csd_data.to(self.device, non_blocking=True)
                labels = labels.squeeze().to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model.forward_classification(csd_data)
                        loss = self.criterion(logits, labels)
                else:
                    logits = self.model.forward_classification(csd_data)
                    loss = self.criterion(logits, labels)
                
                # Metrics
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()
                probabilities = torch.softmax(logits, dim=1)
                
                # Store for analysis
                test_metrics['predictions'].extend(predictions.cpu().numpy())
                test_metrics['labels'].extend(labels.cpu().numpy())
                test_metrics['probabilities'].extend(probabilities.cpu().numpy())
                
                # Accumulate
                test_metrics['total_loss'] += loss.item()
                test_metrics['accuracy'] += accuracy.item()
                num_batches += 1
        
        # Average metrics
        test_metrics['total_loss'] /= num_batches
        test_metrics['accuracy'] /= num_batches
        
        # Detailed metrics
        test_metrics = self._compute_detailed_metrics(test_metrics)
        
        print(f"?? Test Results:")
        print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"   AUC: {test_metrics['auc']:.4f}")
        print(f"   F1 (Macro): {test_metrics['f1_macro']:.4f}")
        
        return test_metrics
    
    def _print_final_summary(self, results: Dict, total_time: float):
        """Enhanced final summary"""
        print(f"\n? FINE-TUNING COMPLETED!")
        print(f"{'='*60}")
        print(f"?? Training Summary:")
        print(f"   Total Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"   Epochs: {results['total_epochs_trained']}")
        print(f"   Session ID: {results['session_id']}")
        print(f"   ?? Attention: {results['model_info']['attention_type']}")
        print(f"   Model Parameters: {results['model_info']['total_parameters']:,}")
        
        print(f"\n?? Best Validation Results:")
        print(f"   Best Accuracy: {results['best_metrics']['best_val_accuracy']:.4f}")
        print(f"   Best AUC: {results['best_metrics']['best_val_auc']:.4f}")
        print(f"   Best F1: {results['best_metrics']['best_val_f1_macro']:.4f}")
        
        print(f"\n?? Final Performance:")
        print(f"   Final Train Acc: {results['final_metrics']['final_train_acc']:.4f}")
        print(f"   Final Val Acc: {results['final_metrics']['final_val_acc']:.4f}")
        
        if 'test_metrics' in results:
            test = results['test_metrics']
            print(f"\n?? Test Performance:")
            print(f"   Test Accuracy: {test['accuracy']:.4f}")
            print(f"   Test AUC: {test['auc']:.4f}")
            print(f"   Test F1: {test['f1_macro']:.4f}")
        
        print(f"\n?? Saved Models:")
        print(f"   Best Accuracy: best_finetune_accuracy.pth")
        print(f"   Best AUC: best_finetune_auc.pth")
        print(f"   Best F1: best_finetune_f1.pth")
        print(f"   Latest: latest_finetune.pth")
        print(f"   Training Log: {self.log_file}")
        print(f"{'='*60}")


# =============== SETUP FUNCTIONS ===============

def setup_finetune_training(train_data_path: str,
                           config: EEGConfig = None,
                           pretrain_checkpoint: str = None,
                           val_data_path: Optional[str] = None,
                           test_data_path: Optional[str] = None,
                           freeze_encoder: bool = False) -> Tuple[EEGConnectivityModel, DataLoader, DataLoader, DataLoader, EEGFinetuneTrainer]:
    """
    ?? Complete fine-tuning setup for TUH Dataset
    
    Args:
        train_data_path: Training data path
        config: EEG configuration (cross/self-attention)
        pretrain_checkpoint: Pre-trained model checkpoint
        val_data_path: Validation data path
        test_data_path: Test data path
        freeze_encoder: Whether to freeze encoder weights
    
    Returns:
        model, train_loader, val_loader, test_loader, trainer
    """
    if config is None:
        config = EEGConfig()
    
    print(f"?? Setting up fine-tuning for TUH Dataset...")
    print(f"   ?? Attention: {'Cross-Attention' if config.USE_CROSS_ATTENTION else 'Self-Attention'}")
    print(f"   Training data: {train_data_path}")
    print(f"   Validation data: {val_data_path}")
    print(f"   Test data: {test_data_path}")
    print(f"   Pre-trained model: {pretrain_checkpoint}")
    print(f"   Freeze encoder: {freeze_encoder}")
    print(f"   Batch size: {config.TRAINING_CONFIG['batch_size']}")
    
    # Create fine-tuning model (includes classification head)
    model = create_finetune_model(config)
    
    # Model analysis
    model_analysis = count_parameters(model)
    print(f"?? Model Analysis:")
    print(f"   Total parameters: {model_analysis['total_parameters']:,}")
    print(f"   Memory estimate: {model_analysis['memory_mb']:.1f} MB")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path
    )
    
    # Create trainer
    trainer = EEGFinetuneTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        test_loader=test_loader,
        pretrain_checkpoint=pretrain_checkpoint
    )
    
    # Optionally freeze encoder
    if freeze_encoder:
        trainer.model.freeze_encoder()
        print(f"?? Encoder frozen - only training classification head")
    
    print(f"? Fine-tuning setup completed!")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset) if val_loader else 0}")
    print(f"   Test samples: {len(test_loader.dataset) if test_loader else 0}")
    print(f"   Training batches: {len(train_loader)}")
    
    return model, train_loader, val_loader, test_loader, trainer


def analyze_finetune_results(log_file_path: str) -> Dict:
    """?? Analyze fine-tuning results"""
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
        best_metrics = final_entry.get('best_metrics', {})
        
        analysis = {
            'training_summary': {
                'total_epochs': len(training_data),
                'session_id': final_entry.get('session_id', 'unknown'),
                'attention_type': final_entry.get('attention_type', 'unknown')
            },
            'best_performance': {
                'best_accuracy': best_metrics.get('best_val_acc', 0),
                'best_auc': best_metrics.get('best_val_auc', 0),
                'best_f1': best_metrics.get('best_val_f1', 0)
            },
            'training_stability': {
                'converged': len(training_data) >= 10,
                'early_stopped': len(training_data) < 50
            }
        }
        
        print(f"\n?? Fine-tuning Analysis Complete:")
        print(f"   Epochs: {analysis['training_summary']['total_epochs']}")
        print(f"   Attention: {analysis['training_summary']['attention_type']}")
        print(f"   Best Accuracy: {analysis['best_performance']['best_accuracy']:.4f}")
        print(f"   Best AUC: {analysis['best_performance']['best_auc']:.4f}")
        print(f"   Best F1: {analysis['best_performance']['best_f1']:.4f}")
        
        return analysis
        
    except Exception as e:
        print(f"? Error analyzing fine-tuning results: {str(e)}")
        return {}


def plot_finetune_curves(log_file_path: str, output_dir: Optional[str] = None):
    """?? Plot fine-tuning curves"""
    
    # Load training data
    training_data = []
    with open(log_file_path, 'r') as f:
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
    train_losses = [entry['train_metrics']['total_loss'] for entry in training_data]
    train_accs = [entry['train_metrics']['accuracy'] for entry in training_data]
    
    val_losses = []
    val_accs = []
    val_aucs = []
    val_f1s = []
    
    for entry in training_data:
        if entry.get('val_metrics'):
            val_losses.append(entry['val_metrics']['total_loss'])
            val_accs.append(entry['val_metrics']['accuracy'])
            val_aucs.append(entry['val_metrics']['auc'])
            val_f1s.append(entry['val_metrics']['f1_macro'])
        else:
            val_losses.append(0)
            val_accs.append(0)
            val_aucs.append(0)
            val_f1s.append(0)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EEG Fine-tuning Progress', fontsize=16)
    
    # Loss curves
    axes[0,0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train')
    if any(val_losses):
        axes[0,0].plot(epochs, val_losses, 'r-', linewidth=2, label='Validation')
    axes[0,0].set_title('Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[0,1].plot(epochs, train_accs, 'b-', linewidth=2, label='Train')
    if any(val_accs):
        axes[0,1].plot(epochs, val_accs, 'r-', linewidth=2, label='Validation')
    axes[0,1].set_title('Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # AUC curve
    if any(val_aucs):
        axes[1,0].plot(epochs, val_aucs, 'g-', linewidth=2)
        axes[1,0].set_title('AUC')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('AUC')
        axes[1,0].grid(True, alpha=0.3)
    
    # F1 curve
    if any(val_f1s):
        axes[1,1].plot(epochs, val_f1s, 'purple', linewidth=2)
        axes[1,1].set_title('F1 Score (Macro)')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('F1 Score')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        plot_path = os.path.join(output_dir, "finetune_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"?? Fine-tuning curves saved to: {plot_path}")
    
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("?? ENHANCED EEG FINE-TUNING TRAINER")
    print("="*80)
    
    # Test setup
    config = EEGConfig()
    
    # Mock dataset for testing
    class MockDataset:
        def __init__(self, size=100):
            self.size = size
            self.dataset_stats = {
                'class_distribution': {0: 60, 1: 40},
                'class_balance': 0.4
            }
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Mock EEG data (20 frequencies)
            return torch.randn(20, 19, 19, 2), torch.tensor([idx % 2])
    
    # Create mock components
    mock_train_dataset = MockDataset(size=80)
    mock_val_dataset = MockDataset(size=20)
    
    train_loader = DataLoader(mock_train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(mock_val_dataset, batch_size=4, shuffle=False)
    
    print("?? Mock Fine-tuning Test Setup:")
    print(f"   Train dataset size: {len(mock_train_dataset)}")
    print(f"   Val dataset size: {len(mock_val_dataset)}")
    print(f"   Batch size: {train_loader.batch_size}")
    print(f"   ?? Cross-Attention: {config.USE_CROSS_ATTENTION}")
    
    # Create model and trainer
    model = create_finetune_model(config)
    trainer = EEGFinetuneTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        pretrain_checkpoint=None  # No pre-trained weights for test
    )
    
    model_analysis = count_parameters(model)
    print(f"\n?? Model Info:")
    print(f"   Parameters: {model_analysis['total_parameters']:,}")
    print(f"   Memory estimate: {model_analysis['memory_mb']:.1f} MB")
    print(f"   Has classification head: {hasattr(model, 'classification_head')}")
    
    # Test single epoch
    print(f"\n?? Testing Single Epoch...")
    test_start = time.time()
    
    # Override num_epochs for quick test
    trainer.num_epochs = 1
    
    try:
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.validate_epoch()
        test_time = time.time() - test_start
        
        print(f"? Single Epoch Test Completed:")
        print(f"   Time: {test_time:.1f}s")
        print(f"   Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"   Train Acc: {train_metrics['accuracy']:.4f}")
        if val_metrics:
            print(f"   Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"   Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"   Val AUC: {val_metrics['auc']:.4f}")
        
        # Test checkpoint saving
        trainer.save_checkpoint(0, train_metrics, val_metrics)
        print(f"   Checkpoint saved successfully")
        
    except Exception as e:
        print(f"? Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("="*80)
    print("?? FINE-TUNING TRAINER TESTING COMPLETED")
    print("="*80)
    
    print(f"\n? Fine-tuning Trainer Ready!")
    print(f"   ?? Cross-Attention & Self-Attention compatible")
    print(f"   ?? TUH Dataset optimized")
    print(f"   ?? Binary classification ready")
    print(f"   ?? Memory optimized")
    print(f"   ?? Detailed metrics & analysis")
    
    print(f"\n?? Ready for Production:")
    print(f"   Use: setup_finetune_training() to start")
    print(f"   Expected: High accuracy on TUH abnormal detection")
    print(f"   Compatible: Both attention types")
    print("="*80)