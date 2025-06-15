#!/usr/bin/env python3
"""
Multi-Scale EEG Connectivity Analysis - Fixed Main Script

¿ÏÀüÇÑ Multi-Scale EEG ½Ã½ºÅÛ - Import ¿¡·¯ ¼öÁ¤:
1. Config ±¸Á¶ ÅëÀÏ
2. Import path ¼öÁ¤
3. È£È¯¼º ¹®Á¦ ÇØ°á
4. Multi-Scale Pre-training (4ÃÊ+8ÃÊ+16ÃÊ)
5. Scale-wise Performance Analysis

Usage:
    python /home/mjkang/cbramod/cnn_time/multiscale_main.py --mode pretrain \
        --train_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/train \
        --val_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/val \
        --output_dir ./multiscale_results
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Multi-Scale imports
    from multiscale_config import MultiScaleEEGConfig
    from models.multiscale_hybrid_model import (
        MultiScaleEEGConnectivityModel,
        create_multiscale_pretrain_model,
        create_multiscale_finetune_model,
        load_pretrained_multiscale_encoder
    )
    from training.multiscale_trainer import (
        MultiScalePretrainTrainer,
        setup_multiscale_pretraining,
        analyze_multiscale_training_progress,
        convert_single_to_multiscale_checkpoint
    )
    print("? Multi-Scale modules imported successfully")
    
except ImportError as e:
    print(f"?? Multi-Scale import failed: {e}")
    print("?? Falling back to single-scale modules...")
    
    # Fallback to single-scale
    from config import EEGConfig as MultiScaleEEGConfig
    from models.hybrid_model import (
        EEGConnectivityModel as MultiScaleEEGConnectivityModel,
        create_pretrain_model as create_multiscale_pretrain_model
    )
    from training.pretrain_trainer import (
        EEGPretrainTrainer as MultiScalePretrainTrainer,
        setup_redesigned_pretraining as setup_multiscale_pretraining,
        analyze_training_results as analyze_multiscale_training_progress
    )
    
    # Mock functions for missing features
    def create_multiscale_finetune_model(*args, **kwargs):
        raise NotImplementedError("Multi-scale finetune not available in fallback mode")
    
    def load_pretrained_multiscale_encoder(*args, **kwargs):
        return False
    
    def convert_single_to_multiscale_checkpoint(*args, **kwargs):
        return False

# Common imports (always available)
from data.dataset import EEGDataset, validate_data_directory, preview_dataset_samples
from utils.losses import EEGLossCalculator
from config import EEGConfig  # Single-scale config for comparison

def create_unified_config(args):
    """ÅëÇÕµÈ config »ý¼º (Multi-Scale ¿ì¼±, Single-Scale È£È¯)"""
    try:
        config = MultiScaleEEGConfig()
        print("?? Using Multi-Scale configuration")
        
        # Multi-Scale config ±¸Á¶ È®ÀÎ ¹× »ý¼º
        if not hasattr(config, 'PRETRAINING_CONFIG'):
            # Single-scale config¿¡¼­ Multi-Scale config·Î º¯È¯
            config.PRETRAINING_CONFIG = getattr(config, 'PRETRAINING_CONFIG', {
                'mask_ratio': 0.5,
                'num_epochs': 50,
                'learning_rate': 5e-4,
                'batch_size': 128,
                'weight_decay': 2e-3
            })
        
        if not hasattr(config, 'MULTISCALE_TRAINING_CONFIG'):
            config.MULTISCALE_TRAINING_CONFIG = {
                'batch_size': 16,
                'num_epochs': 50,
                'learning_rate': 5e-5,
                'weight_decay': 1e-3,
                'gradient_clip_norm': 0.5,
                'scale_sampling_strategy': 'balanced',
                'curriculum_learning': {
                    'start_with_single_scale': True,
                    'single_scale_epochs': 10,
                    'gradual_scale_introduction': True
                }
            }
        
        # Command line arguments Àû¿ë
        if args.mask_ratio:
            if hasattr(config, 'PRETRAINING_CONFIG'):
                config.PRETRAINING_CONFIG['mask_ratio'] = args.mask_ratio
            else:
                setattr(config, 'mask_ratio', args.mask_ratio)
        
        if args.batch_size:
            if hasattr(config, 'MULTISCALE_TRAINING_CONFIG'):
                config.MULTISCALE_TRAINING_CONFIG['batch_size'] = args.batch_size
            if hasattr(config, 'PRETRAINING_CONFIG'):
                config.PRETRAINING_CONFIG['batch_size'] = args.batch_size
        
        if args.learning_rate:
            if hasattr(config, 'MULTISCALE_TRAINING_CONFIG'):
                config.MULTISCALE_TRAINING_CONFIG['learning_rate'] = args.learning_rate
            if hasattr(config, 'PRETRAINING_CONFIG'):
                config.PRETRAINING_CONFIG['learning_rate'] = args.learning_rate
        
        if args.num_epochs:
            if hasattr(config, 'MULTISCALE_TRAINING_CONFIG'):
                config.MULTISCALE_TRAINING_CONFIG['num_epochs'] = args.num_epochs
            if hasattr(config, 'PRETRAINING_CONFIG'):
                config.PRETRAINING_CONFIG['num_epochs'] = args.num_epochs
        
        return config
        
    except Exception as e:
        print(f"?? Failed to create Multi-Scale config: {e}")
        print("?? Using Single-Scale config as fallback...")
        
        # Fallback to single-scale
        config = EEGConfig()
        
        # Ensure compatibility
        if not hasattr(config, 'PRETRAINING_CONFIG'):
            config.PRETRAINING_CONFIG = config.PRETRAINING_CONFIG if hasattr(config, 'PRETRAINING_CONFIG') else {
                'mask_ratio': args.mask_ratio or 0.5,
                'num_epochs': args.num_epochs or 50,
                'learning_rate': args.learning_rate or 1e-4,
                'batch_size': args.batch_size or 32,
                'weight_decay': 1e-3
            }
        
        return config

def multiscale_pretrain_mode(args):
    """Multi-Scale Pre-training ¸ðµå (¼öÁ¤µÊ)"""
    print("="*80)
    print("?? MULTI-SCALE EEG CONNECTIVITY PRE-TRAINING")
    print("="*80)
    
    # ÅëÇÕ Configuration
    config = create_unified_config(args)
    
    # Data validation
    print(f"?? Validating data directory...")
    validation_result = validate_data_directory(args.train_data, config)
    
    if not validation_result['valid']:
        print(f"? Data validation failed: {validation_result['error']}")
        return
    
    print(f"? Data validation passed:")
    print(f"   Files: {validation_result['num_files']}")
    print(f"   Quality score: {validation_result['quality_score']:.3f}")
    print(f"   Class distribution: {validation_result['class_distribution']}")
    
    # Data preview
    if args.preview_data:
        print(f"\n?? Data preview:")
        try:
            preview = preview_dataset_samples(args.train_data, num_samples=3, config=config)
            for sample in preview['samples']:
                print(f"   Sample {sample['index']}: Label={sample['label']}, "
                      f"Magnitude={sample['magnitude_stats']['mean']:.3f}¡¾{sample['magnitude_stats']['std']:.3f}")
        except Exception as e:
            print(f"?? Data preview failed: {e}")
    
    # Setup Multi-Scale pre-training
    print(f"\n?? Setting up Multi-Scale pre-training...")
    try:
        model, train_loader, trainer = setup_multiscale_pretraining(
            data_path=args.train_data,
            config=config,
            mask_ratio=getattr(config, 'mask_ratio', config.PRETRAINING_CONFIG.get('mask_ratio', 0.5)),
            val_data_path=args.val_data,
            resume_from=args.resume_from
        )
        
        print(f"? Multi-Scale setup completed")
        
    except Exception as e:
        print(f"?? Multi-Scale setup failed: {e}")
        print("?? Attempting single-scale fallback...")
        
        # Single-scale fallback
        from training.pretrain_trainer import setup_redesigned_pretraining
        model, train_loader, trainer = setup_redesigned_pretraining(
            data_path=args.train_data,
            config=config,
            mask_ratio=config.PRETRAINING_CONFIG.get('mask_ratio', 0.5),
            val_data_path=args.val_data
        )
        print(f"? Single-scale fallback setup completed")
    
    # Model summary
    try:
        print(f"\n?? Model Information:")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Try to get detailed analysis
        try:
            sample_input = torch.randn(1, 20, 19, 19, 2)
            if hasattr(model, 'get_model_analysis'):
                model_info = model.get_model_analysis(sample_input)
                print(f"   Architecture: Multi-Scale (4s/8s/16s)")
                print(f"   Memory: ~{model_info['model_info']['memory_mb']:.1f} MB")
            else:
                print(f"   Architecture: EEG Connectivity Model")
                print(f"   Memory: ~{sum(p.numel() for p in model.parameters()) * 4 / (1024*1024):.1f} MB")
        except Exception as e:
            print(f"   Architecture: EEG Connectivity Model")
            print(f"   Memory estimate: ~{sum(p.numel() for p in model.parameters()) * 4 / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"?? Model analysis failed: {e}")
    
    # Start training
    print(f"\n?? Starting pre-training...")
    training_results = trainer.train()
    
    # Results summary
    print(f"\n?? Pre-training Results:")
    print(f"   Total epochs: {training_results['total_epochs_trained']}")
    print(f"   Training time: {training_results['total_training_time_hours']:.2f} hours")
    print(f"   Best loss: {training_results['best_metrics']['best_train_loss']:.6f}")
    
    # Scale-specific performance (if available)
    best_metrics = training_results['best_metrics']
    if 'best_multiscale_balance' in best_metrics:
        print(f"   Best multi-scale balance: {best_metrics['best_multiscale_balance']:.6f}")
    if 'best_phase_error' in best_metrics:
        print(f"   Best phase error: {best_metrics['best_phase_error']:.1f}¡Æ")
    
    print(f"   Session ID: {training_results['session_id']}")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, f"multiscale_results_{training_results['session_id']}.json")
        
        # Convert to serializable format
        serializable_results = convert_to_serializable(training_results)
        
        try:
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"?? Results saved to: {results_path}")
        except Exception as e:
            print(f"?? Failed to save results: {e}")
        
        # Generate plots if possible
        if 'training_history' in training_results:
            try:
                plot_multiscale_training_curves(training_results['training_history'], args.output_dir)
            except Exception as e:
                print(f"?? Failed to generate plots: {e}")
    
    # Achievements
    if training_results.get('achievements'):
        print(f"\n?? Achievements:")
        for achievement in training_results['achievements']:
            print(f"   {achievement}")
    
    print(f"\n? Multi-Scale pre-training completed!")
    if args.output_dir:
        print(f"   Results: {args.output_dir}")

def model_conversion_mode(args):
    """Single-Scale ¡æ Multi-Scale ¸ðµ¨ º¯È¯ (¼öÁ¤µÊ)"""
    print("="*80)
    print("?? SINGLE-SCALE TO MULTI-SCALE MODEL CONVERSION")
    print("="*80)
    
    if not args.single_model or not os.path.exists(args.single_model):
        print(f"? Single-scale model not found: {args.single_model}")
        return
    
    config = create_unified_config(args)
    
    print(f"?? Loading single-scale model: {args.single_model}")
    
    # Load single-scale checkpoint
    try:
        checkpoint = torch.load(args.single_model, map_location='cpu')
        single_state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        print(f"? Single-scale model loaded:")
        print(f"   Parameters: {len(single_state_dict)} keys")
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'best_metrics' in checkpoint:
            best_metrics = checkpoint['best_metrics']
            print(f"   Best loss: {best_metrics.get('best_train_loss', 'N/A')}")
    
    except Exception as e:
        print(f"? Failed to load single-scale model: {str(e)}")
        return
    
    # Create Multi-Scale model
    print(f"\n?? Creating Multi-Scale model...")
    try:
        multiscale_model = create_multiscale_pretrain_model(config)
        
        print(f"? Multi-Scale model created:")
        print(f"   Parameters: {sum(p.numel() for p in multiscale_model.parameters()):,}")
        print(f"   Memory: ~{sum(p.numel() for p in multiscale_model.parameters()) * 4 / (1024*1024):.1f} MB")
        
    except Exception as e:
        print(f"? Failed to create Multi-Scale model: {str(e)}")
        return
    
    # Convert weights
    print(f"\n?? Converting model weights...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "converted_multiscale_model.pth")
    
    try:
        success = convert_single_to_multiscale_checkpoint(
            args.single_model, 
            output_path, 
            config
        )
        
        if success:
            print(f"? Conversion completed successfully!")
            print(f"?? Converted model saved to: {output_path}")
        else:
            print(f"? Conversion failed!")
            
    except Exception as e:
        print(f"? Conversion error: {str(e)}")

def multiscale_evaluation_mode(args):
    """Multi-Scale ¸ðµ¨ Æò°¡ (¼öÁ¤µÊ)"""
    print("="*80)
    print("?? MULTI-SCALE MODEL EVALUATION")
    print("="*80)
    
    if not args.model_path or not os.path.exists(args.model_path):
        print(f"? Model not found: {args.model_path}")
        return
    
    config = create_unified_config(args)
    
    # Load model
    print(f"?? Loading Multi-Scale model...")
    try:
        if hasattr(MultiScaleEEGConnectivityModel, 'load_model'):
            model = MultiScaleEEGConnectivityModel.load_model(args.model_path, mode='inference')
        else:
            # Fallback
            model = create_multiscale_pretrain_model(config)
            checkpoint = torch.load(args.model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        print(f"? Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        
    except Exception as e:
        print(f"? Failed to load model: {str(e)}")
        return
    
    # Load evaluation data
    if not args.eval_data:
        print(f"? Evaluation data path required")
        return
    
    try:
        dataset = EEGDataset(args.eval_data, config, apply_masking=False)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=getattr(config, 'batch_size', 16),
            shuffle=False,
            num_workers=2
        )
        
        print(f"?? Evaluation dataset: {len(dataset)} samples, {len(data_loader)} batches")
        
    except Exception as e:
        print(f"? Failed to load evaluation data: {str(e)}")
        return
    
    # Run evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"\n?? Running evaluation...")
    
    all_losses = []
    loss_calculator = EEGLossCalculator(config)
    
    with torch.no_grad():
        for batch_idx, (csd_data, labels) in enumerate(data_loader):
            try:
                csd_data = csd_data.to(device)
                
                # Apply masking
                mask = torch.ones_like(csd_data)
                mask = mask * (torch.rand_like(mask) > 0.5).float()
                
                masked_data = csd_data * mask
                reconstructed = model(masked_data)
                
                # Compute loss
                loss, _ = loss_calculator.compute_total_loss(reconstructed, csd_data, mask)
                all_losses.append(loss.item())
                
            except Exception as e:
                print(f"?? Error in batch {batch_idx}: {str(e)}")
                continue
            
            if batch_idx % 10 == 0:
                print(f"   Processed {batch_idx}/{len(data_loader)} batches")
    
    # Results
    if all_losses:
        avg_loss = np.mean(all_losses)
        print(f"\n?? Evaluation Results:")
        print(f"   Average loss: {avg_loss:.6f} ¡¾ {np.std(all_losses):.6f}")
        print(f"   Samples evaluated: {len(all_losses) * data_loader.batch_size}")
        
        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            results = {
                'average_loss': avg_loss,
                'loss_std': np.std(all_losses),
                'total_samples': len(all_losses) * data_loader.batch_size,
                'all_losses': all_losses
            }
            
            results_path = os.path.join(args.output_dir, "evaluation_results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"?? Results saved to: {results_path}")
    
    print(f"? Evaluation completed!")

def analysis_mode(args):
    """ÈÆ·Ã °á°ú ºÐ¼® (¼öÁ¤µÊ)"""
    print("="*80)
    print("?? TRAINING ANALYSIS")
    print("="*80)
    
    if not args.log_file or not os.path.exists(args.log_file):
        print(f"? Log file not found: {args.log_file}")
        return
    
    print(f"?? Analyzing training log: {args.log_file}")
    
    try:
        analysis_results = analyze_multiscale_training_progress(args.log_file)
        
        if not analysis_results:
            print(f"? Failed to analyze training results")
            return
        
        # Print analysis
        print(f"\n?? Training Analysis Results:")
        
        if 'training_summary' in analysis_results:
            summary = analysis_results['training_summary']
            print(f"   Total epochs: {summary.get('total_epochs', 'N/A')}")
            if 'architecture_type' in summary:
                print(f"   Architecture: {summary['architecture_type']}")
        
        if 'final_performance' in analysis_results:
            final = analysis_results['final_performance']
            print(f"\n?? Final Performance:")
            for key, value in final.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.6f}")
        
        if 'best_performance' in analysis_results:
            best = analysis_results['best_performance']
            print(f"\n?? Best Performance:")
            for key, value in best.items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.6f}")
        
        # Save analysis
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            analysis_path = os.path.join(args.output_dir, "training_analysis.json")
            
            with open(analysis_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            print(f"?? Analysis saved to: {analysis_path}")
        
    except Exception as e:
        print(f"? Analysis failed: {str(e)}")
    
    print(f"? Analysis completed!")

# Utility functions

def convert_to_serializable(obj):
    """Convert numpy/torch types to JSON serializable"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def plot_multiscale_training_curves(training_history, output_dir):
    """Training °î¼± ½Ã°¢È­"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress', fontsize=16)
        
        # Loss curve
        if 'train_loss' in training_history and training_history['train_loss']:
            axes[0,0].plot(training_history['train_loss'], 'b-', linewidth=2)
            axes[0,0].set_title('Training Loss')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].grid(True, alpha=0.3)
        
        # Phase error
        if 'phase_error_degrees' in training_history and training_history['phase_error_degrees']:
            axes[0,1].plot(training_history['phase_error_degrees'], 'r-', linewidth=2)
            axes[0,1].axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Target (<25¡Æ)')
            axes[0,1].set_title('Phase Error')
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('Phase Error (degrees)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Learning rate
        if 'learning_rates' in training_history and training_history['learning_rates']:
            axes[1,0].plot(training_history['learning_rates'], 'g-', linewidth=2)
            axes[1,0].set_title('Learning Rate')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].set_yscale('log')
            axes[1,0].grid(True, alpha=0.3)
        
        # Multi-scale balance (if available)
        if 'multiscale_balance' in training_history and training_history['multiscale_balance']:
            axes[1,1].plot(training_history['multiscale_balance'], 'purple', linewidth=2)
            axes[1,1].axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Target (<0.1)')
            axes[1,1].set_title('Multi-Scale Balance')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Balance Score')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        elif 'correlation' in training_history and training_history['correlation']:
            axes[1,1].plot(training_history['correlation'], 'purple', linewidth=2)
            axes[1,1].set_title('Correlation')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Correlation')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"?? Training curves saved to: {plot_path}")
        
    except Exception as e:
        print(f"?? Failed to generate plots: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Scale EEG Connectivity Analysis - Fixed Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
?? Multi-Scale EEG Connectivity Analysis Examples:

  # Multi-Scale Pre-training
  python multiscale_main.py --mode pretrain \\
      --train_data ./data/train \\
      --val_data ./data/val \\
      --output_dir ./multiscale_results \\
      --preview_data

  # Model conversion
  python multiscale_main.py --mode convert \\
      --single_model ./checkpoints/best_pretrain_model.pth \\
      --output_dir ./converted_models

  # Evaluation
  python multiscale_main.py --mode evaluate \\
      --model_path ./multiscale_results/best_multiscale_model.pth \\
      --eval_data ./data/test \\
      --output_dir ./evaluation_results

  # Analysis
  python multiscale_main.py --mode analyze \\
      --log_file ./logs/training_log.json \\
      --output_dir ./analysis_results

?? Multi-Scale Features:
  ?? 4ÃÊ/8ÃÊ/16ÃÊ temporal scale processing
  ?? Cross-scale attention mechanism
  ?? Curriculum learning support
  ? Memory optimization (mixed precision + checkpointing)
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['pretrain', 'convert', 'evaluate', 'analyze'], 
                       required=True, 
                       help='Operation mode')
    
    # Data paths
    parser.add_argument('--train_data', type=str, help='Training data directory')
    parser.add_argument('--val_data', type=str, help='Validation data directory')
    parser.add_argument('--eval_data', type=str, help='Evaluation data directory')
    
    # Model paths
    parser.add_argument('--model_path', type=str, help='Model path for evaluation')
    parser.add_argument('--single_model', type=str, help='Single-scale model path for conversion')
    
    # Training options
    parser.add_argument('--mask_ratio', type=float, default=0.5, 
                       help='Masking ratio for pre-training (default: 0.5)')
    parser.add_argument('--batch_size', type=int, 
                       help='Batch size (default: from config)')
    parser.add_argument('--learning_rate', type=float, 
                       help='Learning rate (default: from config)')
    parser.add_argument('--num_epochs', type=int, 
                       help='Number of epochs (default: from config)')
    parser.add_argument('--resume_from', type=str, 
                       help='Checkpoint path to resume training from')
    
    # Data options
    parser.add_argument('--preview_data', action='store_true',
                       help='Preview dataset samples before training')
    
    # Analysis options
    parser.add_argument('--log_file', type=str, help='Training log file for analysis')
    
    # Output options
    parser.add_argument('--output_dir', type=str, help='Output directory for results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set device and print system info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"?? Multi-Scale EEG Connectivity Analysis - Fixed Version")
    print(f"???  System Information:")
    print(f"   Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute based on mode
    try:
        if args.mode == 'pretrain':
            if not args.train_data:
                print("? --train_data required for pre-training")
                return
            multiscale_pretrain_mode(args)
            
        elif args.mode == 'convert':
            if not args.single_model:
                print("? --single_model required for conversion")
                return
            if not args.output_dir:
                print("? --output_dir required for conversion")
                return
            model_conversion_mode(args)
            
        elif args.mode == 'evaluate':
            if not args.model_path:
                print("? --model_path required for evaluation")
                return
            if not args.eval_data:
                print("? --eval_data required for evaluation")
                return
            multiscale_evaluation_mode(args)
            
        elif args.mode == 'analyze':
            if not args.log_file:
                print("? --log_file required for analysis")
                return
            analysis_mode(args)
            
    except KeyboardInterrupt:
        print(f"\n?? Process interrupted by user")
    except Exception as e:
        print(f"? Error in {args.mode} mode: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()