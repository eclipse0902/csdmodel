"""
EEG Connectivity Analysis - Main Execution Script

¿ÏÀüÈ÷ Àç¼³°èµÈ ½ÇÇà ½ºÅ©¸³Æ®:
1. Pre-training, Fine-tuning, Evaluation, Analysis ¸ðµå
2. »õ·Î¿î ¾ÆÅ°ÅØÃ³ ¿ÏÀü Áö¿ø
3. Config ±â¹Ý ¸ðµç ¼³Á¤
4. »ó¼¼ÇÑ °á°ú ºÐ¼® ¹× ½Ã°¢È­


python main_pretrain.py --mode pretrain \
    --train_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/train \
    --val_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/val \
    --mask_ratio 0.5 \
    --output_dir ./20fq_results \
    --preview_data

    --resume_from csdmodel/checkpoints/best_pretrain_model.pth \
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EEGConfig
from data.dataset import EEGDataset, create_data_loaders, validate_data_directory, preview_dataset_samples
from models.hybrid_model import (
    EEGConnectivityModel, 
    create_pretrain_model, 
    create_finetune_model, 
    create_inference_model
)
from training.pretrain_trainer import (
    EEGPretrainTrainer, 
    setup_redesigned_pretraining, 
    analyze_training_results
)
from utils.losses import EEGLossCalculator, EEGMetricsCalculator

def pretrain_mode(args):
    """Pre-training ¸ðµå"""
    print("="*80)
    print("?? EEG CONNECTIVITY PRE-TRAINING")
    print("="*80)
    
    # Configuration
    config = EEGConfig()
    if args.config_overrides:
        print(f"?? Applying config overrides: {args.config_overrides}")
        # Apply any config overrides here
    
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
    
    if validation_result['recommendations']:
        print(f"??  Recommendations: {validation_result['recommendations']}")
    
    # Data preview
    if args.preview_data:
        print(f"\n?? Data preview:")
        preview = preview_dataset_samples(args.train_data, num_samples=3, config=config)
        for sample in preview['samples']:
            print(f"   Sample {sample['index']}: Label={sample['label']}, "
                  f"Magnitude={sample['magnitude_stats']['mean']:.3f}¡¾{sample['magnitude_stats']['std']:.3f}")
    
    # Setup pre-training
    print(f"\n??? Setting up pre-training...")
    model, train_loader, trainer = setup_redesigned_pretraining(
        data_path=args.train_data,
        config=config,
        mask_ratio=args.mask_ratio,
        val_data_path=args.val_data
    )
    
    # Model summary
    model_info = model.get_model_analysis(torch.randn(1, 20, 19, 19, 2))
    print(f"\n?? Model Information:")
    print(f"   Architecture: Structured Feature Extraction + Global Attention + Frequency-Specific Heads")
    print(f"   Parameters: {model_info['model_info']['total_parameters']:,}")
    print(f"   Memory: ~{model_info['model_info']['memory_mb']:.1f} MB")
    print(f"   Input shape: {model_info['input_shape']}")
    print(f"   Feature shape: {model_info['feature_shape']}")
    
    # Start training
    print(f"\n?? Starting pre-training...")
    training_results = trainer.train()
    
    # Results summary
    print(f"\n?? Pre-training Results Summary:")
    print(f"   Total epochs: {training_results['total_epochs_trained']}")
    print(f"   Training time: {training_results['total_training_time_hours']:.2f} hours")
    print(f"   Best loss: {training_results['best_metrics']['best_train_loss']:.6f}")
    print(f"   Best phase error: {training_results['best_metrics']['best_phase_error_degrees']:.1f}¡Æ")
    print(f"   Best alpha magnitude: {training_results['best_metrics']['best_alpha_magnitude_error']*100:.1f}%")
    print(f"   Session ID: {training_results['session_id']}")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, f"pretrain_results_{training_results['session_id']}.json")
        
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        print(f"?? Results saved to: {results_path}")
    
    # Achievements
    if training_results['achievements']:
        print(f"\n?? Achievements:")
        for achievement in training_results['achievements']:
            print(f"   {achievement}")
    
    print(f"\n? Pre-training completed successfully!")
    print(f"   Best model: ./checkpoints/best_pretrain_model.pth")

def finetune_mode(args):
    """Fine-tuning ¸ðµå"""
    print("="*80)
    print("?? EEG CONNECTIVITY FINE-TUNING")
    print("="*80)
    
    if not args.pretrain_model or not os.path.exists(args.pretrain_model):
        print(f"? Pre-trained model not found: {args.pretrain_model}")
        return
    
    config = EEGConfig()
    
    # Data setup
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        test_data_path=args.test_data
    )
    
    # Model setup
    print(f"??? Setting up fine-tuning model...")
    model = create_finetune_model(config, pretrain_checkpoint=args.pretrain_model)
    
    if args.freeze_encoder:
        model.freeze_encoder()
        print(f"?? Encoder frozen for fine-tuning")
    
    # Training setup
    device = config.DEVICE
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.FINETUNING_CONFIG['learning_rate'],
        weight_decay=config.FINETUNING_CONFIG['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.FINETUNING_CONFIG['num_epochs']
    )
    
    # Training loop
    print(f"?? Starting fine-tuning...")
    
    best_val_acc = 0.0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(config.FINETUNING_CONFIG['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config.FINETUNING_CONFIG['num_epochs']}")
        
        # Training
        model.train()
        train_loss, train_acc = 0.0, 0.0
        num_train_batches = 0
        
        for csd_data, labels in train_loader:
            csd_data, labels = csd_data.to(device), labels.squeeze().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss, loss_breakdown = model.compute_classification_loss(csd_data, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            train_acc += loss_breakdown['accuracy'].item()
            num_train_batches += 1
        
        train_loss /= num_train_batches
        train_acc /= num_train_batches
        
        # Validation
        if val_loader:
            model.eval()
            val_loss, val_acc = 0.0, 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for csd_data, labels in val_loader:
                    csd_data, labels = csd_data.to(device), labels.squeeze().to(device)
                    
                    loss, loss_breakdown = model.compute_classification_loss(csd_data, labels)
                    
                    val_loss += loss.item()
                    val_acc += loss_breakdown['accuracy'].item()
                    num_val_batches += 1
            
            val_loss /= num_val_batches
            val_acc /= num_val_batches
        else:
            val_loss, val_acc = 0.0, 0.0
        
        # Update history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Scheduler step
        scheduler.step()
        
        # Progress
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        if val_loader:
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                best_model_path = os.path.join(args.output_dir, "best_finetune_model.pth")
                model.save_model(best_model_path, epoch=epoch, additional_info={
                    'val_accuracy': val_acc,
                    'training_history': training_history
                })
                print(f"?? New best model saved: {val_acc:.4f}")
    
    # Final results
    print(f"\n?? Fine-tuning Results:")
    print(f"   Best validation accuracy: {best_val_acc:.4f}")
    print(f"   Final train accuracy: {training_history['train_acc'][-1]:.4f}")
    
    # Test evaluation
    if test_loader:
        print(f"\n?? Evaluating on test set...")
        model.eval()
        test_acc = 0.0
        num_test_batches = 0
        
        with torch.no_grad():
            for csd_data, labels in test_loader:
                csd_data, labels = csd_data.to(device), labels.squeeze().to(device)
                _, loss_breakdown = model.compute_classification_loss(csd_data, labels)
                test_acc += loss_breakdown['accuracy'].item()
                num_test_batches += 1
        
        test_acc /= num_test_batches
        print(f"   Test accuracy: {test_acc:.4f}")
    
    print(f"? Fine-tuning completed successfully!")

def evaluate_mode(args):
    """Evaluation ¸ðµå"""
    print("="*80)
    print("?? EEG CONNECTIVITY EVALUATION")
    print("="*80)
    
    if not args.model_path or not os.path.exists(args.model_path):
        print(f"? Model not found: {args.model_path}")
        return
    
    config = EEGConfig()
    
    # Load model
    print(f"?? Loading model from: {args.model_path}")
    model = EEGConnectivityModel.load_model(args.model_path, mode='inference')
    model.eval()
    
    # Data loading
    if args.eval_data:
        dataset = EEGDataset(args.eval_data, config, apply_masking=False)
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.TRAINING_CONFIG['batch_size'],
            shuffle=False,
            num_workers=4
        )
    else:
        print(f"? Evaluation data path required")
        return
    
    print(f"?? Evaluating on {len(dataset)} samples...")
    
    # Evaluation
    device = config.DEVICE
    model.to(device)
    
    # Metrics
    all_predictions = []
    all_labels = []
    all_losses = []
    
    loss_calculator = EEGLossCalculator(config)
    metrics_calculator = EEGMetricsCalculator(config)
    
    with torch.no_grad():
        for batch_idx, (csd_data, labels) in enumerate(data_loader):
            csd_data = csd_data.to(device)
            labels = labels.squeeze().to(device)
            
            # Forward pass
            if hasattr(model, 'classification_head'):
                # Classification evaluation
                logits = model.forward_classification(csd_data)
                predictions = torch.argmax(logits, dim=1)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_losses.append(loss.item())
            
            else:
                # Reconstruction evaluation (pre-training model)
                # Apply masking for reconstruction evaluation
                mask = torch.ones_like(csd_data)
                mask = mask * (torch.rand_like(mask) > 0.5).float()  # 50% masking
                
                masked_data = csd_data * mask
                reconstructed = model(masked_data)
                
                # Compute reconstruction metrics
                loss, loss_breakdown = loss_calculator.compute_total_loss(
                    reconstructed, csd_data, mask, return_breakdown=True
                )
                
                quality_metrics = metrics_calculator.compute_signal_quality_metrics(
                    reconstructed, csd_data, mask
                )
                
                all_losses.append(loss.item())
                
                if batch_idx == 0:  # Print detailed metrics for first batch
                    print(f"?? Reconstruction Metrics (Sample):")
                    print(f"   Phase Error: {loss_breakdown['phase_error_degrees'].item():.1f}¡Æ")
                    print(f"   Alpha Magnitude Error: {loss_breakdown['alpha_magnitude_error'].item()*100:.1f}%")
                    print(f"   SNR: {quality_metrics['snr_db'].item():.1f} dB")
                    print(f"   Correlation: {quality_metrics['correlation'].item():.3f}")
            
            if batch_idx % 10 == 0:
                print(f"   Processed {batch_idx}/{len(data_loader)} batches")
    
    # Results summary
    avg_loss = np.mean(all_losses)
    
    if all_predictions:  # Classification results
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Class-wise accuracy
        class_accuracies = {}
        for class_idx in [0, 1]:
            class_mask = (all_labels == class_idx)
            if class_mask.sum() > 0:
                class_acc = (all_predictions[class_mask] == all_labels[class_mask]).mean()
                class_accuracies[f'class_{class_idx}'] = class_acc
        
        evaluation_results = {
            'evaluation_type': 'classification',
            'total_samples': len(all_predictions),
            'average_loss': avg_loss,
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'predictions': all_predictions.tolist(),
            'labels': all_labels.tolist()
        }
        
        print(f"\n?? Classification Results:")
        print(f"   Total samples: {len(all_predictions)}")
        print(f"   Average loss: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        for class_name, class_acc in class_accuracies.items():
            print(f"   {class_name} accuracy: {class_acc:.4f}")
    
    else:  # Reconstruction results
        evaluation_results = {
            'evaluation_type': 'reconstruction',
            'total_samples': len(data_loader) * config.TRAINING_CONFIG['batch_size'],
            'average_loss': avg_loss
        }
        
        print(f"\n?? Reconstruction Results:")
        print(f"   Total samples: ~{evaluation_results['total_samples']}")
        print(f"   Average loss: {avg_loss:.6f}")
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        print(f"?? Results saved to: {results_path}")
    
    print(f"? Evaluation completed successfully!")

def analyze_mode(args):
    """Analysis ¸ðµå"""
    print("="*80)
    print("?? EEG CONNECTIVITY ANALYSIS")
    print("="*80)
    
    if not args.log_file or not os.path.exists(args.log_file):
        print(f"? Log file not found: {args.log_file}")
        return
    
    print(f"?? Analyzing training log: {args.log_file}")
    
    # Analyze training results
    analysis_results = analyze_training_results(args.log_file)
    
    if not analysis_results:
        print(f"? Failed to analyze training results")
        return
    
    # Print analysis
    print(f"\n?? Training Analysis Results:")
    print(f"   Session: {analysis_results['training_summary']['session_id']}")
    print(f"   Total epochs: {analysis_results['training_summary']['total_epochs']}")
    print(f"   Converged: {analysis_results['training_stability']['converged']}")
    print(f"   Early stopped: {analysis_results['training_stability']['early_stopped']}")
    
    print(f"\n?? Performance Summary:")
    print(f"   Final loss: {analysis_results['final_performance']['final_loss']:.6f}")
    print(f"   Final phase error: {analysis_results['final_performance']['final_phase_error']:.1f}¡Æ")
    print(f"   Final alpha magnitude: {analysis_results['final_performance']['final_alpha_magnitude_error']:.1f}%")
    print(f"   Final correlation: {analysis_results['final_performance']['final_correlation']:.3f}")
    
    print(f"\n?? Best Performance:")
    print(f"   Best loss: {analysis_results['best_performance']['best_loss']:.6f}")
    print(f"   Best phase error: {analysis_results['best_performance']['best_phase_error']:.1f}¡Æ")
    print(f"   Best alpha magnitude: {analysis_results['best_performance']['best_alpha_magnitude_error']:.1f}%")
    
    # Save analysis
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        analysis_path = os.path.join(args.output_dir, "training_analysis.json")
        
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        print(f"?? Analysis saved to: {analysis_path}")
    
    # Generate training curves (if matplotlib available)
    try:
        plot_training_curves(args.log_file, args.output_dir)
    except Exception as e:
        print(f"??  Could not generate plots: {e}")
    
    print(f"? Analysis completed successfully!")

def plot_training_curves(log_file: str, output_dir: Optional[str] = None):
    """ÈÆ·Ã °î¼± ½Ã°¢È­"""
    
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
    train_losses = [entry['train_metrics']['total_loss'] for entry in training_data]
    phase_errors = [entry['train_metrics']['phase_error_degrees'] for entry in training_data]
    alpha_mag_errors = [entry['train_metrics']['alpha_magnitude_error'] * 100 for entry in training_data]
    correlations = [entry['train_metrics']['correlation'] for entry in training_data]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EEG Pre-training Progress', fontsize=16)
    
    # Loss curve
    axes[0,0].plot(epochs, train_losses, 'b-', linewidth=2)
    axes[0,0].set_title('Training Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)
    
    # Phase error
    axes[0,1].plot(epochs, phase_errors, 'r-', linewidth=2)
    axes[0,1].axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Target (<25¡Æ)')
    axes[0,1].set_title('Phase Error')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Phase Error (degrees)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Alpha magnitude error
    axes[1,0].plot(epochs, alpha_mag_errors, 'g-', linewidth=2)
    axes[1,0].axhline(y=8, color='r', linestyle='--', alpha=0.5, label='Target (<8%)')
    axes[1,0].set_title('Alpha Magnitude Error')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Magnitude Error (%)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Correlation
    axes[1,1].plot(epochs, correlations, 'purple', linewidth=2)
    axes[1,1].axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='Target (>0.8)')
    axes[1,1].set_title('Correlation')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Correlation')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"?? Training curves saved to: {plot_path}")
    
    plt.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="EEG Connectivity Analysis with Redesigned Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-training
  python main_pretrain.py --mode pretrain --train_data ./data/train --mask_ratio 0.5
  
  # Fine-tuning
  python main_pretrain.py --mode finetune --pretrain_model ./checkpoints/best_pretrain_model.pth --train_data ./data/train --val_data ./data/val
  
  # Evaluation
  python main_pretrain.py --mode evaluate --model_path ./checkpoints/best_model.pth --eval_data ./data/test --output_dir ./results
  
  # Analysis
  python main_pretrain.py --mode analyze --log_file ./logs/pretrain_log_20241209_120000_redesigned.json --output_dir ./analysis
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['pretrain', 'finetune', 'evaluate', 'analyze'], 
                       required=True, 
                       help='Operation mode')
    
    # Data paths
    parser.add_argument('--train_data', type=str, help='Training data directory')
    parser.add_argument('--val_data', type=str, help='Validation data directory')
    parser.add_argument('--test_data', type=str, help='Test data directory')
    parser.add_argument('--eval_data', type=str, help='Evaluation data directory')
    
    # Model paths
    parser.add_argument('--model_path', type=str, help='Model checkpoint path for evaluation')
    parser.add_argument('--pretrain_model', type=str, help='Pre-trained model path for fine-tuning')
    
    # Pre-training options
    parser.add_argument('--mask_ratio', type=float, default=0.5, 
                       help='Masking ratio for pre-training (default: 0.5)')
    parser.add_argument('--preview_data', action='store_true',
                       help='Preview dataset samples before training')
    parser.add_argument('--resume_from', type=str, help='Checkpoint path to resume training from')
    # Fine-tuning options
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder during fine-tuning')
    
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
            pretrain_mode(args)
            
        elif args.mode == 'finetune':
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
        print(f"\n??  Process interrupted by user")
    except Exception as e:
        print(f"? Error in {args.mode} mode: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()