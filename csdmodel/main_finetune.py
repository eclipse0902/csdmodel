"""
EEG Connectivity Analysis - Main Fine-tuning Execution Script

TUH Abnormal Dataset Fine-tuning:
1. Cross-Attention & Self-Attention ÀÚµ¿ È£È¯
2. Pre-trained ¸ðµ¨ ·Îµù
3. Binary Classification (Normal vs Abnormal)
4. »ó¼¼ÇÑ ¼º´É ºÐ¼®

»ç¿ë¹ý:
python csdmodel/main_finetune.py --mode finetune \
    --pretrain_model /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/best_pretrain_model.pth \
    --train_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_fq/train \
    --val_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_fq/val \
    --test_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_fq/test \
    --output_dir ./results \
    --freeze_encoder
"""
"""
EEG Connectivity Analysis - Main Fine-tuning Execution Script

TUH Abnormal Dataset Fine-tuning:
1. Cross-Attention & Self-Attention ÀÚµ¿ È£È¯
2. Pre-trained ¸ðµ¨ ·Îµù
3. Binary Classification (Normal vs Abnormal)
4. »ó¼¼ÇÑ ¼º´É ºÐ¼®

»ç¿ë¹ý:
python csdmodel/main_finetune.py --mode finetune \
    --pretrain_model ./checkpoints/best_pretrain_model.pth \
    --train_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_fq/train \
    --val_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_freq/val \
    --test_data /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/test \
    --output_dir ./results \
    --freeze_encoder
"""

import argparse
import os
import sys
import torch
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import EEGConfig
from data.dataset import validate_data_directory, preview_dataset_samples
from training.finetune_trainer import (
    setup_finetune_training, 
    analyze_finetune_results,
    plot_finetune_curves
)
from models.hybrid_model import EEGConnectivityModel

def finetune_mode(args):
    """?? Fine-tuning ¸ðµå"""
    print("="*80)
    print("?? EEG CONNECTIVITY FINE-TUNING")
    print("="*80)
    
    # =============== CONFIGURATION ===============
    config = EEGConfig()
    if args.config_overrides:
        print(f"?? Applying config overrides: {args.config_overrides}")
        # Apply any config overrides here
    
    # ?? Attention type reporting
    attention_type = "Cross-Attention" if config.USE_CROSS_ATTENTION else "Self-Attention"
    print(f"?? Attention Type: {attention_type}")
    print(f"?? Task: Binary Classification (Normal vs Abnormal)")
    
    # =============== PRE-TRAINED MODEL VALIDATION ===============
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
    
    # =============== DATA VALIDATION ===============
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
    val_validation = None
    if args.val_data and os.path.exists(args.val_data):
        val_validation = validate_data_directory(args.val_data, config)
        if val_validation['valid']:
            print(f"? Validation data: {val_validation['num_files']} files")
        else:
            print(f"?? Validation data issues: {val_validation['error']}")
    
    # Test data validation (optional)
    test_validation = None
    if args.test_data and os.path.exists(args.test_data):
        test_validation = validate_data_directory(args.test_data, config)
        if test_validation['valid']:
            print(f"? Test data: {test_validation['num_files']} files")
        else:
            print(f"?? Test data issues: {test_validation['error']}")
    
    # =============== DATA PREVIEW ===============
    if args.preview_data:
        print(f"\n?? Data preview:")
        preview = preview_dataset_samples(args.train_data, num_samples=3, config=config)
        for sample in preview['samples']:
            label_name = "Abnormal" if sample['label'] == 1 else "Normal"
            print(f"   Sample {sample['index']}: {label_name}, "
                  f"Magnitude={sample['magnitude_stats']['mean']:.3f}¡¾{sample['magnitude_stats']['std']:.3f}")
    
    # =============== SETUP FINE-TUNING ===============
    print(f"\n?? Setting up fine-tuning...")
    
    model, train_loader, val_loader, test_loader, trainer = setup_finetune_training(
        train_data_path=args.train_data,
        config=config,
        pretrain_checkpoint=args.pretrain_model,
        val_data_path=args.val_data,
        test_data_path=args.test_data,
        freeze_encoder=args.freeze_encoder
    )
    
    # =============== MODEL SUMMARY ===============
    model_info = model.get_model_analysis(torch.randn(1, 15, 19, 19, 2))
    print(f"\n?? Model Information:")
    print(f"   Architecture: {attention_type} + Classification Head")
    print(f"   Total Parameters: {model_info['model_info']['total_parameters']:,}")
    print(f"   Memory Estimate: ~{model_info['model_info']['memory_mb']:.1f} MB")
    print(f"   Encoder Frozen: {args.freeze_encoder}")
    
    if 'feature_extraction_stats' in model_info:
        print(f"   Feature Extraction: {model_info['feature_extraction_stats']['output_statistics']['feature_norm']:.3f} norm")
    
    # =============== START FINE-TUNING ===============
    print(f"\n?? Starting fine-tuning...")
    
    training_results = trainer.train()
    
    # =============== RESULTS SUMMARY ===============
    print(f"\n?? Fine-tuning Results Summary:")
    print(f"   Total epochs: {training_results['total_epochs_trained']}")
    print(f"   Training time: {training_results['total_training_time_hours']:.2f} hours")
    print(f"   ?? Attention type: {training_results['model_info']['attention_type']}")
    
    best_metrics = training_results['best_metrics']
    print(f"   ?? Best validation performance:")
    print(f"     Accuracy: {best_metrics['best_val_accuracy']:.4f}")
    print(f"     AUC: {best_metrics['best_val_auc']:.4f}")
    print(f"     F1 Score: {best_metrics['best_val_f1_macro']:.4f}")
    
    if 'test_metrics' in training_results:
        test_metrics = training_results['test_metrics']
        print(f"   ?? Test performance:")
        print(f"     Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"     AUC: {test_metrics['auc']:.4f}")
        print(f"     F1 Score: {test_metrics['f1_macro']:.4f}")
    
    print(f"   Session ID: {training_results['session_id']}")
    
    # =============== SAVE RESULTS ===============
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
    
    # =============== PERFORMANCE ANALYSIS ===============
    print(f"\n?? Performance Analysis:")
    
    # Compare with typical EEG classification benchmarks
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
    
    # Attention type specific insights
    if config.USE_CROSS_ATTENTION:
        print(f"   ?? Cross-Attention insights:")
        print(f"     - Should show better interpretability")
        print(f"     - May converge faster than self-attention")
        print(f"     - Look for channel-specific attention patterns")
    else:
        print(f"   ? Self-Attention insights:")
        print(f"     - Standard global attention approach")
        print(f"     - Compare with cross-attention results")
    
    print(f"\n? Fine-tuning completed successfully!")
    print(f"   ?? Best models saved in: {trainer.checkpoint_dir}")
    print(f"   ?? Training log: {trainer.log_file}")

def evaluate_mode(args):
    """?? Evaluation ¸ðµå - Fine-tuned ¸ðµ¨ Æò°¡"""
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
        if 'attention_type' in checkpoint:
            print(f"   ?? Attention: {checkpoint['attention_type']}")
        
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
    from data.dataset import EEGDataset
    from torch.utils.data import DataLoader
    
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
    try:
        if all_probabilities.shape[1] == 2:
            auc_score = roc_auc_score(all_labels, all_probabilities[:, 1])
        else:
            auc_score = 0.0
    except:
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
    
    print(f"\n? Evaluation completed successfully!")

def analyze_mode(args):
    """?? Analysis ¸ðµå - ÈÆ·Ã °á°ú ºÐ¼®"""
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
    print(f"   ?? Attention: {analysis_results['training_summary']['attention_type']}")
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

def compare_mode(args):
    """?? Compare ¸ðµå - Cross vs Self Attention ºñ±³"""
    print("="*80)
    print("?? CROSS-ATTENTION vs SELF-ATTENTION COMPARISON")
    print("="*80)
    
    if not args.cross_log or not args.self_log:
        print(f"? Need both log files for comparison")
        print(f"   Cross-attention log: {args.cross_log}")
        print(f"   Self-attention log: {args.self_log}")
        return
    
    if not os.path.exists(args.cross_log) or not os.path.exists(args.self_log):
        print(f"? Log files not found")
        return
    
    print(f"?? Comparing attention mechanisms...")
    print(f"   Cross-attention: {args.cross_log}")
    print(f"   Self-attention: {args.self_log}")
    
    # Analyze both logs
    cross_results = analyze_finetune_results(args.cross_log)
    self_results = analyze_finetune_results(args.self_log)
    
    if not cross_results or not self_results:
        print(f"? Failed to analyze one or both log files")
        return
    
    # Comparison
    comparison = {
        'cross_attention': cross_results,
        'self_attention': self_results,
        'comparison': {
            'accuracy_diff': cross_results['best_performance']['best_accuracy'] - self_results['best_performance']['best_accuracy'],
            'auc_diff': cross_results['best_performance']['best_auc'] - self_results['best_performance']['best_auc'],
            'f1_diff': cross_results['best_performance']['best_f1'] - self_results['best_performance']['best_f1'],
            'epochs_diff': cross_results['training_summary']['total_epochs'] - self_results['training_summary']['total_epochs']
        }
    }
    
    print(f"\n?? Comparison Results:")
    print(f"{'='*50}")
    print(f"{'Metric':<15} {'Cross-Att':<12} {'Self-Att':<12} {'Difference':<12}")
    print(f"{'='*50}")
    
    cross_best = cross_results['best_performance']
    self_best = self_results['best_performance']
    
    acc_diff = cross_best['best_accuracy'] - self_best['best_accuracy']
    auc_diff = cross_best['best_auc'] - self_best['best_auc']
    f1_diff = cross_best['best_f1'] - self_best['best_f1']
    
    print(f"{'Accuracy':<15} {cross_best['best_accuracy']:<12.4f} {self_best['best_accuracy']:<12.4f} {acc_diff:<12.4f}")
    print(f"{'AUC':<15} {cross_best['best_auc']:<12.4f} {self_best['best_auc']:<12.4f} {auc_diff:<12.4f}")
    print(f"{'F1-Score':<15} {cross_best['best_f1']:<12.4f} {self_best['best_f1']:<12.4f} {f1_diff:<12.4f}")
    
    print(f"\n?? Winner Analysis:")
    winners = []
    if acc_diff > 0.01:
        winners.append("Cross-Attention wins on Accuracy")
    elif acc_diff < -0.01:
        winners.append("Self-Attention wins on Accuracy")
    else:
        winners.append("Accuracy: Tie")
    
    if auc_diff > 0.01:
        winners.append("Cross-Attention wins on AUC")
    elif auc_diff < -0.01:
        winners.append("Self-Attention wins on AUC")
    else:
        winners.append("AUC: Tie")
    
    if f1_diff > 0.01:
        winners.append("Cross-Attention wins on F1")
    elif f1_diff < -0.01:
        winners.append("Self-Attention wins on F1")
    else:
        winners.append("F1: Tie")
    
    for winner in winners:
        print(f"   {winner}")
    
    # Overall winner
    wins_cross = sum(1 for w in winners if "Cross-Attention wins" in w)
    wins_self = sum(1 for w in winners if "Self-Attention wins" in w)
    
    if wins_cross > wins_self:
        print(f"\n?? Overall Winner: Cross-Attention ({wins_cross}/{len(winners)} metrics)")
    elif wins_self > wins_cross:
        print(f"\n?? Overall Winner: Self-Attention ({wins_self}/{len(winners)} metrics)")
    else:
        print(f"\n?? Overall Result: Tie")
    
    # Save comparison
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        comparison_path = os.path.join(args.output_dir, "attention_comparison.json")
        
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"?? Comparison saved to: {comparison_path}")
    
    print(f"\n? Comparison completed successfully!")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="EEG Connectivity Fine-tuning with Cross-Attention Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tuning with Cross-Attention
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
  
  # Compare attention types
  python main_finetune.py --mode compare \
      --cross_log ./logs/finetune_log_*_cross.json \
      --self_log ./logs/finetune_log_*_self.json \
      --output_dir ./comparison
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', 
                       choices=['finetune', 'evaluate', 'analyze', 'compare'], 
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
    parser.add_argument('--cross_log', type=str, help='Cross-attention log file for comparison')
    parser.add_argument('--self_log', type=str, help='Self-attention log file for comparison')
    
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
            
        elif args.mode == 'compare':
            if not args.cross_log or not args.self_log:
                print("? --cross_log and --self_log required for comparison")
                return
            compare_mode(args)
            
    except KeyboardInterrupt:
        print(f"\n?? Process interrupted by user")
    except Exception as e:
        print(f"? Error in {args.mode} mode: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()