"""
EEG Connectivity Analysis - Complete Usage Example

This script demonstrates the complete workflow for training and evaluating
the EEG connectivity analysis model. It shows how to:

1. Set up configuration and data paths
2. Load and preprocess EEG connectivity data
3. Create and train the hybrid model
4. Evaluate model performance
5. Analyze results and generate interpretability outputs

Usage:
    python main.py --mode train --data_dir ./data --config_file config.yaml
    python main.py --mode evaluate --model_path best_model.pth --test_data ./test_data
    python main.py --mode analyze --model_path best_model.pth --sample_data sample.pkl
"""

import argparse
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import EEGConfig
from data.dataset import create_data_loaders, EEGDataset
from models.hybrid_model import EEGHybridModel
from csdmodel.training.pretrain_trainer import EEGTrainer, CrossValidator, TrainingVisualizer, train_eeg_model

def setup_directories(config: EEGConfig):
    """Create necessary directories for the project"""
    config.create_directories()
    print("Project directories created/verified")

def train_model(args):
    """
    Train the EEG connectivity analysis model
    
    Args:
        args: Command line arguments
    """
    print("="*60)
    print("EEG CONNECTIVITY ANALYSIS - TRAINING MODE")
    print("="*60)
    
    # Initialize configuration
    config = EEGConfig()
    
    # Override data paths if provided
    if args.data_dir:
        config.DATA_CONFIG['train_data_path'] = os.path.join(args.data_dir, 'train')
        config.DATA_CONFIG['val_data_path'] = os.path.join(args.data_dir, 'val') 
        config.DATA_CONFIG['test_data_path'] = os.path.join(args.data_dir, 'test')
    
    # Setup directories
    setup_directories(config)
    
    # Print configuration summary
    config.print_config_summary()
    
    try:
        # Create data loaders
        print("\nLoading datasets...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Create model
        print("\nInitializing model...")
        model = EEGHybridModel(config)
        
        # Create trainer
        print("\nInitializing trainer...")
        trainer = EEGTrainer(model, train_loader, val_loader, config)
        
        # Resume from checkpoint if specified
        if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
            trainer.load_checkpoint(args.resume_checkpoint)
            print(f"Resumed training from {args.resume_checkpoint}")
        
        # Start training
        print("\nStarting training...")
        training_results = trainer.train()
        
        # Evaluate on test set
        if test_loader:
            print("\nEvaluating on test set...")
            test_results = trainer.evaluate_on_test(test_loader)
            
            # Save test results
            test_results_path = os.path.join(config.DATA_CONFIG['log_path'], 
                                           f"test_results_{trainer.session_id}.json")
            import json
            with open(test_results_path, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            print(f"Test results saved to {test_results_path}")
        
        # Generate training visualization
        if args.plot_training:
            plot_path = os.path.join(config.DATA_CONFIG['log_path'], 
                                   f"training_curves_{trainer.session_id}.png")
            TrainingVisualizer.plot_training_history(trainer.training_history, plot_path)
        
        print(f"\nTraining completed successfully!")
        print(f"Session ID: {trainer.session_id}")
        print(f"Best validation accuracy: {training_results['best_val_accuracy']:.4f}")
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

def evaluate_model(args):
    """
    Evaluate a trained model
    
    Args:
        args: Command line arguments
    """
    print("="*60)
    print("EEG CONNECTIVITY ANALYSIS - EVALUATION MODE")
    print("="*60)
    
    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError(f"Model path {args.model_path} does not exist")
    
    # Initialize configuration
    config = EEGConfig()
    
    # Override test data path if provided
    if args.test_data:
        config.DATA_CONFIG['test_data_path'] = args.test_data
    
    print(f"Loading model from {args.model_path}")
    
    # Create model and load checkpoint
    model = EEGHybridModel(config)
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print("Loading test dataset...")
    test_dataset = EEGDataset(config.DATA_CONFIG['test_data_path'], apply_masking=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.TRAINING_CONFIG['batch_size'],
        shuffle=False
    )
    
    # Comprehensive evaluation
    print("Performing comprehensive evaluation...")
    evaluation_results = model.evaluate_model(test_loader)
    
    # Print results
    print("\nEvaluation Results:")
    print("="*40)
    from models.classifier import ClassificationMetrics
    ClassificationMetrics.print_metrics(evaluation_results['basic_metrics'])
    
    print(f"\nConfidence Analysis:")
    conf_analysis = evaluation_results['confidence_analysis']
    print(f"Mean Confidence: {conf_analysis['mean_confidence']:.4f}")
    print(f"Confidence Std: {conf_analysis['confidence_std']:.4f}")
    print(f"Low Confidence Samples: {conf_analysis['low_confidence_samples']}")
    
    # Save detailed results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        
        import json
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {results_path}")

def analyze_model(args):
    """
    Analyze model predictions and interpretability
    
    Args:
        args: Command line arguments  
    """
    print("="*60)
    print("EEG CONNECTIVITY ANALYSIS - ANALYSIS MODE")
    print("="*60)
    
    if not args.model_path or not os.path.exists(args.model_path):
        raise ValueError(f"Model path {args.model_path} does not exist")
    
    if not args.sample_data or not os.path.exists(args.sample_data):
        raise ValueError(f"Sample data {args.sample_data} does not exist")
    
    # Initialize configuration
    config = EEGConfig()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = EEGHybridModel(config)
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load sample data
    print(f"Loading sample data from {args.sample_data}")
    import pickle
    with open(args.sample_data, 'rb') as f:
        sample_data = pickle.load(f)
    
    # Convert to tensor and add batch dimension
    csd_data = torch.FloatTensor(sample_data['csd_normalized']).unsqueeze(0)  # (1, 15, 19, 19, 2)
    true_label = sample_data['label']
    
    print(f"Sample info: True label = {true_label} ({'Normal' if true_label == 0 else 'Abnormal'})")
    
    # Make prediction
    print("\nMaking prediction...")
    prediction_results = model.predict(csd_data)
    
    pred_class = prediction_results['predicted_classes'][0].item()
    pred_prob = prediction_results['class_probabilities'][0]
    confidence = prediction_results['confidence_scores'][0].item()
    
    print(f"Predicted class: {pred_class} ({'Normal' if pred_class == 0 else 'Abnormal'})")
    print(f"Confidence: {confidence:.4f}")
    print(f"Class probabilities: Normal={pred_prob[0]:.4f}, Abnormal={pred_prob[1]:.4f}")
    
    # Connectivity pattern analysis
    print("\nAnalyzing connectivity patterns...")
    connectivity_analysis = model.analyze_connectivity_patterns(csd_data)
    
    print("\nTop connectivity patterns:")
    for pair_name, analysis in list(connectivity_analysis['raw_connectivity'].items())[:5]:
        strength = analysis['connectivity_strength'].item()
        dom_freq_idx = analysis['dominant_frequency'].item()
        dom_freq = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29][dom_freq_idx]
        print(f"{pair_name}: Strength={strength:.4f}, Dominant Freq={dom_freq}Hz")
    
    # Stage-wise feature analysis
    print("\nExtracting stage-wise features...")
    features = model.extract_features(csd_data, stage='all')
    
    print(f"Stage 1 features shape: {features['stage1'].shape}")
    print(f"Stage 2 features shape: {features['stage2'].shape}")
    print(f"Stage 3 classifier info: {len(features['stage3'])} components")
    
    # Save analysis results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        analysis_results = {
            'sample_info': {
                'true_label': true_label,
                'predicted_class': pred_class,
                'confidence': confidence,
                'class_probabilities': pred_prob.tolist()
            },
            'connectivity_analysis': {
                k: {
                    'connectivity_strength': v['connectivity_strength'].item() if torch.is_tensor(v['connectivity_strength']) else v['connectivity_strength'],
                    'dominant_frequency': v['dominant_frequency'].item() if torch.is_tensor(v['dominant_frequency']) else v['dominant_frequency']
                } for k, v in connectivity_analysis['raw_connectivity'].items()
            },
            'feature_shapes': {
                'stage1': list(features['stage1'].shape),
                'stage2': list(features['stage2'].shape)
            }
        }
        
        import json
        analysis_path = os.path.join(args.output_dir, "analysis_results.json")
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\nAnalysis results saved to {analysis_path}")

def cross_validate_model(args):
    """
    Perform cross-validation
    
    Args:
        args: Command line arguments
    """
    print("="*60)
    print("EEG CONNECTIVITY ANALYSIS - CROSS VALIDATION")
    print("="*60)
    
    # Initialize configuration
    config = EEGConfig()
    
    if args.data_dir:
        data_path = os.path.join(args.data_dir, 'train')  # Use training data for CV
    else:
        data_path = config.DATA_CONFIG['train_data_path']
    
    # Create dataset
    print(f"Loading dataset from {data_path}")
    dataset = EEGDataset(data_path, apply_masking=True)
    
    # Perform cross-validation
    cv = CrossValidator(config)
    cv_results = cv.k_fold_cross_validation(dataset, k=args.cv_folds)
    
    # Save CV results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        cv_path = os.path.join(args.output_dir, "cross_validation_results.json")
        
        import json
        with open(cv_path, 'w') as f:
            json.dump(cv_results, f, indent=2, default=str)
        print(f"\nCross-validation results saved to {cv_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="EEG Connectivity Analysis")
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'evaluate', 'analyze', 'cross_validate'], 
                       required=True, help='Operation mode')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, help='Directory containing train/val/test data')
    parser.add_argument('--test_data', type=str, help='Path to test data directory')
    parser.add_argument('--sample_data', type=str, help='Path to sample .pkl file for analysis')
    
    # Model paths
    parser.add_argument('--model_path', type=str, help='Path to trained model checkpoint')
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint to resume training')
    
    # Output options
    parser.add_argument('--output_dir', type=str, help='Directory to save results')
    parser.add_argument('--plot_training', action='store_true', help='Generate training plots')
    
    # Cross-validation options
    parser.add_argument('--cv_folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        if args.mode == 'train':
            train_model(args)
        elif args.mode == 'evaluate':
            evaluate_model(args)
        elif args.mode == 'analyze':
            analyze_model(args)
        elif args.mode == 'cross_validate':
            cross_validate_model(args)
            
    except Exception as e:
        print(f"Error in {args.mode} mode: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example usage commands:
"""
# Train a new model
python main.py --mode train --data_dir ./eeg_data --plot_training

# Evaluate trained model
python main.py --mode evaluate --model_path ./checkpoints/best_model.pth --test_data ./eeg_data/test --output_dir ./results

# Analyze specific sample
python main.py --mode analyze --model_path ./checkpoints/best_model.pth --sample_data ./sample.pkl --output_dir ./analysis

# Cross-validation
python main.py --mode cross_validate --data_dir ./eeg_data --cv_folds 5 --output_dir ./cv_results

# Resume training from checkpoint
python main.py --mode train --data_dir ./eeg_data --resume_checkpoint ./checkpoints/checkpoint_epoch_50.pth
"""