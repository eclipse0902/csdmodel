"""
EEG Visualization Runner - Real Model and Dataset
cd /home/mjkang/cbramod/csdmodel_20/visualize/

python reconstruct.py \
    --model_path /home/mjkang/cbramod/csdmodel_20/checkpoints/best_pretrain_model.pth \
    --data_path /remotenas0/database/TUH_Corpus/tuh_eeg_abnormal/v3.0.1/edf/normalized_20freq/val \
    --output_dir ./results \
    --num_samples 3
½ÇÁ¦ ÈÆ·ÃµÈ ¸ðµ¨°ú µ¥ÀÌÅÍ¼ÂÀ» »ç¿ëÇÏ¿© º¹¿ø Ç°Áú ½Ã°¢È­
"""

import argparse
import torch
import numpy as np
import os
import sys
from typing import Optional, Tuple
import random

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))  # /home/mjkang/cbramod/csdmodel/visualize/
csdmodel_dir = os.path.dirname(current_dir)              # /home/mjkang/cbramod/csdmodel/
sys.path.append(csdmodel_dir)

from config import EEGConfig
from models.hybrid_model import EEGConnectivityModel
from data.dataset import EEGDataset
from eeg_visualization import EEGReconstructionVisualizer

def apply_masking(data: torch.Tensor, mask_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    µ¥ÀÌÅÍ¿¡ ¸¶½ºÅ· Àû¿ë
    
    Args:
        data: (15, 19, 19, 2) CSD data
        mask_ratio: ¸¶½ºÅ· ºñÀ²
        
    Returns:
        masked_data, mask
    """
    mask = torch.ones_like(data)
    
    # 361°³ Àü±Ø ½Ö¿¡¼­ ¸¶½ºÅ·ÇÒ °³¼ö
    num_pairs = 19 * 19
    num_to_mask = int(num_pairs * mask_ratio)
    
    # ·£´ý À§Ä¡ ¼±ÅÃ
    mask_indices = torch.randperm(num_pairs)[:num_to_mask]
    
    for idx in mask_indices:
        i, j = idx // 19, idx % 19
        # ¸ðµç ÁÖÆÄ¼ö¿¡¼­ ÇØ´ç Àü±Ø ½Ö ¸¶½ºÅ·
        mask[:, i, j, :] = 0
    
    masked_data = data * mask
    return masked_data, mask

def load_model_and_data(model_path: str, 
                       data_path: str, 
                       config: EEGConfig,
                       num_samples: int = 5) -> Tuple[EEGConnectivityModel, list]:
    """
    ¸ðµ¨°ú µ¥ÀÌÅÍ ·Îµå
    
    Args:
        model_path: ¸ðµ¨ Ã¼Å©Æ÷ÀÎÆ® °æ·Î
        data_path: µ¥ÀÌÅÍ¼Â °æ·Î
        config: EEG ¼³Á¤
        num_samples: ºÐ¼®ÇÒ »ùÇÃ ¼ö
        
    Returns:
        model, data_samples
    """
    print(f"?? Loading model and data...")
    
    # 1. ¸ðµ¨ ·Îµå
    print(f"   Loading model from: {model_path}")
    
    # Pre-training ¸ðµ¨Àº 'pretrain' ¸ðµå·Î ·Îµå
    try:
        model = EEGConnectivityModel.load_model(model_path, mode='pretrain')
    except Exception as e:
        print(f"   Warning: Failed to load as pretrain mode, trying inference mode...")
        try:
            model = EEGConnectivityModel.load_model(model_path, mode='inference')
        except Exception as e2:
            print(f"   Error: Failed to load model in both modes")
            print(f"   Pretrain mode error: {str(e)}")
            print(f"   Inference mode error: {str(e2)}")
            raise e
    
    model.eval()
    print(f"   ? Model loaded successfully")
    
    # 2. µ¥ÀÌÅÍ¼Â ·Îµå
    print(f"   Loading dataset from: {data_path}")
    dataset = EEGDataset(
        data_path=data_path,
        config=config,
        apply_masking=False,  # ¿ì¸®°¡ Á÷Á¢ ¸¶½ºÅ· Àû¿ë
        normalize_data=True
    )
    print(f"   ? Dataset loaded: {len(dataset)} samples")
    
    # 3. »ùÇÃ ¼±ÅÃ
    print(f"   Selecting {num_samples} samples...")
    data_samples = []
    
    # ´Ù¾çÇÑ Å¬·¡½º¿¡¼­ »ùÇÃ ¼±ÅÃ
    class_0_samples = []
    class_1_samples = []
    
    # °¢ Å¬·¡½ºº°·Î »ùÇÃ ¼öÁý
    for i in range(min(len(dataset), 100)):  # ÃÖ´ë 100°³¸¸ Ã¼Å©
        csd_data, label = dataset[i]
        label_value = label.item()
        
        if label_value == 0 and len(class_0_samples) < num_samples // 2 + 1:
            class_0_samples.append((csd_data, label_value, i))
        elif label_value == 1 and len(class_1_samples) < num_samples // 2 + 1:
            class_1_samples.append((csd_data, label_value, i))
        
        if len(class_0_samples) + len(class_1_samples) >= num_samples:
            break
    
    # ¼±ÅÃµÈ »ùÇÃµé °áÇÕ
    all_samples = class_0_samples + class_1_samples
    data_samples = all_samples[:num_samples]
    
    print(f"   ? Selected samples:")
    for i, (_, label, idx) in enumerate(data_samples):
        print(f"      Sample {i}: Index {idx}, Label {label}")
    
    return model, data_samples

def run_single_sample_analysis(model: EEGConnectivityModel,
                              original_csd: torch.Tensor,
                              sample_idx: int,
                              label: int,
                              device: str,
                              mask_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ´ÜÀÏ »ùÇÃ ºÐ¼® ½ÇÇà
    
    Args:
        model: EEG ¸ðµ¨
        original_csd: ¿øº» CSD µ¥ÀÌÅÍ (15, 19, 19, 2)
        sample_idx: »ùÇÃ ÀÎµ¦½º
        label: »ùÇÃ ¶óº§
        device: µð¹ÙÀÌ½º
        mask_ratio: ¸¶½ºÅ· ºñÀ²
        
    Returns:
        original, reconstructed, mask
    """
    print(f"\n?? Analyzing sample {sample_idx} (Label: {label})...")
    
    # Device·Î ÀÌµ¿
    original_csd = original_csd.to(device)
    
    # ¸¶½ºÅ· Àû¿ë
    masked_csd, mask = apply_masking(original_csd, mask_ratio)
    
    print(f"   Applied {mask_ratio*100:.0f}% masking")
    print(f"   Original shape: {original_csd.shape}")
    print(f"   Masked positions: {(mask == 0).sum().item()} / {mask.numel()}")
    
    # ¸ðµ¨ Ãß·Ð
    with torch.no_grad():
        # ¹èÄ¡ Â÷¿ø Ãß°¡
        masked_input = masked_csd.unsqueeze(0)  # (1, 15, 19, 19, 2)
        
        # ¸ðµ¨ forward
        reconstructed_batch = model(masked_input)  # (1, 15, 19, 19, 2)
        
        # ¹èÄ¡ Â÷¿ø Á¦°Å
        reconstructed_csd = reconstructed_batch.squeeze(0)  # (15, 19, 19, 2)
    
    print(f"   ? Reconstruction completed")
    print(f"   Reconstructed shape: {reconstructed_csd.shape}")
    
    # CPU·Î ÀÌµ¿ (½Ã°¢È­¿ë)
    original_csd = original_csd.cpu()
    reconstructed_csd = reconstructed_csd.cpu()
    mask = mask.cpu()
    
    return original_csd, reconstructed_csd, mask

def main():
    """¸ÞÀÎ ½ÇÇà ÇÔ¼ö"""
    parser = argparse.ArgumentParser(description="EEG Reconstruction Visualization with Real Model and Dataset")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./visualization_results',
                       help='Output directory for visualization results')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to analyze')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                       help='Masking ratio (0.0-1.0)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Training log file for progress analysis (optional)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("?? EEG RECONSTRUCTION VISUALIZATION - REAL MODEL & DATA")
    print("="*80)
    
    # Device ¼³Á¤
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"?? Configuration:")
    print(f"   Model: {args.model_path}")
    print(f"   Dataset: {args.data_path}")
    print(f"   Output: {args.output_dir}")
    print(f"   Samples: {args.num_samples}")
    print(f"   Mask ratio: {args.mask_ratio}")
    print(f"   Device: {device}")
    
    # Ãâ·Â µð·ºÅä¸® »ý¼º
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # ¼³Á¤ ·Îµå
        config = EEGConfig()
        
        # ¸ðµ¨°ú µ¥ÀÌÅÍ ·Îµå
        model, data_samples = load_model_and_data(
            args.model_path, args.data_path, config, args.num_samples
        )
        
        # ¸ðµ¨À» device·Î ÀÌµ¿
        model = model.to(device)
        
        # ½Ã°¢È­±â ÃÊ±âÈ­
        visualizer = EEGReconstructionVisualizer(model=model, config=config, device=device)
        
        print(f"\n?? Starting visualization analysis...")
        
        # °¢ »ùÇÃº°·Î ºÐ¼®
        for i, (original_csd, label, sample_idx) in enumerate(data_samples):
            print(f"\n{'='*60}")
            print(f"SAMPLE {i+1}/{len(data_samples)}")
            print(f"{'='*60}")
            
            # ´ÜÀÏ »ùÇÃ ºÐ¼®
            original, reconstructed, mask = run_single_sample_analysis(
                model, original_csd, sample_idx, label, device, args.mask_ratio
            )
            
            # °³º° »ùÇÃ ½Ã°¢È­
            sample_output_dir = os.path.join(args.output_dir, f"sample_{sample_idx}_label_{label}")
            
            print(f"   ?? Generating visualizations...")
            report_files = visualizer.generate_full_report(
                original, reconstructed, mask,
                output_dir=sample_output_dir,
                log_file_path=args.log_file
            )
            
            print(f"   ? Sample {i+1} analysis completed!")
            print(f"      Report: {sample_output_dir}/")
            print(f"      HTML: {report_files.get('html_report', 'N/A')}")
            
            # ¸ÞÆ®¸¯ Ãâ·Â
            metrics = visualizer.compute_reconstruction_metrics(original, reconstructed, mask)
            print(f"   ?? Key Metrics:")
            print(f"      Phase Error: {metrics['phase_error_degrees']:.1f}¡Æ (Target: <25¡Æ)")
            print(f"      Magnitude Error: {metrics['magnitude_relative_error']:.1f}% (Target: <8%)")
            print(f"      Correlation: {metrics['correlation']:.3f} (Target: >0.8)")
            print(f"      SNR: {metrics['snr_db']:.1f} dB (Target: >0 dB)")
        
        # ÀüÃ¼ ¿ä¾à ¸®Æ÷Æ® »ý¼º
        print(f"\n?? VISUALIZATION COMPLETED!")
        print(f"{'='*80}")
        print(f"?? Results saved in: {args.output_dir}/")
        print(f"?? Analyzed {len(data_samples)} samples")
        print(f"?? Open HTML reports to view detailed analysis")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n? Error during visualization:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_interactive_demo():
    """´ëÈ­Çü µ¥¸ð ½ÇÇà"""
    print("?? Interactive Demo Mode")
    print("This will create mock data for demonstration...")
    
    # Mock µ¥ÀÌÅÍ »ý¼º
    config = EEGConfig()
    
    # Mock ¸ðµ¨ (½ÇÁ¦·Î´Â ´õ¹Ì)
    model = None
    
    # Mock µ¥ÀÌÅÍ
    original = torch.randn(20, 19, 19, 2) * 0.3
    
    # Çö½ÇÀûÀÎ ¿¬°á¼º ÆÐÅÏ »ý¼º
    for freq in range(20):
        for i in range(19):
            for j in range(19):
                if i == j:
                    # ´ë°¢¼±Àº ³ôÀº °ª
                    original[freq, i, j, 0] = 1.0 + torch.randn(1) * 0.1
                    original[freq, i, j, 1] = torch.randn(1) * 0.05
                else:
                    # °Å¸® ±â¹Ý ¿¬°á¼º
                    distance = abs(i - j)
                    strength = 0.5 / (1 + distance * 0.1)
                    original[freq, i, j, 0] = strength * (1 + torch.randn(1) * 0.2)
                    original[freq, i, j, 1] = strength * torch.randn(1) * 0.2
    
    # ¸¶½ºÅ· Àû¿ë
    masked_data, mask = apply_masking(original, 0.5)
    
    # Mock º¹¿ø (¿øº» + ³ëÀÌÁî)
    reconstructed = original.clone()
    noise = torch.randn_like(original) * 0.1
    reconstructed = original * mask + (original + noise) * (1 - mask)
    
    # ½Ã°¢È­
    visualizer = EEGReconstructionVisualizer(config=config)
    
    report_files = visualizer.generate_full_report(
        original, reconstructed, mask,
        output_dir="./demo_visualization"
    )
    
    print(f"? Demo completed! Check ./demo_visualization/")
    return report_files

if __name__ == "__main__":
    # ÀÎ¼ö°¡ ¾øÀ¸¸é µ¥¸ð ¸ðµå
    if len(sys.argv) == 1:
        print("No arguments provided. Running interactive demo...")
        run_interactive_demo()
    else:
        main()