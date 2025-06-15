#!/usr/bin/env python3
"""
EEG Model Shape Debugging Test
"""

import torch
import sys
import os
sys.path.append('.')

def test_shape_compatibility():
    """Shape È£È¯¼º Å×½ºÆ®"""
    print("?? Testing Shape Compatibility...")
    
    try:
        from config import EEGConfig
        from models.hybrid_model import create_pretrain_model
        
        config = EEGConfig()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"?? Device: {device}")
        
        # ¸ðµ¨ »ý¼º
        model = create_pretrain_model(config)
        model.to(device)
        
        # Å×½ºÆ® ÀÔ·Âµé
        test_cases = [
            ("Matrix format", torch.randn(2, 15, 19, 19, 2)),
            ("Pair format", torch.randn(2, 361, 15, 2)),
        ]
        
        for case_name, test_input in test_cases:
            print(f"\n?? {case_name}: {test_input.shape}")
            test_input = test_input.to(device)
            
            try:
                # Test get_features
                if case_name == "Matrix format":
                    features = model.get_features(test_input)
                    print(f"   ? get_features: {test_input.shape} ¡æ {features.shape}")
                
                # Test feature_extraction directly
                if case_name == "Pair format":
                    features = model.feature_extraction(test_input)
                    print(f"   ? feature_extraction: {test_input.shape} ¡æ {features.shape}")
                
                # Test get_feature_statistics
                if case_name == "Matrix format":
                    stats = model.feature_extraction.get_feature_statistics(test_input)
                    print(f"   ? get_feature_statistics: shape converted = {stats['input_info']['shape_converted']}")
                    print(f"   ?? Input: {stats['input_info']['original_shape']} ¡æ {stats['input_info']['converted_shape']}")
                    print(f"   ?? Output: {stats['information_preservation']['output_shape']}")
                
            except Exception as e:
                print(f"   ? {case_name} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Test model analysis
        print(f"\n?? Testing Model Analysis...")
        try:
            matrix_input = torch.randn(1, 15, 19, 19, 2).to(device)
            analysis = model.get_model_analysis(matrix_input)
            print(f"   ? Model analysis successful")
            print(f"   ?? Total parameters: {analysis['model_info']['total_parameters']:,}")
            print(f"   ?? Feature shape: {analysis['feature_shape']}")
            print(f"   ?? Input shape: {analysis['input_shape']}")
            
        except Exception as e:
            print(f"   ? Model analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test forward pass
        print(f"\n?? Testing Forward Pass...")
        try:
            matrix_input = torch.randn(2, 15, 19, 19, 2).to(device)
            output = model(matrix_input)
            print(f"   ? Forward pass: {matrix_input.shape} ¡æ {output.shape}")
            
        except Exception as e:
            print(f"   ? Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n?? Shape compatibility testing completed!")
        
    except Exception as e:
        print(f"? Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shape_compatibility()