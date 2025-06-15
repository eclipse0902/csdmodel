"""
EEG Multi-Scale Config - Fixed Version

±âÁ¸ EEGConfig¿Í ¿ÏÀü È£È¯µÇµµ·Ï ¼öÁ¤:
1. PRETRAINING_CONFIG ´©¶ô ¹®Á¦ ÇØ°á
2. ±âÁ¸ config ±¸Á¶ »ó¼Ó
3. Multi-Scale Àü¿ë Ãß°¡ ¼³Á¤
"""

import torch
import os
import numpy as np

# ±âÁ¸ config È£È¯¼ºÀ» À§ÇÑ import
try:
    from config import EEGConfig
    BASE_CONFIG_AVAILABLE = True
except ImportError:
    BASE_CONFIG_AVAILABLE = False
    EEGConfig = object

class MultiScaleEEGConfig(EEGConfig if BASE_CONFIG_AVAILABLE else object):
    """
    Multi-Scale EEG ¼³Á¤ Å¬·¡½º - ±âÁ¸ EEGConfig ¿ÏÀü È£È¯
    """
    
    # =============== ±âº» µ¥ÀÌÅÍ ±¸Á¶ (±âÁ¸ À¯Áö) ===============
    NUM_FREQUENCIES = 20      
    NUM_ELECTRODES = 19         
    NUM_COMPLEX_DIMS = 2        
    NUM_CLASSES = 2
    NUM_PAIRS = NUM_ELECTRODES * NUM_ELECTRODES  # 361
    
    # =============== ±âÁ¸ ¼³Á¤ À¯Áö ===============
    FREQUENCY_FEATURE_DIM = 32
    COMPLEX_FEATURE_DIM = 32
    UNIFIED_FEATURE_DIM = 64
    
    # CNN Configuration (±âÁ¸ È£È¯)
    USE_CNN_BACKBONE = True
    USE_CROSS_ATTENTION = False
    
    # =============== PRETRAINING_CONFIG ÇÊ¼ö Ãß°¡ ===============
    PRETRAINING_CONFIG = {
        'mask_ratio': 0.3,            # Multi-Scale¿ë ÃÖÀûÈ­
        'num_epochs': 50,
        'learning_rate': 5e-4,        
        'batch_size': 64,             # Multi-Scale ¸Þ¸ð¸® °í·Á
        'weight_decay': 2e-3,
        
        # Multi-Scale ¸¶½ºÅ· Àü·«
        'masking_strategy': 'structured',
        'masking_config': {
            'random_prob': 0.7,
            'preserve_diagonal': True,
            'hermitian_symmetric': True,
            'spatial_coherence': True
        }
    }
    
    # =============== Multi-Scale ¼¼±×¸ÕÆ® ¼³Á¤ ===============
    SCALE_CONFIGS = {
        '4s': {
            'segment_length': 4,
            'num_segments': 4,  # 4ÃÊ ¡¿ 4°³
            'receptive_field': 'small',
            'optimization': 'high_freq',
            'temporal_resolution': 'fine',
            'feature_dim': 64
        },
        '8s': {
            'segment_length': 8, 
            'num_segments': 2,  # 8ÃÊ ¡¿ 2°³
            'receptive_field': 'medium',
            'optimization': 'mid_term',
            'temporal_resolution': 'medium',
            'feature_dim': 64
        },
        '16s': {
            'segment_length': 16,
            'num_segments': 1,  # 16ÃÊ ¡¿ 1°³
            'receptive_field': 'large',
            'optimization': 'long_term',
            'temporal_resolution': 'coarse',
            'feature_dim': 64
        }
    }
    
    # =============== Multi-Scale Feature ¼³Á¤ ===============
    MULTISCALE_FEATURE_CONFIG = {
        # Scaleº° µ¶¸³ Feature Extraction
        'scale_processors': {
            '4s': {
                'input_dim': NUM_FREQUENCIES,
                'hidden_dims': [32, 64],
                'output_dim': 64,
                'temporal_conv': {
                    'kernel_size': 3,
                    'stride': 1,
                    'padding': 1
                },
                'optimization_target': 'high_frequency_dynamics'
            },
            '8s': {
                'input_dim': NUM_FREQUENCIES,
                'hidden_dims': [32, 64], 
                'output_dim': 64,
                'temporal_conv': {
                    'kernel_size': 5,
                    'stride': 1,
                    'padding': 2
                },
                'optimization_target': 'rhythm_stability'
            },
            '16s': {
                'input_dim': NUM_FREQUENCIES,
                'hidden_dims': [32, 64],
                'output_dim': 64,
                'temporal_conv': {
                    'kernel_size': 7,
                    'stride': 1, 
                    'padding': 3
                },
                'optimization_target': 'network_transitions'
            }
        },
        
        # Cross-Scale Attention ¼³Á¤
        'cross_scale_attention': {
            'num_heads': 8,
            'attention_dim': 64,
            'dropout': 0.1,
            'use_position_encoding': True
        },
        
        # Multi-Scale Fusion ¼³Á¤
        'fusion_config': {
            'input_dim': 64 * 3,  # 3°³ ½ºÄÉÀÏ °áÇÕ
            'hidden_dims': [128, 64],
            'output_dim': 64,
            'fusion_strategy': 'hierarchical',  # 'concat' | 'hierarchical' | 'attention'
            'scale_weights': {
                '4s': 1.0,
                '8s': 1.0, 
                '16s': 1.0
            }
        }
    }
    
    # =============== ±âÁ¸ Feature Extraction Config À¯Áö ===============
    FEATURE_EXTRACTION_CONFIG = {
        'frequency_processor': {
            'input_dim': NUM_FREQUENCIES,
            'hidden_dims': [40, 32],
            'output_dim': 32,
            'activation': 'gelu',
            'dropout': 0.1
        },
        'complex_processor': {
            'input_dim': NUM_COMPLEX_DIMS,
            'hidden_dims': [16, 32],
            'output_dim': 32,
            'activation': 'gelu',
            'dropout': 0.1,
            'frequency_independent': True,
            'shared_across_frequencies': False
        },
        'fusion_config': {
            'input_dim': 64,
            'hidden_dims': [64],
            'output_dim': 64,
            'activation': 'gelu',
            'dropout': 0.1,
            'use_residual': True
        },
        'frequency_aggregation': 'attention',
        'complex_combination': 'separate'
    }
    
    # =============== CNN Configuration À¯Áö ===============
    CNN_ATTENTION_CONFIG = {
        'input_dim': 64,
        'architecture': 'cnn_spatial',
        'cnn_layers': [
            {'channels': 32, 'kernel': 3, 'padding': 1},
            {'channels': 64, 'kernel': 5, 'padding': 2},
            {'channels': 32, 'kernel': 3, 'padding': 1},
        ],
        'use_spatial_attention': True,
        'attention_reduction': 4,
        'output_dim': 64,
        'use_residual': True,
        'memory_efficient': True
    }
    
    # =============== Multi-Scale Training Configuration ===============
    MULTISCALE_TRAINING_CONFIG = {
        'batch_size': 64,  # Multi-scale·Î ¸Þ¸ð¸® »ç¿ë·® Áõ°¡
        'num_epochs': 50,
        'learning_rate': 5e-5,  # ´õ º¹ÀâÇÑ ¸ðµ¨ÀÌ¹Ç·Î ³·Àº LR
        'weight_decay': 1e-3,
        'gradient_clip_norm': 0.5,
        
        # Multi-scale specific
        'scale_sampling_strategy': 'balanced',  # 'balanced' | 'weighted' | 'curriculum'
        'scale_weights': {
            '4s': 1.0,
            '8s': 1.0,
            '16s': 1.0
        },
        'curriculum_learning': {
            'start_with_single_scale': True,
            'single_scale_epochs': 10,
            'gradual_scale_introduction': True
        }
    }
    
    # =============== Multi-Scale Loss Configuration ===============
    MULTISCALE_LOSS_CONFIG = {
        # Scaleº° µ¶¸³ loss weights
        'scale_loss_weights': {
            '4s': 0.3,
            '8s': 0.4, 
            '16s': 0.3
        },
        
        # Cross-scale consistency loss
        'cross_scale_consistency': {
            'weight': 0.1,
            'consistency_type': 'feature_alignment',
            'temperature': 0.5
        },
        
        # ±âÁ¸ 4°³ ÇÙ½É loss À¯Áö
        'base_loss_weights': {
            'mse': 0.30,
            'magnitude': 0.25,
            'phase': 0.35,
            'coherence': 0.10
        }
    }
    
    # =============== ±âÁ¸ Loss Configuration À¯Áö ===============
    LOSS_CONFIG = {
        'loss_weights': {
            'mse': 0.30,
            'magnitude': 0.25,
            'phase': 0.35,
            'coherence': 0.10
        },
        'magnitude_loss_config': {
            'loss_type': 'l2',
            'relative_weight': 0.6,
            'frequency_weights': {
                'alpha': 1.8,
                'beta1': 1.6,
                'beta2': 1.4,
                'gamma': 1.2,
                'theta': 1.3,
                'delta': 1.0
            }
        },
        'phase_loss_config': {
            'loss_type': 'cosine',  # ¾ÈÁ¤¼ºÀ» À§ÇØ cosine »ç¿ë
            'wrap_around': True,
            'frequency_emphasis': 'alpha',
            'temperature': 1.0
        },
        'coherence_loss_config': {
            'coherence_type': 'magnitude_consistency',
            'spatial_coherence_weight': 0.3,
            'temporal_coherence_weight': 0.7
        }
    }
    
    # =============== Training Configuration À¯Áö ===============
    TRAINING_CONFIG = {
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-3,
        'gradient_clip_norm': 0.5,
        
        'optimizer': 'adamw',
        'optimizer_params': {
            'betas': (0.9, 0.999),
            'eps': 1e-8
        },
        
        'scheduler': 'cosine_with_warmup',
        'scheduler_params': {
            'warmup_epochs': 3,
            'min_lr_ratio': 0.01
        },
        
        'early_stopping_patience': 10,
        'monitor_metric': 'total_loss'
    }
    
    # =============== ±âÁ¸ ¼³Á¤µé À¯Áö ===============
    FREQUENCY_BANDS = {
        'delta': [0, 1, 2, 3],
        'theta': [4, 5, 6, 7],
        'alpha': [8, 9],
        'beta1': [10, 11, 12, 13],
        'beta2': [14, 15],
        'gamma': [16, 17, 18, 19]
    }
    
    ELECTRODE_NAMES = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
    ]
    
    BRAIN_REGIONS = {
        'frontal': [0, 1, 2, 3, 10, 11, 16],
        'central': [4, 5, 17],
        'parietal': [6, 7, 18],
        'temporal': [12, 13, 14, 15],
        'occipital': [8, 9]
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DATA_CONFIG = {
        'train_data_path': './data/train/',
        'val_data_path': './data/val/',
        'test_data_path': './data/test/',
        'checkpoint_path': './checkpoints/',
        'log_path': './logs/',
        'data_key': 'csd'
    }
    
    # =============== Reconstruction Config Ãß°¡ ===============
    RECONSTRUCTION_CONFIG = {
        'input_dim': UNIFIED_FEATURE_DIM,
        'num_frequency_heads': NUM_FREQUENCIES,
        'frequency_head_config': {
            'input_dim': UNIFIED_FEATURE_DIM,
            'hidden_dims': [32, 16],
            'output_dim': NUM_COMPLEX_DIMS,
            'activation': 'gelu',
            'dropout': 0.1,
            'use_batch_norm': False
        },
        'reconstruction_strategy': 'independent_heads',
        'output_activation': None,
        'frequency_specific_weights': True
    }
    
    # =============== Classification Config Ãß°¡ ===============
    CLASSIFICATION_CONFIG = {
        'input_dim': UNIFIED_FEATURE_DIM,
        'hidden_dims': [32, 16],
        'num_classes': NUM_CLASSES,
        'dropout': 0.3,
        
        'pooling_strategy': 'attention',
        'attention_pooling_dim': 16,
        
        'use_brain_region_pooling': False,
        'region_pooling_weights': {
            'frontal': 1.2,
            'central': 1.0, 
            'parietal': 1.1,
            'temporal': 1.3,
            'occipital': 0.9
        }
    }
    
    # =============== Memory Config Ãß°¡ ===============
    MEMORY_CONFIG = {
        'gradient_checkpointing': True,
        'mixed_precision': True,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': True,
        
        # Multi-Scale ÃÖÀûÈ­
        'multiscale_optimization': True,
        'scale_wise_checkpointing': True,
        'cross_scale_memory_sharing': True
    }
    
    @classmethod
    def get_multiscale_model_complexity(cls):
        """Multi-scale ¸ðµ¨ º¹Àâµµ °è»ê"""
        
        # Scaleº° processor parameters
        scale_params = 0
        for scale_name, scale_config in cls.MULTISCALE_FEATURE_CONFIG['scale_processors'].items():
            scale_processor_params = (
                cls.NUM_FREQUENCIES * scale_config['hidden_dims'][0] +
                scale_config['hidden_dims'][0] * scale_config['hidden_dims'][1] +
                scale_config['hidden_dims'][1] * scale_config['output_dim']
            )
            scale_params += scale_processor_params
        
        # Cross-scale attention parameters  
        attention_config = cls.MULTISCALE_FEATURE_CONFIG['cross_scale_attention']
        attention_params = (
            attention_config['attention_dim'] * attention_config['attention_dim'] * 4 *
            attention_config['num_heads']
        )
        
        # Fusion parameters
        fusion_config = cls.MULTISCALE_FEATURE_CONFIG['fusion_config']
        fusion_params = (
            fusion_config['input_dim'] * fusion_config['hidden_dims'][0] +
            fusion_config['hidden_dims'][0] * fusion_config['hidden_dims'][1] +
            fusion_config['hidden_dims'][1] * fusion_config['output_dim']
        )
        
        # ±âÁ¸ single-scale parameters
        single_scale_params = cls._calculate_single_scale_params()
        
        total_params = scale_params + attention_params + fusion_params + single_scale_params
        
        return {
            'scale_processors': scale_params,
            'cross_scale_attention': attention_params,
            'multiscale_fusion': fusion_params,
            'single_scale_backbone': single_scale_params,
            'total': total_params,
            'architecture_type': 'multi_scale_hierarchical'
        }
    
    @classmethod
    def _calculate_single_scale_params(cls):
        """±âÁ¸ single-scale backbone parameters"""
        freq_proc = cls.NUM_FREQUENCIES * cls.FREQUENCY_FEATURE_DIM * 2
        complex_proc = (cls.NUM_COMPLEX_DIMS * cls.COMPLEX_FEATURE_DIM * 2) * cls.NUM_FREQUENCIES
        fusion = cls.UNIFIED_FEATURE_DIM * cls.UNIFIED_FEATURE_DIM
        return freq_proc + complex_proc + fusion
    
    @classmethod
    def validate_multiscale_configuration(cls):
        """Multi-scale ¼³Á¤ °ËÁõ"""
        try:
            # Scale ¼³Á¤ °ËÁõ
            total_segments = sum(config['num_segments'] for config in cls.SCALE_CONFIGS.values())
            assert total_segments == 7, f"Total segments should be 7 (4+2+1), got {total_segments}"
            
            # Feature dimension ÀÏ°ü¼º °ËÁõ
            for scale_name, scale_config in cls.MULTISCALE_FEATURE_CONFIG['scale_processors'].items():
                assert scale_config['output_dim'] == 64, f"All scales should output 64 dims, {scale_name} outputs {scale_config['output_dim']}"
            
            # Loss weight ÇÕ°è °ËÁõ
            scale_weights_sum = sum(cls.MULTISCALE_LOSS_CONFIG['scale_loss_weights'].values())
            assert abs(scale_weights_sum - 1.0) < 1e-6, f"Scale loss weights should sum to 1.0, got {scale_weights_sum}"
            
            base_weights_sum = sum(cls.MULTISCALE_LOSS_CONFIG['base_loss_weights'].values())
            assert abs(base_weights_sum - 1.0) < 1e-6, f"Base loss weights should sum to 1.0, got {base_weights_sum}"
            
            print("? Multi-Scale Configuration validation passed!")
            print(f"   ?? Scales: {list(cls.SCALE_CONFIGS.keys())}")
            print(f"   ?? Segments: {[config['num_segments'] for config in cls.SCALE_CONFIGS.values()]}")
            print(f"   ?? Feature dims: {[config['feature_dim'] for config in cls.SCALE_CONFIGS.values()]}")
            print(f"   ?? Expected parameters: ~{cls.get_multiscale_model_complexity()['total']:,}")
            return True
            
        except Exception as e:
            print(f"? Configuration validation failed: {str(e)}")
            return False
    
    @classmethod
    def print_multiscale_architecture_summary(cls):
        """Multi-scale ¾ÆÅ°ÅØÃ³ ¿ä¾à Ãâ·Â"""
        complexity = cls.get_multiscale_model_complexity()
        
        print("="*80)
        print("?? MULTI-SCALE EEG CONNECTIVITY ANALYSIS - FIXED CONFIG")
        print("="*80)
        print(f"?? Multi-Scale Data Flow:")
        print(f"   4ÃÊ ¡¿ 4°³: High-freq optimized (Small receptive field)")
        print(f"   8ÃÊ ¡¿ 2°³: Mid-term optimized (Temporal CNN added)")  
        print(f"   16ÃÊ ¡¿ 1°³: Long-term optimized (Large receptive field)")
        print(f"   ¡æ Cross-Scale Attention ¡æ Multi-Scale Fusion ¡æ Output")
        print()
        print(f"?? Architecture Details:")
        print(f"   Scale Processors: {complexity['scale_processors']:,} parameters")
        print(f"   Cross-Scale Attention: {complexity['cross_scale_attention']:,} parameters")
        print(f"   Multi-Scale Fusion: {complexity['multiscale_fusion']:,} parameters")
        print(f"   Single-Scale Backbone: {complexity['single_scale_backbone']:,} parameters")
        print(f"   Total Parameters: {complexity['total']:,}")
        print()
        print(f"?? Scale-Specific Optimizations:")
        for scale_name, scale_config in cls.SCALE_CONFIGS.items():
            segments = scale_config['num_segments']
            length = scale_config['segment_length']
            optimization = scale_config['optimization']
            print(f"   {scale_name}: {segments}°³ ¡¿ {length}ÃÊ - {optimization}")
        print()
        print(f"?? Expected Benefits:")
        print(f"   ? 4ÃÊ: Fast dynamics capture")
        print(f"   ?? 8ÃÊ: Rhythm stability analysis") 
        print(f"   ?? 16ÃÊ: Network transition detection")
        print(f"   ?? Cross-Scale: Temporal hierarchy learning")
        print()
        print(f"?? Fixed Issues:")
        print(f"   ? PRETRAINING_CONFIG Ãß°¡µÊ")
        print(f"   ? ±âÁ¸ EEGConfig¿Í ¿ÏÀü È£È¯")
        print(f"   ? Import ¿¡·¯ ÇØ°á")
        print(f"   ? ¸ðµç ÇÊ¼ö ¼³Á¤ Æ÷ÇÔ")
        print("="*80)

if __name__ == "__main__":
    config = MultiScaleEEGConfig()
    config.validate_multiscale_configuration()
    config.print_multiscale_architecture_summary()