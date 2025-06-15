"""
EEG Connectivity Analysis - Complete Enhanced Structured Feature Extraction Module

¸ðµç ´ëÈ­ ³»¿ë ¿ÏÀü ¹Ý¿µ:
1. CNN ±â¹Ý Global Processing (Gradient Infinity ÇØ°á)
2. 20°³ µ¶¸³ Complex Processor (1Hz ¡Á 50Hz)
3. Attention Aggregation ±¸Çö
4. Separate Complex Processing ¿ÏÀü ±¸Çö
5. ±âÁ¸ È£È¯¼º 100% À¯Áö
6. Config ±â¹Ý ¸ðµç ±â´É Á¦¾î
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EEGConfig

class StructuredFeatureExtraction(nn.Module):
    """
    Complete Enhanced Structured Feature Extraction Module
    
    ÇÙ½É Æ¯Â¡:
    1. ÁÖÆÄ¼ö¿Í º¹¼Ò¼ö ±¸Á¶¸¦ ¸ðµÎ º¸Á¸ÇÏ¸ç Ã³¸®
    2. Config ±â¹Ý ¿ÏÀü Á¦¾î (attention/separate Áö¿ø)
    3. 20°³ µ¶¸³ Complex Processor
    4. Attention Aggregation ±¸Çö
    5. ±âÁ¸ È£È¯¼º 100% À¯Áö
    """
    
    def __init__(self, config: EEGConfig = None):
        super(StructuredFeatureExtraction, self).__init__()
        
        if config is None:
            config = EEGConfig()
            
        self.config = config
        self.feature_config = config.FEATURE_EXTRACTION_CONFIG
        
        # =============== PROCESSORS ===============
        self.frequency_processor = DynamicFrequencyProcessor(config)
        self.complex_processor = DynamicComplexProcessor(config)
        
        # =============== AGGREGATION STRATEGIES ===============
        self.freq_aggregation = self.feature_config['frequency_aggregation']
        self.complex_combination = self.feature_config['complex_combination']
        
        # =============== ATTENTION AGGREGATION SETUP ===============
        if self.freq_aggregation == 'attention':
            self.freq_attention = nn.Sequential(
                nn.Linear(config.COMPLEX_FEATURE_DIM, config.COMPLEX_FEATURE_DIM // 4),
                nn.GELU(),
                nn.Linear(config.COMPLEX_FEATURE_DIM // 4, 1),
                nn.Softmax(dim=2)  # dim=2 is frequency dimension
            )
            print(f"? Attention Aggregation enabled")
        
        # =============== FEATURE FUSION ===============
        fusion_input_dim = config.FREQUENCY_FEATURE_DIM + config.COMPLEX_FEATURE_DIM
        fusion_config = self.feature_config['fusion_config']
        
        fusion_layers = []
        prev_dim = fusion_input_dim
        
        for hidden_dim in fusion_config['hidden_dims']:
            fusion_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(fusion_config['dropout'])
            ])
            prev_dim = hidden_dim
        
        fusion_layers.append(nn.Linear(prev_dim, fusion_config['output_dim']))
        
        if fusion_config.get('use_residual', False) and fusion_input_dim == fusion_config['output_dim']:
            self.fusion_layer = ResidualFusion(nn.Sequential(*fusion_layers))
        else:
            self.fusion_layer = nn.Sequential(*fusion_layers)
        
        print(f"?? Enhanced Structured Feature Extraction:")
        print(f"   Input: (pairs, {config.NUM_FREQUENCIES}, {config.NUM_COMPLEX_DIMS})")
        print(f"   Output: (pairs, {config.UNIFIED_FEATURE_DIM})")
        print(f"   Frequency aggregation: {self.freq_aggregation}")
        print(f"   Complex combination: {self.complex_combination}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with full config support
        
        Args:
            x: (batch, 361, 20, 2) - CSD data in pair format
        Returns:
            (batch, 361, unified_feature_dim) - unified features
        """
        batch_size, num_pairs, num_freq, complex_dim = x.shape
        
        # =============== INPUT VALIDATION ===============
        assert num_pairs == self.config.NUM_PAIRS, f"Expected {self.config.NUM_PAIRS} pairs, got {num_pairs}"
        assert num_freq == self.config.NUM_FREQUENCIES, f"Expected {self.config.NUM_FREQUENCIES} freq, got {num_freq}"
        assert complex_dim == self.config.NUM_COMPLEX_DIMS, f"Expected {self.config.NUM_COMPLEX_DIMS} complex, got {complex_dim}"
        
        # =============== COMPLEX PROCESSING (20°³ µ¶¸³) ===============
        complex_features = self.complex_processor(x)  # (batch, 361, 20, 32)
        
        # =============== FREQUENCY AGGREGATION ===============
        if self.freq_aggregation == 'mean':
            aggregated_complex = complex_features.mean(dim=2)  # (batch, 361, 32)
            
        elif self.freq_aggregation == 'max':
            aggregated_complex = complex_features.max(dim=2)[0]  # (batch, 361, 32)
            
        elif self.freq_aggregation == 'attention':
            # ? Attention-based aggregation
            attn_weights = self.freq_attention(complex_features)  # (batch, 361, 20, 1)
            aggregated_complex = (complex_features * attn_weights).sum(dim=2)  # (batch, 361, 32)
            
        else:
            # Fallback to mean
            aggregated_complex = complex_features.mean(dim=2)
        
        # =============== FREQUENCY PROCESSING ===============
        if self.complex_combination == 'mean':
            freq_input = x.mean(dim=-1)  # (batch, 361, 20)
        elif self.complex_combination == 'magnitude':
            freq_input = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-8)  # (batch, 361, 20)
        elif self.complex_combination == 'separate':
            # Separate Ã³¸®ÀÇ °æ¿ì magnitude »ç¿ë (frequency processor¿ë)
            freq_input = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-8)  # (batch, 361, 20)
        else:
            freq_input = x.mean(dim=-1)
        
        freq_features = self.frequency_processor(freq_input)  # (batch, 361, 32)
        
        # =============== FEATURE FUSION ===============
        combined_features = torch.cat([aggregated_complex, freq_features], dim=-1)  # (batch, 361, 64)
        final_features = self.fusion_layer(combined_features)  # (batch, 361, 64)
        
        return final_features
    
    def get_feature_statistics(self, x: torch.Tensor) -> Dict:
        """ÃßÃâµÈ feature Åë°è ºÐ¼®"""
        self.eval()
        
        with torch.no_grad():
            # Shape º¯È¯: (batch, 20, 19, 19, 2) ¡æ (batch, 361, 20, 2)
            original_shape = x.shape
            
            if len(x.shape) == 5 and x.shape[1:] == (20, 19, 19, 2):
                # Matrix format ¡æ Pair format
                batch_size = x.shape[0]
                x_pairs = x.permute(0, 2, 3, 1, 4).reshape(batch_size, 361, 20, 2)
                converted_input = True
            elif len(x.shape) == 4 and x.shape[1:] == (361, 20, 2):
                # Already in pair format
                x_pairs = x
                converted_input = False
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")
            
            # Feature extraction
            features = self.forward(x_pairs)  # (batch, 361, unified_feature_dim)
            
            stats = {
                'input_info': {
                    'original_shape': list(original_shape),
                    'converted_shape': list(x_pairs.shape),
                    'shape_converted': converted_input
                },
                'output_statistics': {
                    'mean': features.mean().item(),
                    'std': features.std().item(),
                    'min': features.min().item(),
                    'max': features.max().item(),
                    'feature_norm': torch.norm(features, dim=-1).mean().item()
                },
                'information_preservation': {
                    'input_variance': x_pairs.var().item(),
                    'output_variance': features.var().item(),
                    'variance_ratio': (features.var() / x_pairs.var()).item(),
                    'input_shape': list(x_pairs.shape),
                    'output_shape': list(features.shape),
                    'compression_ratio': (x_pairs.numel() / features.numel())
                },
                'aggregation_info': {
                    'frequency_aggregation': self.freq_aggregation,
                    'complex_combination': self.complex_combination,
                    'uses_attention': self.freq_aggregation == 'attention',
                    'uses_separate': self.complex_combination == 'separate'
                }
            }
            
        return stats

class DynamicFrequencyProcessor(nn.Module):
    """
    Enhanced Dynamic Frequency Processor
    """
    
    def __init__(self, config):
        super().__init__()
        
        freq_config = config.FEATURE_EXTRACTION_CONFIG['frequency_processor']
        
        self.input_dim = freq_config['input_dim']        # 20
        self.output_dim = freq_config['output_dim']      # 32
        
        # Dynamic MLP
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in freq_config['hidden_dims']:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(freq_config['dropout'])
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        self.processor = nn.Sequential(*layers)
        
        # ÇÐ½À °¡´ÉÇÑ ÁÖÆÄ¼ö Áß¿äµµ
        self.frequency_importance = nn.Parameter(torch.ones(self.input_dim))
        
        print(f"?? Dynamic FrequencyProcessor: {self.input_dim} ¡æ {self.output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, pairs, 20) - ÁÖÆÄ¼ö Â÷¿ø µ¥ÀÌÅÍ
        Returns:
            (batch, pairs, output_dim) - ÁÖÆÄ¼ö features
        """
        device = x.device
        
        # ÇÐ½ÀµÈ ÁÖÆÄ¼ö Áß¿äµµ Àû¿ë
        freq_weights = torch.softmax(self.frequency_importance, dim=0).to(device)
        x_weighted = x * freq_weights.unsqueeze(0).unsqueeze(0)
        
        # ÁÖÆÄ¼ö °ü°è ÇÐ½À
        freq_features = self.processor(x_weighted)
        
        return freq_features

class DynamicComplexProcessor(nn.Module):
    """
    Enhanced DynamicComplexProcessor - 20°³ µ¶¸³ + Separate ¿ÏÀü Áö¿ø
    """
    
    def __init__(self, config):
        super().__init__()
        
        complex_config = config.FEATURE_EXTRACTION_CONFIG['complex_processor']
        
        self.input_dim = complex_config['input_dim']      # 2 (real, imag)
        self.output_dim = complex_config['output_dim']    # 32
        self.frequency_independent = complex_config.get('frequency_independent', True)
        
        # ?? Combination mode ¼³Á¤
        self.combination_mode = config.FEATURE_EXTRACTION_CONFIG.get('complex_combination', 'mean')
        
        if self.frequency_independent:
            # 20°³ ÁÖÆÄ¼öº° µ¶¸³ Ã³¸®±â
            hidden_dims = complex_config.get('hidden_dims', [16, 32])
            
            if self.combination_mode == 'separate':
                # ? Real/Imag °¢°¢ µ¶¸³ processor
                self.real_processors = nn.ModuleList([
                    self._create_single_processor(hidden_dims, input_dim=1)  # Real Àü¿ë
                    for _ in range(20)
                ])
                self.imag_processors = nn.ModuleList([
                    self._create_single_processor(hidden_dims, input_dim=1)  # Imag Àü¿ë  
                    for _ in range(20)
                ])
                
                # Real-Imag À¶ÇÕ ·¹ÀÌ¾î
                self.fusion_layers = nn.ModuleList([
                    self._create_fusion_layer() for _ in range(20)
                ])
                
                print(f"? Separate Complex Processor:")
                print(f"   Real processors: 20°³ (°¢ 1 ¡æ {hidden_dims} ¡æ {self.output_dim})")
                print(f"   Imag processors: 20°³ (°¢ 1 ¡æ {hidden_dims} ¡æ {self.output_dim})")
                print(f"   Fusion layers: 20°³")
            else:
                # ±âÁ¸ ¹æ½Ä (È£È¯¼º)
                self.freq_processors = nn.ModuleList([
                    self._create_single_processor(hidden_dims)
                    for _ in range(20)
                ])
                print(f"? Standard Complex Processor: 20°³ µ¶¸³")
        else:
            # ±âÁ¸ °øÀ¯ ¹æ½Ä (È£È¯¼º)
            self.processor = self._create_single_processor([16, 32])
    
    def _create_single_processor(self, hidden_dims, input_dim=None):
        """´ÜÀÏ processor »ý¼º"""
        if input_dim is None:
            input_dim = self.input_dim  # 2 (real + imag)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim))
        return nn.Sequential(*layers)
    
    def _create_fusion_layer(self):
        """Real-Imag À¶ÇÕ ·¹ÀÌ¾î"""
        return nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),  # 32+32 ¡æ 32
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with complete separate support
        
        Args:
            x: (batch, 361, 20, 2) - CSD data in pair format
        Returns:
            (batch, 361, 20, 32) - processed complex features
        """
        
        if not self.frequency_independent:
            # ±âÁ¸ °øÀ¯ ¹æ½Ä (backward compatibility)
            return self.processor(x)
        
        if self.combination_mode == 'separate':
            return self._separate_forward(x)
        else:
            return self._standard_forward(x)
    
    def _separate_forward(self, x: torch.Tensor) -> torch.Tensor:
        """? Separate Real/Imag processing"""
        batch_size, num_pairs, num_freq, complex_dim = x.shape
        
        # Real°ú Imag ºÐ¸®
        real_part = x[..., 0:1]  # (batch, 361, 20, 1)
        imag_part = x[..., 1:2]  # (batch, 361, 20, 1)
        
        # 20°³ ÁÖÆÄ¼öº° µ¶¸³ Ã³¸®
        frequency_outputs = []
        for freq_idx in range(num_freq):
            # ÇØ´ç ÁÖÆÄ¼öÀÇ Real/Imag ÃßÃâ
            freq_real = real_part[:, :, freq_idx, :]  # (batch, 361, 1)
            freq_imag = imag_part[:, :, freq_idx, :]  # (batch, 361, 1)
            
            # °¢°¢ µ¶¸³ÀûÀ¸·Î Ã³¸®
            real_output = self.real_processors[freq_idx](freq_real)  # (batch, 361, 32)
            imag_output = self.imag_processors[freq_idx](freq_imag)  # (batch, 361, 32)
            
            # Real-Imag À¶ÇÕ
            combined_input = torch.cat([real_output, imag_output], dim=-1)  # (batch, 361, 64)
            fused_output = self.fusion_layers[freq_idx](combined_input)     # (batch, 361, 32)
            
            frequency_outputs.append(fused_output)
        
        # Stack all frequency outputs
        result = torch.stack(frequency_outputs, dim=2)  # (batch, 361, 20, 32)
        return result
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """±âÁ¸ ¹æ½Ä (È£È¯¼º À¯Áö)"""
        batch_size, num_pairs, num_freq, complex_dim = x.shape
        
        # 20°³ ÁÖÆÄ¼öº° µ¶¸³ Ã³¸®
        frequency_outputs = []
        for freq_idx in range(num_freq):
            freq_input = x[:, :, freq_idx, :]  # (batch, 361, 2)
            freq_output = self.freq_processors[freq_idx](freq_input)  # (batch, 361, 32)
            frequency_outputs.append(freq_output)
        
        # Stack all frequency outputs
        result = torch.stack(frequency_outputs, dim=2)  # (batch, 361, 20, 32)
        return result

class ResidualFusion(nn.Module):
    """Residual connection wrapper for fusion"""
    
    def __init__(self, fusion_module):
        super().__init__()
        self.fusion_module = fusion_module
    
    def forward(self, x):
        # x shape: (batch, pairs, input_dim)
        output = self.fusion_module(x)
        
        # Residual connection only if dimensions match
        if output.shape == x.shape:
            return output + x
        else:
            return output

class GlobalAttentionModule(nn.Module):
    """
    CNN-based Global Processing Module (Gradient Infinity ÇØ°á)
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            from config import EEGConfig
            config = EEGConfig()
            
        self.config = config
        self.use_cnn = config.USE_CNN_BACKBONE
        
        if self.use_cnn:
            self._setup_cnn_processing(config)
            print(f"? Using CNN-based Processing (Gradient Infinity ÇØ°á)")
        else:
            self._setup_attention_processing(config)
            print(f"??  Using Attention-based Processing (fallback)")
        
        # ÆÄ¶ó¹ÌÅÍ ºÐ¼®
        total_params = sum(p.numel() for p in self.parameters())
        print(f"?? Global Processing Parameters: {total_params:,}")
    
    def _setup_cnn_processing(self, config):
        """CNN ±â¹Ý °ø°£ Ã³¸® ¼³Á¤"""
        
        cnn_config = config.CNN_ATTENTION_CONFIG
        self.input_dim = cnn_config['input_dim']      # 64
        self.output_dim = cnn_config['output_dim']    # 64
        
        # CNN layers for spatial processing
        layers = []
        input_channels = 1  # °¢ feature dimensionÀ» °³º° Ã³¸®
        
        for layer_config in cnn_config['cnn_layers']:
            output_channels = layer_config['channels']
            kernel_size = layer_config['kernel']
            padding = layer_config['padding']
            
            layers.extend([
                nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
                nn.Dropout2d(0.1)
            ])
            
            input_channels = output_channels
        
        # Final projection back to feature space
        layers.extend([
            nn.Conv2d(input_channels, 1, kernel_size=1),  # Back to single channel
            nn.GELU()
        ])
        
        self.cnn_backbone = nn.Sequential(*layers)
        
        # Feature projection
        self.input_projection = nn.Linear(self.input_dim, self.output_dim) if self.input_dim != self.output_dim else nn.Identity()
        self.output_projection = nn.Linear(self.output_dim, self.output_dim)
        
        # Spatial attention (lightweight)
        if cnn_config.get('use_spatial_attention', True):
            self.spatial_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(1, 1, 1),
                nn.Sigmoid()
            )
        else:
            self.spatial_attention = None
        
        # Residual connection
        self.use_residual = cnn_config.get('use_residual', True)
    
    def _setup_attention_processing(self, config):
        """±âÁ¸ Attention ±â¹Ý Ã³¸® ¼³Á¤ (fallback)"""
        
        attention_config = config.GLOBAL_ATTENTION_CONFIG
        self.input_dim = attention_config['input_dim']
        self.attention_dim = attention_config['attention_dim']
        self.num_heads = attention_config['num_heads']
        self.num_layers = attention_config['num_layers']
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attention_dim,
            nhead=self.num_heads,
            dim_feedforward=attention_config['ffn_hidden_dim'],
            dropout=attention_config['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Projections
        self.input_projection = nn.Linear(self.input_dim, self.attention_dim) if self.input_dim != self.attention_dim else nn.Identity()
        self.output_projection = nn.Linear(self.attention_dim, self.input_dim)
        
        # Residual
        self.use_residual = attention_config.get('use_residual_connections', True)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with CNN or Attention
        
        Args:
            x: (batch, 361, input_dim) features from feature extraction
            mask: Optional attention mask (ignored in CNN mode)
        Returns:
            (batch, 361, output_dim) processed features
        """
        
        if self.use_cnn:
            return self._cnn_forward(x)
        else:
            return self._attention_forward(x, mask)
    
    def _cnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        """CNN ±â¹Ý forward pass"""
        
        batch_size, num_pairs, feature_dim = x.shape
        device = x.device
        
        # Input projection
        x_proj = self.input_projection(x)  # (batch, 361, output_dim)
        
        # Reshape to 19x19 spatial format for CNN
        x_spatial = x_proj.view(batch_size, 19, 19, self.output_dim)
        x_spatial = x_spatial.permute(0, 3, 1, 2)  # (batch, output_dim, 19, 19)
        
        # Process each feature dimension independently
        processed_features = []
        for feat_idx in range(self.output_dim):
            # Extract single feature map
            feat_map = x_spatial[:, feat_idx:feat_idx+1, :, :]  # (batch, 1, 19, 19)
            
            # CNN processing
            processed_feat = self.cnn_backbone(feat_map)  # (batch, 1, 19, 19)
            
            # Spatial attention
            if self.spatial_attention is not None:
                attention_weight = self.spatial_attention(processed_feat)
                processed_feat = processed_feat * attention_weight
            
            processed_features.append(processed_feat)
        
        # Combine processed features
        processed_spatial = torch.cat(processed_features, dim=1)  # (batch, output_dim, 19, 19)
        
        # Reshape back to pair format
        processed_spatial = processed_spatial.permute(0, 2, 3, 1)  # (batch, 19, 19, output_dim)
        processed_pairs = processed_spatial.view(batch_size, 361, self.output_dim)
        
        # Output projection
        output = self.output_projection(processed_pairs)
        
        # Residual connection
        if self.use_residual:
            output = output + x_proj
        
        return output
    
    def _attention_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """±âÁ¸ Attention ±â¹Ý forward pass (fallback)"""
        
        # Input projection
        x_proj = self.input_projection(x)
        
        # Global attention
        attended = self.transformer(x_proj, mask=mask)
        
        # Output projection
        output = self.output_projection(attended)
        
        # Residual connection
        if self.use_residual:
            output = output + x
        
        return output
    
    def get_attention_patterns(self, x: torch.Tensor) -> Dict:
        """Attention ÆÐÅÏ ºÐ¼® (CNN ¸ðµå¿¡¼­´Â spatial patterns)"""
        if self.use_cnn:
            return {
                'processing_type': 'cnn_spatial',
                'spatial_locality': 'optimized',
                'gradient_stability': 'stable',
                'attention_statistics': {
                    'entropy': 1.0,  # Placeholder
                    'sparsity': 0.1   # Placeholder
                }
            }
        else:
            return {
                'processing_type': 'global_attention',
                'gradient_stability': 'potentially_unstable',
                'attention_statistics': {
                    'entropy': 2.0,  # Placeholder
                    'sparsity': 0.3   # Placeholder
                }
            }

if __name__ == "__main__":
    print("="*80)
    print("?? COMPLETE ENHANCED STRUCTURED FEATURE EXTRACTION")
    print("="*80)
    
    # Test configuration
    from config import EEGConfig
    config = EEGConfig()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Feature extraction
    print("\n1. ? Feature Extraction Test:")
    feature_module = StructuredFeatureExtraction(config).to(device)
    sample_input = torch.randn(2, 361, 20, 2).to(device)
    
    features = feature_module(sample_input)
    print(f"   Input: {sample_input.shape}")
    print(f"   Output: {features.shape}")
    print(f"   Parameters: {sum(p.numel() for p in feature_module.parameters()):,}")
    
    # Test 2: Global attention/CNN
    print("\n2. ? Global Processing Test:")
    attention_module = GlobalAttentionModule(config).to(device)
    
    attended = attention_module(features)
    print(f"   Input: {features.shape}")
    print(f"   Output: {attended.shape}")
    print(f"   Parameters: {sum(p.numel() for p in attention_module.parameters()):,}")
    print(f"   Processing type: {'CNN' if config.USE_CNN_BACKBONE else 'Attention'}")
    
    # Test 3: Different aggregation modes
    print("\n3. ? Aggregation Modes Test:")
    test_configs = [
        ('mean', 'mean'),
        ('attention', 'separate'),
        ('mean', 'separate')
    ]
    
    for freq_agg, complex_comb in test_configs:
        config_test = EEGConfig()
        config_test.FEATURE_EXTRACTION_CONFIG['frequency_aggregation'] = freq_agg
        config_test.FEATURE_EXTRACTION_CONFIG['complex_combination'] = complex_comb
        
        try:
            test_module = StructuredFeatureExtraction(config_test)
            test_output = test_module(sample_input.cpu())
            print(f"   ? {freq_agg}/{complex_comb}: {test_output.shape}")
        except Exception as e:
            print(f"   ? {freq_agg}/{complex_comb}: {str(e)}")
    
    # Test 4: Statistics
    print("\n4. ? Feature Statistics:")
    stats = feature_module.get_feature_statistics(sample_input)
    print(f"   Variance ratio: {stats['information_preservation']['variance_ratio']:.3f}")
    print(f"   Compression: {stats['information_preservation']['compression_ratio']:.1f}x")
    print(f"   Uses attention: {stats['aggregation_info']['uses_attention']}")
    print(f"   Uses separate: {stats['aggregation_info']['uses_separate']}")
    
    print("="*80)
    print("? COMPLETE ENHANCED STRUCTURED FEATURE EXTRACTION READY!")
    print("="*80)
    
    print("?? Key Features Implemented:")
    print("   ? CNN-based Global Processing (Gradient Infinity ÇØ°á)")
    print("   ? 20°³ µ¶¸³ Complex Processor (1Hz ¡Á 50Hz)")
    print("   ? Attention Aggregation ¿ÏÀü ±¸Çö")
    print("   ? Separate Complex Processing ¿ÏÀü ±¸Çö")
    print("   ? ±âÁ¸ È£È¯¼º 100% À¯Áö (mean/mean µ¿ÀÛ)")
    print("   ? Config ±â¹Ý ¸ðµç ±â´É Á¦¾î")
    
    print("\n?? Usage:")
    print("   # ±âÁ¸ ¹æ½Ä (È£È¯¼º)")
    print("   config.FEATURE_EXTRACTION_CONFIG['frequency_aggregation'] = 'mean'")
    print("   config.FEATURE_EXTRACTION_CONFIG['complex_combination'] = 'mean'")
    print("")
    print("   # °³¼±µÈ ¹æ½Ä")
    print("   config.FEATURE_EXTRACTION_CONFIG['frequency_aggregation'] = 'attention'")
    print("   config.FEATURE_EXTRACTION_CONFIG['complex_combination'] = 'separate'")
    
    print("\n?? Problem Solutions:")
    print("   ?? CNN ¡æ Gradient ¾ÈÁ¤¼º È®º¸")
    print("   ?? 20°³ µ¶¸³ ¡æ ÁÖÆÄ¼öº° Æ¯¼º º¸Á¸")
    print("   ?? Attention ¡æ Áß¿ä ÁÖÆÄ¼ö ÁýÁß")
    print("   ?? Separate ¡æ Real/Imag À§»ó Á¤º¸ º¸Á¸")
    print("   ?? ¿Ïº®ÇÑ Backward Compatibility")
    print("="*80)