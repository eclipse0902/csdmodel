"""
Multi-Scale Feature Extraction - 4ÃÊ/8ÃÊ/16ÃÊ ÅëÇÕ Ã³¸®

±âÁ¸ StructuredFeatureExtractionÀ» È®ÀåÇÏ¿© Multi-Scale Áö¿ø
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
import os

# ±âÁ¸ imports (½ÇÁ¦ È¯°æ¿¡¼­´Â °æ·Î Á¶Á¤ ÇÊ¿ä)
from structured_feature_extraction import StructuredFeatureExtraction, GlobalAttentionModule
from config import MultiScaleEEGConfig

class ScaleSpecificProcessor(nn.Module):
    """
    Æ¯Á¤ ½ºÄÉÀÏ¿¡ ÃÖÀûÈ­µÈ ÇÁ·Î¼¼¼­
    4ÃÊ/8ÃÊ/16ÃÊ °¢°¢ÀÇ Æ¯¼º¿¡ ¸Â°Ô ¼³°è
    """
    
    def __init__(self, scale_name: str, config: MultiScaleEEGConfig):
        super().__init__()
        
        self.scale_name = scale_name
        self.scale_config = config.MULTISCALE_FEATURE_CONFIG['scale_processors'][scale_name]
        self.segment_config = config.SCALE_CONFIGS[scale_name]
        
        # Scaleº° ÃÖÀûÈ­ ¼³Á¤
        self.num_segments = self.segment_config['num_segments']
        self.segment_length = self.segment_config['segment_length']
        self.optimization_target = self.scale_config['optimization_target']
        
        # Scaleº° Temporal Convolution
        temporal_config = self.scale_config['temporal_conv']
        self.temporal_conv = nn.Conv1d(
            in_channels=config.NUM_FREQUENCIES,
            out_channels=config.NUM_FREQUENCIES,
            kernel_size=temporal_config['kernel_size'],
            stride=temporal_config['stride'],
            padding=temporal_config['padding'],
            groups=config.NUM_FREQUENCIES  # Depthwise convolution
        )
        
        # Scaleº° Feature MLP
        input_dim = self.scale_config['input_dim']
        hidden_dims = self.scale_config['hidden_dims']
        output_dim = self.scale_config['output_dim']
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.feature_mlp = nn.Sequential(*layers)
        
        # Scaleº° Á¤±ÔÈ­
        self.scale_norm = nn.LayerNorm(output_dim)
        
        print(f"?? {scale_name} Processor: {self.num_segments}°³ ¡¿ {self.segment_length}ÃÊ")
        print(f"   Optimization: {self.optimization_target}")
        print(f"   Temporal kernel: {temporal_config['kernel_size']}")
        print(f"   Feature path: {input_dim} ¡æ {hidden_dims} ¡æ {output_dim}")
    
    def forward(self, x: torch.Tensor, segment_idx: int = 0) -> torch.Tensor:
        """
        Scaleº° Æ¯È­ Ã³¸®
        
        Args:
            x: (batch, 361, 20, 2) - CSD data in pair format
            segment_idx: ÇöÀç Ã³¸®ÁßÀÎ ¼¼±×¸ÕÆ® ÀÎµ¦½º
            
        Returns:
            (batch, 361, 64) - scale-specific features
        """
        batch_size, num_pairs, num_freq, complex_dim = x.shape
        
        # 1. º¹¼Ò¼ö¸¦ magnitude·Î º¯È¯ (temporal processing¿ë)
        magnitude = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-8)  # (batch, 361, 20)
        
        # 2. Scaleº° Temporal Convolution
        # (batch, 361, 20) ¡æ (batch, 20, 361) ¡æ Conv1d ¡æ (batch, 20, 361) ¡æ (batch, 361, 20)
        magnitude_t = magnitude.transpose(1, 2)  # (batch, 20, 361)
        temporal_features = self.temporal_conv(magnitude_t)  # (batch, 20, 361)
        temporal_features = temporal_features.transpose(1, 2)  # (batch, 361, 20)
        
        # 3. Scaleº° ÃÖÀûÈ­ Àû¿ë
        if self.optimization_target == 'high_frequency_dynamics':
            # 4ÃÊ: °íÁÖÆÄ dynamics °­Á¶
            freq_weights = torch.tensor([0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2,  # µ¨Å¸-¾ËÆÄ
                                       1.3, 1.4, 1.4, 1.3,  # º£Å¸1
                                       1.2, 1.1,  # º£Å¸2  
                                       1.0, 0.9, 0.9, 0.8], device=x.device)  # °¨¸¶
            
        elif self.optimization_target == 'rhythm_stability':
            # 8ÃÊ: ¸®µë ¾ÈÁ¤¼º (¾ËÆÄ/º£Å¸ °­Á¶)
            freq_weights = torch.tensor([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.5,  # µ¨Å¸-¾ËÆÄ
                                       1.4, 1.3, 1.2, 1.1,  # º£Å¸1
                                       1.0, 0.9,  # º£Å¸2
                                       0.8, 0.7, 0.7, 0.6], device=x.device)  # °¨¸¶
            
        elif self.optimization_target == 'network_transitions':
            # 16ÃÊ: ³×Æ®¿öÅ© ÀüÀÌ (ÀúÁÖÆÄ °­Á¶)
            freq_weights = torch.tensor([1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.1,  # µ¨Å¸-¾ËÆÄ
                                       1.0, 0.9, 0.8, 0.7,  # º£Å¸1
                                       0.6, 0.5,  # º£Å¸2
                                       0.4, 0.4, 0.3, 0.3], device=x.device)  # °¨¸¶
        else:
            freq_weights = torch.ones(num_freq, device=x.device)
        
        # ÁÖÆÄ¼ö °¡ÁßÄ¡ Àû¿ë
        weighted_features = temporal_features * freq_weights.unsqueeze(0).unsqueeze(0)
        
        # 4. Feature MLP Ã³¸®
        scale_features = self.feature_mlp(weighted_features)  # (batch, 361, 64)
        
        # 5. Scale Á¤±ÔÈ­
        scale_features = self.scale_norm(scale_features)
        
        return scale_features

class CrossScaleAttention(nn.Module):
    """
    Cross-Scale Attention - ¼­·Î ´Ù¸¥ ½ºÄÉÀÏ °£ »óÈ£ÀÛ¿ë
    """
    
    def __init__(self, config: MultiScaleEEGConfig):
        super().__init__()
        
        attention_config = config.MULTISCALE_FEATURE_CONFIG['cross_scale_attention']
        
        self.num_heads = attention_config['num_heads']
        self.attention_dim = attention_config['attention_dim']
        self.dropout = nn.Dropout(attention_config['dropout'])
        
        # Multi-head attention for cross-scale interaction
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=attention_config['dropout'],
            batch_first=True
        )
        
        # Scale position encodings
        if attention_config['use_position_encoding']:
            self.register_buffer('scale_positions', self._create_scale_positions())
            self.position_embedding = nn.Linear(3, self.attention_dim)  # 3 scales
        
        # Output projection
        self.output_projection = nn.Linear(self.attention_dim, self.attention_dim)
        self.layer_norm = nn.LayerNorm(self.attention_dim)
        
        print(f"?? Cross-Scale Attention:")
        print(f"   Heads: {self.num_heads}")
        print(f"   Attention dim: {self.attention_dim}")
        print(f"   Position encoding: {attention_config['use_position_encoding']}")
    
    def _create_scale_positions(self):
        """½ºÄÉÀÏº° À§Ä¡ ÀÎÄÚµù »ý¼º"""
        # 4ÃÊ=0, 8ÃÊ=1, 16ÃÊ=2 ÀÇ one-hot encoding
        positions = torch.eye(3)  # (3, 3)
        return positions
    
    def forward(self, scale_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Cross-scale attention Àû¿ë
        
        Args:
            scale_features: {'4s': (batch, 361, 64), '8s': (batch, 361, 64), '16s': (batch, 361, 64)}
            
        Returns:
            Enhanced scale features with cross-scale interactions
        """
        batch_size = list(scale_features.values())[0].shape[0]
        num_pairs = list(scale_features.values())[0].shape[1]
        
        # 1. Stack all scale features
        stacked_features = []
        scale_names = ['4s', '8s', '16s']
        
        for scale_name in scale_names:
            features = scale_features[scale_name]  # (batch, 361, 64)
            stacked_features.append(features)
        
        # (3, batch, 361, 64) ¡æ (batch, 3*361, 64)
        stacked = torch.stack(stacked_features, dim=1)  # (batch, 3, 361, 64)
        stacked = stacked.reshape(batch_size, 3 * num_pairs, self.attention_dim)
        
        # 2. Add position encodings
        if hasattr(self, 'position_embedding'):
            # Create position encoding for each scale
            pos_encodings = []
            for i, scale_name in enumerate(scale_names):
                scale_pos = self.scale_positions[i].unsqueeze(0).repeat(num_pairs, 1)  # (361, 3)
                pos_emb = self.position_embedding(scale_pos)  # (361, 64)
                pos_encodings.append(pos_emb)
            
            pos_encodings = torch.stack(pos_encodings, dim=0)  # (3, 361, 64)
            pos_encodings = pos_encodings.reshape(3 * num_pairs, self.attention_dim)  # (3*361, 64)
            pos_encodings = pos_encodings.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, 3*361, 64)
            
            stacked = stacked + pos_encodings
        
        # 3. Self-attention across scales
        attended, attention_weights = self.multihead_attn(
            query=stacked,
            key=stacked, 
            value=stacked
        )
        
        # 4. Apply dropout and residual connection
        attended = self.dropout(attended)
        attended = attended + stacked  # Residual connection
        attended = self.layer_norm(attended)
        
        # 5. Project output
        attended = self.output_projection(attended)
        
        # 6. Reshape back to scale-specific features
        attended = attended.reshape(batch_size, 3, num_pairs, self.attention_dim)
        
        enhanced_features = {}
        for i, scale_name in enumerate(scale_names):
            enhanced_features[scale_name] = attended[:, i, :, :]  # (batch, 361, 64)
        
        return enhanced_features

class MultiScaleFusion(nn.Module):
    """
    Multi-Scale Fusion - ¿©·¯ ½ºÄÉÀÏÀÇ Æ¯¼ºÀ» ÅëÇÕ
    """
    
    def __init__(self, config: MultiScaleEEGConfig):
        super().__init__()
        
        fusion_config = config.MULTISCALE_FEATURE_CONFIG['fusion_config']
        
        self.fusion_strategy = fusion_config['fusion_strategy']
        self.scale_weights = nn.Parameter(torch.tensor(list(fusion_config['scale_weights'].values())))
        
        input_dim = fusion_config['input_dim']  # 64 * 3 = 192
        hidden_dims = fusion_config['hidden_dims']
        output_dim = fusion_config['output_dim']
        
        if self.fusion_strategy == 'hierarchical':
            # °èÃþÀû À¶ÇÕ: 4s+8s ¡æ intermediate, intermediate+16s ¡æ final
            self.short_term_fusion = nn.Sequential(
                nn.Linear(64 * 2, 64),  # 4s + 8s
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(64)
            )
            
            self.final_fusion = nn.Sequential(
                nn.Linear(64 * 2, output_dim),  # short_term + 16s
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(output_dim)
            )
            
        elif self.fusion_strategy == 'attention':
            # Attention ±â¹Ý À¶ÇÕ
            self.attention_weights = nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Softmax(dim=1)
            )
            
            self.fusion_mlp = nn.Sequential(
                nn.Linear(64, output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.LayerNorm(output_dim)
            )
            
        else:  # 'concat'
            # ´Ü¼ø concat ÈÄ MLP
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.LayerNorm(hidden_dim)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.fusion_mlp = nn.Sequential(*layers)
        
        print(f"?? Multi-Scale Fusion:")
        print(f"   Strategy: {self.fusion_strategy}")
        print(f"   Input dim: {input_dim} (3 scales ¡¿ 64)")
        print(f"   Output dim: {output_dim}")
        print(f"   Scale weights: learnable parameters")
    
    def forward(self, scale_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Multi-scale fusion
        
        Args:
            scale_features: {'4s': (batch, 361, 64), '8s': (batch, 361, 64), '16s': (batch, 361, 64)}
            
        Returns:
            (batch, 361, 64) - fused multi-scale features
        """
        batch_size = list(scale_features.values())[0].shape[0]
        num_pairs = list(scale_features.values())[0].shape[1]
        
        # Apply learnable scale weights
        weighted_features = {}
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        for i, (scale_name, features) in enumerate(scale_features.items()):
            weighted_features[scale_name] = features * scale_weights[i]
        
        if self.fusion_strategy == 'hierarchical':
            # °èÃþÀû À¶ÇÕ
            # 1. Short-term fusion (4s + 8s)
            short_term_concat = torch.cat([
                weighted_features['4s'], 
                weighted_features['8s']
            ], dim=-1)  # (batch, 361, 128)
            
            short_term_fused = self.short_term_fusion(short_term_concat)  # (batch, 361, 64)
            
            # 2. Final fusion (short_term + 16s)
            final_concat = torch.cat([
                short_term_fused,
                weighted_features['16s']
            ], dim=-1)  # (batch, 361, 128)
            
            fused = self.final_fusion(final_concat)  # (batch, 361, 64)
            
        elif self.fusion_strategy == 'attention':
            # Attention ±â¹Ý À¶ÇÕ
            stacked_features = torch.stack([
                weighted_features['4s'],
                weighted_features['8s'], 
                weighted_features['16s']
            ], dim=2)  # (batch, 361, 3, 64)
            
            # Compute attention weights for each scale
            attention_weights = self.attention_weights(stacked_features)  # (batch, 361, 3, 1)
            
            # Apply attention weights
            attended_features = (stacked_features * attention_weights).sum(dim=2)  # (batch, 361, 64)
            
            # Final MLP
            fused = self.fusion_mlp(attended_features)
            
        else:  # 'concat'
            # ´Ü¼ø concat ÈÄ MLP
            concatenated = torch.cat([
                weighted_features['4s'],
                weighted_features['8s'],
                weighted_features['16s']
            ], dim=-1)  # (batch, 361, 192)
            
            fused = self.fusion_mlp(concatenated)  # (batch, 361, 64)
        
        return fused

class MultiScaleStructuredFeatureExtraction(nn.Module):
    """
    Multi-Scale ÅëÇÕ Feature Extraction
    
    ±âÁ¸ StructuredFeatureExtractionÀ» È®ÀåÇÏ¿©:
    1. 4ÃÊ/8ÃÊ/16ÃÊ ½ºÄÉÀÏº° Ã³¸®
    2. Cross-Scale Attention
    3. Multi-Scale Fusion
    4. ±âÁ¸ Single-Scale ¹éº»°ú °áÇÕ
    """
    
    def __init__(self, config: MultiScaleEEGConfig = None):
        super().__init__()
        
        if config is None:
            config = MultiScaleEEGConfig()
        
        self.config = config
        
        # =============== MULTI-SCALE PROCESSORS ===============
        self.scale_processors = nn.ModuleDict({
            '4s': ScaleSpecificProcessor('4s', config),
            '8s': ScaleSpecificProcessor('8s', config), 
            '16s': ScaleSpecificProcessor('16s', config)
        })
        
        # =============== CROSS-SCALE ATTENTION ===============
        self.cross_scale_attention = CrossScaleAttention(config)
        
        # =============== MULTI-SCALE FUSION ===============
        self.multiscale_fusion = MultiScaleFusion(config)
        
        # =============== SINGLE-SCALE BACKBONE (±âÁ¸ À¯Áö) ===============
        self.single_scale_backbone = StructuredFeatureExtraction(config)
        
        # =============== FINAL COMBINATION ===============
        self.final_combination = nn.Sequential(
            nn.Linear(64 * 2, 64),  # multi-scale + single-scale
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(64)
        )
        
        # Model analysis
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"?? Multi-Scale Structured Feature Extraction:")
        print(f"   Scale processors: {len(self.scale_processors)}")
        print(f"   Cross-scale attention: Enabled")
        print(f"   Multi-scale fusion: {config.MULTISCALE_FEATURE_CONFIG['fusion_config']['fusion_strategy']}")
        print(f"   Single-scale backbone: Preserved")
        print(f"   Total parameters: {total_params:,}")
    
    def forward(self, x: torch.Tensor, segment_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Multi-scale forward pass
        
        Args:
            x: (batch, 361, 20, 2) - CSD data in pair format
            segment_info: Optional info about current segments
            
        Returns:
            (batch, 361, 64) - unified multi-scale features
        """
        batch_size, num_pairs, num_freq, complex_dim = x.shape
        
        # =============== MULTI-SCALE PROCESSING ===============
        scale_features = {}
        
        # Process each scale independently
        for scale_name, processor in self.scale_processors.items():
            scale_features[scale_name] = processor(x)  # (batch, 361, 64)
        
        # =============== CROSS-SCALE ATTENTION ===============
        enhanced_scale_features = self.cross_scale_attention(scale_features)
        
        # =============== MULTI-SCALE FUSION ===============
        multiscale_features = self.multiscale_fusion(enhanced_scale_features)  # (batch, 361, 64)
        
        # =============== SINGLE-SCALE BACKBONE ===============
        single_scale_features = self.single_scale_backbone(x)  # (batch, 361, 64)
        
        # =============== FINAL COMBINATION ===============
        combined_features = torch.cat([
            multiscale_features,
            single_scale_features
        ], dim=-1)  # (batch, 361, 128)
        
        final_features = self.final_combination(combined_features)  # (batch, 361, 64)
        
        return final_features
    
    def get_scale_analysis(self, x: torch.Tensor) -> Dict:
        """½ºÄÉÀÏº° Æ¯¼º ºÐ¼®"""
        self.eval()
        
        with torch.no_grad():
            # Individual scale processing
            scale_features = {}
            scale_stats = {}
            
            for scale_name, processor in self.scale_processors.items():
                features = processor(x)
                scale_features[scale_name] = features
                
                scale_stats[scale_name] = {
                    'mean': features.mean().item(),
                    'std': features.std().item(),
                    'max': features.max().item(),
                    'min': features.min().item(),
                    'norm': torch.norm(features, dim=-1).mean().item()
                }
            
            # Cross-scale attention effects
            enhanced_features = self.cross_scale_attention(scale_features)
            
            attention_effects = {}
            for scale_name in scale_features.keys():
                original = scale_features[scale_name]
                enhanced = enhanced_features[scale_name]
                
                change = torch.norm(enhanced - original, dim=-1).mean().item()
                attention_effects[scale_name] = change
            
            # Multi-scale fusion analysis
            fused = self.multiscale_fusion(enhanced_features)
            
            analysis = {
                'scale_statistics': scale_stats,
                'attention_effects': attention_effects,
                'fusion_output': {
                    'mean': fused.mean().item(),
                    'std': fused.std().item(),
                    'norm': torch.norm(fused, dim=-1).mean().item()
                },
                'scale_contributions': {
                    scale_name: F.softmax(self.multiscale_fusion.scale_weights, dim=0)[i].item()
                    for i, scale_name in enumerate(['4s', '8s', '16s'])
                },
                'architecture_info': {
                    'num_scales': len(self.scale_processors),
                    'fusion_strategy': self.config.MULTISCALE_FEATURE_CONFIG['fusion_config']['fusion_strategy'],
                    'cross_attention_heads': self.cross_scale_attention.num_heads,
                    'total_parameters': sum(p.numel() for p in self.parameters())
                }
            }
        
        return analysis

# =============== UTILITY FUNCTIONS ===============

def create_multiscale_data_splits(data: torch.Tensor, config: MultiScaleEEGConfig) -> Dict[str, torch.Tensor]:
    """
    ÀÔ·Â µ¥ÀÌÅÍ¸¦ Multi-Scale ¼¼±×¸ÕÆ®·Î ºÐÇÒ
    
    Args:
        data: (batch, time_steps, 19, 19, 2) - Raw EEG data
        config: Multi-scale configuration
        
    Returns:
        Dictionary of scale-specific data splits
    """
    batch_size, total_time, height, width, complex_dim = data.shape
    
    scale_data = {}
    
    for scale_name, scale_config in config.SCALE_CONFIGS.items():
        segment_length = scale_config['segment_length']
        num_segments = scale_config['num_segments']
        
        # Calculate time steps per segment
        steps_per_segment = total_time // num_segments
        
        segments = []
        for i in range(num_segments):
            start_idx = i * steps_per_segment
            end_idx = start_idx + steps_per_segment
            
            segment = data[:, start_idx:end_idx, :, :, :]
            # Average over time for this segment
            segment_avg = segment.mean(dim=1)  # (batch, 19, 19, 2)
            segments.append(segment_avg)
        
        # Stack segments
        scale_data[scale_name] = torch.stack(segments, dim=1)  # (batch, num_segments, 19, 19, 2)
    
    return scale_data

def convert_to_multiscale_pairs(scale_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Multi-scale data¸¦ pair formatÀ¸·Î º¯È¯
    
    Args:
        scale_data: {'4s': (batch, 4, 19, 19, 2), '8s': (batch, 2, 19, 19, 2), '16s': (batch, 1, 19, 19, 2)}
        
    Returns:
        {'4s': (batch, 361, 20, 2), '8s': (batch, 361, 20, 2), '16s': (batch, 361, 20, 2)}
    """
    scale_pairs = {}
    
    for scale_name, data in scale_data.items():
        batch_size, num_segments, height, width, complex_dim = data.shape
        
        # Reshape to pair format for each segment and average
        segments_pairs = []
        for seg_idx in range(num_segments):
            segment = data[:, seg_idx, :, :, :]  # (batch, 19, 19, 2)
            # Assume 20 frequencies (±âÁ¸ ¸ðµ¨°ú È£È¯¼º)
            segment_expanded = segment.unsqueeze(1).repeat(1, 20, 1, 1, 1)  # (batch, 20, 19, 19, 2)
            
            # Convert to pair format
            segment_pairs = segment_expanded.permute(0, 2, 3, 1, 4).reshape(batch_size, 361, 20, 2)
            segments_pairs.append(segment_pairs)
        
        # Average across segments for this scale
        scale_pairs[scale_name] = torch.stack(segments_pairs, dim=0).mean(dim=0)  # (batch, 361, 20, 2)
    
    return scale_pairs

if __name__ == "__main__":
    print("="*80)
    print("?? MULTI-SCALE FEATURE EXTRACTION")
    print("="*80)
    
    # Test configuration
    config = MultiScaleEEGConfig()
    config.validate_multiscale_configuration()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test multi-scale feature extraction
    print("\n1. ? Multi-Scale Feature Extraction Test:")
    multiscale_extractor = MultiScaleStructuredFeatureExtraction(config).to(device)
    sample_input = torch.randn(2, 361, 20, 2).to(device)
    
    multiscale_features = multiscale_extractor(sample_input)
    print(f"   Input: {sample_input.shape}")
    print(f"   Output: {multiscale_features.shape}")
    print(f"   Parameters: {sum(p.numel() for p in multiscale_extractor.parameters()):,}")
    
    # Test individual scale processors
    print("\n2. ? Individual Scale Processors Test:")
    for scale_name, processor in multiscale_extractor.scale_processors.items():
        scale_output = processor(sample_input)
        print(f"   {scale_name}: {sample_input.shape} ¡æ {scale_output.shape}")
    
    # Test cross-scale attention
    print("\n3. ? Cross-Scale Attention Test:")
    scale_features = {}
    for scale_name, processor in multiscale_extractor.scale_processors.items():
        scale_features[scale_name] = processor(sample_input)
    
    enhanced_features = multiscale_extractor.cross_scale_attention(scale_features)
    print(f"   Original scales: {list(scale_features.keys())}")
    print(f"   Enhanced scales: {list(enhanced_features.keys())}")
    for scale_name in scale_features.keys():
        change = torch.norm(enhanced_features[scale_name] - scale_features[scale_name]).item()
        print(f"   {scale_name} attention effect: {change:.3f}")
    
    # Test multi-scale fusion
    print("\n4. ? Multi-Scale Fusion Test:")
    fused = multiscale_extractor.multiscale_fusion(enhanced_features)
    print(f"   Fusion input: 3 scales ¡¿ {enhanced_features['4s'].shape}")
    print(f"   Fusion output: {fused.shape}")
    print(f"   Fusion strategy: {config.MULTISCALE_FEATURE_CONFIG['fusion_config']['fusion_strategy']}")
    
    # Test scale analysis
    print("\n5. ? Scale Analysis:")
    analysis = multiscale_extractor.get_scale_analysis(sample_input)
    print(f"   Scale contributions:")
    for scale_name, contribution in analysis['scale_contributions'].items():
        print(f"     {scale_name}: {contribution:.3f}")
    print(f"   Cross-attention heads: {analysis['architecture_info']['cross_attention_heads']}")
    print(f"   Total parameters: {analysis['architecture_info']['total_parameters']:,}")
    
    print("="*80)
    print("? MULTI-SCALE FEATURE EXTRACTION READY!")
    print("="*80)
    
    print("?? Key Features Implemented:")
    print("   ? 4ÃÊ/8ÃÊ/16ÃÊ Scale-Specific Processors")
    print("   ? Cross-Scale Attention Mechanism")
    print("   ? Multi-Scale Fusion (Hierarchical/Attention/Concat)")
    print("   ? Single-Scale Backbone Integration")
    print("   ? Scale-Specific Optimizations")
    
    print("\n?? Architecture Benefits:")
    print("   ? 4ÃÊ: High-frequency dynamics capture")
    print("   ?? 8ÃÊ: Rhythm stability analysis")
    print("   ?? 16ÃÊ: Network transition detection")
    print("   ?? Cross-Scale: Temporal hierarchy learning")
    print("   ?? Full compatibility with existing single-scale models")
    print("="*80)