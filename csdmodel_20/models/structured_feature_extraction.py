"""
EEG Connectivity Analysis - Enhanced Structured Feature Extraction Module

ÇÙ½É °³¼±»çÇ×:
1. Config ±â¹Ý Dynamic Depth (2Ãþ ¡æ 3Ãþ+ °¡´É)
2. 4-5M ÆÄ¶ó¹ÌÅÍ Áö¿ø (32Â÷¿ø ¡æ 160Â÷¿ø)
3. ¸Þ¸ð¸® ÃÖÀûÈ­ (Gradient Checkpointing)
4. ±âÁ¸ ±¸Á¶ ¿ÏÀü È£È¯
5. ¸ðµç ´ëÈ­ ³»¿ë ¹Ý¿µ (ComplexProcessor °øÀ¯ Ã¶ÇÐ À¯Áö)
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
from utils.layers import (
    DynamicFrequencyProcessor, 
    DynamicComplexProcessor, 
    DynamicFusionLayer,
    checkpoint,
    get_memory_info,
    count_parameters
)

class StructuredFeatureExtraction(nn.Module):
    """
    Enhanced Structured Feature Extraction Module
    
    ÇÙ½É Æ¯Â¡:
    1. ÁÖÆÄ¼ö¿Í º¹¼Ò¼ö ±¸Á¶¸¦ ¸ðµÎ º¸Á¸ÇÏ¸ç Ã³¸®
    2. Config ±â¹Ý Dynamic Depth (4-5M ÆÄ¶ó¹ÌÅÍ Áö¿ø)
    3. Á¤º¸ ¼Õ½Ç ÃÖ¼ÒÈ­ + ¸Þ¸ð¸® ÃÖÀûÈ­
    4. ¹°¸®Àû ÀÇ¹Ì À¯Áö
    5. Device È£È¯¼º º¸Àå
    """
    
    def __init__(self, config: EEGConfig = None):
        super(StructuredFeatureExtraction, self).__init__()
        
        if config is None:
            config = EEGConfig()
            
        self.config = config
        self.feature_config = config.FEATURE_EXTRACTION_CONFIG
        
        # Memory optimization settings
        self.use_gradient_checkpointing = getattr(config, 'MEMORY_CONFIG', {}).get('gradient_checkpointing', False)
        
        # =============== ENHANCED PROCESSORS (Dynamic Depth) ===============
        self.frequency_processor = DynamicFrequencyProcessor(config)
        self.complex_processor = DynamicComplexProcessor(config)
        
        # =============== AGGREGATION STRATEGIES (±âÁ¸ À¯Áö) ===============
        self.freq_aggregation = self.feature_config['frequency_aggregation']
        self.complex_combination = self.feature_config['complex_combination']
        
        # =============== ENHANCED FEATURE FUSION ===============
        self.fusion_layer = DynamicFusionLayer(config)
        
        # =============== ¸Þ¸ð¸® ÃÖÀûÈ­ ===============
        if self.use_gradient_checkpointing:
            print("?? Gradient Checkpointing enabled for memory optimization")
        
        # ÆÄ¶ó¹ÌÅÍ ºÐ¼®
        param_analysis = count_parameters(self)
        
        print(f"?? Enhanced Structured Feature Extraction:")
        print(f"   Input structure: (pairs, {config.NUM_FREQUENCIES}, {config.NUM_COMPLEX_DIMS})")
        print(f"   Output: (pairs, {config.UNIFIED_FEATURE_DIM})")
        print(f"   Total parameters: {param_analysis['total_parameters']:,}")
        print(f"   Memory estimate: {param_analysis['memory_mb']:.1f} MB")
        print(f"   Frequency aggregation: {self.freq_aggregation}")
        print(f"   Complex combination: {self.complex_combination}")
        print(f"   Gradient checkpointing: {self.use_gradient_checkpointing}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhanced forward pass with memory optimization
        
        Args:
            x: (batch, 361, 15, 2) - CSD data in pair format
        Returns:
            (batch, 361, unified_feature_dim) - unified features
        """
        batch_size, num_pairs, num_freq, complex_dim = x.shape
        
        # =============== INPUT VALIDATION ===============
        assert num_pairs == self.config.NUM_PAIRS, f"Expected {self.config.NUM_PAIRS} pairs, got {num_pairs}"
        assert num_freq == self.config.NUM_FREQUENCIES, f"Expected {self.config.NUM_FREQUENCIES} freq, got {num_freq}"
        assert complex_dim == self.config.NUM_COMPLEX_DIMS, f"Expected {self.config.NUM_COMPLEX_DIMS} complex, got {complex_dim}"
        
        # Device È®ÀÎ
        device = x.device
        
        # =============== MEMORY OPTIMIZED PROCESSING ===============
        if self.use_gradient_checkpointing and self.training:
            # Gradient checkpointingÀ¸·Î ¸Þ¸ð¸® Àý¾à
            features = self._checkpointed_forward(x)
        else:
            # ÀÏ¹Ý forward
            features = self._regular_forward(x)
        
        return features
    
    def _regular_forward(self, x: torch.Tensor) -> torch.Tensor:
        """ÀÏ¹Ý forward pass"""
        
        # =============== COMPLEX PROCESSING (Enhanced) ===============
        # ¸ðµç ÁÖÆÄ¼ö¿¡¼­ °øÀ¯µÇ´Â º¹¼Ò¼ö Ã³¸® (±âÁ¸ Ã¶ÇÐ À¯Áö)
        complex_features = self.complex_processor(x)  # (batch, 361, 15, complex_feature_dim)
        
        # ÁÖÆÄ¼ö Â÷¿ø aggregation
        if self.freq_aggregation == 'mean':
            aggregated_complex = complex_features.mean(dim=2)  # (batch, 361, complex_feature_dim)
        elif self.freq_aggregation == 'max':
            aggregated_complex = complex_features.max(dim=2)[0]
        elif self.freq_aggregation == 'attention':
            # Simple attention aggregation
            attn_weights = torch.softmax(complex_features.mean(dim=-1), dim=-1)  # (batch, 361, 15)
            aggregated_complex = (complex_features * attn_weights.unsqueeze(-1)).sum(dim=2)
        else:
            aggregated_complex = complex_features.mean(dim=2)
        
        # =============== FREQUENCY PROCESSING (Enhanced) ===============
        # Real/Imag °áÇÕÇÏ¿© ÁÖÆÄ¼ö Â÷¿ø Ã³¸®
        if self.complex_combination == 'mean':
            freq_input = x.mean(dim=-1)  # (batch, 361, 15)
        elif self.complex_combination == 'magnitude':
            freq_input = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-8)
        else:
            freq_input = x.mean(dim=-1)
        
        freq_features = self.frequency_processor(freq_input)  # (batch, 361, freq_feature_dim)
        
        # =============== ENHANCED FEATURE FUSION ===============
        # º¹¼Ò¼ö features + ÁÖÆÄ¼ö features °áÇÕ
        combined_features = torch.cat([aggregated_complex, freq_features], dim=-1)  # (batch, 361, total_dim)
        
        # Dynamic fusion
        final_features = self.fusion_layer(combined_features)  # (batch, 361, unified_feature_dim)
        
        return final_features
    
    def _checkpointed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient checkpointingÀ» »ç¿ëÇÑ ¸Þ¸ð¸® È¿À²Àû forward"""
        
        # Complex processing checkpointed
        def complex_processing_func(x):
            return self.complex_processor(x)
        
        complex_features = checkpoint(complex_processing_func, x)
        
        # Aggregation
        if self.freq_aggregation == 'mean':
            aggregated_complex = complex_features.mean(dim=2)
        elif self.freq_aggregation == 'max':
            aggregated_complex = complex_features.max(dim=2)[0]
        else:
            aggregated_complex = complex_features.mean(dim=2)
        
        # Frequency processing checkpointed
        def frequency_processing_func(freq_input):
            return self.frequency_processor(freq_input)
        
        if self.complex_combination == 'mean':
            freq_input = x.mean(dim=-1)
        elif self.complex_combination == 'magnitude':
            freq_input = torch.sqrt(x[..., 0]**2 + x[..., 1]**2 + 1e-8)
        else:
            freq_input = x.mean(dim=-1)
        
        freq_features = checkpoint(frequency_processing_func, freq_input)
        
        # Fusion checkpointed
        def fusion_func(combined):
            return self.fusion_layer(combined)
        
        combined_features = torch.cat([aggregated_complex, freq_features], dim=-1)
        final_features = checkpoint(fusion_func, combined_features)
        
        return final_features
    
    def get_feature_statistics(self, x: torch.Tensor) -> Dict:
        """ÃßÃâµÈ feature Åë°è ºÐ¼® (Enhanced with memory monitoring)"""
        self.eval()
        
        # Memory monitoring
        initial_memory = get_memory_info()
        
        with torch.no_grad():
            # Shape º¯È¯: (batch, 15, 19, 19, 2) ¡æ (batch, 361, 15, 2)
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
                raise ValueError(f"Unexpected input shape: {x.shape}. Expected (batch, 15, 19, 19, 2) or (batch, 361, 15, 2)")
            
            # Feature extraction
            features = self.forward(x_pairs)  # (batch, 361, unified_feature_dim)
            
            # Memory monitoring
            peak_memory = get_memory_info()
            
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
                'memory_usage': {
                    'initial_allocated_gb': initial_memory['allocated'],
                    'peak_allocated_gb': peak_memory['allocated'],
                    'memory_increase_gb': peak_memory['allocated'] - initial_memory['allocated']
                },
                'frequency_analysis': self.frequency_processor.get_frequency_analysis(),
                'complex_analysis': self.complex_processor.get_complex_analysis(),
                'model_parameters': count_parameters(self)
            }
            
        return stats
    
    def get_learned_representations(self) -> Dict:
        """ÇÐ½ÀµÈ representation ºÐ¼® (Enhanced)"""
        analysis = {
            'frequency_importance': self.frequency_processor.get_frequency_analysis(),
            'complex_balance': self.complex_processor.get_complex_analysis(),
            'architecture_info': {
                'frequency_aggregation': self.freq_aggregation,
                'complex_combination': self.complex_combination,
                'use_gradient_checkpointing': self.use_gradient_checkpointing,
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'component_parameters': {
                    'frequency_processor': sum(p.numel() for p in self.frequency_processor.parameters()),
                    'complex_processor': sum(p.numel() for p in self.complex_processor.parameters()),
                    'fusion_layer': sum(p.numel() for p in self.fusion_layer.parameters())
                }
            },
            'memory_optimization': {
                'gradient_checkpointing': self.use_gradient_checkpointing,
                'estimated_memory_mb': count_parameters(self)['memory_mb']
            }
        }
        
        return analysis

class ChannelGroupedCrossAttention(nn.Module):
    """Ã¤³Î ±â¹Ý ±×·ì Cross-Attention"""
    
    def __init__(self, config: EEGConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.UNIFIED_FEATURE_DIM  # 160
        self.num_heads = config.GLOBAL_ATTENTION_CONFIG['num_heads']  # 8
        self.cross_config = config.CROSS_ATTENTION_CONFIG
        
        # 19°³ Ã¤³Îº° Cross-Attention
        self.channel_cross_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=self.num_heads,
                dropout=config.GLOBAL_ATTENTION_CONFIG['dropout'],
                batch_first=True
            ) for _ in range(19)  # 19°³ Ã¤³Î
        ])
        
        # Fusion layer
        fusion_type = self.cross_config.get('fusion_type', 'linear')
        if fusion_type == 'mlp':
            self.fusion = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.feature_dim * 2, self.feature_dim)
            )
        else:
            self.fusion = nn.Linear(self.feature_dim, self.feature_dim)
        
        # Ã¤³Î ¸ÅÇÎ ¹Ì¸® °è»ê (È¿À²¼º)
        self.pair_to_channels = self._create_pair_channel_mapping()
        self.channel_to_pairs = self._create_channel_pair_mapping()
        
        print(f"?? Channel-Grouped Cross-Attention:")
        print(f"   Strategy: {self.cross_config['group_strategy']}")
        print(f"   Fusion: {fusion_type}")
        print(f"   Channels: 19, Heads: {self.num_heads}")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 361, feature_dim)
        Returns:
            (batch, 361, feature_dim)
        """
        batch_size, num_pairs, feature_dim = x.shape
        device = x.device
        
        attended_features = []
        
        for pair_idx in range(361):
            i, j = self.pair_to_channels[pair_idx]  # ÀÌ ½ÖÀÇ Ã¤³Îµé
            
            # Query: ÇöÀç ½Ö
            query = x[:, pair_idx:pair_idx+1, :]  # (batch, 1, feature_dim)
            
            # Key/Value: °ü·Ã Ã¤³Î½Öµé
            if self.cross_config['group_strategy'] == 'shared_channel':
                # °°Àº Ã¤³Î Æ÷ÇÔÇÏ´Â ¸ðµç ½Öµé
                related_pairs_i = self.channel_to_pairs[i]
                related_pairs_j = self.channel_to_pairs[j] 
                related_pairs = list(set(related_pairs_i + related_pairs_j))
            else:
                # ±âº»: ¸ðµç ½Ö (fallback)
                related_pairs = list(range(361))
            
            key_value = x[:, related_pairs, :]  # (batch, related_count, feature_dim)
            
            # Cross-attention (i¹ø Ã¤³ÎÀÇ attention »ç¿ë)
            try:
                attended, _ = self.channel_cross_attentions[i](
                    query, key_value, key_value
                )
                attended_features.append(attended.squeeze(1))
            except Exception as e:
                # Fallback: ¿øº» feature »ç¿ë
                attended_features.append(x[:, pair_idx, :])
        
        # ¸ðµç attended features ÇÕÄ¡±â
        result = torch.stack(attended_features, dim=1)  # (batch, 361, feature_dim)
        
        # Fusion + Residual connection
        if self.cross_config.get('use_residual', True):
            result = self.fusion(result) + x
        else:
            result = self.fusion(result)
        
        return result
    
    def _create_pair_channel_mapping(self):
        """½Ö ÀÎµ¦½º ¡æ (Ã¤³Îi, Ã¤³Îj) ¸ÅÇÎ"""
        mapping = {}
        idx = 0
        for i in range(19):
            for j in range(19):
                mapping[idx] = (i, j)
                idx += 1
        return mapping
    
    def _create_channel_pair_mapping(self):
        """Ã¤³Î ¡æ Æ÷ÇÔÇÏ´Â ½Öµé ¸ÅÇÎ"""
        mapping = {i: [] for i in range(19)}
        idx = 0
        for i in range(19):
            for j in range(19):
                mapping[i].append(idx)  # i¹ø Ã¤³ÎÀÌ Æ÷ÇÔµÈ ½Ö
                if i != j:  # Áßº¹ ¹æÁö
                    mapping[j].append(idx)  # j¹ø Ã¤³ÎÀÌ Æ÷ÇÔµÈ ½Ö
                idx += 1
        return mapping


class GlobalAttentionModule(nn.Module):
    """
    Enhanced Global Attention Module with Cross-Attention Option
    """
    
    def __init__(self, config: EEGConfig = None):
        super(GlobalAttentionModule, self).__init__()
        
        if config is None:
            config = EEGConfig()
            
        self.config = config
        self.attention_config = config.GLOBAL_ATTENTION_CONFIG
        
        # Â÷¿ø ¼³Á¤
        self.input_dim = self.attention_config['input_dim']          # 160
        self.attention_dim = self.attention_config['attention_dim']  # 160
        self.num_heads = self.attention_config['num_heads']          # 8
        self.num_layers = self.attention_config['num_layers']        # 4
        
        # Cross-Attention ¼³Á¤
        self.use_cross_attention = getattr(config, 'USE_CROSS_ATTENTION', False)
        
        # Memory optimization
        self.use_gradient_checkpointing = getattr(config, 'MEMORY_CONFIG', {}).get('gradient_checkpointing', False)
        
        if self.use_cross_attention:
            # Cross-Attention ¸ðµâ
            self.cross_attention_module = ChannelGroupedCrossAttention(config)
            print(f"?? Using Cross-Attention Mode")
        else:
            # ±âÁ¸ Self-Attention ¸ðµâ
            self._setup_self_attention(config)
            print(f"? Using Self-Attention Mode")
        
        # ÆÄ¶ó¹ÌÅÍ ºÐ¼®
        param_analysis = count_parameters(self)
        print(f"?? Global Attention Parameters: {param_analysis['total_parameters']:,}")
    
    def _setup_self_attention(self, config):
        """±âÁ¸ Self-Attention ¼³Á¤"""
        
        # Position encoding
        if self.attention_config['use_position_encoding']:
            if self.attention_config['position_encoding_type'] == 'learned':
                self.position_encoding = nn.Parameter(
                    torch.randn(self.config.NUM_PAIRS, self.attention_dim) * 0.02
                )
            else:
                self.register_buffer('position_encoding', 
                                   self._create_sinusoidal_encoding())
        else:
            self.position_encoding = None
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.attention_dim,
            nhead=self.num_heads,
            dim_feedforward=self.attention_config['ffn_hidden_dim'],
            dropout=self.attention_config['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers,
            enable_nested_tensor=False
        )
        
        # Input/Output projections
        if self.input_dim != self.attention_dim:
            self.input_projection = nn.Linear(self.input_dim, self.attention_dim)
        else:
            self.input_projection = nn.Identity()
            
        self.output_projection = nn.Linear(self.attention_dim, self.input_dim)
        
        # Residual connections
        self.use_residual = self.attention_config['use_residual_connections']
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with Cross-Attention or Self-Attention
        
        Args:
            x: (batch, 361, input_dim) features from feature extraction
            mask: Optional attention mask
        Returns:
            (batch, 361, input_dim) attended features
        """
        batch_size, num_pairs, feature_dim = x.shape
        
        # Input validation
        assert num_pairs == self.config.NUM_PAIRS, f"Expected {self.config.NUM_PAIRS} pairs, got {num_pairs}"
        assert feature_dim == self.input_dim, f"Expected {self.input_dim} features, got {feature_dim}"
        
        if self.use_cross_attention:
            return self._cross_attention_forward(x)
        else:
            return self._self_attention_forward(x, mask)
    
    def _cross_attention_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-Attention forward pass"""
        return self.cross_attention_module(x)
    
    def _self_attention_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """±âÁ¸ Self-Attention forward pass"""
        device = x.device
        
        if self.use_gradient_checkpointing and self.training:
            return self._checkpointed_self_attention_forward(x, mask, device)
        else:
            return self._regular_self_attention_forward(x, mask, device)
    
    def _regular_self_attention_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        """ÀÏ¹Ý Self-Attention forward pass"""
        
        # Input projection
        x_proj = self.input_projection(x)
        
        # Add position encoding
        if self.position_encoding is not None:
            pos_enc = self.position_encoding.to(device)
            x_pos = x_proj + pos_enc.unsqueeze(0)
        else:
            x_pos = x_proj
        
        # Global attention
        attended = self.transformer(x_pos, mask=mask)
        
        # Output projection
        output = self.output_projection(attended)
        
        # Residual connection
        if self.use_residual:
            output = output + x
        
        return output
    
    def _checkpointed_self_attention_forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
        """Gradient checkpointing Self-Attention forward pass"""
        
        x_proj = self.input_projection(x)
        
        if self.position_encoding is not None:
            pos_enc = self.position_encoding.to(device)
            x_pos = x_proj + pos_enc.unsqueeze(0)
        else:
            x_pos = x_proj
        
        def transformer_func(x_input):
            return self.transformer(x_input, mask=mask)
        
        attended = checkpoint(transformer_func, x_pos)
        output = self.output_projection(attended)
        
        if self.use_residual:
            output = output + x
        
        return output
    
    def _create_sinusoidal_encoding(self) -> torch.Tensor:
        """Sinusoidal position encoding »ý¼º"""
        pe = torch.zeros(self.config.NUM_PAIRS, self.attention_dim)
        position = torch.arange(0, self.config.NUM_PAIRS).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.attention_dim, 2).float() * 
                           -(math.log(10000.0) / self.attention_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def get_attention_patterns(self, x: torch.Tensor) -> Dict:
        """Attention pattern ºÐ¼® (Cross-Attention È£È¯)"""
        if self.use_cross_attention:
            return {
                'attention_type': 'cross_attention',
                'channel_groups': 19,
                'cross_attention_active': True,
                'group_strategy': self.config.CROSS_ATTENTION_CONFIG['group_strategy']
            }
        else:
            # ±âÁ¸ Self-Attention ºÐ¼® ÄÚµå
            return self._analyze_self_attention(x)
    
    def _analyze_self_attention(self, x: torch.Tensor) -> Dict:
        """±âÁ¸ Self-Attention ºÐ¼®"""
        self.eval()
        initial_memory = get_memory_info()
        
        with torch.no_grad():
            device = x.device
            x_proj = self.input_projection(x)
            
            if self.position_encoding is not None:
                pos_enc = self.position_encoding.to(device)
                x_pos = x_proj + pos_enc.unsqueeze(0)
            else:
                x_pos = x_proj
            
            # Ã¹ ¹øÂ° layer attention ºÐ¼®
            first_layer = self.transformer.layers[0]
            attn_output, attn_weights = first_layer.self_attn(
                x_pos, x_pos, x_pos, need_weights=True, average_attn_weights=True
            )
            
            peak_memory = get_memory_info()
            
            return {
                'attention_type': 'self_attention',
                'attention_statistics': {
                    'entropy': (-attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1).mean().item(),
                    'sparsity': (attn_weights < 0.1).float().mean().item(),
                    'max_attention': attn_weights.max().item(),
                    'min_attention': attn_weights.min().item(),
                },
                'memory_usage': {
                    'memory_increase_gb': peak_memory['allocated'] - initial_memory['allocated']
                }
            }

# Backward compatibility
FrequencyProcessor = DynamicFrequencyProcessor
ComplexProcessor = DynamicComplexProcessor
FrequencyCNN = StructuredFeatureExtraction

if __name__ == "__main__":
    print("="*80)
    print("?? ENHANCED STRUCTURED FEATURE EXTRACTION MODULE")
    print("="*80)
    
    # Enhanced config for 4-5M parameters
    from config import EEGConfig
    config = EEGConfig()
    
    # Override for testing (4-5M parameter config)
    config.FREQUENCY_FEATURE_DIM = 80
    config.COMPLEX_FEATURE_DIM = 80
    config.UNIFIED_FEATURE_DIM = 160
    
    config.GLOBAL_ATTENTION_CONFIG.update({
        'input_dim': 160,
        'attention_dim': 160, 
        'num_heads': 10,
        'num_layers': 16,
        'ffn_hidden_dim': 640
    })
    
    # Memory optimization
    setattr(config, 'MEMORY_CONFIG', {
        'gradient_checkpointing': True,
        'mixed_precision': True
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Feature extraction Å×½ºÆ®
    print("\n1. Enhanced Feature Extraction Test:")
    feature_module = StructuredFeatureExtraction(config).to(device)
    sample_input = torch.randn(2, 361, 20, 2).to(device)
    
    initial_memory = get_memory_info()
    features = feature_module(sample_input)
    peak_memory = get_memory_info()
    
    print(f"   Input: {sample_input.shape} on {sample_input.device}")
    print(f"   Output: {features.shape} on {features.device}")
    print(f"   Parameters: {sum(p.numel() for p in feature_module.parameters()):,}")
    print(f"   Memory usage: {peak_memory['allocated'] - initial_memory['allocated']:.3f} GB")
    
    # Global attention Å×½ºÆ®
    print("\n2. Enhanced Global Attention Test:")
    attention_module = GlobalAttentionModule(config).to(device)
    
    initial_memory = get_memory_info()
    attended = attention_module(features)
    peak_memory = get_memory_info()
    
    print(f"   Input: {features.shape} on {features.device}")
    print(f"   Output: {attended.shape} on {attended.device}")
    print(f"   Parameters: {sum(p.numel() for p in attention_module.parameters()):,}")
    print(f"   Memory usage: {peak_memory['allocated'] - initial_memory['allocated']:.3f} GB")
    
    # ÅëÇÕ ºÐ¼®
    total_params = (sum(p.numel() for p in feature_module.parameters()) + 
                   sum(p.numel() for p in attention_module.parameters()))
    
    print(f"\n3. Combined Analysis:")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Memory Estimate: {total_params * 4 / (1024**3):.3f} GB")
    
    # ¼º´É ºÐ¼®
    feature_stats = feature_module.get_feature_statistics(sample_input)
    attention_patterns = attention_module.get_attention_patterns(features)
    
    print(f"\n4. Performance Analysis:")
    print(f"   Feature variance ratio: {feature_stats['information_preservation']['variance_ratio']:.3f}")
    print(f"   Attention entropy: {attention_patterns['attention_statistics']['entropy']:.3f}")
    print(f"   Attention sparsity: {attention_patterns['attention_statistics']['sparsity']:.3f}")
    print(f"   Memory efficiency: Gradient checkpointing enabled")
    
    print("="*80)
    print("? Enhanced Structured Feature Extraction Ready!")
    print("   - 4-5M parameter support")
    print("   - Memory optimization")
    print("   - Dynamic depth configuration")
    print("   - Complete backward compatibility")
    print("="*80)