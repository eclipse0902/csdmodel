"""
EEG Connectivity Analysis - Enhanced Dynamic Layer Utilities with Complete Separate Support

ÇÙ½É ±â´É:
1. Config ±â¹Ý Dynamic MLP »ý¼º
2. 4-5M ÆÄ¶ó¹ÌÅÍ Áö¿øÀ» À§ÇÑ ±íÀº ³×Æ®¿öÅ©
3. ? Complete Separate Complex Processing ±¸Çö
4. ¸Þ¸ð¸® ÃÖÀûÈ­ ±â´É
5. ±âÁ¸ ±¸Á¶ ¿ÏÀü È£È¯
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Dict, Any
import math

def get_activation(activation: str) -> nn.Module:
    """È°¼ºÈ­ ÇÔ¼ö ¼±ÅÃ"""
    activations = {
        'relu': nn.ReLU(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(0.01),
        'swish': nn.SiLU(),
        'elu': nn.ELU()
    }
    
    if activation.lower() not in activations:
        raise ValueError(f"Unsupported activation: {activation}. Choose from {list(activations.keys())}")
    
    return activations[activation.lower()]

def build_mlp(input_dim: int,
              output_dim: int, 
              hidden_dims: Optional[List[int]] = None,
              activation: str = 'gelu',
              dropout: float = 0.1,
              use_batch_norm: bool = False,
              use_residual: bool = False,
              final_activation: bool = False) -> nn.Module:
    """
    Config ±â¹Ý Dynamic MLP »ý¼º
    
    Args:
        input_dim: ÀÔ·Â Â÷¿ø
        output_dim: Ãâ·Â Â÷¿ø
        hidden_dims: È÷µç ·¹ÀÌ¾î Â÷¿øµé [64, 32] -> input->64->32->output
        activation: È°¼ºÈ­ ÇÔ¼ö¸í
        dropout: µå·Ó¾Æ¿ô È®·ü
        use_batch_norm: ¹èÄ¡ Á¤±ÔÈ­ »ç¿ë ¿©ºÎ
        use_residual: ÀÜÂ÷ ¿¬°á »ç¿ë ¿©ºÎ (input_dim == output_dimÀÏ ¶§¸¸)
        final_activation: ¸¶Áö¸· Ãâ·Â¿¡µµ È°¼ºÈ­ Àû¿ë ¿©ºÎ
        
    Returns:
        nn.Module: µ¿ÀûÀ¸·Î »ý¼ºµÈ MLP
    """
    
    if hidden_dims is None:
        # ±âº»°ª: ´Ü¼øÇÑ 2Ãþ MLP
        hidden_dims = [max(input_dim, output_dim)]
    
    # ÀüÃ¼ Â÷¿ø ¼ø¼­: input -> hidden_dims -> output
    all_dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    
    # °¢ ·¹ÀÌ¾î »ý¼º
    for i in range(len(all_dims) - 1):
        # Linear layer
        layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
        
        # ¸¶Áö¸· ·¹ÀÌ¾î°¡ ¾Æ´Ï°Å³ª final_activation=TrueÀÎ °æ¿ì
        if i < len(all_dims) - 2 or final_activation:
            # Batch normalization (¼±ÅÃÀû)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(all_dims[i + 1]))
            
            # Activation
            layers.append(get_activation(activation))
            
            # Dropout (¸¶Áö¸· ·¹ÀÌ¾î°¡ ¾Æ´Ò ¶§¸¸)
            if i < len(all_dims) - 2 and dropout > 0:
                layers.append(nn.Dropout(dropout))
    
    mlp = nn.Sequential(*layers)
    
    # Residual connection wrapper
    if use_residual and input_dim == output_dim:
        return ResidualMLP(mlp, input_dim)
    
    return mlp

class ResidualMLP(nn.Module):
    """ÀÜÂ÷ ¿¬°áÀÌ ÀÖ´Â MLP wrapper"""
    
    def __init__(self, mlp: nn.Module, feature_dim: int):
        super().__init__()
        self.mlp = mlp
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., feature_dim)
        residual = x
        output = self.mlp(x)
        
        # Residual connection
        if output.shape == residual.shape:
            return output + residual
        else:
            # Â÷¿øÀÌ ´Ù¸£¸é residual ¾øÀÌ
            return output

class DynamicFrequencyProcessor(nn.Module):
    """
    Config ±â¹Ý Dynamic Frequency Processor
    
    4-5M ÆÄ¶ó¹ÌÅÍ Áö¿øÀ» À§ÇØ ±íÀº ³×Æ®¿öÅ© °¡´É
    """
    
    def __init__(self, config):
        super().__init__()
        
        freq_config = config.FEATURE_EXTRACTION_CONFIG['frequency_processor']
        
        self.input_dim = freq_config['input_dim']        # 20
        self.output_dim = freq_config['output_dim']      # 32
        
        # Dynamic MLP »ý¼º
        hidden_dims = freq_config.get('hidden_dims', [self.output_dim])
        
        self.processor = build_mlp(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=hidden_dims,
            activation=freq_config.get('activation', 'gelu'),
            dropout=freq_config.get('dropout', 0.1),
            use_batch_norm=freq_config.get('use_batch_norm', False),
            use_residual=freq_config.get('use_residual', False)
        )
        
        # ÇÐ½À °¡´ÉÇÑ ÁÖÆÄ¼ö Áß¿äµµ °¡ÁßÄ¡
        self.frequency_importance = nn.Parameter(torch.ones(self.input_dim))
        
        # ÁÖÆÄ¼ö ´ë¿ªº° ÀÓº£µù (±âÁ¸ À¯Áö)
        self.frequency_bands = config.FREQUENCY_BANDS
        self.band_embeddings = nn.ModuleDict({
            band: nn.Linear(len(indices), 8)  # 4->8·Î È®Àå 
            for band, indices in self.frequency_bands.items()
        })
        
        print(f"?? Dynamic FrequencyProcessor: {self.input_dim} ¡æ {self.output_dim}")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, pairs, 20) - ÁÖÆÄ¼ö Â÷¿ø µ¥ÀÌÅÍ
        Returns:
            (batch, pairs, output_dim) - ÁÖÆÄ¼ö features
        """
        # Device È£È¯¼º
        device = x.device
        
        # ÇÐ½ÀµÈ ÁÖÆÄ¼ö Áß¿äµµ Àû¿ë
        freq_weights = torch.softmax(self.frequency_importance, dim=0).to(device)
        x_weighted = x * freq_weights.unsqueeze(0).unsqueeze(0)
        
        # ÁÖÆÄ¼ö °ü°è ÇÐ½À
        freq_features = self.processor(x_weighted)
        
        return freq_features
    
    def get_frequency_analysis(self) -> Dict:
        """ÇÐ½ÀµÈ ÁÖÆÄ¼ö Áß¿äµµ ºÐ¼® (±âÁ¸ À¯Áö)"""
        with torch.no_grad():
            weights = torch.softmax(self.frequency_importance, dim=0)
            
            analysis = {
                'frequency_weights': weights.cpu().numpy(),
                'band_importance': {},
                'most_important_freq': torch.argmax(weights).item(),
                'frequency_entropy': (-weights * torch.log(weights + 1e-8)).sum().item(),
                'total_parameters': sum(p.numel() for p in self.parameters())
            }
            
            # ÁÖÆÄ¼ö ´ë¿ªº° Áß¿äµµ
            for band, indices in self.frequency_bands.items():
                band_weight = weights[indices].mean().item()
                analysis['band_importance'][band] = band_weight
            
        return analysis

class DynamicComplexProcessor(nn.Module):
    """
    ? COMPLETE Enhanced DynamicComplexProcessor - ÁÖÆÄ¼öº° µ¶¸³ Ã³¸® + Separate ¿ÏÀü ±¸Çö
    
    ÇÙ½É °³¼±»çÇ×:
    - ±âÁ¸: ¸ðµç ÁÖÆÄ¼ö °øÀ¯ ¡æ 20°³ µ¶¸³ Ã³¸®±â
    - ? Ãß°¡: Real/Imag separate Ã³¸® ¿ÏÀü ±¸Çö
    - ±âÁ¸ Å¬·¡½º ÀÌ¸§ ¿ÏÀü À¯Áö (È£È¯¼º)
    """
    
    def __init__(self, config):
        super().__init__()
        
        complex_config = config.FEATURE_EXTRACTION_CONFIG['complex_processor']
        
        self.input_dim = complex_config['input_dim']      # 2 (real, imag)
        self.output_dim = complex_config['output_dim']    # 32
        self.frequency_independent = complex_config.get('frequency_independent', True)
        
        # ? Separate ¸ðµå ¼³Á¤
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
                
                # ? Real-Imag À¶ÇÕ ·¹ÀÌ¾î (°¢ ÁÖÆÄ¼öº°)
                self.fusion_layers = nn.ModuleList([
                    self._create_fusion_layer() for _ in range(20)
                ])
                
                print(f"? Separate Complex Processor:")
                print(f"   Real processors: 20°³ (°¢ 1 ¡æ {hidden_dims} ¡æ {self.output_dim})")
                print(f"   Imag processors: 20°³ (°¢ 1 ¡æ {hidden_dims} ¡æ {self.output_dim})")
                print(f"   Fusion layers: 20°³")
                
            elif self.combination_mode in ['mean', 'magnitude']:
                # ±âÁ¸ ¹æ½Ä (È£È¯¼º) - 20°³ µ¶¸³ÀÌÁö¸¸ Real+Imag ÇÔ²² Ã³¸®
                self.freq_processors = nn.ModuleList([
                    self._create_single_processor(hidden_dims)
                    for _ in range(20)
                ])
                print(f"? Standard Complex Processor: 20°³ µ¶¸³ ({self.combination_mode} ¹æ½Ä)")
            else:
                # Fallback to mean
                self.freq_processors = nn.ModuleList([
                    self._create_single_processor(hidden_dims)
                    for _ in range(20)
                ])
                print(f"??  Unknown combination mode '{self.combination_mode}', using standard processing")
        else:
            # ±âÁ¸ °øÀ¯ ¹æ½Ä (backward compatibility)
            hidden_dims = complex_config.get('hidden_dims', [16, 32])
            self.processor = self._create_single_processor(hidden_dims)
            print(f"??  Using shared complex processor (not recommended)")
    
    def _create_single_processor(self, hidden_dims, input_dim=None):
        """´ÜÀÏ processor »ý¼º"""
        if input_dim is None:
            input_dim = self.input_dim  # 2 (real + imag)
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _create_fusion_layer(self):
        """? Real-Imag À¶ÇÕ ·¹ÀÌ¾î - »óÈ£ÀÛ¿ë ÇÐ½À"""
        return nn.Sequential(
            nn.Linear(self.output_dim * 2, self.output_dim),  # 32+32 ¡æ 32
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.output_dim, self.output_dim),      # Ãß°¡ º¯È¯
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ? Enhanced forward pass with complete separate support
        
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
        """? Separate Real/Imag processing - ¿ÏÀü ±¸Çö"""
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
            
            # ? °¢°¢ µ¶¸³ÀûÀ¸·Î Ã³¸® (Real°ú ImagÀÇ ¼­·Î ´Ù¸¥ Æ¯¼º ÇÐ½À)
            real_output = self.real_processors[freq_idx](freq_real)  # (batch, 361, 32)
            imag_output = self.imag_processors[freq_idx](freq_imag)  # (batch, 361, 32)
            
            # ? Real-Imag Áö´ÉÀû À¶ÇÕ (´Ü¼ø concatÀÌ ¾Æ´Ñ ÇÐ½ÀµÈ »óÈ£ÀÛ¿ë)
            combined_input = torch.cat([real_output, imag_output], dim=-1)  # (batch, 361, 64)
            fused_output = self.fusion_layers[freq_idx](combined_input)     # (batch, 361, 32)
            
            frequency_outputs.append(fused_output)
        
        # Stack all frequency outputs
        result = torch.stack(frequency_outputs, dim=2)  # (batch, 361, 20, 32)
        return result
    
    def _standard_forward(self, x: torch.Tensor) -> torch.Tensor:
        """±âÁ¸ ¹æ½Ä (È£È¯¼º À¯Áö) - 20°³ µ¶¸³ÀÌÁö¸¸ Real+Imag ÇÔ²² Ã³¸®"""
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
    
    def get_complex_analysis(self) -> Dict:
        """? º¹¼Ò¼ö Ã³¸® ºÐ¼® (enhanced with separate info)"""
        analysis = {
            'combination_mode': self.combination_mode,
            'frequency_independent': self.frequency_independent,
            'total_parameters': sum(p.numel() for p in self.parameters()),
        }
        
        if self.combination_mode == 'separate':
            # ? Separate ¸ðµå »ó¼¼ ºÐ¼®
            analysis.update({
                'processing_type': 'separate_real_imag',
                'real_processors': len(self.real_processors),
                'imag_processors': len(self.imag_processors),
                'fusion_layers': len(self.fusion_layers),
                'real_params': sum(p.numel() for processor in self.real_processors for p in processor.parameters()),
                'imag_params': sum(p.numel() for processor in self.imag_processors for p in processor.parameters()),
                'fusion_params': sum(p.numel() for layer in self.fusion_layers for p in layer.parameters()),
                'advantages': [
                    'Real°ú ImagÀÇ ¼­·Î ´Ù¸¥ ¹°¸®Àû Æ¯¼º °¢°¢ ÃÖÀûÈ­',
                    'ºñ¼±Çü »óÈ£ÀÛ¿ë ÇÐ½À °¡´É',
                    'À§»ó Á¤º¸ ¿Ïº® º¸Á¸',
                    'EEG connectivityÀÇ º¹¼Ò¼ö Æ¯¼º ¿ÏÀü È°¿ë'
                ]
            })
        elif hasattr(self, 'freq_processors'):
            # Ç¥ÁØ µ¶¸³ Ã³¸®
            analysis.update({
                'processing_type': 'standard_independent',
                'standard_processors': len(self.freq_processors),
                'params_per_processor': sum(p.numel() for p in self.freq_processors[0].parameters())
            })
        else:
            # °øÀ¯ Ã³¸® (ºñÃßÃµ)
            analysis.update({
                'processing_type': 'shared_legacy',
                'warning': 'Shared processing not recommended for optimal performance'
            })
        
        return analysis

class DynamicFusionLayer(nn.Module):
    """
    Config ±â¹Ý Dynamic Fusion Layer
    
    ÁÖÆÄ¼ö Æ¯¼º°ú º¹¼Ò¼ö Æ¯¼ºÀ» À¶ÇÕ
    """
    
    def __init__(self, config):
        super().__init__()
        
        fusion_config = config.FEATURE_EXTRACTION_CONFIG['fusion_config']
        
        self.input_dim = fusion_config['input_dim']       # 64 (32+32)
        self.output_dim = fusion_config['output_dim']     # 64
        self.use_residual = fusion_config.get('use_residual', True)
        
        # Dynamic MLP »ý¼º
        hidden_dims = fusion_config.get('hidden_dims', [self.output_dim])
        
        self.fusion_layer = build_mlp(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=hidden_dims,
            activation=fusion_config.get('activation', 'gelu'),
            dropout=fusion_config.get('dropout', 0.1),
            use_residual=self.use_residual and (self.input_dim == self.output_dim)
        )
        
        # Residual projection (Â÷¿øÀÌ ´Ù¸¦ ¶§)
        if self.use_residual and self.input_dim != self.output_dim:
            self.residual_projection = nn.Linear(self.input_dim, self.output_dim)
        else:
            self.residual_projection = None
        
        print(f"?? Dynamic FusionLayer: {self.input_dim} ¡æ {self.output_dim}")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Residual: {self.use_residual}")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, pairs, input_dim) - °áÇÕµÈ features
        Returns:
            (batch, pairs, output_dim) - À¶ÇÕµÈ features
        """
        # Fusion Ã³¸®
        fused = self.fusion_layer(x)
        
        # Residual connection
        if self.use_residual:
            if self.residual_projection is not None:
                residual = self.residual_projection(x)
            else:
                residual = x
            
            if fused.shape == residual.shape:
                fused = fused + residual
        
        return fused

# Gradient checkpointing utility
class CheckpointFunction(torch.autograd.Function):
    """¸Þ¸ð¸® È¿À²ÀûÀÎ gradient checkpointing"""
    
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors
    
    @staticmethod
    def backward(ctx, *output_grads):
        for i, arg in enumerate(ctx.input_tensors):
            if torch.is_tensor(arg):
                ctx.input_tensors[i] = arg.detach().requires_grad_(arg.requires_grad)
        
        with torch.enable_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        
        if isinstance(output_tensors, torch.Tensor):
            output_tensors = (output_tensors,)
        
        torch.autograd.backward(output_tensors, output_grads)
        
        grads = []
        for inp in ctx.input_tensors:
            if torch.is_tensor(inp):
                grads.append(inp.grad)
            else:
                grads.append(None)
        
        return (None, None) + tuple(grads)

def checkpoint(function, *args):
    """Gradient checkpointing wrapper"""
    return CheckpointFunction.apply(function, len(args), *args)

# ¸Þ¸ð¸® ÃÖÀûÈ­ utilities
def get_memory_info():
    """ÇöÀç GPU ¸Þ¸ð¸® »ç¿ë·® ¹ÝÈ¯"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'cached': torch.cuda.memory_reserved() / 1024**3,     # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {'allocated': 0, 'cached': 0, 'max_allocated': 0}

def clear_memory():
    """GPU ¸Þ¸ð¸® Á¤¸®"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ¸ðµ¨ Å©±â °è»ê utility
def count_parameters(model: nn.Module) -> Dict[str, int]:
    """¸ðµ¨ ÆÄ¶ó¹ÌÅÍ ¼ö »ó¼¼ ºÐ¼®"""
    total_params = 0
    trainable_params = 0
    
    param_dict = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if module_params > 0:
                param_dict[name] = {
                    'total': module_params,
                    'trainable': module_trainable,
                    'type': module.__class__.__name__
                }
                
                total_params += module_params
                trainable_params += module_trainable
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_breakdown': param_dict,
        'memory_mb': total_params * 4 / (1024 * 1024)  # float32 ±âÁØ
    }

if __name__ == "__main__":
    print("="*80)
    print("? ENHANCED DYNAMIC LAYER UTILITIES WITH COMPLETE SEPARATE SUPPORT")
    print("="*80)
    
    # Mock config for testing
    class MockConfig:
        FEATURE_EXTRACTION_CONFIG = {
            'frequency_processor': {
                'input_dim': 20,
                'hidden_dims': [40, 32],
                'output_dim': 32,
                'activation': 'gelu',
                'dropout': 0.1
            },
            'complex_processor': {
                'input_dim': 2,
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
            'complex_combination': 'separate'  # ? Test separate mode
        }
        FREQUENCY_BANDS = {
            'delta': [0, 1, 2, 3],
            'theta': [4, 5, 6, 7],
            'alpha': [8, 9],
            'beta1': [10, 11, 12, 13],
            'beta2': [14, 15],
            'gamma': [16, 17, 18, 19]
        }
    
    config = MockConfig()
    
    # 1. Dynamic MLP Builder Å×½ºÆ®
    print("\n1. ? Dynamic MLP Builder Test:")
    simple_mlp = build_mlp(32, 16, hidden_dims=[64, 32])
    print(f"   Simple MLP: 32 ¡æ [64, 32] ¡æ 16")
    print(f"   Parameters: {sum(p.numel() for p in simple_mlp.parameters()):,}")
    
    # 2. ? Separate Complex Processor Å×½ºÆ® (ÇÙ½É!)
    print("\n2. ? SEPARATE Complex Processor Test:")
    complex_processor = DynamicComplexProcessor(config)
    
    # Test input: (batch=2, pairs=361, freq=20, complex=2)
    test_input = torch.randn(2, 361, 20, 2)
    print(f"   Input: {test_input.shape}")
    
    try:
        output = complex_processor(test_input)
        print(f"   ? Output: {output.shape}")
        print(f"   ? Separate processing working!")
        
        # ºÐ¼®
        analysis = complex_processor.get_complex_analysis()
        print(f"   Mode: {analysis['combination_mode']}")
        print(f"   Type: {analysis.get('processing_type', 'unknown')}")
        print(f"   Total params: {analysis['total_parameters']:,}")
        if 'real_params' in analysis:
            print(f"   Real processors: {analysis['real_params']:,} params")
            print(f"   Imag processors: {analysis['imag_params']:,} params")
            print(f"   Fusion layers: {analysis['fusion_params']:,} params")
            
    except Exception as e:
        print(f"   ? Error: {str(e)}")
    
    # 3. È£È¯¼º Å×½ºÆ®
    print("\n3. ? Backward Compatibility Test:")
    test_configs = [
        ('mean', 'mean'),
        ('mean', 'separate'),
        ('attention', 'separate')
    ]
    
    for freq_agg, complex_comb in test_configs:
        config.FEATURE_EXTRACTION_CONFIG['frequency_aggregation'] = freq_agg
        config.FEATURE_EXTRACTION_CONFIG['complex_combination'] = complex_comb
        
        try:
            test_processor = DynamicComplexProcessor(config)
            test_output = test_processor(test_input)
            print(f"   ? {freq_agg}/{complex_comb}: {test_output.shape}")
        except Exception as e:
            print(f"   ? {freq_agg}/{complex_comb}: {str(e)}")
    
    # 4. Memory utilities Å×½ºÆ®
    print("\n4. ? Memory Utilities Test:")
    memory_info = get_memory_info()
    print(f"   GPU Memory: {memory_info}")
    
    # 5. Parameter counting Å×½ºÆ®
    print("\n5. ? Parameter Analysis Test:")
    param_info = count_parameters(complex_processor)
    print(f"   Total parameters: {param_info['total_parameters']:,}")
    print(f"   Memory estimate: {param_info['memory_mb']:.1f} MB")
    
    print("="*80)
    print("? ENHANCED LAYER UTILITIES READY!")
    print("="*80)
    
    print("?? Key Features Implemented:")
    print("   ? Complete Separate Real/Imag processing")
    print("   ? 20°³ µ¶¸³ ÁÖÆÄ¼ö processors")
    print("   ? Config-based dynamic MLP generation")
    print("   ? Memory optimization tools")
    print("   ? Parameter analysis utilities")
    print("   ? 100% Backward compatibility")
    
    print("\n?? Usage for Separate Mode:")
    print("   config.FEATURE_EXTRACTION_CONFIG['complex_combination'] = 'separate'")
    print("   # Real°ú Imag °¢°¢ µ¶¸³ Ã³¸® + Áö´ÉÀû À¶ÇÕ")
    
    print("\n?? Problem Solutions:")
    print("   ?? Separate ¡æ Real/Imag À§»ó Á¤º¸ ¿Ïº® º¸Á¸")
    print("   ?? 20°³ µ¶¸³ ¡æ 1Hz ¡Á 50Hz ÁÖÆÄ¼öº° Æ¯¼º ÇÐ½À")
    print("   ?? ºñ¼±Çü À¶ÇÕ ¡æ Real-Imag »óÈ£ÀÛ¿ë ÇÐ½À")
    print("   ?? ¹°¸®Àû ÀÇ¹Ì ¡æ µ¿À§»ó/Á÷±³À§»ó ¼ººÐ °¢°¢ ÃÖÀûÈ­")
    print("   ?? ¿Ïº®ÇÑ È£È¯¼º ¡æ ±âÁ¸ mean/magnitude ¹æ½Ä À¯Áö")
    
    print("\n?? Performance Comparison:")
    print("   Mean:     ´Ü¼ø Æò±Õ (À§»ó Á¤º¸ ¼Õ½Ç)")
    print("   Magnitude: Å©±â¸¸ º¸Á¸ (À§»ó ¼Õ½Ç)")
    print("   ? Separate: Real+Imag ¿ÏÀü º¸Á¸ + »óÈ£ÀÛ¿ë ÇÐ½À")
    
    print("\n?? EEG Connectivity ÃÖÀûÈ­:")
    print("   ? Real part: µ¿À§»ó ¼ººÐ (in-phase)")
    print("   ? Imag part: Á÷±³À§»ó ¼ººÐ (quadrature)")
    print("   ? Fusion: µÎ ¼ººÐ°£ ºñ¼±Çü »óÈ£ÀÛ¿ë")
    print("   ? Result: À§»ó °ü°è ¿Ïº® º¸Á¸")
    
    print("="*80)