"""
EEG Connectivity Analysis - Dynamic Layer Utilities

ÇÙ½É ±â´É:
1. Config ±â¹Ý Dynamic MLP »ý¼º
2. 4-5M ÆÄ¶ó¹ÌÅÍ Áö¿øÀ» À§ÇÑ ±íÀº ³×Æ®¿öÅ©
3. ¸Þ¸ð¸® ÃÖÀûÈ­ ±â´É
4. ±âÁ¸ ±¸Á¶ ¿ÏÀü È£È¯
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
        
        self.input_dim = freq_config['input_dim']        # 15
        self.output_dim = freq_config['output_dim']      # 80 (È®ÀåµÊ)
        
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
            x: (batch, pairs, 15) - ÁÖÆÄ¼ö Â÷¿ø µ¥ÀÌÅÍ
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
    Config ±â¹Ý Dynamic Complex Processor
    
    4-5M ÆÄ¶ó¹ÌÅÍ Áö¿øÀ» À§ÇØ ±íÀº ³×Æ®¿öÅ© °¡´É
    ±âÁ¸ "¸ðµç ÁÖÆÄ¼ö °øÀ¯" Ã¶ÇÐ À¯Áö
    """
    
    def __init__(self, config):
        super().__init__()
        
        complex_config = config.FEATURE_EXTRACTION_CONFIG['complex_processor']
        
        self.input_dim = complex_config['input_dim']      # 2 (real, imag)
        self.output_dim = complex_config['output_dim']    # 80 (È®ÀåµÊ)
        
        # Dynamic MLP »ý¼º
        hidden_dims = complex_config.get('hidden_dims', [40, self.output_dim])
        
        self.processor = build_mlp(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=hidden_dims,
            activation=complex_config.get('activation', 'gelu'),
            dropout=complex_config.get('dropout', 0.1),
            use_batch_norm=complex_config.get('use_batch_norm', False)
        )
        
        # ÇÐ½À °¡´ÉÇÑ Real/Imag ±ÕÇü °¡ÁßÄ¡ (±âÁ¸ À¯Áö)
        self.complex_balance = nn.Parameter(torch.ones(self.input_dim))
        
        # º¹¼Ò¼ö magnitude¿Í phase Á¤º¸ ÇÐ½À (È®Àå)
        self.magnitude_processor = nn.Linear(1, 8)   # 4->8·Î È®Àå
        self.phase_processor = nn.Linear(1, 8)       # 4->8·Î È®Àå
        self.complex_fusion = nn.Linear(self.output_dim + 16, self.output_dim)  # 16+16 ¡æ output
        
        print(f"?? Dynamic ComplexProcessor: {self.input_dim} ¡æ {self.output_dim}")
        print(f"   Hidden dims: {hidden_dims}")
        print(f"   Shared across all frequencies: True")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, pairs, freq, 2) - º¹¼Ò¼ö µ¥ÀÌÅÍ
        Returns:
            (batch, pairs, freq, output_dim) - º¹¼Ò¼ö features
        """
        batch_size, num_pairs, num_freq, complex_dim = x.shape
        device = x.device
        
        # Flatten for batch processing
        x_flat = x.reshape(-1, complex_dim)  # (batch*pairs*freq, 2)
        
        # ÇÐ½ÀµÈ Real/Imag ±ÕÇü Àû¿ë
        complex_weights = torch.softmax(self.complex_balance, dim=0).to(device)
        x_weighted = x_flat * complex_weights.unsqueeze(0)
        
        # ±âº» Real/Imag Ã³¸®
        complex_features = self.processor(x_weighted)  # (batch*pairs*freq, output_dim)
        
        # Magnitude¿Í Phase Á¤º¸ Ãß°¡ Ã³¸®
        real = x_flat[:, 0:1]
        imag = x_flat[:, 1:2]
        
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-8)
        phase = torch.atan2(imag, real + 1e-8)
        
        mag_features = self.magnitude_processor(magnitude)   # (batch*pairs*freq, 8)
        phase_features = self.phase_processor(phase)        # (batch*pairs*freq, 8)
        
        # ¸ðµç feature °áÇÕ
        all_features = torch.cat([complex_features, mag_features, phase_features], dim=1)
        final_features = self.complex_fusion(all_features)  # (batch*pairs*freq, output_dim)
        
        # Reshape back
        output = final_features.reshape(batch_size, num_pairs, num_freq, self.output_dim)
        
        return output
    
    def get_complex_analysis(self) -> Dict:
        """ÇÐ½ÀµÈ º¹¼Ò¼ö ±ÕÇü ºÐ¼® (±âÁ¸ À¯Áö)"""
        with torch.no_grad():
            weights = torch.softmax(self.complex_balance, dim=0)
            
            analysis = {
                'real_weight': weights[0].item(),
                'imag_weight': weights[1].item(),
                'real_imag_ratio': (weights[0] / weights[1]).item(),
                'complex_balance_entropy': (-weights * torch.log(weights + 1e-8)).sum().item(),
                'total_parameters': sum(p.numel() for p in self.parameters())
            }
            
        return analysis

class DynamicFusionLayer(nn.Module):
    """
    Config ±â¹Ý Dynamic Fusion Layer
    
    ÁÖÆÄ¼ö Æ¯¼º°ú º¹¼Ò¼ö Æ¯¼ºÀ» À¶ÇÕ
    """
    
    def __init__(self, config):
        super().__init__()
        
        fusion_config = config.FEATURE_EXTRACTION_CONFIG['fusion_config']
        
        self.input_dim = fusion_config['input_dim']       # 160 (80+80)
        self.output_dim = fusion_config['output_dim']     # 160
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
    print("?? DYNAMIC LAYER UTILITIES")
    print("="*80)
    
    # MLP builder Å×½ºÆ®
    print("\n1. Dynamic MLP Builder Test:")
    
    # °£´ÜÇÑ MLP
    simple_mlp = build_mlp(32, 16, hidden_dims=[64, 32])
    print(f"   Simple MLP: 32 ¡æ [64, 32] ¡æ 16")
    print(f"   Parameters: {sum(p.numel() for p in simple_mlp.parameters()):,}")
    
    # º¹ÀâÇÑ MLP (4-5M ÆÄ¶ó¹ÌÅÍ¿ë)
    complex_mlp = build_mlp(160, 80, hidden_dims=[160, 160, 120, 80], 
                           use_residual=True, use_batch_norm=True)
    print(f"   Complex MLP: 160 ¡æ [160, 160, 120, 80] ¡æ 80")
    print(f"   Parameters: {sum(p.numel() for p in complex_mlp.parameters()):,}")
    
    # Å×½ºÆ® ½ÇÇà
    test_input = torch.randn(4, 361, 32)
    test_output = simple_mlp(test_input)
    print(f"   Test: {test_input.shape} ¡æ {test_output.shape}")
    
    print("\n2. Memory Utilities Test:")
    memory_info = get_memory_info()
    print(f"   GPU Memory: {memory_info}")
    
    print("\n3. Parameter Count Test:")
    param_info = count_parameters(complex_mlp)
    print(f"   Total parameters: {param_info['total_parameters']:,}")
    print(f"   Memory estimate: {param_info['memory_mb']:.1f} MB")
    
    print("="*80)
    print("? Dynamic Layer Utilities Ready!")
    print("   - Config-based MLP generation")
    print("   - Memory optimization tools") 
    print("   - Parameter analysis utilities")
    print("   - 4-5M parameter support")
    print("="*80)