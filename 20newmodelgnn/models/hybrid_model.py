"""
EEG Connectivity Analysis - Unified Hybrid Model

ÀüÃ¼ ¸ðµ¨ ÅëÇÕ:
1. Structured Feature Extraction (Stage 1)
2. Global Attention (Stage 2) 
3. Frequency-Specific Reconstruction (Stage 3)
4. Pre-training°ú Fine-tuning ¸ðµÎ Áö¿ø
5. Config ¿ÏÀü ÀÇÁ¸ ¼³°è
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_gnn import EEGConfig
from models.structured_feature_extraction_gnn import StructuredFeatureExtraction, GlobalAttentionModule
from models.reconstruction_head import FrequencySpecificReconstructionHead

class ClassificationHead(nn.Module):
    """
    Fine-tuning¿ë Classification Head
    361°³ Ã¤³Î ½ÖÀ» ÇÏ³ªÀÇ class·Î ºÐ·ù
    """
    
    def __init__(self, config: EEGConfig):
        super(ClassificationHead, self).__init__()
        
        self.config = config
        self.class_config = config.CLASSIFICATION_CONFIG
        
        self.input_dim = self.class_config['input_dim']          # 32
        self.hidden_dims = self.class_config['hidden_dims']      # [64, 32, 16]
        self.num_classes = self.class_config['num_classes']      # 2
        self.pooling_strategy = self.class_config['pooling_strategy']  # 'attention'
        
        # =============== GLOBAL POOLING LAYER ===============
        if self.pooling_strategy == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.input_dim, self.class_config['attention_pooling_dim']),
                nn.Tanh(),
                nn.Linear(self.class_config['attention_pooling_dim'], 1),
                nn.Softmax(dim=1)
            )
        elif self.pooling_strategy == 'mean':
            self.attention_pooling = None
        elif self.pooling_strategy == 'max':
            self.attention_pooling = None
        
        # =============== BRAIN REGION AWARE POOLING ===============
        if self.class_config['use_brain_region_pooling']:
            self.brain_regions = config.BRAIN_REGIONS
            self.region_weights = self.class_config['region_pooling_weights']
            self.region_poolings = nn.ModuleDict()
            
            for region_name in self.brain_regions.keys():
                self.region_poolings[region_name] = nn.Linear(self.input_dim, self.input_dim // 4)
            
            # Region fusion
            total_region_dim = len(self.brain_regions) * (self.input_dim // 4)
            self.region_fusion = nn.Linear(total_region_dim, self.input_dim)
        
        # =============== CLASSIFICATION LAYERS ===============
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.class_config['dropout']),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, self.num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        print(f"? Classification Head:")
        print(f"   Pooling: {self.pooling_strategy}")
        print(f"   Architecture: {self.input_dim} ¡æ {' ¡æ '.join(map(str, self.hidden_dims))} ¡æ {self.num_classes}")
        print(f"   Brain region pooling: {self.class_config['use_brain_region_pooling']}")
        print(f"   Dropout: {self.class_config['dropout']}")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, 361, 32) - features from global attention
        Returns:
            (batch, num_classes) - classification logits
        """
        batch_size, num_pairs, feature_dim = features.shape
        
        # =============== GLOBAL POOLING ===============
        if self.pooling_strategy == 'attention':
            # Attention-based pooling
            attention_weights = self.attention_pooling(features)  # (batch, 361, 1)
            pooled_features = (features * attention_weights).sum(dim=1)  # (batch, 32)
            
        elif self.pooling_strategy == 'mean':
            # Simple mean pooling
            pooled_features = features.mean(dim=1)  # (batch, 32)
            
        elif self.pooling_strategy == 'max':
            # Max pooling
            pooled_features = features.max(dim=1)[0]  # (batch, 32)
        
        # =============== BRAIN REGION AWARE POOLING ===============
        if self.class_config['use_brain_region_pooling']:
            region_features = []
            
            for region_name, electrode_indices in self.brain_regions.items():
                # Get pairs involving this brain region
                region_pairs = []
                for i in electrode_indices:
                    for j in range(19):  # All pairs with electrode i
                        pair_idx = i * 19 + j
                        if pair_idx < num_pairs:
                            region_pairs.append(pair_idx)
                
                if region_pairs:
                    region_pairs = torch.tensor(region_pairs, device=features.device)
                    region_feat = features[:, region_pairs, :].mean(dim=1)  # (batch, 32)
                    
                    # Apply region-specific processing
                    processed_region = self.region_poolings[region_name](region_feat)  # (batch, 8)
                    
                    # Apply region weight
                    weight = self.region_weights.get(region_name, 1.0)
                    weighted_region = processed_region * weight
                    
                    region_features.append(weighted_region)
            
            if region_features:
                combined_regions = torch.cat(region_features, dim=1)  # (batch, num_regions*8)
                fused_regions = self.region_fusion(combined_regions)  # (batch, 32)
                
                # Combine with global pooling
                pooled_features = (pooled_features + fused_regions) / 2
        
        # =============== CLASSIFICATION ===============
        logits = self.classifier(pooled_features)  # (batch, num_classes)
        
        return logits

class EEGConnectivityModel(nn.Module):
    """
    ÅëÇÕ EEG Connectivity Model
    
    ¸ðµåº° »ç¿ë¹ý:
    1. Pre-training: forward() ¡æ reconstruction loss
    2. Fine-tuning: forward_classification() ¡æ classification loss
    3. Feature extraction: get_features() ¡æ representation learning
    """
    
    def __init__(self, config: EEGConfig = None, mode: str = 'pretrain'):
        super(EEGConnectivityModel, self).__init__()
        
        if config is None:
            config = EEGConfig()
            
        self.config = config
        self.mode = mode  # 'pretrain', 'finetune', 'inference'
        
        # =============== CORE BACKBONE (3 STAGES) ===============
        self.feature_extraction = StructuredFeatureExtraction(config)
        self.global_attention = GlobalAttentionModule(config)
        
        # =============== TASK-SPECIFIC HEADS ===============
        if mode in ['pretrain', 'inference']:
            self.reconstruction_head = FrequencySpecificReconstructionHead(config)
        
        if mode in ['finetune', 'inference']:
            self.classification_head = ClassificationHead(config)
        
        # =============== MODEL INFO ===============
        self.model_info = self._get_model_info()
        
        print(f"? EEG Connectivity Model ({mode} mode):")
        print(f"   Backbone: Feature Extraction + Global Attention")
        if hasattr(self, 'reconstruction_head'):
            print(f"   Reconstruction: 15 frequency-specific heads")
        if hasattr(self, 'classification_head'):
            print(f"   Classification: ¡æ {config.NUM_CLASSES} classes")
        print(f"   Total parameters: {self.model_info['total_parameters']:,}")
        print(f"   Memory footprint: ~{self.model_info['memory_mb']:.1f} MB")
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Pre-training forward pass
        
        Args:
            x: (batch, 15, 19, 19, 2) - raw CSD data
            return_features: whether to return intermediate features
            
        Returns:
            reconstructed: (batch, 15, 19, 19, 2)
            features: (batch, 361, 32) - if return_features=True
        """
        if not hasattr(self, 'reconstruction_head'):
            raise ValueError(f"Reconstruction head not available in {self.mode} mode")
        
        # =============== BACKBONE PROCESSING ===============
        features = self.get_features(x)  # (batch, 361, 32)
        
        # =============== RECONSTRUCTION ===============
        reconstructed_pairs = self.reconstruction_head(features)  # (batch, 361, 15, 2)
        
        # =============== RESHAPE TO ORIGINAL FORMAT ===============
        batch_size = x.shape[0]
        # (batch, 361, 15, 2) ¡æ (batch, 19, 19, 15, 2) ¡æ (batch, 15, 19, 19, 2)
        reconstructed = reconstructed_pairs.reshape(batch_size, 19, 19, 20, 2)
        reconstructed = reconstructed.permute(0, 3, 1, 2, 4)  # (batch, 15, 19, 19, 2)
        
        if return_features:
            return reconstructed, features
        return reconstructed
    
    def forward_classification(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Fine-tuning classification forward pass
        
        Args:
            x: (batch, 15, 19, 19, 2) - raw CSD data
            return_features: whether to return intermediate features
            
        Returns:
            logits: (batch, num_classes)
            features: (batch, 361, 32) - if return_features=True
        """
        if not hasattr(self, 'classification_head'):
            raise ValueError(f"Classification head not available in {self.mode} mode")
        
        # =============== BACKBONE PROCESSING ===============
        features = self.get_features(x)  # (batch, 361, 32)
        
        # =============== CLASSIFICATION ===============
        logits = self.classification_head(features)  # (batch, num_classes)
        
        if return_features:
            return logits, features
        return logits
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract unified features (backbone only)
        
        Args:
            x: (batch, 15, 19, 19, 2) - raw CSD data
            
        Returns:
            features: (batch, 361, 32) - unified features
        """
        batch_size = x.shape[0]
        
        # =============== INPUT VALIDATION ===============
        expected_shape = (batch_size, 20, 19, 19, 2)
        if x.shape != expected_shape:
            raise ValueError(f"Expected input shape {expected_shape}, got {x.shape}")
        
        # =============== RESHAPE TO PAIR FORMAT ===============
        # (batch, 15, 19, 19, 2) ¡æ (batch, 361, 15, 2)
        x_pairs = x.permute(0, 2, 3, 1, 4).reshape(batch_size, 361, 20, 2)
        
        # =============== STAGE 1: STRUCTURED FEATURE EXTRACTION ===============
        extracted_features = self.feature_extraction(x_pairs)  # (batch, 361, 32)
        
        # =============== STAGE 2: GLOBAL ATTENTION ===============
        attended_features = self.global_attention(extracted_features)  # (batch, 361, 32)
        
        return attended_features
    
    def compute_pretrain_loss(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Pre-training loss °è»ê (reconstruction)
        
        Args:
            x: (batch, 15, 19, 19, 2) - original CSD data
            mask: (batch, 15, 19, 19, 2) - masking pattern
            
        Returns:
            total_loss: scalar tensor
            loss_breakdown: detailed loss components
        """
        # Apply masking to input
        masked_x = x * mask
        
        # Forward pass
        reconstructed = self.forward(masked_x)
        
        # Compute reconstruction loss
        total_loss, loss_breakdown = self.reconstruction_head.compute_simplified_loss(
            reconstructed, x, mask, return_breakdown=True
        )
        
        return total_loss, loss_breakdown
    
    def compute_classification_loss(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Fine-tuning loss °è»ê (classification)
        
        Args:
            x: (batch, 15, 19, 19, 2) - CSD data
            labels: (batch,) - class labels
            
        Returns:
            total_loss: scalar tensor
            loss_breakdown: detailed metrics
        """
        # Forward pass
        logits = self.forward_classification(x)
        
        # Classification loss
        ce_loss = F.cross_entropy(logits, labels)
        
        # Additional metrics
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean()
            
            # Class-wise accuracy
            class_accuracies = {}
            for class_idx in range(self.config.NUM_CLASSES):
                class_mask = (labels == class_idx)
                if class_mask.sum() > 0:
                    class_acc = (predictions[class_mask] == labels[class_mask]).float().mean()
                    class_accuracies[f'class_{class_idx}_accuracy'] = class_acc
        
        loss_breakdown = {
            'total_loss': ce_loss,
            'cross_entropy_loss': ce_loss,
            'accuracy': accuracy,
            'predictions': predictions,
            **class_accuracies
        }
        
        return ce_loss, loss_breakdown
    
    def load_pretrained_encoder(self, pretrain_checkpoint_path: str, strict: bool = False) -> bool:
        """
        Pre-trained encoder ·Îµå (fine-tuning¿ë)
        
        Args:
            pretrain_checkpoint_path: pre-trained model checkpoint path
            strict: whether to strictly match parameter names
            
        Returns:
            success: whether loading was successful
        """
        try:
            checkpoint = torch.load(pretrain_checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out reconstruction head parameters
            encoder_state_dict = {}
            for key, value in state_dict.items():
                if not key.startswith('reconstruction_head'):
                    encoder_state_dict[key] = value
            
            # Load encoder parameters
            missing_keys, unexpected_keys = self.load_state_dict(encoder_state_dict, strict=False)
            
            if not strict:
                # Filter out expected missing keys (reconstruction head, classification head)
                expected_missing = [k for k in missing_keys if 
                                  k.startswith('reconstruction_head') or k.startswith('classification_head')]
                unexpected_missing = [k for k in missing_keys if k not in expected_missing]
                
                if unexpected_missing:
                    print(f"Warning: Unexpected missing keys: {unexpected_missing}")
                
                if unexpected_keys:
                    print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
            print(f"? Pre-trained encoder loaded successfully!")
            print(f"   Checkpoint: {pretrain_checkpoint_path}")
            print(f"   Loaded parameters: {len(encoder_state_dict)}")
            
            return True
            
        except Exception as e:
            print(f"? Failed to load pre-trained encoder: {str(e)}")
            return False
    
    def freeze_encoder(self):
        """Encoder ÆÄ¶ó¹ÌÅÍ °íÁ¤ (fine-tuning ½Ã »ç¿ë)"""
        for param in self.feature_extraction.parameters():
            param.requires_grad = False
        for param in self.global_attention.parameters():
            param.requires_grad = False
        
        print("?? Encoder parameters frozen for fine-tuning")
    
    def unfreeze_encoder(self):
        """Encoder ÆÄ¶ó¹ÌÅÍ ÇØÁ¦"""
        for param in self.feature_extraction.parameters():
            param.requires_grad = True
        for param in self.global_attention.parameters():
            param.requires_grad = True
        
        print("?? Encoder parameters unfrozen")
    
    def _get_model_info(self) -> Dict:
        """¸ðµ¨ Á¤º¸ ¼öÁý"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Componentº° ÆÄ¶ó¹ÌÅÍ ¼ö
        component_params = {
            'feature_extraction': sum(p.numel() for p in self.feature_extraction.parameters()),
            'global_attention': sum(p.numel() for p in self.global_attention.parameters()),
        }
        
        if hasattr(self, 'reconstruction_head'):
            component_params['reconstruction_head'] = sum(p.numel() for p in self.reconstruction_head.parameters())
        
        if hasattr(self, 'classification_head'):
            component_params['classification_head'] = sum(p.numel() for p in self.classification_head.parameters())
        
        # ¸Þ¸ð¸® ÃßÁ¤ (float32 ±âÁØ)
        memory_mb = total_params * 4 / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'component_parameters': component_params,
            'memory_mb': memory_mb,
            'mode': self.mode
        }
    
    def get_model_analysis(self, sample_input: torch.Tensor) -> Dict:
        """¸ðµ¨ ÀüÃ¼ ºÐ¼® (Device È£È¯¼º º¸Àå)"""
        self.eval()
        
        # ?? Device È£È¯¼º: sample_inputÀ» ¸ðµ¨°ú °°Àº device·Î ÀÌµ¿
        if next(self.parameters()).device != sample_input.device:
            sample_input = sample_input.to(next(self.parameters()).device)
        
        with torch.no_grad():
            # Feature extraction ºÐ¼®
            features = self.get_features(sample_input)
            feature_stats = self.feature_extraction.get_feature_statistics(sample_input)
            
            # Global attention ºÐ¼®
            attention_stats = self.global_attention.get_attention_patterns(features)
            
            analysis = {
                'model_info': self.model_info,
                'feature_extraction_stats': feature_stats,
                'global_attention_stats': attention_stats,
                'feature_shape': list(features.shape),
                'input_shape': list(sample_input.shape),
                'device_info': {
                    'model_device': str(next(self.parameters()).device),
                    'input_device': str(sample_input.device),
                    'feature_device': str(features.device)
                }
            }
            
            # Reconstruction head ºÐ¼® (ÀÖ´Â °æ¿ì)
            if hasattr(self, 'reconstruction_head'):
                freq_analysis = self.reconstruction_head.get_frequency_analysis()
                analysis['reconstruction_stats'] = freq_analysis
            
            return analysis
    
    def save_model(self, save_path: str, epoch: Optional[int] = None, additional_info: Optional[Dict] = None):
        """¸ðµ¨ ÀúÀå"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_info': self.model_info,
            'mode': self.mode
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, save_path)
        print(f"?? Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, checkpoint_path: str, mode: Optional[str] = None) -> 'EEGConnectivityModel':
        """¸ðµ¨ ·Îµå"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        config = checkpoint.get('config')
        if config is None:
            config = EEGConfig()
            print("Warning: Config not found in checkpoint, using default")
        
        if mode is None:
            mode = checkpoint.get('mode', 'inference')
        
        model = cls(config=config, mode=mode)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"?? Model loaded from {checkpoint_path}")
        print(f"   Mode: {mode}")
        print(f"   Parameters: {model.model_info['total_parameters']:,}")
        
        return model

# Factory functions for different use cases

def create_pretrain_model(config: EEGConfig = None) -> EEGConnectivityModel:
    """Pre-training¿ë ¸ðµ¨ »ý¼º"""
    return EEGConnectivityModel(config=config, mode='pretrain')

def create_finetune_model(config: EEGConfig = None, pretrain_checkpoint: Optional[str] = None) -> EEGConnectivityModel:
    """Fine-tuning¿ë ¸ðµ¨ »ý¼º"""
    model = EEGConnectivityModel(config=config, mode='finetune')
    
    if pretrain_checkpoint:
        success = model.load_pretrained_encoder(pretrain_checkpoint)
        if not success:
            print("Warning: Failed to load pre-trained weights, training from scratch")
    
    return model

def create_inference_model(config: EEGConfig = None) -> EEGConnectivityModel:
    """Inference¿ë ¸ðµ¨ »ý¼º (¸ðµç head Æ÷ÇÔ)"""
    return EEGConnectivityModel(config=config, mode='inference')

# Backward compatibility
EEGHybridModel = EEGConnectivityModel

if __name__ == "__main__":
    print("="*80)
    print("??? UNIFIED EEG CONNECTIVITY MODEL")
    print("="*80)
    
    config = EEGConfig()
    
    # Pre-training ¸ðµ¨ Å×½ºÆ®
    print("\n1. Pre-training Model Test:")
    pretrain_model = create_pretrain_model(config)
    sample_input = torch.randn(2, 20, 19, 19, 2)
    
    reconstructed = pretrain_model(sample_input)
    print(f"   Input: {sample_input.shape}")
    print(f"   Reconstructed: {reconstructed.shape}")
    
    # Loss °è»ê Å×½ºÆ®
    mask = torch.ones_like(sample_input)
    mask = mask * (torch.rand_like(mask) > 0.5).float()  # 50% masking
    
    loss, loss_breakdown = pretrain_model.compute_pretrain_loss(sample_input, mask)
    print(f"   Pretrain Loss: {loss.item():.6f}")
    print(f"   Phase Error: {loss_breakdown['phase_error_degrees'].item():.1f}¡Æ")
    
    # Fine-tuning ¸ðµ¨ Å×½ºÆ®
    print("\n2. Fine-tuning Model Test:")
    finetune_model = create_finetune_model(config)
    labels = torch.randint(0, 2, (2,))
    
    logits = finetune_model.forward_classification(sample_input)
    print(f"   Input: {sample_input.shape}")
    print(f"   Logits: {logits.shape}")
    
    cls_loss, cls_breakdown = finetune_model.compute_classification_loss(sample_input, labels)
    print(f"   Classification Loss: {cls_loss.item():.6f}")
    print(f"   Accuracy: {cls_breakdown['accuracy'].item():.3f}")
    
    # ¸ðµ¨ ºÐ¼®
    print("\n3. Model Analysis:")
    analysis = pretrain_model.get_model_analysis(sample_input)
    
    print(f"   Total Parameters: {analysis['model_info']['total_parameters']:,}")
    print(f"   Memory Footprint: {analysis['model_info']['memory_mb']:.1f} MB")
    print(f"   Feature Variance Ratio: {analysis['feature_extraction_stats']['information_preservation']['variance_ratio']:.3f}")
    print(f"   Attention Entropy: {analysis['global_attention_stats']['attention_statistics']['entropy']:.3f}")
    
    # Component º° ÆÄ¶ó¹ÌÅÍ
    print(f"\n4. Component Parameters:")
    for component, params in analysis['model_info']['component_parameters'].items():
        print(f"   {component}: {params:,}")
    
    print("="*80)