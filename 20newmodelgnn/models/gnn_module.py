import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class ElectrodeGraphBuilder:
    """EEG Àü±Ø ±â¹Ý ±×·¡ÇÁ ±¸Á¶ »ý¼º"""
    
    def __init__(self, config):
        self.config = config
        self.gnn_config = config.GNN_CONFIG
        self.electrode_names = config.ELECTRODE_NAMES
        self.num_electrodes = len(self.electrode_names)
        
        # Àü±Ø À§Ä¡ Á¤º¸ (19°³ Àü±ØÀÇ 2D ÁÂÇ¥)
        self.electrode_positions = self._get_electrode_positions()
        
        # ±×·¡ÇÁ Å¸ÀÔº° ÀÎÁ¢ Çà·Ä »ý¼º
        self.adjacency_matrix = self._build_adjacency_matrix()
        
    def _get_electrode_positions(self):
        """19°³ Àü±ØÀÇ Ç¥ÁØ 10-20 ½Ã½ºÅÛ À§Ä¡"""
        positions = {
            'FP1': (-0.3, 0.9), 'FP2': (0.3, 0.9),
            'F7': (-0.7, 0.5), 'F3': (-0.3, 0.5), 'FZ': (0.0, 0.5), 'F4': (0.3, 0.5), 'F8': (0.7, 0.5),
            'T3': (-0.9, 0.0), 'C3': (-0.3, 0.0), 'CZ': (0.0, 0.0), 'C4': (0.3, 0.0), 'T4': (0.9, 0.0),
            'T5': (-0.7, -0.5), 'P3': (-0.3, -0.5), 'PZ': (0.0, -0.5), 'P4': (0.3, -0.5), 'T6': (0.7, -0.5),
            'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
        }
        
        return torch.tensor([positions[name] for name in self.electrode_names], dtype=torch.float32)
    
    def _build_adjacency_matrix(self):
        """±×·¡ÇÁ Å¸ÀÔ¿¡ µû¸¥ ÀÎÁ¢ Çà·Ä »ý¼º"""
        graph_type = self.gnn_config.get('graph_type', 'electrode_connectivity')
        
        if graph_type == 'fully_connected':
            # ¿ÏÀü ¿¬°á ±×·¡ÇÁ
            adj = torch.ones(self.num_electrodes, self.num_electrodes) - torch.eye(self.num_electrodes)
            
        elif graph_type == 'distance_based':
            # °Å¸® ±â¹Ý ¿¬°á
            threshold = self.gnn_config.get('distance_threshold', 3.0)
            adj = self._distance_based_adjacency(threshold)
            
        elif graph_type == 'electrode_connectivity':
            # Àü±Ø ¿¬°á¼º ±â¹Ý (±âº»)
            adj = self._electrode_connectivity_adjacency()
            
        else:
            adj = torch.eye(self.num_electrodes)  # ±âº»°ª
        
        return adj
    
    def _distance_based_adjacency(self, threshold):
        """°Å¸® ±â¹Ý ÀÎÁ¢ Çà·Ä"""
        distances = torch.cdist(self.electrode_positions, self.electrode_positions)
        adj = (distances <= threshold).float() - torch.eye(self.num_electrodes)
        return adj
    
    def _electrode_connectivity_adjacency(self):
        """Àü±Ø ¿¬°á¼º ±â¹Ý ÀÎÁ¢ Çà·Ä (10-20 ½Ã½ºÅÛ)"""
        # Àü±Ø °£ ¿¬°á ±ÔÄ¢ Á¤ÀÇ
        connections = [
            ('FP1', ['F7', 'F3']), ('FP2', ['F8', 'F4']),
            ('F7', ['FP1', 'F3', 'T3']), ('F3', ['FP1', 'F7', 'FZ', 'C3']),
            ('FZ', ['F3', 'F4', 'CZ']), ('F4', ['FP2', 'FZ', 'F8', 'C4']),
            ('F8', ['FP2', 'F4', 'T4']), ('T3', ['F7', 'C3', 'T5']),
            ('C3', ['F3', 'T3', 'CZ', 'P3']), ('CZ', ['FZ', 'C3', 'C4', 'PZ']),
            ('C4', ['F4', 'CZ', 'T4', 'P4']), ('T4', ['F8', 'C4', 'T6']),
            ('T5', ['T3', 'P3', 'O1']), ('P3', ['C3', 'T5', 'PZ', 'O1']),
            ('PZ', ['CZ', 'P3', 'P4']), ('P4', ['C4', 'PZ', 'T6', 'O2']),
            ('T6', ['T4', 'P4', 'O2']), ('O1', ['T5', 'P3']), ('O2', ['P4', 'T6'])
        ]
        
        adj = torch.zeros(self.num_electrodes, self.num_electrodes)
        electrode_idx = {name: i for i, name in enumerate(self.electrode_names)}
        
        for electrode, neighbors in connections:
            if electrode in electrode_idx:
                i = electrode_idx[electrode]
                for neighbor in neighbors:
                    if neighbor in electrode_idx:
                        j = electrode_idx[neighbor]
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0  # ¹«¹æÇâ ±×·¡ÇÁ
        
        return adj

class GraphAttentionLayer(nn.Module):
    """Custom Graph Attention Layer for EEG connectivity"""
    
    def __init__(self, input_dim, output_dim, num_heads=8, dropout=0.1, use_edge_features=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_edge_features = use_edge_features
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        
        # Edge features (if used)
        if use_edge_features:
            self.edge_encoder = nn.Linear(1, self.head_dim)  # Distance feature
        
        # Attention mechanism
        self.attention_dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(output_dim, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: (num_nodes, input_dim) - node features
            edge_index: (2, num_edges) - edge connections
            edge_attr: (num_edges, edge_dim) - edge features
        """
        batch_size, num_nodes, feature_dim = x.shape
        
        # Reshape for multi-head attention
        Q = self.W_q(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.einsum('bnhd,bmhd->bnmh', Q, K) / math.sqrt(self.head_dim)
        
        # Apply edge mask (only compute attention for connected nodes)
        if edge_index is not None:
            attention_mask = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)
            attention_mask[:, edge_index[0], edge_index[1]] = 1.0
            attention_scores = attention_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, float('-inf'))
        
        # Apply edge features
        if edge_attr is not None and self.use_edge_features:
            edge_features = self.edge_encoder(edge_attr)  # (num_edges, head_dim)
            # Add edge features to attention scores (simplified)
            for i, (src, dst) in enumerate(edge_index.t()):
                attention_scores[:, src, dst, :] += edge_features[i]
        
        # Softmax attention
        attention_weights = F.softmax(attention_scores, dim=2)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.einsum('bnmh,bmhd->bnhd', attention_weights, V)
        attended = attended.contiguous().view(batch_size, num_nodes, self.output_dim)
        
        # Output projection and residual connection
        output = self.output_proj(attended)
        output = self.layer_norm(output + x)  # Residual connection
        
        return output

class EEGGraphNeuralNetwork(nn.Module):
    """EEG¿ë Graph Neural Network ¸ðµâ"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.gnn_config = config.GNN_CONFIG
        self.input_dim = self.gnn_config['input_dim']
        self.output_dim = self.gnn_config['output_dim']
        
        # Graph builder
        self.graph_builder = ElectrodeGraphBuilder(config)
        
        # Learnable adjacency matrix
        if self.gnn_config.get('adjacency_type') == 'learnable':
            initial_adj = self.graph_builder.adjacency_matrix
            self.adjacency_params = nn.Parameter(initial_adj.clone())
        else:
            self.register_buffer('adjacency_matrix', self.graph_builder.adjacency_matrix)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        current_dim = self.input_dim
        
        for layer_config in self.gnn_config['gnn_layers']:
            layer_type = layer_config['type']
            hidden_dim = layer_config['hidden_dim']
            
            if layer_type == 'GAT':
                num_heads = layer_config.get('heads', 8)
                layer = GraphAttentionLayer(
                    current_dim, hidden_dim, num_heads,
                    dropout=self.gnn_config.get('attention_dropout', 0.1),
                    use_edge_features=self.gnn_config.get('use_edge_features', True)
                )
            elif layer_type == 'GCN':
                layer = nn.Linear(current_dim, hidden_dim)  # Simplified GCN
            
            self.gnn_layers.append(layer)
            current_dim = hidden_dim
        
        # Final projection
        if current_dim != self.output_dim:
            self.final_proj = nn.Linear(current_dim, self.output_dim)
        else:
            self.final_proj = nn.Identity()
        
        # Edge dropout
        self.edge_dropout = nn.Dropout(self.gnn_config.get('edge_dropout', 0.2))
        
        print(f"?? EEG Graph Neural Network:")
        print(f"   Input/Output: {self.input_dim} ¡æ {self.output_dim}")
        print(f"   GNN layers: {len(self.gnn_layers)}")
        print(f"   Graph type: {self.gnn_config.get('graph_type', 'electrode_connectivity')}")
        print(f"   Learnable adjacency: {self.gnn_config.get('adjacency_type') == 'learnable'}")
    
    def forward(self, x):
        """
        Args:
            x: (batch, 361, input_dim) - pair features
        Returns:
            (batch, 361, output_dim) - graph-processed features
        """
        batch_size, num_pairs, feature_dim = x.shape
        
        # Convert pair format to electrode graph format
        # (batch, 361, dim) ¡æ (batch, 19, 19, dim) ¡æ electrode processing
        x_electrode = x.view(batch_size, 19, 19, feature_dim)
        
        # Get adjacency matrix
        if hasattr(self, 'adjacency_params'):
            adj_matrix = torch.sigmoid(self.adjacency_params)  # Learnable
        else:
            adj_matrix = self.adjacency_matrix
        
        # Create edge index from adjacency matrix
        edge_index = adj_matrix.nonzero().t().contiguous()
        
        # Apply edge dropout during training
        if self.training:
            edge_mask = torch.rand(edge_index.shape[1], device=edge_index.device) > self.gnn_config.get('edge_dropout', 0.2)
            edge_index = edge_index[:, edge_mask]
        
        # Process each pair location through electrode graph
        processed_pairs = []
        
        for i in range(19):
            for j in range(19):
                pair_idx = i * 19 + j
                pair_features = x[:, pair_idx, :]  # (batch, feature_dim)
                
                # Create electrode-wise representation for this pair
                electrode_features = torch.zeros(batch_size, 19, feature_dim, device=x.device)
                electrode_features[:, i, :] = pair_features  # Source electrode
                electrode_features[:, j, :] += pair_features  # Target electrode
                
                # Apply GNN layers
                for layer in self.gnn_layers:
                    if isinstance(layer, GraphAttentionLayer):
                        electrode_features = layer(electrode_features, edge_index)
                    else:  # GCN or Linear
                        electrode_features = F.relu(layer(electrode_features))
                
                # Extract processed pair representation
                processed_pair = (electrode_features[:, i, :] + electrode_features[:, j, :]) / 2
                processed_pairs.append(processed_pair)
        
        # Stack all processed pairs
        output = torch.stack(processed_pairs, dim=1)  # (batch, 361, output_dim)
        
        # Final projection
        output = self.final_proj(output)
        
        # Residual connection
        if self.gnn_config.get('use_residual', True) and output.shape == x.shape:
            output = output + x
        
        return output
    
    def get_graph_analysis(self):
        """±×·¡ÇÁ ±¸Á¶ ºÐ¼®"""
        if hasattr(self, 'adjacency_params'):
            adj_matrix = torch.sigmoid(self.adjacency_params)
        else:
            adj_matrix = self.adjacency_matrix
        
        with torch.no_grad():
            num_edges = adj_matrix.sum().item()
            sparsity = num_edges / (19 * 19)
            
            analysis = {
                'num_nodes': 19,
                'num_edges': int(num_edges),
                'sparsity': sparsity,
                'density': sparsity,
                'learnable_adjacency': hasattr(self, 'adjacency_params'),
                'total_parameters': sum(p.numel() for p in self.parameters())
            }
            
            if hasattr(self, 'adjacency_params'):
                analysis['adjacency_weights_stats'] = {
                    'mean': adj_matrix.mean().item(),
                    'std': adj_matrix.std().item(),
                    'min': adj_matrix.min().item(),
                    'max': adj_matrix.max().item()
                }
        
        return analysis