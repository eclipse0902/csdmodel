#!/usr/bin/env python3
"""
visualize_gnn.py - µ¶¸³ ½ÇÇà GNN ½Ã°¢È­

»ç¿ë¹ý:
1. ¸ðµ¨ °æ·Î¸¸ ¼öÁ¤
2. python visualize_gnn.py ½ÇÇà
3. ÀÚµ¿À¸·Î ¸ðµç ºÐ¼® ¹× ½Ã°¢È­ »ý¼º

°£´ÜÇÏ°í µ¶¸³ÀûÀÎ GNN »óÅÂ ºÐ¼® µµ±¸
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
import os
import sys
from datetime import datetime

# ÇÁ·ÎÁ§Æ® °æ·Î Ãß°¡ (ÇÊ¿äÇÑ °æ¿ì)
sys.path.append('.')
sys.path.append('./models')
sys.path.append('./config')

# ================================
# ?? ¼³Á¤ ºÎºÐ - ¿©±â¸¸ ¼öÁ¤ÇÏ¼¼¿ä!
# ================================

MODEL_PATH = "/home/mjkang/cbramod/20newmodelgnn/checkpoints/best_pretrain_model.pth"  # ?? ¸ðµ¨ °æ·Î
SAVE_DIR = "./gnn_analysis"                           # °á°ú ÀúÀå Æú´õ
SAMPLE_DATA_SHAPE = (1, 20, 19, 19, 2)              # »ùÇÃ µ¥ÀÌÅÍ Å©±â

# ================================

class SimpleGNNVisualizer:
    """°£´ÜÇÏ°í µ¶¸³ÀûÀÎ GNN ½Ã°¢È­ µµ±¸"""
    
    def __init__(self):
        # EEG Àü±Ø À§Ä¡ (19°³, 10-20 ½Ã½ºÅÛ)
        self.electrode_positions = {
            'FP1': (-0.3, 0.9), 'FP2': (0.3, 0.9),
            'F7': (-0.7, 0.5), 'F3': (-0.3, 0.5), 'FZ': (0.0, 0.5), 
            'F4': (0.3, 0.5), 'F8': (0.7, 0.5),
            'T3': (-0.9, 0.0), 'C3': (-0.3, 0.0), 'CZ': (0.0, 0.0), 
            'C4': (0.3, 0.0), 'T4': (0.9, 0.0),
            'T5': (-0.7, -0.5), 'P3': (-0.3, -0.5), 'PZ': (0.0, -0.5), 
            'P4': (0.3, -0.5), 'T6': (0.7, -0.5),
            'O1': (-0.3, -0.9), 'O2': (0.3, -0.9)
        }
        
        self.electrode_names = [
            'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'
        ]
        
        # ³ú ¿µ¿ª »ö»ó
        self.region_colors = {
            'frontal': '#FF6B6B',    # »¡°­
            'central': '#4ECDC4',    # Ã»·Ï
            'parietal': '#45B7D1',   # ÆÄ¶û
            'temporal': '#96CEB4',   # ÃÊ·Ï
            'occipital': '#FFEAA7'   # ³ë¶û
        }
        
        self.brain_regions = {
            'frontal': [0, 1, 2, 3, 10, 11, 16],
            'central': [4, 5, 17],
            'parietal': [6, 7, 18],
            'temporal': [12, 13, 14, 15],
            'occipital': [8, 9]
        }
        
        print("?? Simple GNN Visualizer initialized")
    
    def extract_adjacency_matrix(self, model):
        """¸ðµ¨¿¡¼­ ÀÎÁ¢ Çà·Ä ÃßÃâ"""
        try:
            # 1. global_attention.spatial_processor °æ·Î
            if hasattr(model, 'global_attention'):
                if hasattr(model.global_attention, 'spatial_processor'):
                    gnn = model.global_attention.spatial_processor
                    if hasattr(gnn, 'adjacency_params'):
                        adj = torch.sigmoid(gnn.adjacency_params).detach().cpu()
                        print("? Found learnable adjacency matrix")
                        return adj
                    elif hasattr(gnn, 'adjacency_matrix'):
                        adj = gnn.adjacency_matrix.detach().cpu()
                        print("? Found fixed adjacency matrix")
                        return adj
            
            # 2. Á÷Á¢ spatial_processor °æ·Î
            if hasattr(model, 'spatial_processor'):
                gnn = model.spatial_processor
                if hasattr(gnn, 'adjacency_params'):
                    adj = torch.sigmoid(gnn.adjacency_params).detach().cpu()
                    print("? Found learnable adjacency matrix")
                    return adj
                elif hasattr(gnn, 'adjacency_matrix'):
                    adj = gnn.adjacency_matrix.detach().cpu()
                    print("? Found fixed adjacency matrix")
                    return adj
            
            # 3. ¸ðµç ¸ðµâ °Ë»ö
            for name, module in model.named_modules():
                if hasattr(module, 'adjacency_params'):
                    adj = torch.sigmoid(module.adjacency_params).detach().cpu()
                    print(f"? Found adjacency matrix in {name}")
                    return adj
                elif hasattr(module, 'adjacency_matrix'):
                    adj = module.adjacency_matrix.detach().cpu()
                    print(f"? Found adjacency matrix in {name}")
                    return adj
            
            print("??  No adjacency matrix found")
            return None
            
        except Exception as e:
            print(f"? Error extracting adjacency matrix: {e}")
            return None
    
    def analyze_graph_stats(self, adj_matrix):
        """±×·¡ÇÁ Åë°è ºÐ¼®"""
        if isinstance(adj_matrix, torch.Tensor):
            adj = adj_matrix.numpy()
        else:
            adj = adj_matrix
        
        # ´ë°¢¼± Á¦°Å
        adj_no_diag = adj.copy()
        np.fill_diagonal(adj_no_diag, 0)
        
        # ±âº» Åë°è
        n_nodes = adj.shape[0]
        n_edges = np.sum(adj_no_diag > 0.1)
        density = n_edges / (n_nodes * (n_nodes - 1))
        
        stats = {
            'nodes': n_nodes,
            'edges': int(n_edges),
            'density': density,
            'sparsity': 1 - density,
            'mean_strength': np.mean(adj_no_diag),
            'max_strength': np.max(adj_no_diag),
            'std_strength': np.std(adj_no_diag)
        }
        
        return stats, adj_no_diag
    
    def plot_electrode_layout(self, save_path):
        """Àü±Ø ¹èÄ¡µµ"""
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
        
        # ¸Ó¸® À±°û
        head_circle = Circle((0, 0), 1.1, fill=False, color='black', linewidth=2)
        ax.add_patch(head_circle)
        
        # ÄÚ
        ax.plot([0, 0], [1.1, 1.3], 'k-', linewidth=3)
        
        # ±Í
        left_ear = Circle((-1.1, 0), 0.1, fill=False, color='black', linewidth=2)
        right_ear = Circle((1.1, 0), 0.1, fill=False, color='black', linewidth=2)
        ax.add_patch(left_ear)
        ax.add_patch(right_ear)
        
        # Àü±Øµé
        for i, name in enumerate(self.electrode_names):
            x, y = self.electrode_positions[name]
            
            # ³ú ¿µ¿ª »ö»ó
            color = 'gray'
            for region, indices in self.brain_regions.items():
                if i in indices:
                    color = self.region_colors[region]
                    break
            
            electrode = Circle((x, y), 0.08, color=color, alpha=0.8, zorder=3)
            ax.add_patch(electrode)
            ax.text(x, y, name, ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white', zorder=4)
        
        # ¹ü·Ê
        legend_elements = []
        for region, color in self.region_colors.items():
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=10, 
                                            label=region.capitalize()))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1))
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('EEG Electrode Layout (10-20 System)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"?? Electrode layout saved: {save_path}")
    
    def plot_adjacency_heatmap(self, adj_matrix, save_path):
        """ÀÎÁ¢ Çà·Ä È÷Æ®¸Ê"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if isinstance(adj_matrix, torch.Tensor):
            adj = adj_matrix.numpy()
        else:
            adj = adj_matrix
        
        # 1. ÀüÃ¼ Çà·Ä
        im1 = ax1.imshow(adj, cmap='viridis', aspect='equal')
        ax1.set_title('Full Adjacency Matrix', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Target Electrode')
        ax1.set_ylabel('Source Electrode')
        ax1.set_xticks(range(len(self.electrode_names)))
        ax1.set_yticks(range(len(self.electrode_names)))
        ax1.set_xticklabels(self.electrode_names, rotation=45, ha='right')
        ax1.set_yticklabels(self.electrode_names)
        plt.colorbar(im1, ax=ax1, shrink=0.6)
        
        # 2. ´ë°¢¼± Á¦°Å
        adj_no_diag = adj.copy()
        np.fill_diagonal(adj_no_diag, 0)
        
        im2 = ax2.imshow(adj_no_diag, cmap='plasma', aspect='equal')
        ax2.set_title('Adjacency Matrix (No Self-Connections)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Target Electrode')
        ax2.set_ylabel('Source Electrode')
        ax2.set_xticks(range(len(self.electrode_names)))
        ax2.set_yticks(range(len(self.electrode_names)))
        ax2.set_xticklabels(self.electrode_names, rotation=45, ha='right')
        ax2.set_yticklabels(self.electrode_names)
        plt.colorbar(im2, ax=ax2, shrink=0.6)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"?? Adjacency heatmap saved: {save_path}")
    
    def plot_brain_network(self, adj_matrix, threshold=0.5, save_path=None):
        """³ú ¿¬°á¼º ³×Æ®¿öÅ©"""
        if isinstance(adj_matrix, torch.Tensor):
            adj = adj_matrix.numpy()
        else:
            adj = adj_matrix
        
        fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
        
        # ¸Ó¸® À±°û
        head_circle = Circle((0, 0), 1.1, fill=False, color='black', linewidth=2)
        ax.add_patch(head_circle)
        ax.plot([0, 0], [1.1, 1.3], 'k-', linewidth=3)
        
        # ´ë°¢¼± Á¦°Å
        adj_no_diag = adj.copy()
        np.fill_diagonal(adj_no_diag, 0)
        
        # ÀÚµ¿ ÀÓ°è°ª (»óÀ§ 30% ¿¬°á)
        if np.any(adj_no_diag > 0):
            auto_threshold = np.percentile(adj_no_diag[adj_no_diag > 0], 70)
            threshold = max(threshold, auto_threshold)
        
        # ¿¬°á¼± ±×¸®±â
        n_connections = 0
        for i in range(len(self.electrode_names)):
            for j in range(i+1, len(self.electrode_names)):
                if adj_no_diag[i, j] > threshold:
                    x1, y1 = self.electrode_positions[self.electrode_names[i]]
                    x2, y2 = self.electrode_positions[self.electrode_names[j]]
                    
                    strength = adj_no_diag[i, j]
                    alpha = 0.3 + 0.7 * (strength / adj_no_diag.max())
                    linewidth = 0.5 + 3 * (strength / adj_no_diag.max())
                    
                    ax.plot([x1, x2], [y1, y2], 'b-', 
                           alpha=alpha, linewidth=linewidth, zorder=1)
                    n_connections += 1
        
        # Àü±Øµé
        for i, name in enumerate(self.electrode_names):
            x, y = self.electrode_positions[name]
            
            # ³ú ¿µ¿ª »ö»ó
            color = 'gray'
            for region, indices in self.brain_regions.items():
                if i in indices:
                    color = self.region_colors[region]
                    break
            
            # ³ëµå Å©±â (¿¬°á °­µµ¿¡ ºñ·Ê)
            node_strength = adj_no_diag[i, :].sum() + adj_no_diag[:, i].sum()
            total_strength = adj_no_diag.sum() * 2
            node_size = 0.06 + 0.04 * (node_strength / total_strength) if total_strength > 0 else 0.06
            
            electrode = Circle((x, y), node_size, color=color, alpha=0.9, zorder=3)
            ax.add_patch(electrode)
            ax.text(x, y, name, ha='center', va='center', 
                   fontsize=7, fontweight='bold', color='white', zorder=4)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Á¦¸ñ°ú Åë°è
        density = np.sum(adj_no_diag > threshold) / (adj_no_diag.size - len(adj_no_diag))
        ax.set_title(f'Brain Connectivity Network\n'
                    f'Connections: {n_connections} | Density: {density:.3f} | Threshold: {threshold:.3f}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"?? Brain network saved: {save_path}")
        else:
            plt.show()
        
        return n_connections, density
    
    def create_summary_dashboard(self, adj_matrix, stats, save_path):
        """¿ä¾à ´ë½Ãº¸µå"""
        fig = plt.figure(figsize=(20, 12))
        
        # 2x3 ·¹ÀÌ¾Æ¿ô
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Àü±Ø ¹èÄ¡
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_mini_electrode_layout(ax1)
        
        # 2. ÀÎÁ¢ Çà·Ä
        ax2 = fig.add_subplot(gs[0, 1])
        adj_no_diag = stats[1]  # adj_no_diag from analyze_graph_stats
        im = ax2.imshow(adj_no_diag, cmap='viridis', aspect='equal')
        ax2.set_title('Adjacency Matrix', fontweight='bold')
        ax2.set_xticks([])
        ax2.set_yticks([])
        plt.colorbar(im, ax=ax2, shrink=0.6)
        
        # 3. ³ú ³×Æ®¿öÅ©
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_mini_brain_network(ax3, adj_no_diag)
        
        # 4. Åë°è ¿ä¾à
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_statistics_text(ax4, stats[0])
        
        # 5. ¿¬°á °­µµ È÷½ºÅä±×·¥
        ax5 = fig.add_subplot(gs[1, 1])
        non_zero = adj_no_diag[adj_no_diag > 0]
        if len(non_zero) > 0:
            ax5.hist(non_zero, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax5.set_title('Connection Strength Distribution', fontweight='bold')
            ax5.set_xlabel('Connection Strength')
            ax5.set_ylabel('Frequency')
            ax5.grid(True, alpha=0.3)
        
        # 6. ³ú ¿µ¿ªº° ¿¬°á¼º
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_region_analysis(ax6, adj_no_diag)
        
        plt.suptitle('GNN Analysis Dashboard', fontsize=18, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"?? Summary dashboard saved: {save_path}")
    
    def _plot_mini_electrode_layout(self, ax):
        """¹Ì´Ï Àü±Ø ¹èÄ¡µµ"""
        head_circle = Circle((0, 0), 1.1, fill=False, color='black', linewidth=1)
        ax.add_patch(head_circle)
        
        for i, name in enumerate(self.electrode_names):
            x, y = self.electrode_positions[name]
            color = 'gray'
            for region, indices in self.brain_regions.items():
                if i in indices:
                    color = self.region_colors[region]
                    break
            
            electrode = Circle((x, y), 0.06, color=color, alpha=0.8)
            ax.add_patch(electrode)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Electrode Layout', fontweight='bold')
    
    def _plot_mini_brain_network(self, ax, adj_matrix):
        """¹Ì´Ï ³ú ³×Æ®¿öÅ©"""
        head_circle = Circle((0, 0), 1.1, fill=False, color='black', linewidth=1)
        ax.add_patch(head_circle)
        
        threshold = np.percentile(adj_matrix[adj_matrix > 0], 80) if np.any(adj_matrix > 0) else 0
        
        for i in range(len(self.electrode_names)):
            for j in range(i+1, len(self.electrode_names)):
                if adj_matrix[i, j] > threshold:
                    x1, y1 = self.electrode_positions[self.electrode_names[i]]
                    x2, y2 = self.electrode_positions[self.electrode_names[j]]
                    ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.6, linewidth=1)
        
        for i, name in enumerate(self.electrode_names):
            x, y = self.electrode_positions[name]
            color = 'gray'
            for region, indices in self.brain_regions.items():
                if i in indices:
                    color = self.region_colors[region]
                    break
            
            electrode = Circle((x, y), 0.05, color=color, alpha=0.8)
            ax.add_patch(electrode)
        
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Brain Network', fontweight='bold')
    
    def _plot_statistics_text(self, ax, stats):
        """Åë°è ÅØ½ºÆ®"""
        ax.axis('off')
        
        text = f"""
GRAPH STATISTICS

Nodes: {stats['nodes']}
Edges: {stats['edges']}
Density: {stats['density']:.4f}
Sparsity: {stats['sparsity']:.4f}

CONNECTION STRENGTH
Mean: {stats['mean_strength']:.4f}
Max: {stats['max_strength']:.4f}
Std: {stats['std_strength']:.4f}

INTERPRETATION
"""
        
        # ÇØ¼® Ãß°¡
        if stats['density'] > 0.3:
            text += "??  High density (may overfit)\n"
        elif stats['density'] < 0.05:
            text += "??  Low density (may underfit)\n"
        else:
            text += "? Good density range\n"
        
        if stats['mean_strength'] > 0.7:
            text += "?? Strong connections\n"
        elif stats['mean_strength'] < 0.1:
            text += "??  Weak connections\n"
        else:
            text += "? Moderate connections\n"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        ax.set_title('Summary Statistics', fontweight='bold')
    
    def _plot_region_analysis(self, ax, adj_matrix):
        """³ú ¿µ¿ªº° ºÐ¼®"""
        region_internal = []
        region_names = []
        
        for region, indices in self.brain_regions.items():
            internal_adj = adj_matrix[np.ix_(indices, indices)]
            internal_density = np.sum(internal_adj > 0.1) / internal_adj.size
            region_internal.append(internal_density)
            region_names.append(region.capitalize())
        
        bars = ax.bar(region_names, region_internal, color=list(self.region_colors.values()), alpha=0.7)
        ax.set_title('Regional Internal Connectivity', fontweight='bold')
        ax.set_ylabel('Internal Density')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # °ª Ç¥½Ã
        for bar, val in zip(bars, region_internal):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

def load_model(model_path):
    """¸ðµ¨ ·Îµå"""
    try:
        print(f"?? Loading model from: {model_path}")
        
        # Ã¼Å©Æ÷ÀÎÆ® ·Îµå
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Config ÃßÃâ
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("? Config found in checkpoint")
        else:
            print("??  No config in checkpoint, using default")
            from config import EEGConfig
            config = EEGConfig()
        
        # ¸ðµ¨ »ý¼º
        from models.hybrid_model import EEGConnectivityModel
        
        # ¸ðµå °áÁ¤
        if 'mode' in checkpoint:
            mode = checkpoint['mode']
        else:
            mode = 'inference'
        
        model = EEGConnectivityModel(config=config, mode=mode)
        
        # »óÅÂ ·Îµå
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"? Model loaded successfully!")
        print(f"   Mode: {mode}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
        
    except Exception as e:
        print(f"? Failed to load model: {e}")
        return None

def main():
    """¸ÞÀÎ ½ÇÇà ÇÔ¼ö"""
    print("?? GNN Visualizer - Standalone Mode")
    print("="*50)
    
    # Ã¼Å©Æ÷ÀÎÆ® °æ·Î È®ÀÎ
    if not os.path.exists(MODEL_PATH):
        print(f"? Model not found: {MODEL_PATH}")
        print("?? Please update MODEL_PATH in the script")
        return
    
    # °á°ú Æú´õ »ý¼º
    os.makedirs(SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ¸ðµ¨ ·Îµå
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    # ½Ã°¢È­ µµ±¸ ÃÊ±âÈ­
    visualizer = SimpleGNNVisualizer()
    
    # ÀÎÁ¢ Çà·Ä ÃßÃâ
    print("\n?? Extracting adjacency matrix...")
    adj_matrix = visualizer.extract_adjacency_matrix(model)
    
    if adj_matrix is None:
        print("? Could not extract adjacency matrix")
        return
    
    # Åë°è ºÐ¼®
    print("\n?? Analyzing graph statistics...")
    stats, adj_no_diag = visualizer.analyze_graph_stats(adj_matrix)
    
    print(f"?? Graph Statistics:")
    print(f"   Nodes: {stats['nodes']}")
    print(f"   Edges: {stats['edges']}")
    print(f"   Density: {stats['density']:.4f}")
    print(f"   Mean strength: {stats['mean_strength']:.4f}")
    
    # ½Ã°¢È­ »ý¼º
    print(f"\n?? Creating visualizations...")
    
    # 1. Àü±Ø ¹èÄ¡µµ
    layout_path = f"{SAVE_DIR}/electrode_layout_{timestamp}.png"
    visualizer.plot_electrode_layout(layout_path)
    
    # 2. ÀÎÁ¢ Çà·Ä È÷Æ®¸Ê
    heatmap_path = f"{SAVE_DIR}/adjacency_heatmap_{timestamp}.png"
    visualizer.plot_adjacency_heatmap(adj_matrix, heatmap_path)
    
    # 3. ³ú ³×Æ®¿öÅ©
    network_path = f"{SAVE_DIR}/brain_network_{timestamp}.png"
    n_conn, density = visualizer.plot_brain_network(adj_matrix, save_path=network_path)
    
    # 4. ¿ä¾à ´ë½Ãº¸µå
    dashboard_path = f"{SAVE_DIR}/summary_dashboard_{timestamp}.png"
    visualizer.create_summary_dashboard(adj_matrix, (stats, adj_no_diag), dashboard_path)
    
    # 5. ÅØ½ºÆ® ¸®Æ÷Æ®
    report_path = f"{SAVE_DIR}/gnn_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("GNN ANALYSIS REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_PATH}\n\n")
        
        f.write("GRAPH STATISTICS:\n")
        f.write("-"*30 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nCONNECTIONS DISPLAYED: {n_conn}\n")
        f.write(f"EFFECTIVE DENSITY: {density:.4f}\n\n")
        
        f.write("GENERATED FILES:\n")
        f.write("-"*30 + "\n")
        f.write(f"Electrode Layout: {layout_path}\n")
        f.write(f"Adjacency Heatmap: {heatmap_path}\n")
        f.write(f"Brain Network: {network_path}\n")
        f.write(f"Summary Dashboard: {dashboard_path}\n")
        f.write(f"Report: {report_path}\n")
    
    print(f"\n? Analysis complete!")
    print(f"?? Results saved in: {SAVE_DIR}")
    print(f"?? Graph density: {stats['density']:.4f}")
    print(f"?? Active connections: {n_conn}")
    print(f"?? Mean connection strength: {stats['mean_strength']:.4f}")
    
    # ÇØ¼® Ãâ·Â
    print(f"\n?? Interpretation:")
    if stats['density'] > 0.3:
        print("   ??  High density - model may be overfitting")
    elif stats['density'] < 0.05:
        print("   ??  Low density - model may be underfitting")  
    else:
        print("   ? Good density range - model seems well-balanced")
    
    if stats['mean_strength'] > 0.7:
        print("   ?? Strong connections - high confidence in learned patterns")
    elif stats['mean_strength'] < 0.1:
        print("   ??  Weak connections - model may need more training")
    else:
        print("   ? Moderate connections - reasonable learning progress")
    
    print(f"\n?? Quick Assessment:")
    score = 0
    if 0.05 <= stats['density'] <= 0.3:
        score += 1
    if 0.1 <= stats['mean_strength'] <= 0.7:
        score += 1
    if stats['edges'] > 10:
        score += 1
    
    if score == 3:
        print("   ?? Excellent - Model shows healthy graph learning!")
    elif score == 2:
        print("   ?? Good - Model is learning reasonable patterns")
    elif score == 1:
        print("   ?? Fair - Model needs more training or tuning")
    else:
        print("   ?? Poor - Consider adjusting hyperparameters")

if __name__ == "__main__":
    main()

"""
?? »ç¿ë¹ý:

1. ¸ðµ¨ °æ·Î ¼³Á¤:
   MODEL_PATH = "./checkpoints/best_pretrain_model.pth"

2. ½ÇÇà:
   python visualize_gnn.py

3. °á°ú È®ÀÎ:
   ./gnn_analysis/ Æú´õ¿¡ ¸ðµç ºÐ¼® °á°ú ÀúÀå

?? Ä¿½ºÅÍ¸¶ÀÌÂ¡:
- MODEL_PATH: ºÐ¼®ÇÒ ¸ðµ¨ °æ·Î
- SAVE_DIR: °á°ú ÀúÀå Æú´õ
- threshold °ª Á¶Á¤À¸·Î ¿¬°á¼± Ç¥½Ã ¼öÁØ º¯°æ

?? »ý¼ºµÇ´Â ÆÄÀÏµé:
- electrode_layout_*.png: Àü±Ø ¹èÄ¡µµ
- adjacency_heatmap_*.png: ÀÎÁ¢ Çà·Ä È÷Æ®¸Ê
- brain_network_*.png: ³ú ¿¬°á¼º ³×Æ®¿öÅ©
- summary_dashboard_*.png: Á¾ÇÕ ´ë½Ãº¸µå
- gnn_report_*.txt: ÅØ½ºÆ® ºÐ¼® ¸®Æ÷Æ®

?? ÇØ¼® °¡ÀÌµå:
- Density 0.05-0.3: ÀûÀýÇÑ Èñ¼Ò¼º ?
- Mean strength 0.1-0.7: °Ç°­ÇÑ ¿¬°á °­µµ ?  
- °­ÇÑ ¿¬°áÀÌ ÀÇ¹Ì ÀÖ´Â Àü±Ø ½ÖÀÎÁö È®ÀÎ
- ³ú ¿µ¿ªº° ¿¬°á ÆÐÅÏÀÇ »ý¹°ÇÐÀû Å¸´ç¼º °ËÅä
"""