"""
PanGu Drug Model Architecture Visualization
Generates a comprehensive diagram showing the complete cVAE architecture
for molecular graph to SELFIES translation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_model_diagram():
    """Create a detailed architecture diagram for the PanGu Drug Model"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E8F4FD',
        'encoder': '#FFE5B4',
        'latent': '#D4EDDA',
        'decoder': '#F8D7DA',
        'output': '#D1ECF1',
        'arrows': '#2C3E50',
        'text': '#2C3E50'
    }
    
    # Title
    ax.text(10, 19.5, 'PanGu Drug Model: cVAE Architecture', 
            fontsize=24, fontweight='bold', ha='center', color=colors['text'])
    ax.text(10, 19.0, 'Molecular Graph → Latent Space → SELFIES String', 
            fontsize=16, ha='center', color=colors['text'])
    
    # Input Section
    ax.text(3, 17.5, 'Input: Molecular Graph', fontsize=16, fontweight='bold', ha='center')
    
    # Graph representation
    graph_box = FancyBboxPatch((0.5, 15.5), 5, 1.5, boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(graph_box)
    ax.text(3, 16.25, 'Graph Structure\n• Nodes (Atoms)\n• Edges (Bonds)\n• Node Features\n  - Atomic Number\n  - Degree\n  - Charge\n  - Hybridization\n  - Aromaticity\n  - Radicals', 
            fontsize=10, ha='center', va='center')
    
    # Encoder Section
    ax.text(10, 17.5, 'Graph Transformer Encoder', fontsize=16, fontweight='bold', ha='center')
    
    # Encoder layers visualization
    y_start = 15
    encoder_layers = []
    for i in range(10):
        if i < 8:
            color = colors['encoder']
            alpha = 1.0
        else:
            color = 'lightgray'
            alpha = 0.5
            
        layer = FancyBboxPatch((8.5, y_start - i*0.6), 3, 0.4, 
                             boxstyle="round,pad=0.05", facecolor=color, 
                             edgecolor='black', alpha=alpha, linewidth=1.5)
        ax.add_patch(layer)
        encoder_layers.append(layer)
        
        if i in [0, 1, 2, 3, 4, 5, 7, 9]:  # Selected layers
            ax.text(10, y_start - i*0.6 + 0.2, f'Layer {i+1}*', fontsize=9, ha='center', fontweight='bold')
        else:
            ax.text(10, y_start - i*0.6 + 0.2, f'Layer {i+1}', fontsize=9, ha='center')
    
    # Encoder details
    ax.text(12.5, 15.5, 'TransformerConv\n+ BatchNorm\n+ ReLU\nMulti-head Attention\n8×256 output', 
            fontsize=9, ha='left', va='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', edgecolor='gray'))
    
    # Latent Space Section
    ax.text(3, 8.5, 'Variational Latent Space', fontsize=16, fontweight='bold', ha='center')
    
    # Latent space boxes
    mu_box = FancyBboxPatch((1, 6.5), 4, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['latent'], edgecolor='green', linewidth=2)
    ax.add_patch(mu_box)
    ax.text(3, 7, 'μ (Mean)\nLatent Vector\n128-dim', fontsize=11, ha='center', fontweight='bold')
    
    var_box = FancyBboxPatch((1, 5), 4, 1, boxstyle="round,pad=0.1", 
                           facecolor=colors['latent'], edgecolor='green', linewidth=2)
    ax.add_patch(var_box)
    ax.text(3, 5.5, 'log(σ²) (Log Variance)\nLatent Vector\n128-dim', fontsize=11, ha='center', fontweight='bold')
    
    # Reparameterization trick
    reparam_box = FancyBboxPatch((0.5, 3.5), 5, 1, boxstyle="round,pad=0.1", 
                                facecolor='lightyellow', edgecolor='orange', linewidth=2)
    ax.add_patch(reparam_box)
    ax.text(3, 4, 'Reparameterization\nz = μ + ε·σ\nε ~ N(0,1)', fontsize=11, ha='center', fontweight='bold')
    
    # Decoder Section
    ax.text(10, 8.5, 'Transformer Decoder', fontsize=16, fontweight='bold', ha='center')
    
    # Decoder layers
    decoder_y_start = 6.5
    for i in range(6):
        if i == 0:
            label = f'Layer {i+1}\n(Absolute PE)'
        else:
            label = f'Layer {i+1}\n(Relative PE)'
            
        layer = FancyBboxPatch((8.5, decoder_y_start - i*0.6), 3, 0.4, 
                             boxstyle="round,pad=0.05", facecolor=colors['decoder'], 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(layer)
        ax.text(10, decoder_y_start - i*0.6 + 0.2, label, fontsize=9, ha='center')
    
    # Decoder details
    ax.text(12.5, 5.5, 'Transformer Decoder\n+ Positional Encoding\nMulti-head Attention\nCross-attention\n128-dim hidden', 
            fontsize=9, ha='left', va='center', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='white', edgecolor='gray'))
    
    # Output Section
    ax.text(16.5, 8.5, 'Output: SELFIES', fontsize=16, fontweight='bold', ha='center')
    
    output_box = FancyBboxPatch((14.5, 6.5), 4, 1.5, boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(16.5, 7.25, 'SELFIES String\nSMILES-like\nRepresentation\nCharacter Sequence\nOutput Vocab: 128', 
            fontsize=10, ha='center', va='center')
    
    # Condition Vector
    cond_box = FancyBboxPatch((14.5, 4.5), 4, 1, boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='purple', linewidth=2, alpha=0.7)
    ax.add_patch(cond_box)
    ax.text(16.5, 5, 'Condition Vector\n(Placeholder)\nControl Generation', fontsize=10, ha='center')
    
    # Arrows
    arrow_props = dict(arrowstyle='-|>', color=colors['arrows'], lw=2)
    
    # Input to Encoder
    ax.annotate('', xy=(8.5, 16.25), xytext=(5.5, 16.25), 
                arrowprops=arrow_props)
    
    # Encoder to Latent
    ax.annotate('', xy=(3, 13), xytext=(10, 9.5), 
                arrowprops=dict(arrowstyle='-|>', color='green', lw=2))
    
    # Latent to Reparameterization
    ax.annotate('', xy=(3, 6.5), xytext=(3, 4.5), 
                arrowprops=dict(arrowstyle='-|>', color='orange', lw=2))
    
    # Reparameterization to Decoder
    ax.annotate('', xy=(10, 4), xytext=(5.5, 4), 
                arrowprops=dict(arrowstyle='-|>', color='blue', lw=2))
    
    # Condition to Decoder
    ax.annotate('', xy=(12, 5.5), xytext=(16.5, 5.5), 
                arrowprops=dict(arrowstyle='-|>', color='purple', lw=2, alpha=0.7))
    
    # Decoder to Output
    ax.annotate('', xy=(14.5, 7.25), xytext=(12, 5.5), 
                arrowprops=arrow_props)
    
    # Loss Functions
    loss_box = FancyBboxPatch((6, 1), 8, 1.5, boxstyle="round,pad=0.1", 
                             facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(10, 1.75, 'Training Loss\n• Reconstruction Loss (MSE)\n• KL Divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)\n• Total: L = L_recon + β·L_KL (β=1.0)', 
            fontsize=10, ha='center', va='center', color='darkred', fontweight='bold')
    
    # Model parameters
    params_box = FancyBboxPatch((0.5, 0.2), 19, 0.6, boxstyle="round,pad=0.1", 
                               facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(params_box)
    ax.text(10, 0.5, 'Key Parameters: Encoder: 6 layers, 4 heads, 256-dim hidden | Decoder: 6 layers, 4 heads, 256-dim hidden | Latent: 128-dim | Batch: 32 | LR: 1e-4 | Epochs: 10', 
            fontsize=9, ha='center', va='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('pangu_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_detailed_layer_diagram():
    """Create a detailed view of encoder and decoder layers"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Encoder detailed view
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_title('Graph Transformer Encoder Details', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Encoder layer components
    for i in range(3):
        y = 10 - i*3
        
        # Input
        ax1.text(1, y+0.5, f'Layer {i+1} Input', fontsize=12, ha='center')
        ax1.add_patch(patches.Rectangle((0.5, y), 1, 0.8, fill=False))
        
        # TransformerConv
        ax1.text(3, y+0.5, 'TransformerConv\nMulti-head Attention', fontsize=10, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        # BatchNorm
        ax1.text(5, y+0.5, 'BatchNorm1d', fontsize=10, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        # ReLU
        ax1.text(6.5, y+0.5, 'ReLU', fontsize=10, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        
        # Output
        ax1.text(8.5, y+0.5, 'Layer Output', fontsize=12, ha='center')
        ax1.add_patch(patches.Rectangle((8, y), 1, 0.8, fill=False))
        
        # Arrows
        ax1.annotate('', xy=(2, y+0.4), xytext=(1.5, y+0.4), arrowprops=dict(arrowstyle='->'))
        ax1.annotate('', xy=(4, y+0.4), xytext=(3.5, y+0.4), arrowprops=dict(arrowstyle='->'))
        ax1.annotate('', xy=(5.5, y+0.4), xytext=(5, y+0.4), arrowprops=dict(arrowstyle='->'))
        ax1.annotate('', xy=(7, y+0.4), xytext=(6.5, y+0.4), arrowprops=dict(arrowstyle='->'))
        ax1.annotate('', xy=(8, y+0.4), xytext=(7.5, y+0.4), arrowprops=dict(arrowstyle='->'))
    
    # Decoder detailed view
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title('Transformer Decoder Details', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Decoder layer components
    for i in range(2):
        y = 10 - i*4
        
        # Input
        ax2.text(1, y+0.5, f'Decoder Layer {i+1}', fontsize=12, ha='center')
        ax2.add_patch(patches.Rectangle((0.5, y), 1, 0.8, fill=False))
        
        # Self-attention
        ax2.text(3, y+0.5, 'Masked\nSelf-Attention', fontsize=10, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
        
        # Cross-attention
        ax2.text(5, y+0.5, 'Cross-Attention\n(Encoder Output)', fontsize=10, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
        
        # Feed-forward
        ax2.text(7, y+0.5, 'Feed-Forward\nNetwork', fontsize=10, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
        
        # Output
        ax2.text(9, y+0.5, 'Layer Output', fontsize=12, ha='center')
        ax2.add_patch(patches.Rectangle((8.5, y), 1, 0.8, fill=False))
        
        # Arrows
        ax2.annotate('', xy=(2, y+0.4), xytext=(1.5, y+0.4), arrowprops=dict(arrowstyle='->'))
        ax2.annotate('', xy=(4, y+0.4), xytext=(3.5, y+0.4), arrowprops=dict(arrowstyle='->'))
        ax2.annotate('', xy=(6, y+0.4), xytext=(5.5, y+0.4), arrowprops=dict(arrowstyle='->'))
        ax2.annotate('', xy=(8, y+0.4), xytext=(7.5, y+0.4), arrowprops=dict(arrowstyle='->'))
        ax2.annotate('', xy=(8.5, y+0.4), xytext=(8, y+0.4), arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.savefig('pangu_layer_details.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("Generating PanGu Drug Model architecture diagrams...")
    
    # Create main architecture diagram
    main_fig = create_model_diagram()
    print("[OK] Main architecture diagram saved as 'pangu_model_architecture.png'")
    
    # Create detailed layer diagram
    detail_fig = create_detailed_layer_diagram()
    print("[OK] Detailed layer diagram saved as 'pangu_layer_details.png'")
    
    print("Visualization complete!")