"""
Magnetic Topological Material Classifier
=======================================
A deep learning framework for simultaneously predicting magnetic ordering and topological classification
of crystalline materials using graph neural networks and transformers.
"""

import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add, scatter_mean
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Materials science libraries
import pymatgen as pmg
from pymatgen.core.structure import Structure
from pymatgen.core import Element
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

# Load environment variables
load_dotenv()
api_key = os.getenv("MP_API_KEY")

# Constants
ORDER_ENCODE = {"NM": 0, "AFM": 1, "FM": 2, "FiM": 2}  # FiM grouped with FM
TOPO_ENCODE = {False: 0, True: 1}  # Non-topological vs topological

# Global parameters
PARAMS = {
    'max_radius': 8.0,        # Maximum cutoff radius for atom connections
    'n_norm': 35,             # Normalization factor
    'hidden_dim': 128,        # Hidden dimension size
    'num_heads': 4,           # Number of attention heads
    'num_layers': 3,          # Number of transformer layers
    'batch_size': 8,          # Batch size for training
    'lr': 0.0005,             # Learning rate
    'weight_decay': 0.01,     # Weight decay for regularization
    'max_epochs': 200,        # Maximum training epochs
    'early_stop_patience': 15  # Patience for early stopping
}


#===============================
# DATA STRUCTURES AND PROCESSING
#===============================

class MaterialData(Data):
    """
    Custom PyTorch Geometric Data class for material structure data with proper batching.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.num_nodes
        if key == 'cell_index':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def get_element_properties(symbol):
    """Get key properties for an element by symbol."""
    try:
        elem = Element(symbol)
        return {
            'Z': elem.Z,
            'group': elem.group,
            'row': elem.row,
            'atomic_radius': elem.atomic_radius or 0.0,
            'atomic_mass': elem.atomic_mass or 0.0,
            'electronegativity': elem.electronegativity or 0.0,
            'is_magnetic': int(symbol in ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Gd', 'Dy', 'Ho', 'Er', 'Tm', 'Yb'])
        }
    except Exception:
        # Default values if element properties can't be retrieved
        return {'Z': 0, 'group': 0, 'row': 0, 'atomic_radius': 0.0, 
                'atomic_mass': 0.0, 'electronegativity': 0.0, 'is_magnetic': 0}


def extract_structure_features(structure):
    """Extract features from a pymatgen Structure object."""
    # Symmetry features
    analyzer = SpacegroupAnalyzer(structure)
    spacegroup = analyzer.get_space_group_number()
    point_group = analyzer.get_point_group_symbol()
    has_inversion = int(analyzer.has_inversion())
    
    # Magnetic features
    mag_elements = ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Gd', 'Dy', 'Ho', 'Er', 'Tm', 'Yb']
    mag_count = sum(1 for site in structure if site.species_string in mag_elements)
    mag_fraction = mag_count / len(structure)
    
    # Lattice features
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    volume = structure.lattice.volume
    density = structure.density
    
    return {
        'spacegroup': spacegroup,
        'point_group_id': hash(point_group) % 100,  # Simple hash for point group
        'has_inversion': has_inversion,
        'mag_fraction': mag_fraction,
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'volume': volume,
        'density': density
    }


def structure_to_graph(structure, max_radius=8.0):
    """
    Convert a pymatgen Structure to a graph representation.
    
    Args:
        structure: pymatgen Structure object
        max_radius: Maximum bond distance to consider
        
    Returns:
        x: Node features tensor
        edge_index: Edge connectivity tensor
        edge_attr: Edge features tensor
        pos: Node positions tensor
        structure_features: Global structure features tensor
    """
    num_sites = len(structure)
    
    # Node features: 7 features per atom
    node_features = []
    for site in structure:
        element = site.species_string
        props = get_element_properties(element)
        
        # Feature vector for each atom: element properties
        features = [
            props['Z'] / 100,  # Normalized atomic number
            props['group'] / 18,  # Normalized group
            props['row'] / 7,  # Normalized row
            props['atomic_radius'] / 2.0 if props['atomic_radius'] else 0,  # Normalized radius
            props['electronegativity'] / 4.0 if props['electronegativity'] else 0,  # Normalized electronegativity
            props['atomic_mass'] / 250.0,  # Normalized mass
            float(props['is_magnetic'])  # Is magnetic element
        ]
        node_features.append(features)
    
    # Node positions
    positions = torch.tensor(structure.cart_coords, dtype=torch.float)
    
    # Create edges based on distance
    src_list = []
    dst_list = []
    edge_attr_list = []
    
    # For each pair of atoms, check if they're within max_radius
    for i in range(num_sites):
        for j in range(num_sites):
            if i == j:  # Skip self-loops for now
                continue
                
            # Get the distance considering periodic boundary conditions
            dist = structure.get_distance(i, j)
            
            if dist <= max_radius:
                src_list.append(i)
                dst_list.append(j)
                
                # Edge features: distance, direction vector (normalized)
                direction = positions[j] - positions[i]
                direction_norm = torch.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                
                # Create edge feature vector:
                # [distance, dx, dy, dz]
                edge_attr_list.append([dist / max_radius] + direction.tolist())
    
    # If no edges were found, create self-loops to avoid errors
    if not src_list:
        for i in range(num_sites):
            src_list.append(i)
            dst_list.append(i)
            edge_attr_list.append([0.0, 0.0, 0.0, 0.0])  # Self-loop has zero features
    
    # Convert to tensors
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Extract global structure features
    structure_feats = extract_structure_features(structure)
    structure_features = torch.tensor([
        structure_feats['spacegroup'] / 230,  # Normalize by max space group
        structure_feats['point_group_id'] / 100,
        float(structure_feats['has_inversion']),
        structure_feats['mag_fraction'],
        structure_feats['a'] / 20.0,  # Normalize lattice parameters
        structure_feats['b'] / 20.0,
        structure_feats['c'] / 20.0,
        structure_feats['alpha'] / 180.0,
        structure_feats['beta'] / 180.0,
        structure_feats['gamma'] / 180.0,
        structure_feats['volume'] / 1000.0,
        structure_feats['density'] / 20.0
    ], dtype=torch.float).unsqueeze(0).repeat(num_sites, 1)
    
    return x, edge_index, edge_attr, positions, structure_features


def process_structures(structures, materials_ids=None, formulas=None):
    """
    Process a list of structures into graph data objects.
    
    Args:
        structures: List of pymatgen Structure objects
        materials_ids: List of Material Project IDs (optional)
        formulas: List of chemical formulas (optional)
        
    Returns:
        data_list: List of MaterialData objects
    """
    data_list = []
    
    for i, structure in enumerate(structures):
        print(f"Processing structure {i+1}/{len(structures)}", end="\r", flush=True)
        
        try:
            # Extract magnetic ordering
            mag_analyzer = CollinearMagneticStructureAnalyzer(structure)
            ordering = mag_analyzer.ordering.name
            magnetic_y = ORDER_ENCODE.get(ordering, 0)
            
            # Get material ID and formula if provided
            material_id = materials_ids[i] if materials_ids else f"struct_{i}"
            formula = formulas[i] if formulas else structure.composition.reduced_formula
            
            # Convert structure to graph representation
            x, edge_index, edge_attr, pos, structure_features = structure_to_graph(
                structure, max_radius=PARAMS['max_radius']
            )
            
            # Create data object
            data = MaterialData(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=pos,
                structure_features=structure_features,
                magnetic_y=torch.tensor([magnetic_y], dtype=torch.long),
                topological_y=torch.tensor([0], dtype=torch.long),  # Default to non-topological
                material_id=material_id,
                formula=formula,
                num_atoms=len(structure)
            )
            
            data_list.append(data)
            
        except Exception as e:
            print(f"\nError processing structure {i}: {e}")
            continue
    
    print(f"\nProcessed {len(data_list)}/{len(structures)} structures successfully")
    return data_list


def fetch_topological_labels(data_list, api_key):
    """
    Fetch topological classifications for materials using the Materials Project API.
    
    Args:
        data_list: List of MaterialData objects
        api_key: Materials Project API key
        
    Returns:
        data_list: Updated list with topological labels
    """
    materials_with_topo_info = 0
    
    with MPRester(api_key=api_key) as mpr:
        for i, data in enumerate(data_list):
            material_id = data.material_id
            
            try:
                # Skip if not a real MP ID
                if not material_id.startswith("mp-"):
                    continue
                    
                # Query Materials Project API
                result = mpr.materials.summary.search(material_ids=[material_id])
                
                if result and hasattr(result[0], "is_topological"):
                    label = result[0].is_topological
                    data.topological_y = torch.tensor([TOPO_ENCODE[label]], dtype=torch.long)
                    materials_with_topo_info += 1
                    print(f"Found topological info for {material_id}: {label}")
                    
            except Exception as e:
                print(f"Error retrieving topological info for {material_id}: {e}")
    
    print(f"Added topological labels for {materials_with_topo_info} materials")
    return data_list


def load_and_process_data(mp_structures_file, api_key=None):
    """
    Load and process materials data.
    
    Args:
        mp_structures_file: Path to saved structures file
        api_key: Materials Project API key for topological data
        
    Returns:
        processed_data: List of processed MaterialData objects
    """
    print(f"Loading structures from {mp_structures_file}")
    mp_structures_dict = torch.load(mp_structures_file, weights_only=False)
    
    structures = mp_structures_dict['structures']
    materials_ids = mp_structures_dict['materials_id']
    formulas = mp_structures_dict['formulas']
    
    print(f"Loaded {len(structures)} structures")
    
    # Process structures to graph data
    processed_data = process_structures(structures, materials_ids, formulas)
    
    # Fetch topological labels if API key is provided
    if api_key:
        processed_data = fetch_topological_labels(processed_data, api_key)
    
    return processed_data


def prepare_datasets(data_list, train_ratio=0.8, val_ratio=0.1):
    """
    Split data into training, validation and test sets.
    
    Args:
        data_list: List of MaterialData objects
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
    """
    # Balance datasets by magnetic ordering
    data_by_order = {0: [], 1: [], 2: []}
    
    for data in data_list:
        magnetic_y = data.magnetic_y.item()
        data_by_order[magnetic_y].append(data)
    
    print(f"Data distribution by magnetic ordering:")
    for order, items in data_by_order.items():
        order_name = {0: "NM", 1: "AFM", 2: "FM/FiM"}[order]
        print(f"  {order_name}: {len(items)} structures")
    
    # Find minimum count to ensure balanced classes
    min_count = min(len(items) for items in data_by_order.values())
    balanced_data = []
    
    for order, items in data_by_order.items():
        random.shuffle(items)
        balanced_data.extend(items[:min_count])
    
    # Shuffle balanced dataset
    random.shuffle(balanced_data)
    
    # Split into train/val/test
    n = len(balanced_data)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    
    train_data = balanced_data[:train_size]
    val_data = balanced_data[train_size:train_size + val_size]
    test_data = balanced_data[train_size + val_size:]
    
    print(f"Dataset splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=PARAMS['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=PARAMS['batch_size'])
    test_loader = DataLoader(test_data, batch_size=PARAMS['batch_size'])
    
    return train_loader, val_loader, test_loader


#===============================
# MODEL ARCHITECTURE
#===============================

class AttentionLayer(MessagePassing):
    """
    Graph attention layer for materials science applications.
    """
    def __init__(self, hidden_dim, num_heads=4, edge_dim=4):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, key, value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature projection
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, x, edge_index, edge_attr):
        # First attention block with residual connection
        identity = x
        out = self.ln1(x)
        out = self._attention_block(out, edge_index, edge_attr)
        out = out + identity
        
        # Feed-forward block with residual connection
        identity = out
        out = self.ln2(out)
        out = self.ffn(out)
        return out + identity
    
    def _attention_block(self, x, edge_index, edge_attr):
        # Project inputs to queries, keys, values
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Process edge attributes
        edge_weights = self.edge_proj(edge_attr).unsqueeze(-1)  # [E, num_heads, 1]
        
        # Propagate through the graph
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_weights=edge_weights)
        
        # Project output back to original dimension
        return self.output_proj(out.view(-1, self.hidden_dim))
    
    def message(self, q_i, k_j, v_j, edge_weights, index, ptr, size_i):
        # Compute attention scores
        attention = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
        
        # Multiply by edge weights (based on edge features)
        attention = attention.unsqueeze(-1) * edge_weights
        
        # Apply softmax to normalize scores
        alpha = F.softmax(attention, dim=0)
        
        # Apply attention weights to values
        return alpha * v_j


class MagneticTopologicalTransformer(nn.Module):
    """
    Transformer-based model for predicting magnetic ordering and topological class.
    """
    def __init__(self, node_dim=7, structure_dim=12, hidden_dim=128, edge_dim=4, num_heads=4, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.structure_proj = nn.Linear(structure_dim, hidden_dim)
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(hidden_dim, num_heads, edge_dim)
            for _ in range(num_layers)
        ])
        
        # Output heads
        self.magnetic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # NM, AFM, FM/FiM
        )
        
        self.topological_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Not TI, TI
        )
    
    def forward(self, x, edge_index, edge_attr, structure_features, batch):
        # Project node and structure features
        h_nodes = self.node_proj(x)
        h_struct = self.structure_proj(structure_features)
        
        # Combine node features with structure features
        h = h_nodes + h_struct
        
        # Apply attention layers
        for layer in self.attention_layers:
            h = layer(h, edge_index, edge_attr)
        
        # Global pooling
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        h_global = scatter_mean(h, batch, dim=0)
        
        # Predict magnetic ordering and topological class
        magnetic_pred = self.magnetic_head(h_global)
        topological_pred = self.topological_head(h_global)
        
        return magnetic_pred, topological_pred


#===============================
# TRAINING AND EVALUATION
#===============================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=15, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return True
            
        if val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = val_loss
            self.counter = 0
            return True


def train_epoch(model, dataloader, optimizer, device, alpha=0.5):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    magnetic_loss_total = 0
    topological_loss_total = 0
    
    # Class weights to handle imbalance
    magnetic_class_weights = torch.tensor([1.0, 1.2, 1.2], device=device)
    topological_class_weights = torch.tensor([1.0, 1.5], device=device)
    
    # Loss functions
    magnetic_criterion = nn.CrossEntropyLoss(weight=magnetic_class_weights)
    topological_criterion = nn.CrossEntropyLoss(weight=topological_class_weights)
    
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        magnetic_pred, topological_pred = model(
            batch.x, 
            batch.edge_index, 
            batch.edge_attr, 
            batch.structure_features,
            batch.batch
        )
        
        # Compute losses
        magnetic_loss = magnetic_criterion(magnetic_pred, batch.magnetic_y.squeeze())
        topological_loss = topological_criterion(topological_pred, batch.topological_y.squeeze())
        
        # Combined loss with weighting parameter alpha
        loss = alpha * magnetic_loss + (1 - alpha) * topological_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += loss.item() * batch.num_graphs
        magnetic_loss_total += magnetic_loss.item() * batch.num_graphs
        topological_loss_total += topological_loss.item() * batch.num_graphs
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader.dataset)
    avg_magnetic_loss = magnetic_loss_total / len(dataloader.dataset)
    avg_topological_loss = topological_loss_total / len(dataloader.dataset)
    
    return avg_loss, avg_magnetic_loss, avg_topological_loss


def validate(model, dataloader, device, alpha=0.5):
    """Validate the model"""
    model.eval()
    total_loss = 0
    magnetic_loss_total = 0
    topological_loss_total = 0
    
    magnetic_preds = []
    magnetic_targets = []
    topological_preds = []
    topological_targets = []
    
    magnetic_class_weights = torch.tensor([1.0, 1.2, 1.2], device=device)
    topological_class_weights = torch.tensor([1.0, 1.5], device=device)
    
    magnetic_criterion = nn.CrossEntropyLoss(weight=magnetic_class_weights)
    topological_criterion = nn.CrossEntropyLoss(weight=topological_class_weights)
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            magnetic_pred, topological_pred = model(
                batch.x, 
                batch.edge_index, 
                batch.edge_attr, 
                batch.structure_features,
                batch.batch
            )
            
            # Compute losses
            magnetic_loss = magnetic_criterion(magnetic_pred, batch.magnetic_y.squeeze())
            topological_loss = topological_criterion(topological_pred, batch.topological_y.squeeze())
            
            # Combined loss
            loss = alpha * magnetic_loss + (1 - alpha) * topological_loss
            
            # Track losses
            total_loss += loss.item() * batch.num_graphs
            magnetic_loss_total += magnetic_loss.item() * batch.num_graphs
            topological_loss_total += topological_loss.item() * batch.num_graphs
            
            # Track predictions for metrics
            magnetic_preds.append(magnetic_pred.argmax(dim=1).cpu())
            magnetic_targets.append(batch.magnetic_y.squeeze().cpu())
            topological_preds.append(topological_pred.argmax(dim=1).cpu())
            topological_targets.append(batch.topological_y.squeeze().cpu())
    
    # Concatenate predictions and targets
    magnetic_preds = torch.cat(magnetic_preds)
    magnetic_targets = torch.cat(magnetic_targets)
    topological_preds = torch.cat(topological_preds)
    topological_targets = torch.cat(topological_targets)
    
    # Calculate metrics
    magnetic_acc = accuracy_score(magnetic_targets, magnetic_preds)
    magnetic_f1 = f1_score(magnetic_targets, magnetic_preds, average='macro')
    topological_acc = accuracy_score(topological_targets, topological_preds)
    topological_f1 = f1_score(topological_targets, topological_preds, average='macro')
    
    # Calculate average losses
    avg_loss = total_loss / len(dataloader.dataset)
    avg_magnetic_loss = magnetic_loss_total / len(dataloader.dataset)
    avg_topological_loss = topological_loss_total / len(dataloader.dataset)
    
    return (avg_loss, avg_magnetic_loss, avg_topological_loss, 
            magnetic_acc, magnetic_f1, topological_acc, topological_f1)


def train_model(model, train_loader, val_loader, device, model_save_path="./model"):
    """
    Train the model with early stopping.
    
    Args:
        model: MagneticTopologicalTransformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on (CPU or GPU)
        model_save_path: Directory to save model checkpoints
        
    Returns:
        model: Trained model
        history: Training history
    """
    # Create save directory if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=PARAMS['lr'], 
        weight_decay=PARAMS['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10
    )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=PARAMS['early_stop_patience'])
    
    # Initialize best model tracker
    best_val_loss = float('inf')
    best_model_path = os.path.join(model_save_path, "best_model.pt")
    
    # Initialize training history
    history = {
        'train_loss': [], 
        'val_loss': [],
        'train_magnetic_loss': [], 
        'val_magnetic_loss': [],
        'train_topological_loss': [], 
        'val_topological_loss': [],
        'magnetic_acc': [],
        'magnetic_f1': [],
        'topological_acc': [],
        'topological_f1': []
    }
    
    # Training loop
    start_time = time.time()
    print("Starting training...")
    
    for epoch in range(PARAMS['max_epochs']):
        epoch_start = time.time()
        
        # Train one epoch
        train_loss, train_magnetic_loss, train_topological_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        (val_loss, val_magnetic_loss, val_topological_loss, 
         magnetic_acc, magnetic_f1, topological_acc, topological_f1) = val_metrics
        
        # Update learning rate
        scheduler.step(val_loss)  # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_magnetic_loss'].append(train_magnetic_loss)
        history['val_magnetic_loss'].append(val_magnetic_loss)
        history['train_topological_loss'].append(train_topological_loss)
        history['val_topological_loss'].append(val_topological_loss)
        history['magnetic_acc'].append(magnetic_acc)
        history['magnetic_f1'].append(magnetic_f1)
        history['topological_acc'].append(topological_acc)
        history['topological_f1'].append(topological_f1)

        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{PARAMS['max_epochs']} - {epoch_time:.2f}s - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
            f"Magnetic Loss: {train_magnetic_loss:.4f}/{val_magnetic_loss:.4f} - "
            f"Topological Loss: {train_topological_loss:.4f}/{val_topological_loss:.4f} - "
            f"Magnetic Acc: {magnetic_acc:.4f} - Magnetic F1: {magnetic_f1:.4f} - "
            f"Topological Acc: {topological_acc:.4f} - Topological F1: {topological_f1:.4f}")

        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss
            }, os.path.join(PARAMS['checkpoint_dir'], 'best_model.pth'))
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")
            
        # Early stopping check
        if val_loss > best_val_loss and epoch > PARAMS['min_epochs']:
            early_stop_counter += 1
            if early_stop_counter >= PARAMS['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            early_stop_counter = 0


def test_model(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    magnetic_loss = 0
    topological_loss = 0
    
    all_magnetic_preds = []
    all_magnetic_labels = []
    all_topological_preds = []
    all_topological_labels = []
    
    with torch.no_grad():  # No gradient calculation needed for testing
        for batch_idx, (data, magnetic_target, topological_target) in enumerate(test_loader):
            data = data.to(device)
            magnetic_target = magnetic_target.to(device)
            topological_target = topological_target.to(device)
            
            # Forward pass
            magnetic_output, topological_output = model(data)
            
            # Calculate losses
            batch_magnetic_loss = F.binary_cross_entropy_with_logits(magnetic_output, magnetic_target)
            batch_topological_loss = F.binary_cross_entropy_with_logits(topological_output, topological_target)
            batch_loss = batch_magnetic_loss + batch_topological_loss
            
            # Accumulate losses
            test_loss += batch_loss.item()
            magnetic_loss += batch_magnetic_loss.item()
            topological_loss += batch_topological_loss.item()
            
            # Store predictions and labels for metrics calculation
            magnetic_preds = (torch.sigmoid(magnetic_output) > 0.5).float().cpu().numpy()
            topological_preds = (torch.sigmoid(topological_output) > 0.5).float().cpu().numpy()
            
            all_magnetic_preds.extend(magnetic_preds)
            all_magnetic_labels.extend(magnetic_target.cpu().numpy())
            all_topological_preds.extend(topological_preds)
            all_topological_labels.extend(topological_target.cpu().numpy())
    
    # Calculate average losses
    test_loss /= len(test_loader)
    magnetic_loss /= len(test_loader)
    topological_loss /= len(test_loader)
    
    # Convert lists to arrays for scikit-learn metrics
    all_magnetic_preds = np.array(all_magnetic_preds)
    all_magnetic_labels = np.array(all_magnetic_labels)
    all_topological_preds = np.array(all_topological_preds)
    all_topological_labels = np.array(all_topological_labels)
    
    # Calculate metrics
    magnetic_acc = accuracy_score(all_magnetic_labels, all_magnetic_preds)
    magnetic_f1 = f1_score(all_magnetic_labels, all_magnetic_preds, average='weighted')
    topological_acc = accuracy_score(all_topological_labels, all_topological_preds)
    topological_f1 = f1_score(all_topological_labels, all_topological_preds, average='weighted')
    
    # Print results
    print(f"Test Results:")
    print(f"Total Loss: {test_loss:.4f}")
    print(f"Magnetic Loss: {magnetic_loss:.4f}, Accuracy: {magnetic_acc:.4f}, F1 Score: {magnetic_f1:.4f}")
    print(f"Topological Loss: {topological_loss:.4f}, Accuracy: {topological_acc:.4f}, F1 Score: {topological_f1:.4f}")
    
    # Return all metrics
    return {
        'test_loss': test_loss,
        'magnetic_loss': magnetic_loss,
        'topological_loss': topological_loss,
        'magnetic_acc': magnetic_acc,
        'magnetic_f1': magnetic_f1,
        'topological_acc': topological_acc,
        'topological_f1': topological_f1,
        'magnetic_preds': all_magnetic_preds,
        'magnetic_labels': all_magnetic_labels,
        'topological_preds': all_topological_preds,
        'topological_labels': all_topological_labels
    }

# Usage example:
def run_test():
    # Load the best saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YourModelClass().to(device)
    
    checkpoint = torch.load(os.path.join(PARAMS['checkpoint_dir'], 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create test data loader
    test_dataset = YourDataset(data_path=PARAMS['test_data_path'])
    test_loader = DataLoader(test_dataset, batch_size=PARAMS['batch_size'], shuffle=False)
    
    # Run test
    test_results = test_model(model, test_loader, device)
    
    # Additional analysis
    # Confusion matrix for magnetic predictions
    magnetic_cm = confusion_matrix(test_results['magnetic_labels'], test_results['magnetic_preds'])
    print("Magnetic Confusion Matrix:")
    print(magnetic_cm)
    
    # Confusion matrix for topological predictions
    topological_cm = confusion_matrix(test_results['topological_labels'], test_results['topological_preds'])
    print("Topological Confusion Matrix:")
    print(topological_cm)
    
    # You can add more detailed analysis here, like precision, recall, etc.
    
    # Optional: Visualize some predictions
    visualize_predictions(model, test_loader, device, num_samples=5)
    
    return test_results

# Optional: Visualization function
def visualize_predictions(model, data_loader, device, num_samples=5):
    model.eval()
    samples_seen = 0
    
    with torch.no_grad():
        for data, magnetic_target, topological_target in data_loader:
            if samples_seen >= num_samples:
                break
                
            data = data.to(device)
            magnetic_output, topological_output = model(data)
            
            # Convert to predictions
            magnetic_preds = (torch.sigmoid(magnetic_output) > 0.5).float().cpu().numpy()
            topological_preds = (torch.sigmoid(topological_output) > 0.5).float().cpu().numpy()
            
            # For each sample in the batch
            for i in range(min(len(data), num_samples - samples_seen)):
                samples_seen += 1
                
                # Here you could visualize the input data and the predictions
                print(f"Sample {samples_seen}:")
                print(f"Magnetic - True: {magnetic_target[i].cpu().numpy()}, Predicted: {magnetic_preds[i]}")
                print(f"Topological - True: {topological_target[i].cpu().numpy()}, Predicted: {topological_preds[i]}")
                
                # If your data is image-like, you could display it with matplotlib
                # plt.figure()
                # plt.imshow(data[i].cpu().numpy().transpose(1, 2, 0))  # Assuming data is in [C, H, W] format
                # plt.title(f"Magnetic: {magnetic_preds[i]}, Topo: {topological_preds[i]}")
                # plt.show()

if __name__ == "__main__":
    test_results = run_test()