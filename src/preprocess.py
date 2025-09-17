import os
import torch
import logging
import numpy as np
from torch_geometric.datasets import OgbnDatasets, Reddit, PPI, Planetoid
from torch_geometric.transforms import ToUndirected, NormalizeFeatures
from sklearn.decomposition import PCA
from torch_geometric.utils import add_self_loops, remove_self_loops, degree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_dataset(name, root='data/'):
    path = os.path.join(root, name)
    if name.startswith('ogbn'):
        dataset = OgbnDatasets(root=root, name=name)
    elif name == 'Reddit':
        dataset = Reddit(path)
    elif name == 'PPI':
        dataset = PPI(path)
    elif name == 'Cora':
        dataset = Planetoid(root=root, name='Cora')
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return dataset

def preprocess_data(data, name):
    logging.info(f"Preprocessing data for {name}...")
    
    # L2 normalization
    if data.x is not None:
        data.x = F.normalize(data.x, p=2, dim=1)

    # Remove self-loops, add symmetric edges (undirected)
    edge_index, _ = remove_self_loops(data.edge_index)
    transform = ToUndirected()
    data = transform(data)
    
    # Add degree as a feature
    deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.float)
    data.degree = deg
    
    # PPI specific: PCA
    if name == 'PPI':
        logging.info("Applying PCA to PPI features...")
        pca = PCA(n_components=256)
        data.x = torch.from_numpy(pca.fit_transform(data.x.numpy())).float()

    if name == 'ogbn-papers100M-10%':
        logging.info("Creating 10% subset for ogbn-papers100M...")
        split_idx = data.get_edge_split() # OGB uses this for link prediction, node for node classification
        split_idx_node = data.get_idx_split()
        train_idx = split_idx_node['train']
        subset_train_idx = train_idx[:int(0.1 * len(train_idx))]
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[subset_train_idx] = True
        data.train_mask = train_mask
    else:
        # Ensure train_mask exists for all datasets
        if not hasattr(data, 'train_mask'):
            # Create a default split for datasets that don't have one
            num_nodes = data.num_nodes
            indices = torch.randperm(num_nodes)
            train_size = int(0.6 * num_nodes)
            val_size = int(0.2 * num_nodes)
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size:train_size+val_size]] = True
            data.test_mask[indices[train_size+val_size:]] = True

    # Ensure y is long tensor
    if data.y.dtype != torch.long and name != 'PPI':
        data.y = data.y.long().squeeze()
    
    # Get number of classes
    if name == 'PPI':
        data.num_classes = data.y.shape[1]
    else:
        data.num_classes = len(torch.unique(data.y))

    return data

def create_superhub_dataset(base_data, hub_degree=1000000):
    logging.info(f"Creating SuperHub dataset from Reddit with hub degree {hub_degree}...")
    data = base_data.clone()
    num_nodes = data.num_nodes
    
    # Add one new node
    new_node_id = num_nodes
    x_new = data.x.mean(dim=0, keepdim=True) # Avg feature for the new node
    data.x = torch.cat([data.x, x_new], dim=0)
    
    # Connect to random existing nodes
    connections = torch.randperm(num_nodes)[:hub_degree]
    hub_edges_to = torch.stack([torch.full_like(connections, new_node_id), connections], dim=0)
    hub_edges_from = torch.stack([connections, torch.full_like(connections, new_node_id)], dim=0)
    
    data.edge_index = torch.cat([data.edge_index, hub_edges_to, hub_edges_from], dim=1)
    
    # Assign label (majority class of neighbors)
    neighbor_labels = data.y[connections]
    majority_label = torch.mode(neighbor_labels).values.item()
    y_new = torch.tensor([majority_label], dtype=torch.long)
    data.y = torch.cat([data.y, y_new], dim=0)
    
    # Update masks and num_nodes
    data.num_nodes += 1
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        mask = getattr(data, mask_name)
        setattr(data, mask_name, torch.cat([mask, torch.tensor([False])], dim=0))

    return data

def load_and_preprocess_data(dataset_name, data_dir='data/'):
    processed_dir = os.path.join(data_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Adjust name for papers100m to load the base dataset first
    base_name = 'ogbn-papers100M' if dataset_name == 'ogbn-papers100M-10%' else dataset_name
    
    # Handle SuperHub-1M case
    if dataset_name == 'SuperHub-1M':
        cache_file = os.path.join(processed_dir, 'SuperHub-1M.pt')
        if os.path.exists(cache_file):
            logging.info(f"Loading cached processed data for {dataset_name}...")
            return cache_file
        # Load base Reddit first
        reddit_path = load_and_preprocess_data('Reddit', data_dir)
        reddit_data = torch.load(reddit_path)
        data = create_superhub_dataset(reddit_data)
        torch.save(data, cache_file)
        logging.info(f"Saved processed data for {dataset_name} to {cache_file}")
        return cache_file

    cache_file = os.path.join(processed_dir, f"{dataset_name.replace('-', '_')}.pt")
    if os.path.exists(cache_file):
        logging.info(f"Loading cached processed data for {dataset_name}...")
        return cache_file

    logging.info(f"Downloading and processing dataset: {base_name}")
    try:
        dataset = get_dataset(base_name, root=data_dir)
        data = dataset[0]
    except Exception as e:
        logging.error(f"Failed to download or load dataset {base_name}. Error: {e}")
        raise ConnectionError(f"Could not access dataset {base_name}. Check connection and OGB/PyG installation.")

    data = preprocess_data(data, dataset_name)
    
    torch.save(data, cache_file)
    logging.info(f"Saved processed data for {dataset_name} to {cache_file}")
    
    return cache_file
