
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def load_cora(root='./data'):
    
    dataset = Planetoid(
        root=root,
        name='Cora',
        transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"\nData statistics:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Training nodes: {data.train_mask.sum().item()}")
    print(f"  Validation nodes: {data.val_mask.sum().item()}")
    print(f"  Test nodes: {data.test_mask.sum().item()}")
    
    return dataset, data


def get_device():
    
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


if __name__ == "__main__":
    
    dataset, data = load_cora()
    print(f"\nNode features shape: {data.x.shape}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Labels shape: {data.y.shape}")
