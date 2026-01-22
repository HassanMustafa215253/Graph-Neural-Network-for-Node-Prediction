
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):

    
    def __init__(self, num_features, num_classes, hidden_channels=16, dropout=0.5):
        super(GCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        
        # First GCN layer with ReLU and dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            x = self.conv1(x, edge_index)
            x = F.relu(x)
        return x


if __name__ == "__main__":
    # Test the model
    model = GCN(num_features=1433, num_classes=7)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
