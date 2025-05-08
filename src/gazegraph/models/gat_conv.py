
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

class GATBackbone(nn.Module):
    '''
    An encoder backbone which can be modified with a head for different tasks
    or RNN (see ordered future tasks)
    '''
    def __init__(self, input_dim, hidden_dim, edge_dim, num_heads=1, num_layers=3, res_connect=False):
        super().__init__() 
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.GATLayers = nn.ModuleList([])
        for _ in range(num_layers):
            self.GATLayers.append(
                gnn.GATv2Conv(in_channels=input_dim, out_channels=hidden_dim // num_heads, edge_dim=edge_dim, heads=num_heads, residual=res_connect)
            )
            input_dim = hidden_dim

    def forward(self, x, edge_index, edge_attr, batch):
        for GATLayer in self.GATLayers:
            x = GATLayer(x, edge_index, edge_attr, batch)
            x = F.relu(x)
        return x

class GATForClassification(nn.Module):
    def __init__(self, num_classes, input_dim, hidden_dim, edge_dim, num_heads, num_layers, res_connect):
        super().__init__()
        self.GAT = GATBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            res_connect=res_connect,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.GAT(x, edge_index, edge_attr, batch)
        x = gnn.global_mean_pool(x, batch)
        #NOTE if doing BCE, use BCEWithLogits. This is done to keep one model for both BCE and CE loss
        x = self.fc(x)
        return x