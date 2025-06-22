import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.typing import Size

from typing import Dict


class GATBackbone(nn.Module):
    """
    An encoder backbone which can be modified with a head for different tasks
    or RNN (see ordered future tasks)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        edge_dim,
        num_heads=1,
        num_layers=3,
        res_connect=False,
        heterogeneous=False,
        node_types=None,
        metadata=None,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.heterogeneous = heterogeneous

        if self.heterogeneous:
            self.lin_dict = nn.ModuleDict()
            for node_type in node_types:
                self.lin_dict[node_type] = gnn.Linear(-1, input_dim)

        self.GATLayers = nn.ModuleList([])
        for _ in range(num_layers):
            if self.heterogeneous:
                self.GATLayers.append(
                    gnn.HGTConv(input_dim, hidden_dim, metadata, num_heads)
                )
            else:
                self.GATLayers.append(
                    gnn.GATv2Conv(
                        in_channels=input_dim,
                        out_channels=hidden_dim // num_heads,
                        edge_dim=edge_dim,
                        heads=num_heads,
                        residual=res_connect,
                    )
                )
            input_dim = hidden_dim

    def forward(
        self, 
        x: torch.Tensor | Dict[str, torch.Tensor], 
        edge_index: torch.Tensor | Dict[str, torch.Tensor], 
        edge_attr: torch.Tensor | Dict[str, torch.Tensor] | None, 
        batch: Size | Dict[str, Size]
    ):
        """
        if heterogeneous
            x, edge_index should be dicts;
            edge_attr should be a dict if edge features exist, else None;
            batch should be a dict of torch_geometric.Size objects
        else
            x, edge_index should be tensors;
            edge_attr should be a tensor if edge features exist, else None
            batch should be a Size instance returned by torch_geometric.loader.DataLoader
        """
        if self.heterogeneous:
            for node_type, data in x.items():
                x[node_type] = self.lin_dict[node_type](data).relu_()

        for GATLayer in self.GATLayers:
            if self.heterogeneous:
                x = GATLayer(x, edge_index)
            else:
                x = GATLayer(x, edge_index, edge_attr, batch)
                x = F.relu(x)
        return x


class GATForClassification(nn.Module):
    def __init__(
        self,
        num_classes,
        input_dim,
        hidden_dim,
        edge_dim,
        num_heads,
        num_layers,
        res_connect=False,
        heterogeneous=False,
        node_types=None,
        metadata=None,
    ):
        super().__init__()

        self.heterogeneous = heterogeneous

        self.GAT = GATBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            res_connect=res_connect,
            heterogeneous=heterogeneous,
            node_types=node_types,
            metadata=metadata,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, 
        x: torch.Tensor | Dict[str, torch.Tensor], 
        edge_index: torch.Tensor | Dict[str, torch.Tensor], 
        edge_attr: torch.Tensor | Dict[str, torch.Tensor] | None, 
        batch: Size | Dict[str, Size]
    ):
        """
        if heterogeneous
            x, edge_index should be dicts;
            edge_attr should be a dict if edge features exist, else None;
            batch should be a dict of torch_geometric.Size objects
        else
            x, edge_index, edge_attr should be tensors;
            batch should be a Size instance returned by torch_geometric.loader.DataLoader
        """
        x = self.GAT(x, edge_index, edge_attr, batch)
        if self.heterogeneous:
            x = gnn.global_mean_pool(x['action'], batch['action'])
        else:
            x = gnn.global_mean_pool(x, batch)
            
        #NOTE if training with BCE, use BCEWithLogits.
        # This is done to keep one model for both BCE and CE loss
        x = self.fc(x)
        return x
