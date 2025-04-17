import random
import torch
from torch_geometric.data import Data

### Loading ###
def load_datasets(path_to_data: str):
    """
    Returns train and test sets from dataset path
    """
    dataset = torch.load(path_to_data)
    train_set, test_set = dataset['train'], dataset['val']
    return train_set, test_set

### Node dropping ###
def edge_idx_to_adj_list(edge_index, num_nodes=None):
    """
    Remaps the edge_index tensor to an adjacency list 
    where each node index maps to a list of node indices it shares an edge with

    Args:
        edge_index: a tensor of shape (2), num_nodes) following torch geometric standard
        num_nodes: the number of nodes in the graph (if known ahead of time)
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    adj_lists = [[] for _ in range(num_nodes)]
    for u,v in edge_index.transpose(0,1):
        u, v = u.item(), v.item()
        adj_lists[u].append(v)
    return adj_lists

def tarjans(adj_lists: list[list[int]]):
    """
    Tarjan's algorithm 

    Args:
        adj_list: adjacency list of graph (see format of edge_idx_to_adj_list() function)
    """
    n = len(adj_lists)
    disc_time = [0] * n
    low = [0] * n
    ap = [False] * n
    time = [0]

    def dfs_AP(u, parent):
        children = 0
        time[0] += 1
        disc_time[u] = low[u] = time[0]

        for v in adj_lists[u]:
            if v == parent:
                continue
            if disc_time[v] == 0:
                children += 1
                dfs_AP(v, u)
                low[u] = min(low[u], low[v])

                # articulation point condition (excluding root)
                if parent != u and disc_time[u] <= low[v]:
                    ap[u] = True
            else:
                # update low[u] if v was already visited
                low[u] = min(low[u], disc_time[v])

        # special case for root node
        if parent == u and children > 1:
            ap[u] = True

    for u in range(n):
        if disc_time[u] == 0:
            dfs_AP(u, u)

    return {i for i, is_ap in enumerate(ap) if is_ap}

def node_dropping(x, edge_index, edge_attr, max_droppable):
    # select random number of nodes to drop
    num_nodes = x.shape[0]
    adj_lists = edge_idx_to_adj_list(edge_index, num_nodes)
    not_droppable = tarjans(adj_lists)  # find articulation points which would disconnect the graph
    drop_candidates = [u for u in range(num_nodes) if u not in not_droppable]

    max_droppable = min(max_droppable, len(drop_candidates))
    if max_droppable == 0:
        return None

    dropped_nodes = random.sample(drop_candidates, random.randint(1,max_droppable))

    # drop nodes
    node_mask = torch.ones(num_nodes, dtype=torch.bool)
    node_mask[dropped_nodes] = False
    x_aug = x[node_mask]

    if x_aug.size(0) == 0:
        return None

    # remap indices since otherwise we would relabel the nodesx but not account for edge relabeling
    # (-1 means dropped)
    new_idx_map = torch.full((num_nodes,), -1, dtype=torch.long)
    new_idx_map[node_mask] = torch.arange(node_mask.sum())

    # keep edges where both src and dest are in the node mask (non-dropped)
    src = edge_index[0]
    dst = edge_index[1]
    valid_edge_mask = node_mask[src] & node_mask[dst]

    # filter edges and attributes
    edge_index_aug = edge_index[:, valid_edge_mask]
    edge_attr_aug = edge_attr[valid_edge_mask]

    # remap to new node indices using pytorch black magic
    edge_index_aug = new_idx_map[edge_index_aug]

    return x_aug, edge_index_aug, edge_attr_aug

### Dataloader ###
def get_graph_dataset(
    data,
    task_mode='future_actions',
    num_classes=106,
    node_drop_p=0.0,
    max_droppable=0
):
    """
    Load the data corresponding to the task mode into a list of torch_geometric.data.Data instances

    Args:
        data: the data dictionary containing x, edge_idx, edge_attr, and y
        task_mode: the mode to use ("future_actions", "future_actions_ordered", "next_action"), (default: "future_actions")
        num_classes: the number of classes present in the future_actions_ordered task
        node_drop_p: the probability of which a node is dropped (default = 0.0, i.e. no dropping)
        max_droppable: the maximum number of nodes which will be dropped from the graph during node dropping

    Returns
        graph_data: list of graph data compatible with torch_geometric.loader.Dataloader
    """

    graph_data = []
    xs, edge_indices, edge_attrs = data['x'], data['edge_index'], data['edge_attr']
    ys = [label[task_mode] for label in data['y']]

    node_drop_p = max(min(node_drop_p, 1), 0)

    if task_mode == 'future_actions_ordered':
        raise NotImplementedError
        # need to refactor the padding / make a better RNN model
        """
        ys = [torch.unique_consecutive(y) for y in ys]
        ys = nn.utils.rnn.pad_sequence(ys, padding_value=num_classes, batch_first=True)
        """

    for x, edge_index, edge_attr, y in zip(xs, edge_indices, edge_attrs, ys):
        graph_data.append(
            Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        )

        if random.random() < node_drop_p:
            data_aug = node_dropping(x, edge_index, edge_attr, max_droppable)
            if data_aug is None:
                continue
            x_aug, edge_index_aug, edge_attr_aug = data_aug
            graph_data.append(
                Data(x=x_aug, edge_index=edge_index_aug, edge_attr=edge_attr_aug, y=y)
            )

    return graph_data