import cv2
import torch
import numpy as np
from collections import deque
from egtea_gaze.utils import resolution

from graph.node import Node

""" 
Helper functions for all things graph-related
"""

# Graph Search
def dfs(start_node: Node) -> set[Node]:
    visited = set()
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor, _, _ in node.neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)

    return visited

def get_all_nodes(start_node: Node, mode: str='dfs') -> list[Node]:
    """
    Returns all nodes in the graph by running dfs or bfs starting from `start_node`
    Args:
        start_node (Node): pointer to node in graph to start search from
        mode (str): the search method to use (e.g. 'bfs', 'dfs').
                                Currently only supports 'dfs' which is the default.
                                Things will break if you use anything else atm
    """
    if mode == 'dfs':
        all_nodes = list(dfs(start_node))
    elif mode == 'bfs':
        #TODO add bfs
        raise NotImplementedError

    return all_nodes

# Adding nodes to scene graph
def get_angle_bin(x: float, y: float, num_bins: int=8) -> float:
    """
    Split [0, 2pi) into `num_bins` bins and return the bin closest to arctan(y/x)
    Args:
        x (float): the denominator for arctan
        y (float): the numerator for arctan
        num_bins (int): if > 0, use bins. 
                                  if = -1, simply returns arctan(y/x). 
                                  default = 8
    Returns:
        the bin (discrete) or exact angle (fine-grained) in radians
    """
    # place in range [0, 2pi)
    shift_angle = lambda x: x + 2*np.pi if x < 0 else x
    theta = shift_angle(np.arctan2(y, x))

    if num_bins == -1: # fine-grained, use raw angle
        return theta

    bins = [i * 2 * np.pi / num_bins for i in range(num_bins)]
    bin_width = 2 * np.pi / num_bins
    bin_index = round(theta / bin_width)
    return bins[bin_index] if bin_index < num_bins else bins[0]

def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def ransac(kp1, kp2, matches):
    #TODO change hardcoded ransac params
    if len(matches) < 10:
        return None, []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 7.0)

    inliers = [matches[i] for i in range(len(matches)) if mask[i]]

    return H, inliers

def merge(
    curr_node: Node, 
    keypoints: list, 
    descriptors: list, 
    visit: list[int], 
    label: str, 
    inlier_thresh: float, 
    one_label_assumption: bool=True
):
    """
    Checks all nodes starting from `curr_node` for matching node to prevent duplicates
    If there is a match, simply add visit to matching node.
    If there isn't, signal that a new node should be created. 
    Args:
        curr_node (Node): the most recently added node in the graph
        keypoints (list[MatLike]): a list of keypoints (e.g. those returned from SIFT)
        descriptors (list[MatLike]): a list of descriptors (e.g. those returned from SIFT)
        visit (list[int]): a list containing the first and last frame of (potentially new) node
        label (str): the label of the (potentially new) object
        inlier_thresh (float): minimum bound for inlier ratio
        one_label_assumption (bool): whether we assume the scene has only one of each object class => smaller, more accurate graphs if true
    Returns:
        The matching node if a match is indeed found.
        None if no good matches exist (and a new node should created)
    """
    if curr_node.object_label == 'root':
        return None

    visited = set([curr_node])
    queue = deque([curr_node])

    #TODO for now, just using first frame's features.
    # later we can try middle frame, average, concat, etc
    kp1, des1 = keypoints[0], descriptors[0]
    
    most_likely_match = None
    while queue:
        node = queue.popleft()

        if node.object_label == label:
            if one_label_assumption:
                most_likely_match = node
                break
            kp2, des2 = node.keypoints[0], node.descriptors[0]
            if not (des1 is None or des2 is None):
                matches = match_features(des1, des2)
                H, inliers = ransac(kp1, kp2, matches)

                if H is not None:
                    inlier_ratio = len(inliers) / len(matches)

                    #TODO get rid of break and use best inlier ratio so far or even a combo of the two?
                    if inlier_ratio > inlier_thresh:
                        most_likely_match = node
                        break

        for neighbor, _, _ in node.neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    if most_likely_match is not None:
        most_likely_match.add_new_visit(visit)
        
    return most_likely_match

def update_graph(
    curr_node: Node, 
    label_counts: dict[str, int], 
    visit: list[int], 
    keypoints: list, 
    descriptors: list, 
    prev_gaze_pos: tuple[int, int], 
    curr_gaze_pos: tuple[int, int], 
    # the following three are lists for mutability => allows dynamic construction of dataset
    edge_data: list[torch.Tensor],
    edge_index: list[list[int]],
    num_nodes: list[int], 
    num_bins: int=8, 
    inlier_thresh: float=0.3
):
    """
    Update the scene graph.
    First, check if the object already exists in the scene via keypoint matching
    If it does, merge it (update visits attribute of existing node), else, create a new node
    Finally, add the new (or existing node) to the neighbors of the current node
    Args:
        curr_node (Node): the node currently being pointed to (the last object registered to the scene)
        label_counts (dict{str, int}): a dictionary of all detected objects during fixation and their counts
        visit (list[int]): a list containing the first and last frame of object fixation for the new object
        keypoints (list[MatLike]): list of keypoints per frame returned by SIFT for the new object
        descriptors (list[MatLike]): list of descriptors per frame returned by SIFT for the new object
        prev_gaze_pos (tuple[int, int]): the previous gaze x, y of the last registered object
        curr_gaze_pos (tuple[int, int]): the gaze x, y of the new object
        edge_data (list[Tensor]): the edge features for each edge
        edge_index (list[list[int]]): a 2 x num_edges mapping of edges i.e. (x[0][i], x[1][i]) represent edge i
                                        Bidirectional edges require two separate entries
        num_nodes (list[int]): the size/number of nodes of the current graph
        num_bins (int, optional): the number of potential angles that can be used to describe relative position 
                                    between current and new node. See get_angle_bin() for options.
        inlier_thresh (float, optional): inlier threshold for RANSAC in (0,1). Default is 0.3
    Returns:
        The next node (either new or existing)
    """
    prev_x, prev_y, curr_x, curr_y = prev_gaze_pos[0], prev_gaze_pos[1], curr_gaze_pos[0], curr_gaze_pos[1]
    dx, dy = curr_x - prev_x, curr_y - prev_y
    angle = get_angle_bin(dx, dy, num_bins)
    distance = np.sqrt(dx ** 2 + dy ** 2)

    # merge node if it exists
    most_likely_label = max(label_counts, key=label_counts.get)
    next_node = merge(curr_node, keypoints, descriptors, visit, most_likely_label, inlier_thresh)

    if next_node is None:
        next_node = Node(
            id = num_nodes[0],
            object_label=most_likely_label, 
            visits=[visit],
            keypoints=keypoints,
            descriptors=descriptors
        )
        num_nodes[0] += 1

    # link curr and next node, ignoring potential self-loops and duplicate edges
    if next_node != curr_node and not curr_node.has_neighbor(next_node):
        curr_node.add_neighbor(next_node, angle, distance) 

        # Essentially, we want root to be the sentinel node / entry point of the graph => one-way neighbor.
        if curr_node.object_label != 'root':  
            # NOTE: now, the angle and distance don't have any meaning, as the relevant info is in the edge_data.
            # in previous versions, they were "edge" features, but I've been lazy about deprecating them
            next_node.add_neighbor(curr_node, angle + np.pi if angle < np.pi else angle - np.pi, distance)

            prev_x, prev_y, curr_x, curr_y = prev_x / resolution[0], prev_y / resolution[1], curr_x / resolution[0], curr_y / resolution[1]
            edge_data.append(torch.tensor([prev_x, prev_y, curr_x, curr_y]))
            edge_data.append(torch.tensor([prev_x, prev_y, curr_x, curr_y]))

            edge_index[0].extend([curr_node.id, next_node.id])
            edge_index[1].extend([next_node.id, curr_node.id])

    return next_node

# Visualization
def print_levels(start_node: Node, use_degrees: bool=True):
    """
    Prints all nodes in the graph separated by level/distance from start
    """
    visited = set([start_node])
    queue = deque([(start_node, 'none', 'none')])
    
    curr_depth = 0
    while queue:
        level_size = len(queue)
        print(f'Depth: {curr_depth}')
        for _ in range(level_size):
            node, prev_obj, theta = queue.popleft()
            print('-----------------')
            print(f'Object: {node.object_label}')
            print(f'Visited at: {node.visits}')
            print(f'Visited from: {prev_obj}')
            if type(theta) == float and use_degrees:
                theta = theta * 180 / np.pi
            print(f'Angle from prev: {theta}')
            
            for neighbor, t, _ in node.neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, node.object_label, t))
        print('================')
        curr_depth += 1
    
"""
def draw_networkx_double_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items

def vis_graph(nodes):
    '''
    Args:
        nodes (list[Node]): a list of nodes 
    '''
    graph = nx.DiGraph()

    get_name = lambda node: str(sorted(node.objects, key=node.objects.get, reverse=True)[:2]) + ' || ' + str(node.visits)

    # create dot nodes from nodes
    for node in nodes:
        node_name = get_name(node)
        graph.add_node(node_name, visits=node.visits)
    
    # merge neighbors
    for node in nodes:
        for dir, neighbor in node.neighbors.items():
            node_name = get_name(node)
            neighbor_name = get_name(neighbor)
            graph.add_edge(node_name, neighbor_name, x_dir=dir)
    
    # save
    pos = nx.spring_layout(graph, seed=5)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph, pos, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)

    curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
    straight_edges = list(set(graph.edges()) - set(curved_edges))
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=straight_edges)
    arc_rad = 0.25
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=curved_edges,
        connectionstyle=f'arc3, rad = {arc_rad}'
    )

    edge_weights = nx.get_edge_attributes(graph, 'x_dir')
    curved_edge_labels = {edge: edge_weights[edge] for edge in curved_edges}
    straight_edge_labels = {edge: edge_weights[edge] for edge in straight_edges}
    draw_networkx_double_edge_labels(
        graph, pos, ax=ax,
        edge_labels=curved_edge_labels, rotate=False,
        rad=arc_rad
    )
    nx.draw_networkx_edge_labels(
        graph, pos, ax=ax, edge_labels=straight_edge_labels, rotate=False
    )

    plt.savefig('graph.png', format='PNG')
"""