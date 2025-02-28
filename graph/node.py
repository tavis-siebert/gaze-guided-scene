
### Node Class ###
class Node:
    """
    Attributes:
        id (int): the node's index within the adjacency list
        object_label (str): the object class/label (e.g. "cup")
        visits: (list[list[int]]): list of visits (each visit is a list of size 2 where the first entry is the first frame the object was visited, and the second is the last)
        keypoints (list[MatLike]): list of keypoints per frame returned by SIFT
        descriptors (list[MatLike]): list of descriptors per frame returned by SIFT
        neighbors (list(list[Node, float, float])): list of neighbors (each neighbor is a list of size 3 containing neighbor, angle, distance in that order)
    """
    def __init__(
        self, 
        id: int,
        object_label: str='', 
        visits: list[list[int]]=None, 
        keypoints: list=None, 
        descriptors: list=None, 
        neighbors: list=None
    ):
        self.id = id
        self.object_label = object_label
        self.visits = [] if visits is None else visits
        self.keypoints = [] if keypoints is None else keypoints
        self.descriptors = [] if descriptors is None else descriptors
        self.neighbors = [] if neighbors is None else neighbors

    
    def set_object_label(self, label):
        self.object_label = label

    # def set_end_of_curr_visit(self, frame):
        # self.visits[-1][1] = frame

    def add_new_visit(self, visit):
        self.visits.append(visit)
    
    def add_new_feature(self, kp, des):
        self.keypoints.append(kp)
        self.descriptors.append(des)

    def add_neighbor(self, neighbor, angle, distance):
        #TODO distance, angle are becoming deprecated
        self.neighbors.append([neighbor, angle, distance])

    def has_neighbor(self, node):
        neighbors = [n[0] for n in self.neighbors]
        return True if node in neighbors else False