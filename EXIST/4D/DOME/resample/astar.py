# Sample code from https://www.redblobgames.com/pathfinding/a-star/
import heapq
import collections
import numpy as np
import math

class SimpleGraph:
    def __init__(self):
        self.edges = {}
    
    def neighbors(self, id):
        return self.edges[id]


class Queue:
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, x):
        self.elements.append(x)
    
    def get(self):
        return self.elements.popleft()

# utility functions for dealing with square grids
def from_id_width(id, width):
    return (id % width, id // width)

def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = ">"
        if x2 == x1 - 1: r = "<"
        if y2 == y1 + 1: r = "v"
        if y2 == y1 - 1: r = "^"
    if 'start' in style and id == style['start']: r = "A"
    if 'goal' in style and id == style['goal']: r = "Z"
    if 'path' in style and id in style['path']: r = "@"
    if id in graph.walls: r = "#" * width
    return r

def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
        print()


class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls
    
    def neighbors(self, id):
        (x, y) = id
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        if (x + y) % 2 == 0: results.reverse() # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

    @classmethod
    def from_voxel_bev_map(cls, mask,mark_margin_obs=True):
        # assert mask.dtype == np.bool and mask.ndim == 2
        width,height = mask.shape[:2]
        grid = cls(width, height)
        if mark_margin_obs:
            mask[:,[0,-1]]=0
            mask[[0,-1],:]=0
        grid.walls = [(x,y) for x,y in np.argwhere(mask==0)]
        return grid


class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}
    
    def cost(self, from_node, to_node):
        
        return self.weights.get(to_node, 1)
    @classmethod
    def from_voxel_bev_map(cls, mask,cost_map=None,mark_margin_obs=True):
        # assert mask.dtype == np.bool and mask.ndim == 2
        width,height = mask.shape[:2]
        if mark_margin_obs:
            mask[:,[0,-1]]=0
            mask[[0,-1],:]=0
        grid = cls(width, height)
        grid.walls = [(x,y) for x,y in np.argwhere(mask==0)]
        if cost_map is not None:
            grid.weights = {(x,y):cost_map[x,y] for x in range(width) for y in range(height)}
        return grid

    

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

# def heuristic(a, b):
#     (x1, y1) = a
#     (x2, y2) = b
#     return abs(x1 - x2) + abs(y1 - y2)
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

# def a_star_search_distance_cost(graph,distancecostmap, start, goal):
#     frontier = PriorityQueue()
#     frontier.put(start, 0)
#     came_from = {}
#     cost_so_far = {}
#     distance_cost = {}
#     came_from[start] = None
#     cost_so_far[start] = 0
    
#     while not frontier.empty():
#         current = frontier.get()
        
#         if current == goal:
#             break
        
#         for next in graph.neighbors(current):
#             new_cost = cost_so_far[current] + graph.cost(current, next)
#             if next not in cost_so_far or new_cost < cost_so_far[next]:
#                 cost_so_far[next] = new_cost
#                 distance_cost
#                 priority = new_cost + heuristic(goal, next)
#                 frontier.put(next, priority)
#                 came_from[next] = current
    
#     return came_from, cost_so_far

def breadth_first_search_3(graph, start, goal):
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current

    return came_from



def distcost(distancecostmap,x, y, safty_value=2,w=50):
    # large safty value makes the path more away from the wall
    # However, if it is too large, almost grid will get max cost
    # which leads to eliminate the meaning of distance cost.
    max_distance_cost = np.max(distancecostmap)
    distance_cost = max_distance_cost-distancecostmap[x][y]
    #if distance_cost > (max_distance_cost/safty_value):
    #    distance_cost = 1000
    #    return distance_cost
    return w * distance_cost # E5 223 - 50

# start, goal = (1, 4), (7, 8)
# came_from, cost_so_far = a_star_search(diagram4, start, goal)
# draw_grid(diagram4, width=3, point_to=came_from, start=start, goal=goal)
# print()
# draw_grid(diagram4, width=3, number=cost_so_far, start=start, goal=goal)
# print()
# draw_grid(diagram4, width=3, path=reconstruct_path(came_from, start=start, goal=goal))



def create_bev_from_pc(occ_pc_road, resolution_2d, max_dist):
    """
    Create BEV map from point cloud
    Args:
        occ_pc_road: point cloud of the road
        resolution_2d: resolution of the BEV map
        max_dist: maximum distance to the road
    Returns:
        bev_road: BEV map of the road, Binary map
        nearest_distance: nearest distance to the road
        (x_min, y_min): minimum coordinates of the BEV map
    """
    # set z=0 and compute grid parameters
    occ_pc_road = occ_pc_road[:,:2]
    x_min, x_max = occ_pc_road[:,0].min(), occ_pc_road[:,0].max()
    y_min, y_max = occ_pc_road[:,1].min(), occ_pc_road[:,1].max()
    n_x = int((x_max - x_min) / resolution_2d)
    n_y = int((y_max - y_min) / resolution_2d)

    # Create grid coordinates efficiently
    x_coords = np.linspace(x_min + resolution_2d/2, x_max - resolution_2d/2, n_x)
    y_coords = np.linspace(y_min + resolution_2d/2, y_max - resolution_2d/2, n_y)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    grid_coords = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Fit KNN model and compute distances
    knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree', n_jobs=-1).fit(occ_pc_road)
    distances, _ = knn.kneighbors(grid_coords)
    mean_distances = distances.mean(axis=1)

    # Create and populate nd and occ_2d arrays
    nearest_distance = mean_distances.reshape(n_y, n_x)
    bev_road = (nearest_distance < max_dist).astype(np.int32)

    return bev_road, nearest_distance, (x_min, y_min)