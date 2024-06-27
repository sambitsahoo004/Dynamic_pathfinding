import math
import random
import cv2
import matplotlib.pyplot as plt

# (70,550) --> (525,55) for path_1.png
# (55,55)  --> (540,540) for path_2.png

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0


class RRTStar:
    def __init__(self, start, goal, image, max_iterations, step_size, goal_sample_rate, connect_circle_radius):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.image = image
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.connect_circle_radius = connect_circle_radius
        self.node_list_start = [self.start]
        self.node_list_goal = [self.goal]

    # Plotting the path and returning it
    def plan(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap='gray')
        ax.plot(self.start.x, self.start.y, 'go')
        ax.plot(self.goal.x, self.goal.y, 'ro')

        for i in range(self.max_iterations):
            if random.random() < self.goal_sample_rate:
                if random.random() < .5:
                    rnd_node = Node(self.goal.x, self.goal.y)
                    new_node = self.connect(rnd_node, self.node_list_start, ax, 'b-')
                    if new_node:
                        path = self.check(new_node, ax, self.node_list_goal, 0)
                        if path:
                            return path
                else:
                    rnd_node = Node(self.start.x, self.start.y)
                    new_node = self.connect(rnd_node, self.node_list_goal, ax, 'm-')
                    if new_node:
                        path = self.check(new_node, ax, self.node_list_start, 1)
                        if path:
                            return path

            else:
                rnd_node = Node(random.uniform(0, self.image.shape[1]), random.uniform(0, self.image.shape[0]))
                new_node = self.connect(rnd_node, self.node_list_start, ax, 'b-')
                if new_node:
                    path = self.check(new_node, ax, self.node_list_goal, 0)
                    if path:
                        return path

                new_node = self.connect(rnd_node, self.node_list_goal, ax, 'm-')
                if new_node:
                    path = self.check(new_node, ax, self.node_list_start, 1)
                    if path:
                        return path

        plt.show()
        return None

    # Adding a randomly generated node to a tree and rewiring if possible 
    def connect(self, rnd_node, node_list, ax, colour):
        nearest_node = self.nearest_node(rnd_node, node_list)
        new_node = self.steer(nearest_node, rnd_node, self.step_size)

        if self.check_collision(nearest_node, new_node):
            near_nodes = self.near_nodes(new_node, self.connect_circle_radius, node_list)
            min_cost_node = nearest_node
            min_cost = nearest_node.cost + \
                self.distance(nearest_node, new_node)

            # Checking if parent node can be rewired so that cost of new_node is cheaper
            for node in near_nodes:
                if self.check_collision(node, new_node) and node.cost + self.distance(node, new_node) < min_cost:
                    min_cost_node = node
                    min_cost = node.cost + self.distance(node, new_node)

            new_node.parent = min_cost_node
            new_node.cost = min_cost_node.cost + \
                self.distance(min_cost_node, new_node)
            node_list.append(new_node)

            # Checking if nearby nodes can be rewired to the new_node in order to reduce cost of the nearby nodes
            for node in near_nodes:
                if self.check_collision(new_node, node) and new_node.cost + self.distance(new_node, node) < self.get_cost(node):
                    node.parent = new_node
                    node.cost = new_node.cost + self.distance(new_node, node)

            ax.plot([new_node.x, min_cost_node.x], [new_node.y, min_cost_node.y], colour)
            plt.pause(0.001)
            return new_node

        return None

    # Checking if a connection can be made between the two trees
    def check(self, new_node, ax, node_list, k):
        near_nodes = self.near_nodes(new_node, self.step_size, node_list)
        if self.nearest_node(new_node, node_list) not in near_nodes:
            near_nodes.append(self.nearest_node(new_node, node_list))

        for node in near_nodes:
            connect_node = node
            if self.check_collision(new_node, connect_node):
                if k: # Connecting from goal tree to the start tree
                    path = self.extract_path(connect_node, new_node)
                else: # Connecting from start tree to the goal tree
                    path = self.extract_path(new_node, connect_node)
                ax.plot([x for (x, y) in path], [y for (x, y) in path], color='yellow', linestyle='-', linewidth=3)
                plt.pause(0.001)
                plt.show()
                return path

        return None # Returning None if no connection can be made bw start and goal trees

    # Returns closest node to rnd_node
    def nearest_node(self, rnd_node, node_list):
        min_dist = float('inf')
        min_node = None

        for node in node_list:
            dist = self.distance(node, rnd_node)
            if dist < min_dist:
                min_dist = dist
                min_node = node

        return min_node

    # Steers a new node into existence by extending a link (of step_size) towards to_node
    def steer(self, from_node, to_node, step_size):
        dist = self.distance(from_node, to_node)

        if dist <= step_size:
            return to_node

        ratio = step_size / dist
        x = from_node.x + (to_node.x - from_node.x) * ratio
        y = from_node.y + (to_node.y - from_node.y) * ratio

        return Node(x, y)

    # Returns a list of nodes close to "node" (lie within "radius")
    def near_nodes(self, node, radius, node_list):
        near_nodes = []
        for near_node in node_list:
            if self.distance(near_node, node) <= radius:
                near_nodes.append(near_node)

        return near_nodes

    # Function to get the cost of a node from a tree
    def get_cost(self, node):
        cost = 0.0
        if node.parent:
            cost = node.parent.cost + self.distance(node, node.parent)

        return cost

    # Distance between any two nodes
    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    #Checking if the link from from_node to to_node is collision free
    def check_collision(self, from_node, to_node):
        x1, y1 = int(from_node.x), int(from_node.y)
        x2, y2 = int(to_node.x), int(to_node.y)

        points = self.bresenham_line(x1, y1, x2, y2)
        for point in points:
            x, y = point
            if (self.image[y, x] == [0, 0, 0]).all():
                return False  # Collision detected

        return True  # No collision

    # Line approximation
    def bresenham_line(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        points = []
        while x0 != x1 or y0 != y1:
            points.append((x0, y0))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        points.append((x0, y0))
        return points
    
    # To connect the start tree (from node1) and the goal tree (from node2) once a connection can be established and returns the path
    def extract_path(self, node1, node2):
        path = []
        node = node1
        while node:
            path.append((node.x, node.y))
            node = node.parent
        path = path[::-1]

        node = node2
        while node:
            path.append((node.x, node.y))
            node = node.parent

        print("Cost of Path:", node1.cost + node2.cost+self.distance(node1, node2))

        return path


# Example usage
start = (55,55)
goal = (540,540)
image = cv2.imread("path_2.png", 1)

rrt_star = RRTStar(start, goal, image, max_iterations=5000,step_size=10, goal_sample_rate=0.1, connect_circle_radius=50)
path = rrt_star.plan()

if path is None:
    print("No valid path found")
else:
    print("Path found:", path)
