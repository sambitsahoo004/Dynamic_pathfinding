import math
import random
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, image_path, max_iterations, step_size, goal_sample_rate, connect_circle_radius):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.image_path = image_path  # Store image path
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.connect_circle_radius = connect_circle_radius
        self.node_list = [self.start]
        self.tree = KDTree([(self.start.x, self.start.y)])
        self.image = cv2.imread("path_2.png", cv2.IMREAD_GRAYSCALE)  # Load image here

    def plan(self):
        fig, ax = plt.subplots()
        
        if self.image is None:
            raise ValueError(f"Failed to load image from {self.image_path}")

        ax.imshow(self.image, cmap='gray')
        ax.plot(self.start.x, self.start.y, 'go')
        ax.plot(self.goal.x, self.goal.y, 'ro')

        found_path = False
        path = None
        min_cost = float('inf')

        for i in range(self.max_iterations):
            if random.random() < self.goal_sample_rate:
                rnd_node = Node(self.goal.x, self.goal.y)
            else:
                rnd_node = Node(random.uniform(0, self.image.shape[1]), random.uniform(0, self.image.shape[0]))

            new_node = self.connect(rnd_node, self.node_list, ax)
            if new_node:
                if self.distance(new_node, self.goal) <= self.step_size:
                    final_node = self.steer(new_node, self.goal, self.step_size)
                    if self.check_collision(new_node, final_node):
                        final_node.parent = new_node
                        if not found_path or final_node.cost < min_cost:
                            path = self.extract_path(final_node)
                            found_path = True
                            min_cost = final_node.cost

            if found_path:
                self.draw_path(path, ax, 'red')
                plt.pause(0.001)
                for line in ax.lines[-1:]:  # Remove the previous path to draw the new one
                    line.remove()

        if path:
            self.draw_path(path, ax, 'yellow')
            plt.show()
            return path

        plt.show()
        return None

    def connect(self, rnd_node, node_list, ax):
        nearest_node = self.nearest_node(rnd_node, node_list)
        new_node = self.steer(nearest_node, rnd_node, self.step_size)

        if self.check_collision(nearest_node, new_node):
            near_nodes = self.near_nodes(new_node, self.connect_circle_radius, node_list)
            min_cost_node = nearest_node
            min_cost = nearest_node.cost + self.distance(nearest_node, new_node)

            for node in near_nodes:
                if self.check_collision(node, new_node) and node.cost + self.distance(node, new_node) < min_cost:
                    min_cost_node = node
                    min_cost = node.cost + self.distance(node, new_node)

            new_node.parent = min_cost_node
            new_node.cost = min_cost_node.cost + self.distance(min_cost_node, new_node)
            node_list.append(new_node)
            self.tree = KDTree([(node.x, node.y) for node in node_list])

            for node in near_nodes:
                if self.check_collision(new_node, node) and new_node.cost + self.distance(new_node, node) < self.get_cost(node):
                    node.parent = new_node
                    node.cost = new_node.cost + self.distance(new_node, node)

            ax.plot([new_node.x, min_cost_node.x], [new_node.y, min_cost_node.y], 'b-')
            plt.pause(0.001)
            return new_node

        return None

    def nearest_node(self, rnd_node, node_list):
        _, idx = self.tree.query((rnd_node.x, rnd_node.y))
        return node_list[idx]

    def steer(self, from_node, to_node, step_size):
        dist = self.distance(from_node, to_node)

        if dist <= step_size:
            return to_node

        ratio = step_size / dist
        x = from_node.x + (to_node.x - from_node.x) * ratio
        y = from_node.y + (to_node.y - from_node.y) * ratio

        return Node(x, y)

    def near_nodes(self, node, radius, node_list):
        idxs = self.tree.query_ball_point((node.x, node.y), radius)
        return [node_list[i] for i in idxs]

    def get_cost(self, node):
        cost = 0.0
        if node.parent:
            cost = node.parent.cost + self.distance(node, node.parent)

        return cost

    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def check_collision(self, from_node, to_node):
        x1, y1 = int(from_node.x), int(from_node.y)
        x2, y2 = int(to_node.x), int(to_node.y)

        points = self.bresenham_line(x1, y1, x2, y2)
        for point in points:
            x, y = point
            if self.image[y, x] == 0:
                return False

        return True

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
    
    def extract_path(self, node):
        path = []
        while node:
            path.append((node.x, node.y))
            node = node.parent
        return path[::-1]
    
    def get_path_cost(self, path):
        cost = 0.0
        for i in range(len(path) - 1):
            cost += self.distance(Node(*path[i]), Node(*path[i+1]))
        return cost
    
    def draw_path(self, path, ax, color):
        ax.plot([x for (x, y) in path], [y for (x, y) in path], color=color, linestyle='-', linewidth=2)

# Example usage
if __name__ == "__main__":
    start = (55, 55)
    goal = (540, 540)
    image = cv2.imread("path_2.png", cv2.IMREAD_GRAYSCALE)

    rrt_star = RRTStar(start, goal, image, max_iterations=5000, step_size=5, goal_sample_rate=0.1, connect_circle_radius=100)
    path = rrt_star.plan()

    if path is None:
        print("No valid path found")
    else:
        print("Path found:", path)

