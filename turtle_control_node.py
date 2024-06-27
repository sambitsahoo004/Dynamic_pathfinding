import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
import json
from math import pow, atan2, sqrt
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
    def __init__(self, start, goal, image, max_iterations, step_size, goal_sample_rate, connect_circle_radius):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.image = image
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.connect_circle_radius = connect_circle_radius
        self.node_list = [self.start]

    def plan(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image, cmap='gray')
        ax.plot(self.start.x, self.start.y, 'go')
        ax.plot(self.goal.x, self.goal.y, 'ro')

        for i in range(self.max_iterations):
            rnd_node = self.sample()
            new_node = self.connect(rnd_node)
            if new_node:
                path = self.rewire(new_node)
                if path:
                    self.draw_final_path(ax, path)
                    return path

        plt.show()
        return None

    def sample(self):
        if random.random() < self.goal_sample_rate:
            return Node(self.goal.x, self.goal.y)
        else:
            return Node(random.uniform(0, self.image.shape[1]), random.uniform(0, self.image.shape[0]))

    def connect(self, rnd_node):
        nearest_node = self.nearest_node(rnd_node)
        new_node = self.steer(nearest_node, rnd_node)

        if self.check_collision(nearest_node, new_node):
            near_nodes = self.near_nodes(new_node)
            min_cost_node = nearest_node
            min_cost = nearest_node.cost + self.distance(nearest_node, new_node)

            for node in near_nodes:
                if self.check_collision(node, new_node) and node.cost + self.distance(node, new_node) < min_cost:
                    min_cost_node = node
                    min_cost = node.cost + self.distance(node, new_node)

            new_node.parent = min_cost_node
            new_node.cost = min_cost
            self.node_list.append(new_node)

            return new_node

        return None

    def rewire(self, new_node):
        near_nodes = self.near_nodes(new_node)
        for node in near_nodes:
            if self.check_collision(new_node, node) and new_node.cost + self.distance(new_node, node) < node.cost:
                node.parent = new_node
                node.cost = new_node.cost + self.distance(new_node, node)

        if self.distance(new_node, self.goal) <= self.step_size:
            final_node = self.steer(new_node, self.goal)
            if self.check_collision(new_node, final_node):
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + self.distance(new_node, final_node)
                return self.extract_path()

        return None

    def nearest_node(self, rnd_node):
        kdtree = KDTree([(node.x, node.y) for node in self.node_list])
        _, idx = kdtree.query([rnd_node.x, rnd_node.y])
        return self.node_list[idx]

    def steer(self, from_node, to_node):
        dist = self.distance(from_node, to_node)
        if dist <= self.step_size:
            return to_node

        ratio = self.step_size / dist
        x = from_node.x + (to_node.x - from_node.x) * ratio
        y = from_node.y + (to_node.y - from_node.y) * ratio

        return Node(x, y)

    def near_nodes(self, node):
        radius = self.connect_circle_radius
        kdtree = KDTree([(nd.x, nd.y) for nd in self.node_list])
        indices = kdtree.query_ball_point([node.x, node.y], radius)
        return [self.node_list[i] for i in indices]

    def distance(self, node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def check_collision(self, from_node, to_node):
        x1, y1 = int(from_node.x), int(from_node.y)
        x2, y2 = int(to_node.x), int(to_node.y)

        points = self.bresenham_line(x1, y1, x2, y2)
        for point in points:
            x, y = point
            if (self.image[y, x] == 0):
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

    def extract_path(self):
        path = []
        node = self.goal
        while node:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()
        return path

    def draw_final_path(self, ax, path):
        ax.plot([x for (x, y) in path], [y for (x, y) in path], color='yellow', linestyle='-', linewidth=3)
        plt.pause(0.001)
        plt.show()

class TurtleControlNode:
    def __init__(self):
        rospy.init_node('turtle_control_node')
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_pose)
        self.pose = Pose()
        self.rate = rospy.Rate(10)

        # PID controller constants
        self.Kp = 5.0  # Proportional gain
        self.Ki = 0.002 # Integral gain
        self.Kd = 0.01  # Derivative gain

        self.prev_error = 0.0  # Previous error for derivative term
        self.total_error = 0.0  # Accumulated error for integral term

    def update_pose(self, data):
        self.pose = data
        self.pose.x = round(self.pose.x, 4)
        self.pose.y = round(self.pose.y, 4)

    def euclidean_distance(self, target_x, target_y):
        return sqrt(pow((target_x - self.pose.x), 2) + pow((target_y - self.pose.y), 2))

    def linear_vel(self, target_x, target_y, constant=3.0, const=1.0):
        return constant * self.euclidean_distance(target_x, target_y) + const

    def steering_angle(self, target_x, target_y):
        return atan2(target_y - self.pose.y, target_x - self.pose.x)

    def angular_vel(self, target_x, target_y):
        error = self.steering_angle(target_x, target_y) - self.pose.theta

        # Proportional term
        P = self.Kp * error

        # Integral term
        self.total_error += error
        I = self.Ki * self.total_error

        # Derivative term
        D = self.Kd * (error - self.prev_error)
        self.prev_error = error

        return P + I + D

    def move_to_point(self, image, target_x, target_y):
        distance_tolerance = 0.1
        vel_msg = Twist()

        while self.euclidean_distance(target_x, target_y) >= distance_tolerance:
            vel_msg.linear.x = self.linear_vel(target_x, target_y)
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0

            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = self.angular_vel(target_x, target_y)

            self.velocity_publisher.publish(vel_msg)
            position_image = image.copy()
            position_x = int(self.pose.x * image.shape[1] / 11.0)
            position_y = int((11.0 - self.pose.y) * image.shape[0] / 11.0)
            cv2.circle(position_image, (position_x, position_y), 5, (0, 0, 255), -1)
            cv2.imshow("Position", position_image)
            cv2.waitKey(1)
            self.rate.sleep()

        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)

    def follow_path(self, path):
        for point in path:
            target_x = point[0] / 11.0
            target_y = (11.0 - point[1]) / 11.0
            self.move_to_point(target_x, target_y)

def main():
    # Read the image
    image = cv2.imread("path_2.png", 0)  # Load the image in grayscale

    # Define start and goal points (adjust as needed)
    start_points = [(55, 55)]
    goal_points = [(540, 540)]

    # Initialize ROS node and TurtleControlNode
    rospy.init_node('turtle_control_node')
    turtle_control_node = TurtleControlNode()

    # Apply the RRT* algorithm for each pair of start and goal points
    for start_point, goal_point in zip(start_points, goal_points):
        # Apply the RRT* algorithm
        rrt_star = RRTStar(start_point, goal_point, image, max_iterations=5000, step_size=10, goal_sample_rate=0.1, connect_circle_radius=50)
        path = rrt_star.plan()

        # Move the turtle along the path if a valid path is found
        if path:
            turtle_control_node.follow_path(path)

    # Wait for a key press to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


