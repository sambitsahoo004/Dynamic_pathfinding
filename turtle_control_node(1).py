import math
import random
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from math import sqrt, atan2

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


class TurtleControlNode:
    def __init__(self):
        rospy.init_node('turtle_control_node', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_pose)
        self.pose = Pose()
        self.rate = rospy.Rate(10)

        # PID controller constants
        self.Kp = 10.0  # Proportional gain
        self.Ki = 0.2   # Integral gain
        self.Kd = 2.0   # Derivative gain

        self.prev_error = 0.0  # Previous error for derivative term
        self.total_error = 0.0  # Accumulated error for integral term

        # Load the image
        self.image = cv2.imread("path_2.png", cv2.IMREAD_GRAYSCALE)

        rospy.loginfo("TurtleControlNode initialized")

    def update_pose(self, data):
        self.pose = data
        self.pose.x = round(self.pose.x, 4)
        self.pose.y = round(self.pose.y, 4)
        rospy.loginfo(f"Pose updated: x={self.pose.x}, y={self.pose.y}, theta={self.pose.theta}")

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

    def move_to_point(self, target_x, target_y):
        rospy.loginfo(f"Moving to point: x={target_x}, y={target_y}")
        distance_tolerance = 0.1

        vel_msg = Twist()

        while self.euclidean_distance(target_x, target_y) >= distance_tolerance:
            vel_msg.linear.x = self.linear_vel(target_x, target_y)
            vel_msg.linear.y = 0
            vel_msg.linear.z = 0

            vel_msg.angular.x = 0
            vel_msg.angular.y = 0
            vel_msg.angular.z = self.angular_vel(target_x, target_y)

            rospy.loginfo(f"Publishing velocity: linear.x={vel_msg.linear.x}, angular.z={vel_msg.angular.z}")
            self.velocity_publisher.publish(vel_msg)

            # Display the turtle's position on the image
            position_image = self.image.copy()
            position_x = int(self.pose.x * position_image.shape[1] / 11.0)
            position_y = int((11.0 - self.pose.y) * position_image.shape[0] / 11.0)
            cv2.circle(position_image, (position_x, position_y), 5, (0, 0, 255), -1)
            cv2.imshow("Turtle Position", position_image)
            cv2.waitKey(1)

            self.rate.sleep()

        vel_msg.linear.x = 0
        vel_msg.angular.z = 0
        self.velocity_publisher.publish(vel_msg)
        rospy.loginfo("Reached the point")


def main():
    start = (55, 55)
    goal = (540, 540)
    image_path = "path_2.png"

    rrt_star = RRTStar(start, goal, image_path, max_iterations=1000, step_size=10, goal_sample_rate=0.1, connect_circle_radius=50)
    path = rrt_star.plan()

    if path is None:
        rospy.loginfo("No valid path found")
    else:
        rospy.loginfo(f"Path found: {path}")

        turtle_control_node = TurtleControlNode()

        for point in path:
            target_x = point[0] / rrt_star.image.shape[1] * 11.0
            target_y = 11.0 - point[1] / rrt_star.image.shape[0] * 11.0
            turtle_control_node.move_to_point(target_x, target_y)

if __name__ == "__main__":
    main()

