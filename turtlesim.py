#!/usr/bin/env python3

import cv2
import numpy as np
import heapq
import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
from math import pow, atan2, sqrt

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

def find_circle_center(image, color):
    # Convert image to BGR color space
    bgr_image = image.copy()

    # Define color range based on the given color in BGR format
    lower_color = np.array(color, dtype=np.uint8)
    upper_color = np.array(color, dtype=np.uint8)

    
    mask = cv2.inRange(bgr_image, lower_color, upper_color)

   
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

   
    centers = []
    for contour in contours:
        
        M = cv2.moments(contour)

        
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            centers.append((cX, cY))

    return centers

def is_obstacle(image, position):
    # Check if the given position in the image is an obstacle (black pixel)
    pixel = image[position[1], position[0]]
    return np.array_equal(pixel, [0, 0, 0])

def path_planning(image, start, end):
    # Create the start and end nodes
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize the open and closed lists
    open_list = []
    closed_list = []

    # Add the start node to the open list
    heapq.heappush(open_list, start_node)

    # Run the BiT* algorithm
    while open_list:
        # Get the current node from the open list
        current_node = heapq.heappop(open_list)

        # Add the current node to the closed list
        closed_list.append(current_node)

        # Check if the current node is the goal node
        if current_node == end_node:
            # Path found, trace back the path
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            path.reverse()
            return path

        # Generate neighboring nodes
        neighbors = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Check if the new position is within the image boundaries
            if node_position[0] < 0 or node_position[0] >= image.shape[1] or node_position[1] < 0 or node_position[1] >= image.shape[0]:
                continue

            # Check if the new position is an obstacle
            if is_obstacle(image, node_position):
                continue

            # Create a new node
            new_node = Node(current_node, node_position)

            # Calculate the cost to move to the neighbor
            new_node.g = current_node.g + 1
            new_node.h = ((new_node.position[0] - end_node.position[0]) ** 2) + ((new_node.position[1] - end_node.position[1]) ** 2)
            new_node.f = new_node.g + new_node.h

            # Check if the neighbor is already in the closed list
            if new_node in closed_list:
                continue

            # Check if the neighbor is already in the open list
            existing_node = next((node for node in open_list if node == new_node), None)
            if existing_node is not None and new_node.g >= existing_node.g:
                continue

            # Add the neighbor to the open list
            heapq.heappush(open_list, new_node)

    return []  # No path found

class TurtleControlNode:
    def __init__(self):
        rospy.init_node('turtle_control_node')
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.update_pose)
        self.pose = Pose()
        self.rate = rospy.Rate(10)

        # PID controller constants
        self.Kp = 5.0  # Proportional gain
        self.Ki = 0.1 # Integral gain
        self.Kd = 1.0  # Derivative gain

        self.prev_error = 0.0  # Previous error for derivative term
        self.total_error = 0.0  # Accumulated error for integral term

    def update_pose(self, data):
        self.pose = data
        self.pose.x = round(self.pose.x, 4)
        self.pose.y = round(self.pose.y, 4)

    def euclidean_distance(self, target_x, target_y):
        return sqrt(pow((target_x - self.pose.x), 2) + pow((target_y - self.pose.y), 2))

    def linear_vel(self, target_x, target_y, constant=3.0, const = 1.0):
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

   

# Download the image from the provided Google Drive link
import requests
from io import BytesIO

file_id = '1hF_TvZSbJG5z7_CKnmPJOfH1QM_gDXNm'
url = f'https://drive.google.com/uc?id={file_id}'
response = requests.get(url)
image_data = response.content
image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

# Define the colors for the start and goal points in BGR format
start_color = (76, 177, 34)  # Green color
goal_color = (36, 28, 237)  # Red color

# Find the center coordinates of the start and goal points
start_points = find_circle_center(image, start_color)
goal_points = find_circle_center(image, goal_color)

# Initialize the turtle control node
turtle_control_node = TurtleControlNode()

# Apply the BiT* algorithm for each pair of start and goal points
for start_point, goal_point in zip(start_points, goal_points):
    # Apply the BiT* algorithm
    path = path_planning(image, start_point, goal_point)

    # Move the turtle along the path
    for point in path:
       target_x = point[0] / image.shape[1] * 11.0
       target_y = 11.0 - point[1] / image.shape[0] * 11.0
       turtle_control_node.move_to_point(image, target_x, target_y)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
