"""RRT* Path Planner for the articulated G2T vehicle, incorporating obstacle inflation and kinematic turning constraints."""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np


class Node:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent: Optional[Node] = None


class RRTStar:
    def __init__(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        landmarks: np.ndarray,          # (N, 2) estimated landmark centers from SLAM
        landmark_radius: float,         # nominal obstacle radius (e.g. 0.30 m)
        world_size: Tuple[float, float], # (width, height)
        walls: np.ndarray,              # (M, 4) [x0, y0, x1, y1] boundary walls
        L0: float = 1.20,
        max_steer_rad: float = math.radians(35.0),
        clearance: float = 0.65,         # W0/2 (0.40m) + margin (0.25m) for articulated sweep
        max_step: float = 1.0,
        search_radius: float = 2.0,
        goal_sample_rate: float = 0.15,
        goal_tolerance: float = 0.5,
        max_nodes: int = 1500
    ):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.landmarks = landmarks
        self.landmark_radius = landmark_radius
        self.world_size = world_size
        self.walls = walls
        self.L0 = L0
        self.max_steer_rad = max_steer_rad
        self.clearance = clearance
        self.max_step = max_step
        self.search_radius = search_radius
        self.goal_sample_rate = goal_sample_rate
        self.goal_tolerance = goal_tolerance
        self.max_nodes = max_nodes

        self.nodes: List[Node] = [self.start]
        self.R_min = L0 / math.tan(max_steer_rad) # minimum turning radius

    def _sample(self) -> Tuple[float, float]:
        """Sample a point from the state space, biasing toward the goal."""
        if random.random() < self.goal_sample_rate:
            return self.goal.x, self.goal.y
        w, h = self.world_size
        # sample within the bounding walls minus the clearance
        x = random.uniform(-w / 2.0 + self.clearance, w / 2.0 - self.clearance)
        y = random.uniform(-h / 2.0 + self.clearance, h / 2.0 - self.clearance)
        return x, y

    def _in_collision(self, x: float, y: float) -> bool:
        """Check if point (x,y) violates boundary walls or collides with landmarks."""
        w, h = self.world_size
        # Bounding wall collision
        if abs(x) > w / 2.0 - self.clearance or abs(y) > h / 2.0 - self.clearance:
            return True
        
        # Landmark collision
        if len(self.landmarks) > 0:
            dists = np.hypot(self.landmarks[:, 0] - x, self.landmarks[:, 1] - y)
            if np.any(dists <= self.landmark_radius + self.clearance):
                return True
        return False

    def _is_segment_collision_free(self, n1: Node, n2: Node) -> bool:
        """Discretize the line segment and check for collisions."""
        dist = math.hypot(n2.x - n1.x, n2.y - n1.y)
        steps = int(max(5, dist / 0.05))
        for i in range(steps + 1):
            t = i / steps
            x = n1.x + (n2.x - n1.x) * t
            y = n1.y + (n2.y - n1.y) * t
            if self._in_collision(x, y):
                return False
        return True

    def _get_nearest_node(self, x: float, y: float) -> Node:
        dists = [math.hypot(n.x - x, n.y - y) for n in self.nodes]
        return self.nodes[np.argmin(dists)]

    def _steer(self, n_from: Node, x_to: float, y_to: float) -> Node:
        dist = math.hypot(x_to - n_from.x, y_to - n_from.y)
        if dist <= self.max_step:
            return Node(x_to, y_to)
        
        angle = math.atan2(y_to - n_from.y, x_to - n_from.x)
        x_new = n_from.x + self.max_step * math.cos(angle)
        y_new = n_from.y + self.max_step * math.sin(angle)
        return Node(x_new, y_new)

    def _check_kinematic_feasibility(self, parent: Node, child: Node) -> bool:
        """Check if turning radius constraint is respected at parent.
        
        If parent has a parent (grandparent), we check the angle difference between
        the grandparent->parent segment and parent->child segment.
        """
        if parent.parent is None:
            return True # start node has no parent, so any angle is fine
        
        gp = parent.parent
        dx1, dy1 = parent.x - gp.x, parent.y - gp.y
        dx2, dy2 = child.x - parent.x, child.y - parent.y
        
        len1 = math.hypot(dx1, dy1)
        len2 = math.hypot(dx2, dy2)
        if len1 < 1e-4 or len2 < 1e-4:
            return True
            
        dot = dx1 * dx2 + dy1 * dy2
        cos_angle = np.clip(dot / (len1 * len2), -1.0, 1.0)
        angle_diff = math.acos(cos_angle)
        
        # Max angle difference based on R_min and segment length:
        # For a circular arc of radius R_min, angle change is segment_length / R_min
        max_angle = len2 / self.R_min
        return angle_diff <= max_angle

    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """Run the RRT* planning algorithm."""
        for i in range(self.max_nodes):
            x_rand, y_rand = self._sample()
            n_nearest = self._get_nearest_node(x_rand, y_rand)
            n_new = self._steer(n_nearest, x_rand, y_rand)

            if self._in_collision(n_new.x, n_new.y):
                continue

            if not self._is_segment_collision_free(n_nearest, n_new):
                continue

            # Find neighboring nodes
            neighbors = []
            for n in self.nodes:
                d = math.hypot(n.x - n_new.x, n.y - n_new.y)
                if d <= self.search_radius:
                    neighbors.append(n)

            # Choose parent with minimum cost
            min_cost = n_nearest.cost + math.hypot(n_new.x - n_nearest.x, n_new.y - n_nearest.y)
            best_parent = n_nearest

            for n in neighbors:
                cost = n.cost + math.hypot(n_new.x - n.x, n_new.y - n.y)
                if cost < min_cost:
                    if self._is_segment_collision_free(n, n_new):
                        if self._check_kinematic_feasibility(n, n_new):
                            min_cost = cost
                            best_parent = n

            n_new.parent = best_parent
            n_new.cost = min_cost
            self.nodes.append(n_new)

            # Rewire neighbors
            for n in neighbors:
                cost = n_new.cost + math.hypot(n.x - n_new.x, n.y - n_new.y)
                if cost < n.cost:
                    if self._is_segment_collision_free(n_new, n):
                        if self._check_kinematic_feasibility(n_new, n):
                            n.parent = n_new
                            n.cost = cost

            # Check if goal is reached
            if math.hypot(n_new.x - self.goal.x, n_new.y - self.goal.y) <= self.goal_tolerance:
                # Add goal node to tree if it's not already connected
                n_goal = Node(self.goal.x, self.goal.y)
                if self._is_segment_collision_free(n_new, n_goal):
                    n_goal.parent = n_new
                    n_goal.cost = n_new.cost + math.hypot(n_goal.x - n_new.x, n_goal.y - n_new.y)
                    self.nodes.append(n_goal)
                    print(f"[RRT*] Path found in {i} iterations!")
                    return self._extract_path(n_goal)

        # If goal wasn't reached, try to connect the closest node to goal
        closest_node = self.nodes[0]
        min_dist_to_goal = math.hypot(closest_node.x - self.goal.x, closest_node.y - self.goal.y)
        for n in self.nodes:
            d = math.hypot(n.x - self.goal.x, n.y - self.goal.y)
            if d < min_dist_to_goal:
                closest_node = n
                min_dist_to_goal = d
        
        if min_dist_to_goal <= 1.5: # if fairly close, connect it
            n_goal = Node(self.goal.x, self.goal.y)
            if self._is_segment_collision_free(closest_node, n_goal):
                n_goal.parent = closest_node
                n_goal.cost = closest_node.cost + math.hypot(n_goal.x - closest_node.x, n_goal.y - closest_node.y)
                self.nodes.append(n_goal)
                print(f"[RRT*] Close fallback path found!")
                return self._extract_path(n_goal)

        print("[RRT*] Path NOT found!")
        return None

    def _extract_path(self, node: Node) -> List[Tuple[float, float]]:
        path = []
        curr = node
        while curr is not None:
            path.append((curr.x, curr.y))
            curr = curr.parent
        return path[::-1]

    def shortcut_path(self, path: List[Tuple[float, float]], iterations: int = 150) -> List[Tuple[float, float]]:
        """Apply shortcutting to smooth the path while maintaining feasibility."""
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        for _ in range(iterations):
            if len(smoothed) <= 2:
                break
            # Pick two random indexes
            i = random.randint(0, len(smoothed) - 3)
            j = random.randint(i + 2, len(smoothed) - 1)
            
            n1 = Node(smoothed[i][0], smoothed[i][1])
            n2 = Node(smoothed[j][0], smoothed[j][1])
            
            # Check collision
            if self._is_segment_collision_free(n1, n2):
                # Also check kinematic feasibility if parent segments exist
                feasible = True
                if i > 0:
                    n_parent = Node(smoothed[i-1][0], smoothed[i-1][1])
                    n_parent.parent = None # we don't care about earlier segments here
                    # check turning constraint between parent->n1 and n1->n2
                    n1.parent = n_parent
                    feasible = self._check_kinematic_feasibility(n1, n2)
                
                if feasible and j < len(smoothed) - 1:
                    # check turning constraint between n1->n2 and n2->n_child
                    n_child = Node(smoothed[j+1][0], smoothed[j+1][1])
                    n2.parent = n1
                    feasible = self._check_kinematic_feasibility(n2, n_child)
                    
                if feasible:
                    # Remove intermediate nodes
                    smoothed = smoothed[:i+1] + smoothed[j:]
                    
        return smoothed
