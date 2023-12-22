# Written by Caelan Garrett, modified a bit by Tomas Lozano-Perez
# yapf: disable
from math import cos, pi, sin
from random import randint, random, uniform
from time import time

from .geometry import AABB, Line, Object, Point, Polygon, convex_hull
from .robot import ConfigurationSpace


class TreeNode:

    def __init__(self, value, parent=None):
        self.value = value  # tuple of floats representing a configuration
        self.parent = parent  # another TreeNode
        self.children = []  # list of TreeNodes


class RRT:
    """RRT Tree."""

    def __init__(self, root: TreeNode, cspace: ConfigurationSpace):
        self.root = root  # root TreeNode
        self.cspace = cspace  # robot.ConfigurationSpace
        self.size = 1  # int length of path
        self.max_recursion = 1000  # int length of longest possible path

    def add_configuration(self, parent_node, child_value):
        child_node = TreeNode(child_value, parent_node)
        parent_node.children.append(child_node)
        self.size += 1
        return child_node

    # Brute force nearest, handles general distance functions
    def nearest(self, configuration):
        """
        Finds the nearest node by distance to configuration in the
        configuration space.

        Args:
            configuration: tuple of floats representing a configuration of a
                robot

        Returns:
            closest: TreeNode. the closest node in the configuration space
                to configuration
            distance: float. distance from configuration to closest
        """
        assert self.cspace.valid_configuration(configuration)

        def recur(node, depth=0):
            closest, distance = node, self.cspace.distance(
                node.value, configuration)
            if depth < self.max_recursion:
                for child in node.children:
                    (child_closest, child_distance) = recur(child, depth + 1)
                    if child_distance < distance:
                        closest = child_closest
                        distance = child_distance
            return closest, distance

        return recur(self.root)[0]

    def draw(self, window, color='black'):

        def recur(node, depth=0):
            node_point = Point(node.value[0], node.value[1])
            if depth < self.max_recursion:
                for child in node.children:
                    child_point = Point(child.value[0], child.value[1])
                    Line(node_point, child_point).draw(window,
                                                       color=color,
                                                       width=1)
                    recur(child, depth + 1)

        recur(self.root)


class Problem:
    """Defines a motion-planning problem."""

    def __init__(self,
                 x,
                 y,
                 robot,
                 obstacles,
                 start,
                 goal,
                 cspace,
                 display_tree=False):
        """
        Defines a motion planning problem.

        Args:
            x: float. the width of the map's area
            y: float. the height of the map's area
            robot: a robot.Robot instance
            obstacles: list of geometry.Objects self.robot can't move through
            start: tuple of floats: starting configuration of self.robot
            goal: tuple of floats: goal configuration of self.robot
            cspace: robot.Configuration space of self.robot
            display_tree: bool. if True, draw the generated plan trees
        """
        self.x = x
        self.y = y
        self.robot = robot
        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.region = AABB(Point(0, 0), Point(x, y))
        self.cspace = cspace
        self.display_tree = display_tree

        assert self.valid_configuration(self.start)
        assert self.valid_configuration(self.goal)

    # Generate a regular polygon that does not collide with other
    # obstacles or the robot at start and goal.
    def generate_random_regular_poly(self,
                                     num_verts,
                                     radius,
                                     angle=uniform(-pi, pi)):
        """
        Generates a regular polygon that does not collide with other
        obstacles or the robot at start and goal. This polygon is added to
        self.obstacles. To make it random, keep the default angle argument.

        Args:
            num_verts: int. the number of vertices of the polygon >= 3
            radius: float. the distance from the center of the polygon
                to any vertex > 0
            angle: float. the angle in radians between the origin and
                the first vertex. the default is a random value between
                -pi and pi.
        """
        (min_verts, max_verts) = num_verts
        (min_radius, max_radius) = radius
        assert not (min_verts < 3 or min_verts > max_verts or min_radius <= 0
               or min_radius > max_radius)
        reference = Point(random() * self.x, random() * self.y)
        distance = uniform(min_radius, max_radius)
        sides = randint(min_verts, max_verts)
        obj = Object(reference, [
            Polygon([
                Point(distance * cos(angle + 2 * n * pi / sides),
                      distance * sin(angle + 2 * n * pi / sides))
                for n in range(sides)
            ])
        ])
        if any([obj.collides(current) for current in self.obstacles]) or \
                obj.collides(self.robot.configuration(self.start)) or \
                obj.collides(self.robot.configuration(self.goal)) or \
                not self.region.contains(obj):
            self.generate_random_regular_poly(num_verts, radius, angle=angle)
        else:
            self.obstacles.append(obj)

    # Generate a polygon that does not collide with other
    # obstacles or the robot at start and goal.
    def generate_random_poly(self, num_verts, radius):
        """
        Generates a random polygon that does not collide with other
        obstacles or the robot at start and goal. This polygon is added to
        self.obstacles.

        Args:
            num_verts: int. the number of vertices of the polygon >= 3
            radius: float. a reference distance between the origin and some
                vertex of the polygon > 0
        """
        (min_verts, max_verts) = num_verts
        (min_radius, max_radius) = radius
        assert not (min_verts < 3 or min_verts > max_verts or min_radius <= 0
               or min_radius > max_radius)

        reference = Point(random() * self.x, random() * self.y)
        verts = randint(min_verts, max_verts)
        points = [Point(0, 0)]
        for i in range(verts):
            angle = 2 * pi * random()
            points.append(((max_radius - min_radius) * random() + min_radius)
                          * Point(cos(angle), sin(angle)))
        obj = Object(reference, [Polygon(convex_hull(points))])
        if any([obj.collides(current) for current in self.obstacles]) or \
                obj.collides(self.robot.configuration(self.start)) or \
                obj.collides(self.robot.configuration(self.goal)) or \
                not self.region.contains(obj):
            self.generate_random_poly(num_verts, radius)
        else:
            self.obstacles.append(obj)

    def valid_configuration(self, configuration):
        """
        Checks if the given configuration is valid in this Problem's
        configuration space and doesn't collide with any obstacles.

        Args:
            configuration: tuple of floats - tuple describing a robot
                configuration

        Returns:
            bool. True if the given configuration is valid, False otherwise
        """
        return self.cspace.valid_configuration(configuration) \
            and not self.collide(configuration)

    def collide(self, configuration):  # Check collision for configuration
        """
        Checks if the given configuration collides with any of this
        Problem's obstacles.

        Args:
            configuration: tuple of floats - tuple describing a robot
                configuration

        Returns:
            bool. True if the given configuration is in collision,
            False otherwise
        """
        config_robot = self.robot.configuration(configuration)
        return any([config_robot.collides(obstacle)
                    for obstacle in self.obstacles]) or \
            not self.region.contains(config_robot)

    def test_path(self, start, end):  # check path for collisions
        """
        Checks if the path from start to end collides with any obstacles.

        Args:

            start: tuple of floats - tuple describing robot's start
                configuration
            end: tuple of floats - tuple describing robot's end configuration

        Returns:
            bool. False if the path collides with any obstacles, True otherwise
        """
        path = self.cspace.path(start, end)
        for configuration in path:
            if self.collide(configuration):
                return False
        return True

    def safe_path(self, start, end):  # returns subset of path that is safe
        """
        Checks if the path from start to end collides with any obstacles.

        Args:
            start: tuple of floats - tuple describing robot's start
                configuration
            end: tuple of floats - tuple describing robot's end configuration

        Returns:
            list of tuples along the path that are not in collision.
        """
        path = self.cspace.path(start, end)
        safe_path = []
        for configuration in path:
            if self.collide(configuration):
                return safe_path
            safe_path.append(configuration)
        return safe_path

    def path_distance(self, path):  # add up distances along path
        """
        Adds up the distance along the given path.

        Args:
            path: list of tuples describing configurations of the robot

        Returns:
            float. the total distance along the path
        """
        distance = 0
        for i in range(len(path) - 1):
            distance += self.cspace.distance(path[i], path[i + 1])
        return distance

    # Given a path (list of configurations) return a (shorter) path
    def smooth_path(self, path, attempts=100):  # random short-cutting
        """
        Given a path (list of configurations), return a possibly shorter
        path.

        Args:

            path: list of tuples describing configurations of the robot
            attemps: int. the number of times to try to smooth the path

        Returns:
            list of tuples describing configurations of the robot that is
            possibly shorter than the given path
        """
        #########################
        # Your code here

        smoothed_path = path
        for attempt in range(attempts):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = randint(0, len(smoothed_path) - 1)
            j = randint(0, len(smoothed_path) - 1)
            if i == j or abs(i - j) == 1:
                continue
            one, two = i, j
            if j < i:
                one, two = j, i
            if self.test_path(smoothed_path[one], smoothed_path[two]):
                smoothed_path = smoothed_path[:one + 1] + smoothed_path[two:]
        return smoothed_path

        #########################
        raise NotImplemented

    def draw_robot_path(self, path, color=None):
        """
        Draws the robot in each configuration along a path in a Tkinter
        window.

        Args:
            path: list of tuples describing configurations
                of the robot
            color: string. Tkinter color of the robot
        """
        for i in range(1, len(path) - 1):  # don't draw start and end
            self.robot.configuration(path[i]).draw(self.window, color)
            self.window.update()
            input('Next?')

    def run_and_display(self, method, display=True):
        """
        Runs the Problem with the given methods and draws the path in a
        Tkinter window.

        Args:
            method: list of strings of the methods to try, must be "rrt" or
                "birrt"
            display: bool. if True, draw the Problem in a Tkinter window

        Returns:
            bool. True if a collision free path is found, False otherwise
        """

        def draw_problem():
            for obs in self.obstacles:
                obs.draw(self.window, 'red')
            self.robot.configuration(self.start).draw(self.window, 'orange')
            self.robot.configuration(self.goal).draw(self.window, 'green')

        # if display:
        #     self.window = DrawingWindow(600, 600, 0, self.x, 0, self.y,
        #     'RRT Planning')
        #     draw_problem()
        t1 = time()
        if method == 'rrt':
            path = self.rrt_planning()
        elif method == 'birrt':
            path = self.bidirectional_rrt_planning()
        else:
            raise NotImplemented
        print('RRT took', time() - t1, 'seconds')
        if display and self.display_tree:
            for rrt in self.rrts:
                rrt.draw(self.window)

        if path is None:
            print('No path found')
            return False
        else:
            print(
                'Path found with ' + str(len(path) - 1)
                + ' movements of distance ', self.path_distance(path))
            smooth_path = self.smooth_path(path)
            print(
                'Smoothed path found with ' + str(len(smooth_path) - 1)
                + ' movements of distance ', self.path_distance(smooth_path))
            # interpolated smooth path
            spath = []
            for i in range(1, len(smooth_path)):
                spath.extend(
                    self.cspace.path(smooth_path[i - 1], smooth_path[i]))
            # make sure path is collision free
            if any([self.collide(c) for c in spath]):
                print('Collision in smoothed path')
                return False
            if display:
                self.draw_robot_path(path, color='yellow')
                self.window.clear()
                draw_problem()
                self.draw_robot_path(spath, color='gold')
                while input('End? (y or n)') != 'y':
                    pass
        return True
