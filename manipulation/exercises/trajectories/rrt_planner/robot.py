# yapf: disable
import random
from math import ceil, isfinite, sqrt

from .geometry import Object, Point, Pose


class Robot:
    """A Robot is a set of convex polygons."""

    def __init__(self, polys):
        self.polys = polys
        self.radius = None

    # Creates an Object with reference point at the specified conf.
    def configuration(self, configuration):  # [x, y, theta]
        assert len(configuration) == 3
        return Object(Point(configuration[0], configuration[1]),
                      [poly.rotate(configuration[2]) for poly in self.polys])

    def get_radius(self):  # radius of robot
        if self.radius is None:
            self.radius = max([poly.get_radius() for poly in self.polys])
        return self.radius

    def distance(self, configuration):  # configuration distance
        return sqrt(configuration[0] * configuration[0]
                    + configuration[1] * configuration[1]
                    ) + self.get_radius() * abs(configuration[2])

    def __repr__(self):
        return 'Robot: (' + str(self.polys) + ')'

    def __hash__(self):
        return str(self).__hash__()

    __str__ = __repr__


class RobotArm:
    """A robot arm with rotating joints."""

    def __init__(self, reference, joints):
        self.reference = reference  # a Point
        # The joint Point is location of joint at zero configuration
        # The joint Polygon is link relative to that location
        # This definition could be generalized a bit...
        self.joints = joints  # [(Point, Polygon)...]

    # Creates an instance of robot at the specified configuration
    def configuration(self, configuration):  # [theta_1, theta_2, ..., theta_n]
        assert len(configuration) == len(self.joints)
        polys = []
        origin = None
        angle = None
        for i in range(len(configuration)):
            (joint, link) = self.joints[i]
            if origin is None:
                angle = configuration[i]
                origin = joint.rotate(angle)
            else:
                origin += joint.rotate(angle)
                angle += configuration[i]
            polys.append(link.at_pose(Pose(origin, angle)))
        return Object(self.reference, polys)

    def distance(self, configuration):  # configuration distance
        assert len(configuration) == len(self.joints)
        return max([
            abs(configuration[i]) * self.joints[i][1].get_radius()
            for i in range(len(configuration))
        ])

    def __repr__(self):
        return 'RobotArm: (' + self.reference + ', ' + str(self.joints) + ')'

    def __hash__(self):
        return str(self).__hash__()

    __str__ = __repr__


# Below are classes for defining the ranges of values of the joints.
# These classes are generic, they don't depend on the particular
# robot.


class Range:
    """range of values, handles wraparound."""

    def __init__(self, low, high, wrap_around=False):
        self.low = low
        self.high = high
        assert isfinite(self.low) and isfinite(self.high), \
            "Range must be finite; perhaps you need to define a finite range then enable wrap_around (e.g. for continuous revolute joints)?"
        self.wrap_around = wrap_around

    def difference(self, one, two):  # difference (with wraparound)
        if self.wrap_around:
            if one < two:
                if abs(two - one) < abs((self.low - one) + (two - self.high)):
                    return two - one
                else:
                    return (self.low - one) + (two - self.high)
            else:
                if abs(two - one) < abs((self.high - one) + (two - self.low)):
                    return two - one
                else:
                    return (self.high - one) + (two - self.low)
        else:
            return two - one

    def in_range(self, value):  # test if in range
        if self.wrap_around:
            altered = value
            while not (self.low <= altered <= self.high):
                if altered > self.high:
                    altered -= (self.high - self.low)
                else:
                    altered += (self.high - self.low)
            return altered
        else:
            if self.contains(value):
                return value
            else:
                return None

    def sample(self):  # sample random value
        return (self.high - self.low) * random.random() + self.low

    def contains(self, x):
        return self.wrap_around or (x >= self.low and x <= self.high)


class ConfigurationSpace:
    """Cspace for robot (ranges and distance)"""

    def __init__(self, cspace_ranges : Range, robot_distance, max_steps):
        self.cspace_ranges = cspace_ranges
        self.robot_distance = robot_distance
        self.max_diff_on_path = max_steps

    def distance(self, one: tuple, two: tuple):  # distance in Cspace
        a = [
            self.cspace_ranges[i].difference(one[i], two[i])
            for i in range(len(self.cspace_ranges))
        ]
        return self.robot_distance(tuple(a))

    def path(self, one: tuple, two: tuple):  # linear interpolation
        assert self.valid_configuration(one) and self.valid_configuration(two)
        diffs = [
            self.cspace_ranges[i].difference(one[i], two[i])
            for i in range(len(self.cspace_ranges))
        ]
        samples = max([
            int(ceil(abs(diff) / max_diff))
            for diff, max_diff in zip(diffs, self.max_diff_on_path)
        ])
        samples = max(2, samples)
        linear_interpolation = [diff / (samples - 1.0) for diff in diffs]
        path = [one]
        for s in range(1, samples - 1):
            sample = tuple([
                self.cspace_ranges[i].in_range(one[i]
                                               + s * linear_interpolation[i])
                for i in range(len(self.cspace_ranges))
            ])
            #  return path
            path.append(sample)
        return path + [two]

    def sample(self):  # sample random configuration
        """
        Samples a random configuration in this ConfigurationSpace.

        Returns:
            3-tuple of floats (x, y, theta) in this ConfigurationSpace
        """
        return tuple([r.sample() for r in self.cspace_ranges])

    def valid_configuration(self, config):  # validity test (in ranges)
        return len(config) == len(self.cspace_ranges) and \
            all([self.cspace_ranges[i].contains(config[i])
                 for i in range(len(self.cspace_ranges))])
