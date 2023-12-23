# %%
import numpy as np
from rrt_planner.manipulation_station_collision_checker import (
    ManipulationStationCollisionChecker,
)
from rrt_planner.robot import ConfigurationSpace, Range
from rrt_planner.rrt_planning import Problem


class IiwaProblem(Problem):
    def __init__(
        self,
        q_start: np.array,
        q_goal: np.array,
        gripper_setpoint: float,
        left_door_angle: float,
        right_door_angle: float,
        is_visualizing=False,
    ):
        self.gripper_setpoint = gripper_setpoint
        self.left_door_angle = left_door_angle
        self.right_door_angle = right_door_angle
        self.is_visualizing = is_visualizing

        self.collision_checker = ManipulationStationCollisionChecker(
            is_visualizing=is_visualizing
        )

        # Construct configuration space for IIWA.
        plant = self.collision_checker.plant
        nq = 7
        joint_limits = np.zeros((nq, 2))
        for i in range(nq):
            joint = plant.GetJointByName("iiwa_joint_%i" % (i + 1))
            joint_limits[i, 0] = joint.position_lower_limits()
            joint_limits[i, 1] = joint.position_upper_limits()

        range_list = []
        for joint_limit in joint_limits:
            range_list.append(Range(joint_limit[0], joint_limit[1]))

        def l2_distance(q: tuple):
            sum = 0
            for q_i in q:
                sum += q_i**2
            return np.sqrt(sum)

        max_steps = nq * [np.pi / 180 * 2]  # three degrees
        cspace_iiwa = ConfigurationSpace(range_list, l2_distance, max_steps)

        # Call base class constructor.
        Problem.__init__(
            self,
            x=10,  # not used.
            y=10,  # not used.
            robot=None,  # not used.
            obstacles=None,  # not used.
            start=tuple(q_start),
            goal=tuple(q_goal),
            cspace=cspace_iiwa,
        )

    def collide(self, configuration):
        q = np.array(configuration)
        return self.collision_checker.ExistsCollision(
            q,
            self.gripper_setpoint,
            self.left_door_angle,
            self.right_door_angle,
        )

    def run_planner(self, method: str):
        path = None
        if method == "rrt":
            path = self.rrt_planning()
        elif method == "birrt":
            path = self.bidirectional_rrt_planning()
        else:
            raise NotImplementedError

        if path is None:
            print("No path found")
            return None
        else:
            print(
                "Path found with " + str(len(path) - 1) + " movements of distance ",
                self.path_distance(path),
            )
            smooth_path = self.smooth_path(path)
            print(
                "Smoothed path found with "
                + str(len(smooth_path) - 1)
                + " movements of distance ",
                self.path_distance(smooth_path),
            )
            # interpolated smooth path
            spath = []
            for i in range(1, len(smooth_path)):
                spath.extend(self.cspace.path(smooth_path[i - 1], smooth_path[i]))

            # make sure path is collision free
            if any([self.collide(c) for c in spath]):
                print("Collision in smoothed path")
                return None

            return spath

    def visualize_path(self, path):
        # show path in meshcat
        for q in path:
            q = np.array(q)
            self.collision_checker.DrawStation(
                q,
                self.gripper_setpoint,
                self.left_door_angle,
                self.right_door_angle,
            )
            input("next?")
