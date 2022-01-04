# Note: My goal is to move this to it's own repo: See Drake issue #15508.

import gym
import numpy as np
from typing import Callable, Union
import warnings

from pydrake.common import RandomGenerator
from pydrake.systems.sensors import ImageRgba8U
from pydrake.systems.framework import (
    Context,
    InputPort,
    InputPortIndex,
    OutputPort,
    OutputPortIndex,
    PortDataType,
    System,
)
from pydrake.systems.analysis import Simulator, SimulatorStatus


class DrakeGymEnv(gym.Env):
    """
    DrakeGymEnv provides a gym.Env interface for a Drake System (often a
    Diagram) using a Simulator.
    """

    def __init__(self,
                 simulator: Union[Simulator, Callable[[RandomGenerator],
                                                      Simulator]],
                 time_step: float,
                 action_space: gym.spaces.space,
                 observation_space: gym.spaces.space,
                 reward: Union[Callable[[System, Context], float],
                               OutputPortIndex, str],
                 action_port_id: Union[InputPort, InputPortIndex, str] = None,
                 observation_port_id: Union[OutputPortIndex, str] = None,
                 render_rgb_port_id: Union[OutputPortIndex, str] = None):
        """
        Args:
            system: A Drake System
            time_step: Each call to step() will advance the simulator by
                `time_step` seconds.
            reward: The reward can be specified in one of two
                ways: (1) by passing a callable with the signature
                `value = reward(context)` or (2) by passing a scalar
                vector-valued output port of `system`.
            action_port: An input port of `system` compatible with the
                action_space.  Each Env *must* have an action port; passing
                `None` defaults to using the *first* input port (inspired by
                `InputPortSelection.kUseFirstInputIfItExists`).
            action_space: Defines the `gym.spaces.space` for the actions.  If
                action_port is vector-valued, then passing `None` defaults to a
                gym.spaces.Box of the correct dimension with bounds at negative
                and positive infinity.  Note: Stable Baselines 3 strongly
                encourages normalizing the action_space to [-1, 1].
            observation_port: An output port of `system` compatible with the
                observation_space. Each Env *must* have an observation port (it
                seems that gym doesn't support empty observation spaces /
                open-loop policies); passing `None` defaults to using the
                *first* input port (inspired by
                `OutputPortSelection.kUseFirstOutputIfItExists`).
            observation_space: Defines the gym.spaces.space for the
                observations.  If observation_port is vector-valued, then
                passing `None` defaults to a gym.spaces.Box of the correct
                dimension with bounds at negative and positive infinity.
            render_rgb_port: An optional output port of `system` that returns
                an `ImageRgba8U`; often the `color_image` port of Drake's
                `RgbdSensor`.  When not `None`, this enables the environment
                `render_mode` `rgb_array`.

        Notes (using `env` as an instance of this class):
        - You may set simulator/integrator preferences by using `env.simulator`
          directly.
        - The `done` condition returned by `step()` is always False by
          default.  Use `env.simulator.set_monitor()` to use Drake's monitor
          functionality for specifying termination conditions.
        - You may additionally wish to directly set `env.reward_range` and/or
          `env.spec`.  See the docs for gym.Env for more details.
        """
        if isinstance(simulator, Simulator):
            self.simulator = simulator
            self.make_simulator = None
        elif callable(simulator):
            self.simulator = None
            self.make_simulator = simulator
        else:
            raise ValueError("Invalid simulator argument")

        assert time_step > 0
        self.time_step = time_step

        assert isinstance(action_space, gym.spaces.Space)
        self.action_space = action_space

        assert isinstance(observation_space, gym.spaces.Space)
        self.observation_space = observation_space

        if isinstance(reward, (OutputPortIndex, str)):
            self.reward_port_id = reward
            self.reward = None
        elif callable(reward):
            self.reward_port_id = None
            self.reward = reward
        else:
            raise ValueError("Invalid reward argument")

        if action_port_id:
            assert isinstance(action_port_id, (InputPortIndex, str))
            self.action_port_id = action_port_id
        else:
            self.action_port_id = InputPortIndex(0)

        if observation_port_id:
            assert isinstance(observation_port_id, (OutputPortIndex, str))
            self.observation_port_id = observation_port_id
        else:
            self.observation_port_id = OutputPortIndex(0)

        self.metadata['render.modes'] = ['human', 'ascii']

        # (Maybe) setup rendering
        if render_rgb_port_id:
            assert isinstance(render_rgb_port_id, (OutputPortIndex, str))
            self.metadata['render.modes'].append('rgb_array')
        self.render_rgb_port_id = render_rgb_port_id

        self.generator = RandomGenerator()

        if self.simulator:
            self._setup()

    def _setup(self):
        """Completes the setup once we have a self.simulator."""
        system = self.simulator.get_system()

        # Setup action port
        if self.action_port_id:
            if isinstance(self.action_port_id, InputPortIndex):
                self.action_port = system.get_input_port(self.action_port_id)
            else:
                self.action_port = system.GetInputPort(self.action_port_id)
        if self.action_port.get_data_type() == PortDataType.kVectorValued:
            assert np.array_equal(self.action_space.shape,
                                  [self.action_port.size()])

        def get_output_port(id):
            if isinstance(id, OutputPortIndex):
                return system.get_output_port(id)
            return system.GetOutputPort(id)

        # Setup observation port
        if self.observation_port_id:
            self.observation_port = get_output_port(self.observation_port_id)
        if self.observation_port.get_data_type() == PortDataType.kVectorValued:
            assert np.array_equal(self.observation_space.shape,
                                  [self.observation_port.size()])

        # Note: We require that there is no direct feedthrough action_port to
        # observation_port.  Unfortunately, HasDirectFeedthrough returns false
        # positives, and would produce noisy warnings.

        # Setup reward
        if self.reward_port_id:
            reward_port = get_output_port(self.reward_port_id)
            self.reward = lambda system, context: reward_port.Eval(context)[0]

        # (Maybe) setup rendering port
        if self.render_rgb_port_id:
            self.render_rgb_port = get_output_port(self.render_rgb_port_id)
            assert self.render_rgb_port.get_data_type() == \
                PortDataType.kAbstractValued
            assert isinstance(self.render_rgb_port.Allocate().get_value(),
                              ImageRgba8U)

    def step(self, action):
        """
        Implements gym.Env.step to advance the simulation forward by one
        `self.time_step`.

        Args:
            action: an element from self.action_space
        """
        assert self.simulator, "You must call reset() first"

        context = self.simulator.get_context()
        time = context.get_time()

        self.action_port.FixValue(context, action)
        catch = False
        try:
            status = self.simulator.AdvanceTo(time + self.time_step)
        except RuntimeError as e:
            if "MultibodyPlant's discrete update solver failed to converge" \
                    not in e.args[0]:
                raise
            warnings.warn("Calling Done after catching RuntimeError:")
            warnings.warn(e.args[0])
            catch = True

        observation = self.observation_port.Eval(context)
        reward = self.reward(self.simulator.get_system(), context)
        done = catch or status.reason() == \
            SimulatorStatus.ReturnReason.kReachedTerminationCondition
        info = dict()

        return observation, reward, done, info

    def reset(self):
        """
        If a callable "simulator factory" was passed to the constructor, then a
        new simulator is created.  Otherwise this method simply resets the
        `simulator` and its Context.
        """
        if self.make_simulator:
            self.simulator = self.make_simulator(self.generator)
            self._setup()

        context = self.simulator.get_mutable_context()
        context.SetTime(0)
        self.simulator.get_system().SetRandomContext(context, self.generator)
        self.simulator.Initialize()
        # Note: The output port will be evaluated without fixing the input port.
        observations = self.observation_port.Eval(context)
        return observations

    def render(self, mode='human'):
        """
        Rendering in `human` mode is accomplished by calling Publish on
        `system`.  This should cause visualizers inside the System (e.g.
        MeshcatVisualizer, PlanarSceneGraphVisualizer, etc.) to draw their
        outputs.  To be fully compliant, those visualizers should set their
        default publishing period to `np.inf` (do not publish periodically).

        Rendering in `ascii` mode calls __repr__ on the system Context.

        Rendering in `rgb_array` mode is enabled by passing a compatible
        `render_rgb_port` to the class constructor.
        """
        assert self.simulator, "You must call reset() first"

        if mode == 'human':
            self.simulator.get_system().Publish(self.simulator.get_context())
            return
        elif mode == 'ansi':
            return __repr__(self.simulator.get_context())
        elif mode == 'rgb_array':
            assert self.render_rgb_port, \
                "You must set render_rgb_port in the constructor"
            return self.render_rgb_port.Eval(
                self.simulator.get_context()).data[:, :, :3]
        else:
            super(DrakeGymEnv, self).render(mode=mode)

    def seed(self, seed=None):
        """Implements gym.Env.seed using Drake's RandomGenerator."""
        if seed:
            self.generator = RandomGenerator(seed)
        else:
            seed = self.generator()
        # Note: One could call self.action_space.seed(self.generator()) here,
        # but it appears that is not the standard approach:
        # https://github.com/openai/gym/issues/681
        return [seed]
