from typing import List, Dict, Any, Optional, Callable, Union

from vista.entities.sensors.EventCamera import EventCamera
import numpy as np
from collections import deque

from .Dynamics import State, BicycleDynamics, CurvilinearDynamics, curvature2steering, \
                      curvature2tireangle, tireangle2curvature, update_with_perfect_controller
from ..Entity import Entity

from ..sensors import BaseSensor, Camera
try: # hacky way to exclude dependencies on non-camera sensors
    from ..sensors import Lidar, EventCamera
except:
    Lidar = State
    EventCamera = State
from ...core import World, Trace
from ...utils import transform, logging, misc


class Car(Entity):
    """ The class of a car agent. This object lives in the :class:`World` and is attached
    to a trace object that provides pointers to dataset for data-driven simulation and zero
    or one or more sensor objects that synthesize sensory measurement. The update of vehicle
    state is handled in this object.

    Args:
        world (World): The world that this agent lives in.
        car_config (Dict): Configuration of the car. An example (default) is,

            >>> DEFAULT_CONFIG = {
                'length': 5.,
                'width': 2.,
                'wheel_base': 2.78,
                'steering_ratio': 14.7,
                'use_curvilinear_dynamics': False,
                'lookahead_road': True,
                'road_buffer_size': 1e5,
                'control_mode': 'delta-v',
            }

    Example Usage (always make sure reset is called first to initialize vehicle state
    or pointer to the dataset for data-driven simulation)::

        >>> car = world.spawn_agent(car_config)
        >>> car.spawn_camera(camera_config)
        >>> world.reset()
        >>> car.step_dynamics(action) # update vehicle states
        >>> car.step_sensors() # do sensor capture
        >>> observation = car.observations # fetch sensory measurement
        >>> car.step_dataset() # simply get next frame in the dataset without synthesis

    """
    DEFAULT_CONFIG = {
        'length': 5.,
        'width': 2.,
        'wheel_base': 2.78,
        'steering_ratio': 14.7,
        'use_curvilinear_dynamics': True,
        'lookahead_road': True,
        'road_buffer_size': 1e5,
        'control_mode': 'omega-a', #THIS IS WHERE YOU CHANGE CONTROL MODE
    }

    def __init__(self, world: World, car_config: Dict) -> None:
        super(Car, self).__init__()

        print("this is running editable vista")

        # Car configuration
        car_config = misc.merge_dict(car_config, self.DEFAULT_CONFIG)
        self._config = car_config

        # Pointer to the parent vista.World object where the agent lives
        self._parent: World = world

        # The trace to be associated with, updated every reset
        self._trace: Trace = None

        # A list of sensors attached to this agent (List[vista.Sensor]).
        self._sensors: List[BaseSensor] = []

        # State dynamics for tracking the virtual and human agent
        self._relative_state: State = State()
        if self._config['use_curvilinear_dynamics']:
            self._ego_dynamics: CurvilinearDynamics = CurvilinearDynamics()
        else:
            self._ego_dynamics: BicycleDynamics = BicycleDynamics()
        self._human_dynamics: BicycleDynamics = BicycleDynamics()

        # Properties of a car
        self._length: float = self.config['length']
        self._width: float = self.config['width']
        self._wheel_base: float = self.config['wheel_base']
        self._steering_ratio: float = self.config['steering_ratio']
        self._control_mode: str = self.config['control_mode']
        self._speed: float = 0.
        self._curvature: float = 0.
        self._steering: float = 0.
        self._tire_angle: float = 0.
        self._human_speed: float = 0.
        self._human_curvature: float = 0.
        self._human_steering: float = 0.
        self._human_tire_angle: float = 0.
        self._timestamp: float = 0.
        self._frame_number: int = 0
        self._trace_index: int = 0
        self._segment_index: int = 0
        self._frame_index: int = 0
        self._observations: Dict[str, Any] = dict()
        self._done: bool = False

        # Privileged information
        if self._config['use_curvilinear_dynamics']:
            assert self.config['lookahead_road'] and self.config[
                'road_buffer_size'] == 1e5, 'With new (curvilinear dynamics, each agent' + \
                ' (precisely should be at least the ego agent) should set lookahead_road ' + \
                ' = True and road_buffer_size = 1e5'
        if self.config['lookahead_road']:
            road_buffer_size = int(self.config['road_buffer_size'])
            self._road: deque[np.ndarray] = deque(maxlen=road_buffer_size)
            self._road_frame_idcs: deque[int] = deque(maxlen=road_buffer_size)
            self._road_dynamics: BicycleDynamics = BicycleDynamics()

    def spawn_camera(self, cam_config: Dict) -> Camera:
        """ Spawn and attach a camera to this car.

        Args:
            cam_config (Dict): Configuration of camera and rendering. For
                               more details, please check the doc of
                               :class:`Camera` sensor.

        Returns:
            Camera: a vista camera sensor object spawned.

        """
        name = cam_config['name']
        logging.info(f'Spawn a new camera {name} in car ({self.id})')
        cam = Camera(attach_to=self, config=cam_config)
        self._sensors.append(cam)

        return cam

    def spawn_lidar(self, lidar_config: Dict) -> Lidar:
        """ Spawn and attach a LiDAR to this car.

        Args:
            lidar_config (Dict): Configuration of LiDAR. For more details,
                                 please check the doc of :class:`Lidar` sensor.

        Returns:
            Lidar: a vista Lidar sensor object spawned.

        """
        name = lidar_config['name']
        logging.info(f'Spawn a new lidar {name} in car ({self.id})')
        lidar = Lidar(attach_to=self, config=lidar_config)
        self._sensors.append(lidar)

        return lidar

    def spawn_event_camera(self, event_cam_config: Dict) -> EventCamera:
        """ Spawn and attach an event camera to this car.

        Args:
            lidar_config (Dict): Configuration of event camera. For more details, 
                                 please check the doc of :class:`EventCamera` sensor.

        Returns:
            EventCamera: a vista event camera sensor object spawned.

        """
        name = event_cam_config['name']
        logging.info(f'Spawn a new event camera {name} in car ({self.id})')
        event_cam = EventCamera(attach_to=self, config=event_cam_config)
        self._sensors.append(event_cam)

        return event_cam

    def reset(self,
              trace_index: int,
              segment_index: int,
              frame_index: int,
              initial_dynamics_fn: Optional[Callable] = None,
              step_sensors: Optional[bool] = True) -> None:
        """ Reset the car. This involves pointing to somewhere in the dataset for later-on
        data-driven simulation, initializing vehicle state, and resetting all sensors attached
        to this car. If ``lookahead_road = True``, the road cache will also be reset.

        Args:
            trace_index (int): A pointer to which trace to be simulated on.
            segment_index (int): A pointer to which segment in a trace to be simulated on.
            frame_index (int): A pointer to which frame in a segment to be simulated on.
            initial_dynamics_fn (Callable): A function to initialize vehicle state. The
                function takes x, y, yaw, steering (tire angle), and speed as inputs. Default
                is set to ``None``, which initialize vehicle with the same state as the dataset.
            step_sensors (bool): Whether to step sensor; default is set to ``True``.

        """
        logging.info(f'Car ({self.id}) reset')

        # Update pointers to dataset
        self._trace = self.parent.traces[trace_index]
        self._timestamp = self.trace.get_master_timestamp(
            segment_index, frame_index)
        self._frame_number = self.trace.get_master_frame_number(
            segment_index, frame_index)
        self._trace_index = trace_index
        self._segment_index = segment_index
        self._frame_index = frame_index

        self._done = False

        # Reset states and dynamics
        self._human_speed = self.trace.f_speed(self.timestamp)
        self._human_curvature = self.trace.f_curvature(self.timestamp)
        self._human_steering = curvature2steering(self.human_curvature,
                                                  self.wheel_base,
                                                  self.steering_ratio)
        self._human_tire_angle = curvature2tireangle(self.human_curvature,
                                                     self.wheel_base)
        self.human_dynamics.update(0., 0., 0., self.human_tire_angle,
                                   self.human_speed)

        if hasattr(self, '_road'): # Reset for privileged information
            self._road.clear()
            self._road.append(self.human_dynamics.numpy())
            self._road_dynamics = self.human_dynamics.copy()
            self._road_frame_idcs.clear()
            self._road_frame_idcs.append(self.frame_index)
            self._update_road()

        initial_dynamics = self.human_dynamics.numpy().copy()
        if initial_dynamics_fn is not None:
            initial_dynamics = initial_dynamics_fn(*initial_dynamics)
        if self._config['use_curvilinear_dynamics']:
            if self.parent.agents[0].id == self.id: # only compute reference for ego-agent
                ref_xyphi = self.road[:, :3].copy()
                exceed_end = False
                ref_curvature = []
                frame_index_it = self.frame_index
                while not exceed_end:
                    exceed_end, ts = self.trace.get_master_timestamp(
                        self.segment_index, frame_index_it, check_end=True)
                    ref_curvature.append(self.trace.f_curvature(ts))
                    frame_index_it += 1
                ref_curvature = np.array(ref_curvature)[:, None]
                ref = np.concatenate([ref_xyphi, ref_curvature], axis=1)
            else:
                ref = self.parent.agents[0].ego_dynamics._ref.copy()
            self.ego_dynamics.set_reference(ref)

            self.ego_dynamics.reset()
        self.ego_dynamics.update(*initial_dynamics)
        self._speed = self.human_speed
        self._curvature = self.human_curvature
        self._steering = self.human_steering
        self._tire_angle = curvature2tireangle(self.curvature, self.wheel_base)

        latlongyaw = transform.compute_relative_latlongyaw(
            self.ego_dynamics.numpy()[:3],
            self.human_dynamics.numpy()[:3])
        self._relative_state.update(*latlongyaw)

        # Reset sensors
        for sensor in self.sensors:
            if isinstance(
                    sensor,
                    Camera) and self._trace.multi_sensor.main_camera is None:
                self._trace.multi_sensor.set_main_sensor('camera', sensor.name)
            elif isinstance(
                    sensor,
                    Lidar) and self._trace.multi_sensor.main_lidar is None:
                self._trace.multi_sensor.set_main_sensor('lidar', sensor.name)
            elif isinstance(
                    sensor, EventCamera
            ) and self._trace.multi_sensor.main_event_camera is None:
                self._trace.multi_sensor.set_main_sensor(
                    'event_camera', sensor.name)

            sensor.reset()

        # Update observation
        if step_sensors:
            self.step_sensors()

    def step_dataset(self, step_dynamics=True):
        """ Step through the dataset without rendering. This is basically
        fetching the next frame from the dataset. Normally, it is called
        when doing imitation learning.

        Args:
            step_dynamics (bool): Whether to update vehicle state; default
                is set to ``True``.

        Raises:
            NotImplementedError: if any attached sensor has no implemented
                function for stepping through dataset.

        """
        logging.info(f'Car ({self.id}) step based on dataset')

        # Step by incrementing frame number
        ts = self.timestamp
        frame_index = self.frame_index + 1
        exceed_end, self._timestamp = self.trace.get_master_timestamp(
            self.segment_index, frame_index, check_end=True)
        if exceed_end:  # trigger trace done terminatal condition
            self._done = True
            logging.info(f'Car ({self.id}) exceed the end of trace')
        else:
            self._frame_index = frame_index
            self._frame_number = self.trace.get_master_frame_number(
                self.segment_index, self.frame_index)

            self._human_speed = self.trace.f_speed(self.timestamp)
            self._human_curvature = self.trace.f_curvature(self.timestamp)
            self._human_steering = curvature2steering(self.human_curvature,
                                                      self.wheel_base,
                                                      self.steering_ratio)
            self._human_tire_angle = curvature2tireangle(
                self.human_curvature, self.wheel_base)

            self._speed = self._human_speed
            self._curvature = self._human_curvature
            self._steering = self._human_steering
            self._tire_angle = self._human_tire_angle

            # Step human dynamics if queried
            if step_dynamics:
                current_state = [
                    curvature2tireangle(self.human_curvature, self.wheel_base),
                    self.human_speed
                ]
                update_with_perfect_controller(current_state,
                                               self.timestamp - ts,
                                               self._human_dynamics)
                if self._config['use_curvilinear_dynamics']:
                    raise NotImplementedError
                else:
                    self._ego_dynamics = self.human_dynamics.copy()

            # Get image frame
            self._observations = dict()
            for sensor in self.sensors:
                if type(sensor) not in [Camera, Lidar, EventCamera]:
                    raise NotImplementedError(
                        f'Sensor {sensor} is not supported in step dataset')
                self._observations[sensor.name] = sensor.capture(
                    self.timestamp)

    def step_dynamics(self,
                      action: np.ndarray,
                      dt: Optional[float] = 1 / 30.,
                      update_road: Optional[bool] = True, past_actions = None) -> None:
        """ Update vehicle state given control command based on vehicle dynamics
        and update timestamp, which is then used to update pointer to the dataset
        for data-driven simulation.

        Args:
            action (np.ndarray): Control command (curvature and speed).
            dt (float): Elapsed time.
            update_road (bool): Whether to update road cache; default is
                set to ``True``.

        """
        assert not self.done, 'Agent status is done. Please call reset first.'
        logging.info(f'Car ({self.id}) step dynamics')

        # Parse action
        if self.config['use_curvilinear_dynamics']:
            #assert self.control_mode == 'omega-a', 'Curvilinear dynamics only allow a-omega-control for now' #FIXME testing kappa-v rn
            pass
        action = np.array(action).reshape(-1)
        assert action.shape[0] == 2
        if self.control_mode == 'kappa-v':
            desired_curvature, desired_speed = action
            desired_tire_angle = curvature2tireangle(desired_curvature,
                                                     self.wheel_base)
        elif self.control_mode == 'delta-v':
            desired_tire_angle, desired_speed = action
            print("now within Car.py step_dynamics, within delta-v, action is ", action)
        elif self.control_mode == 'omega-a':
            tire_velocity, acceleration = action
        else:
            raise NotImplementedError(f'Unrecognized mode {self.control_mode}')

        # Run low-level controller and step vehicle dynamics
        if self.control_mode in ['kappa-v', 'delta-v']:
            # TODO: non-perfect low-level controller
            logging.debug('Using perfect low-level controller now')
            desired_state = [desired_tire_angle, desired_speed]
            if past_actions is None:
                update_with_perfect_controller(desired_state, dt, self._ego_dynamics)
            else:
                update_with_perfect_controller(desired_state, dt, self._ego_dynamics, past_actions = past_actions)
            print("line 395", desired_state)
        elif self.control_mode == 'omega-a':
            self._ego_dynamics.step(tire_velocity, acceleration, dt)
        else:
            raise NotImplementedError(f'Unrecognized mode {self.control_mode}')

        # Update based on vehicle dynamics feedback #FIXME not updating ego_dynamics
        self._tire_angle = self.ego_dynamics.steering
        self._speed = self.ego_dynamics.speed
        self._curvature = tireangle2curvature(self.tire_angle, self.wheel_base)
        self._steering = curvature2steering(self.curvature, self.wheel_base,
                                            self.steering_ratio)


        # Update human (reference) dynamics for assoication with the trace / dataset
        human = self.human_dynamics.copy()
        top2_closest = dict(dist=deque([np.inf, np.inf], maxlen=2),
                            dynamics=deque([None, None], maxlen=2),
                            timestamp=deque([None, None], maxlen=2),
                            index=deque([None, None], maxlen=2))
        index = self.frame_index
        ts = self.trace.get_master_timestamp(self.segment_index, index)
        while True:
            dist = np.linalg.norm(human.numpy()[:2] -
                                  self.ego_dynamics.numpy()[:2])
            if dist < top2_closest['dist'][1]:
                if dist < top2_closest['dist'][0]:
                    top2_closest['dist'].appendleft(dist)
                    top2_closest['dynamics'].appendleft(human.copy())
                    top2_closest['timestamp'].appendleft(ts)
                    top2_closest['index'].appendleft(index)
                else:
                    top2_closest['dist'][1] = dist
                    top2_closest['dynamics'][1] = human.copy()
                    top2_closest['timestamp'][1] = ts
                    top2_closest['index'][1] = index
            elif dist < 10:  # to prevent from in larger-scope search
                break

            next_index = index + 1 * int(np.sign(dt))
            exceed_end, next_ts = self.trace.get_master_timestamp(
                self.segment_index, next_index, check_end=True)
            if exceed_end:  # trigger trace done terminatal condition
                self._done = True
                logging.info(f'Car ({self.id}) exceed the end of trace')
                break

            current_state = [
                curvature2tireangle(self.trace.f_curvature(ts),
                                    self.wheel_base),
                self.trace.f_speed(ts)
            ]
            update_with_perfect_controller(current_state, next_ts - ts, human)

            index = next_index
            ts = next_ts

        self._human_dynamics = top2_closest['dynamics'][0].copy()
        self._frame_index = top2_closest['index'][0]
        self._frame_number = self.trace.get_master_frame_number(
            self.segment_index, self.frame_index)

        # Update timestamp based on where the car position is with respect to
        # course distance of trace (may not be exactly the same as any of
        # timestamps in the dataset)
        latlongyaw_closest = transform.compute_relative_latlongyaw(
            self.ego_dynamics.numpy()[:3],
            top2_closest['dynamics'][0].numpy()[:3])
        latlongyaw_second_closest = transform.compute_relative_latlongyaw(
            self.ego_dynamics.numpy()[:3],
            top2_closest['dynamics'][1].numpy()[:3])
        ratio = abs(latlongyaw_second_closest[1]) / (
            abs(latlongyaw_closest[1]) + abs(latlongyaw_second_closest[1]))
        self._timestamp = (ratio * top2_closest['timestamp'][0] +
                           (1. - ratio) * top2_closest['timestamp'][1])

        # Update human control based on current timestamp
        self._human_speed = self.trace.f_speed(self.timestamp)
        self._human_curvature = self.trace.f_curvature(self.timestamp)
        self._human_steering = curvature2steering(self.human_curvature,
                                                  self.wheel_base,
                                                  self.steering_ratio)
        self._human_tire_angle = curvature2tireangle(self.human_curvature,
                                                     self.wheel_base)

        # Update relative transformation between human and ego dynamics
        self._relative_state.update(*latlongyaw_closest)

        # Update privileged information (in global coordinates)
        if update_road and hasattr(self, '_road'):
            self._update_road()

    def step_sensors(self) -> None:
        """ Update sensor measurement given current state of the vehicle. """
        logging.info(f'Car ({self.id}) step sensors')
        self._observations = dict()
        for sensor in self.sensors:
            self._observations[sensor.name] = sensor.capture(self.timestamp)

    def _update_road(self) -> None:
        exceed_end = False
        get_timestamp = lambda _idx: self.trace.get_master_timestamp(
            self.segment_index, _idx, check_end=True)
        while self._road_frame_idcs[-1] < (
                self.frame_index + self._road.maxlen / 2.) and not exceed_end:
            exceed_end, ts = get_timestamp(self._road_frame_idcs[-1])
            self._road_frame_idcs.append(self._road_frame_idcs[-1] + 1)
            exceed_end, next_ts = get_timestamp(self._road_frame_idcs[-1])

            state = [
                curvature2tireangle(self.trace.f_curvature(ts),
                                    self.wheel_base),
                self.trace.f_speed(ts)
            ]
            update_with_perfect_controller(state, next_ts - ts,
                                           self._road_dynamics)
            next_road = self._road_dynamics.numpy()
            self._road.append(next_road)

    @property
    def trace(self) -> Trace:
        """ The :class:`Trace` currently associated with the car. """
        return self._trace

    @property
    def sensors(self) -> List[BaseSensor]:
        """ All sensors attached to this car. """
        return self._sensors

    @property
    def relative_state(self) -> State:
        """ Relative transform between ``ego_dynamics`` and ``human_dynamics``. """
        return self._relative_state

    @property
    def ego_dynamics(self) -> Union[BicycleDynamics, CurvilinearDynamics]:
        """ Current simulated vehicle state. """
        return self._ego_dynamics

    @property
    def human_dynamics(self) -> BicycleDynamics:
        """ Vehicle state of the current pointer to the dataset (human trajectory). """
        return self._human_dynamics

    @property
    def length(self) -> float:
        """ Car length. """
        return self._length

    @property
    def width(self) -> float:
        """ Car width. """
        return self._width

    @property
    def wheel_base(self) -> float:
        """ Wheel base. """
        return self._wheel_base

    @property
    def steering_ratio(self) -> float:
        """ Steering ratio. """
        return self._steering_ratio

    @property
    def speed(self) -> float:
        """ Speed of simulated trajectory (this car) in current timestamp. """
        return self._speed

    @property
    def curvature(self) -> float:
        """ Curvature of simulated trajectory (this car) in current timestamp. """
        return self._curvature

    @property
    def steering(self) -> float:
        """ Steering angle of simulated trajectory (this car) in current timestamp. """
        return self._steering

    @property
    def tire_angle(self) -> float:
        """ Tire angle of simulated trajectory (this car) in current timestamp. """
        return self._tire_angle

    @property
    def human_speed(self) -> float:
        """ Speed of human trajectory in current timestamp. """
        return self._human_speed

    @property
    def human_curvature(self) -> float:
        """ Curvature of human trajectory in current timestamp. """
        return self._human_curvature

    @property
    def human_steering(self) -> float:
        """ Steering angle of human trajectory in current timestamp. """
        return self._human_steering

    @property
    def human_tire_angle(self) -> float:
        """ Tire angle of human trajectory in current timestamp. """
        return self._human_tire_angle

    @property
    def timestamp(self) -> float:
        """ Current timestamp (normally ROS timestamp). This serves as a
        continuous pointer to the dataset as opposed to ``trace_index``,
        ``segment_index``, and ``frame_index``. """
        return self._timestamp

    @property
    def frame_number(self) -> int:
        """ Current frame number. Note that this is different from ``frame_index``
        as it is a different pointer based on how we define frame in the ``master_sensor``
        instead of a pointer to the dataset. There is only one unique pointer to the dataset,
        which can be mapped to (potentially) different pointers to the frame number in
        different sensors. """
        return self._frame_number

    @property
    def trace_index(self) -> int:
        """ Current pointer to the trace. """
        return self._trace_index

    @property
    def segment_index(self) -> int:
        """ Current pointer to the segment in the current trace. """
        return self._segment_index

    @property
    def frame_index(self) -> int:
        """ Current pointer to the frame in the current segment. """
        return self._frame_index

    @property
    def observations(self) -> Dict[str, Any]:
        """ Sensory measurement at current timestamp. """
        return self._observations

    @property
    def done(self) -> bool:
        """ Whether exceeding the end of the trace currently associated with the car. """
        return self._done

    @property
    def road(self) -> np.ndarray:
        """ Road cache if ``lookahead_road = True`` otherwise ``None``. """
        return np.array(self._road) if hasattr(self, '_road') else None

    @property
    def config(self) -> Dict:
        """ Configuration of this car. """
        return self._config

    @property
    def control_mode(self) -> str:
        """ Control mode of this car. """
        return self._control_mode

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} (id={self.id})> ' + \
               f'width: {self.width} ' + \
               f'length: {self.length} ' + \
               f'wheel_base: {self.wheel_base} ' + \
               f'steering_ratio: {self.steering_ratio} ' + \
               f'speed: {self.speed} ' + \
               f'curvature: {self.curvature} '