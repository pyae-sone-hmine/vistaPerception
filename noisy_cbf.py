from privileged_noisy_controller import CBF # this makes it noisy
import argparse
import numpy as np
import os
import copy
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import box as Box
from shapely import affinity

import vista
from vista.entities.sensors.camera_utils.ViewSynthesis import DepthModes
from vista.utils import logging
from vista.tasks import MultiAgentBase
from vista.utils import transform
# from BarrierNet.Driving.eval_tools.utils import extract_logs
from vista.entities.agents.Dynamics import curvature2steering, curvature2tireangle



# import sys
# sys.path.append('/home/gridsan/phmine/BarrierNet/Driving/models/')
# from BarrierNet.Driving.models.barrier_net import LitModel

# print("Successful import of BarrierNet")

def main(args):
    # Initialize the simulator
    trace_config = dict(
        road_width=4*3, # changed this TODO
        reset_mode='default',
        master_sensor='camera_front',
    )
    car_config = dict(
        length=5. * 0.5, # changed this TODO
        width=2. * 0.5, # changed this TODO
        wheel_base=2.78,
        steering_ratio=14.7,
        lookahead_road=True,
        use_curvilinear_dynamics = True,
    )
    examples_path = os.path.dirname(os.path.realpath(__file__))
    sensors_config = [
        dict(
            type='camera',
            # camera params
            name='camera_front',
            size=(200, 320),
            # rendering params
            depth_mode=DepthModes.FIXED_PLANE,
            use_lighting=False,
        )
    ]
    task_config = dict(n_agents=2,
                       mesh_dir=args.mesh_dir,
                       init_dist_range=[20, 5.], # used to be [6,10]
                       overlap_threshold= 0.001,
                       init_lat_noise_range=[-1., 1.])
    display_config = dict(road_buffer_size=1000, )

    ego_car_config = copy.deepcopy(car_config)
    ego_car_config['lookahead_road'] = True
    ego_car_config['use_curvilinear_dynamics'] = True
    env = MultiAgentBase(trace_paths=args.trace_paths,
                         trace_config=trace_config,
                         car_configs=[car_config] * task_config['n_agents'],
                         sensors_configs=[sensors_config] + [[]] *
                         (task_config['n_agents'] - 1),
                         task_config=task_config,
                         logging_level='DEBUG')

    print("world is ", env.world)

    # Run
    env.reset()
    # if args.use_display: #FIXME CHANGE THIS BACK
    if True: # FIXME CHANGE THIS BACK
        display = vista.Display(env.world, display_config=display_config)
        display.reset()  # reset should be called after env reset
    if args.visualize_privileged_info:
        fig, axes = plt.subplots(1, task_config['n_agents'])
        for ai, agent in enumerate(env.world.agents):
            axes[ai].set_title(f'Agent ({agent.id})')
        artists = dict()
        fig.tight_layout()
        fig.show()

    has_video_writer = args.out_path is not None
    if has_video_writer:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    
    frame_idx = 0
    done = False

    # FIXME ADDED THIS CODE SECTION
    past_actions = dict()
    for agent in env.world.agents:
        past_actions[agent.id] = [0,0]

    while not done and frame_idx <= 2500:

        # seeing if we can access agents, debugging
        for agent in env.world.agents:
            print("agent id is ", agent.id)
            print("ego dynamics are ",agent.ego_dynamics)
            print("agent is ", agent)
            print("agent type is ", type(agent))

        # see what extract_log does and what it outputs
        # ground_truth = extract_logs(env, env.world.agents[0]) # hopefully ego-car
        # print(ground_truth)
        # follow nominal trajectories for all agents
        #actions = generate_human_actions(env.world) 
        actions = cbf_actions(env.world, frame_idx) 

        observations, rewards, dones, infos = env.step(actions, past_actions = past_actions) # FIXME past_actions added
        done = np.any(list(dones.values()))

        # fetch priviliged information (road, all cars' states)
        privileged_info = dict()
        for agent in env.world.agents:
            privileged_info[agent.id] = fetch_privileged_info(env.world, agent)

        if args.visualize_privileged_info:
            for ai, (aid, pinfo) in enumerate(privileged_info.items()):
                agent = [_a for _a in env.world.agents if _a.id == aid][0]

                update_road_vis(pinfo[0], axes[ai], artists, f'{aid}:road')

                other_car_dims = [(_a.width, _a.length)
                                  for _a in env.world.agents
                                  if _a.id != agent.id]
                ego_car_dim = (agent.width, agent.length)
                update_car_vis(pinfo[1], other_car_dims, ego_car_dim, axes[ai],
                               artists, f'{aid}:ado_car')

                print(aid, pinfo[1])

            fig.canvas.draw()
            if not args.use_display:
                plt.pause(0.03)

        # vista visualization
        # if args.use_display: # we change this to always true
        if True: # FIXME hardcoded thisf
            if frame_idx % 1 == 0 and has_video_writer:
                img = display.render()
                filename = f"frame{frame_idx}.jpg"
                os.chdir(args.out_path)
                cv2.imwrite(filename, img)
            key = cv2.waitKey(20)
            if key == ord('q'):
                break

        past_actions = copy.deepcopy(actions)

        print("Frame", frame_idx)
        frame_idx += 1

def state2poly(state, car_dim):
    """ Convert vehicle state to polygon """
    poly = Box(state[0] - car_dim[0] / 2., state[1] - car_dim[1] / 2.,
               state[0] + car_dim[0] / 2., state[1] + car_dim[1] / 2.)
    poly = affinity.rotate(poly, np.degrees(state[2]))
    return poly


def update_car_vis(other_states, other_car_dims, ego_car_dim, ax, artists,
                   name_prefix):
    # clear car visualization at previous timestamp
    for existing_name in artists.keys():
        if name_prefix in existing_name:
            artists[existing_name].remove()

    # initialize some helper object
    colors = list(cm.get_cmap('Set1').colors)
    poly_i = 0

    # plot ego car (reference pose; always at the center)
    ego_poly = state2poly([0., 0., 0.], ego_car_dim)
    artists[f'{name_prefix}_{poly_i:0d}'], = ax.plot(
        ego_poly.exterior.coords.xy[0],
        ego_poly.exterior.coords.xy[1],
        c=colors[poly_i],
    )
    poly_i += 1

    # plot ado cars
    for other_state, other_car_dim in zip(other_states, other_car_dims):
        other_poly = state2poly(other_state, other_car_dim)
        artists[f'{name_prefix}_{poly_i:0d}'], = ax.plot(
            other_poly.exterior.coords.xy[0],
            other_poly.exterior.coords.xy[1],
            c=colors[poly_i],
        )
        poly_i += 1


def update_road_vis(road, ax, artists, name):
    if name in artists.keys():
        artists[name].remove()
    artists[name], = ax.plot(road[:, 0],
                             road[:, 1],
                             c='k',
                             linewidth=2,
                             linestyle='dashed')
    ax.set_xlim(-10., 10.)
    ax.set_ylim(-20., 20.)


def fetch_privileged_info(world, agent):
    # Get ado cars state w.r.t. agent
    other_agents = [_a for _a in world.agents if _a.id != agent.id]
    other_states = []
    for other_agent in other_agents:
        other_latlongyaw = transform.compute_relative_latlongyaw(
            other_agent.ego_dynamics.numpy()[:3],
            agent.ego_dynamics.numpy()[:3])
        other_states.append(other_latlongyaw)

    # Get road w.r.t. the agent
    road = np.array(agent.road)[:, :3].copy()
    ref_pose = agent.ego_dynamics.numpy()[:3]
    road_in_agent = np.array(
        [transform.compute_relative_latlongyaw(_v, ref_pose) for _v in road])

    return road_in_agent, other_states

# work with omega-a control mode
def generate_human_actions(world):
    actions = dict()
    for agent in world.agents:
        # tire_angle = curvature2tireangle(agent.human_curvature, agent.wheel_base)
        # steering = curvature2steering(agent.human_curvature, agent.wheel_base, agent.steering_ratio)

        # omega = agent.ego_dynamics.tire_velocity
        # acceleration = agent.ego_dynamics.acceleration
        actions[agent.id] = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
        ])
    print("actions are ", actions)
    return actions


def cbf_actions(world, frame):
    input_dict = {"p_oa": [2., 1], "use_lane_following_cbf": True, "p_lf": [1., 1.],
                  "lf_cbf_threshold": 2, "use_lane_following_clf": True,
                  "lf_clf_params": [1., 1., 10., 10.], "use_desired_speed_clf": True,
                  "ds_clf_params": [10, 8, 0]}
    
    controller = CBF(input_dict)

    ego_agent = world.agents[0]
    tire_velocity, acceleration = controller(ego_agent, world)

    # check to make sure we don't go to a super small acceleration:
    if abs(acceleration) < 1e-02 and acceleration != 0:
        if acceleration < 0:
            acceleration = -1e-02
        elif acceleration > 0:
            acceleration = 1e-02

    # have the ego car wait for the 
    if frame < 60:
        tire_velocity, acceleration = 0,0

    actions = dict()
    actions[ego_agent.id] = np.array([tire_velocity, acceleration])

    # only care about movement of ego_car, all non_egos are static
    if frame <= 20:
        for non_ego_agent in world.agents[1:]:
            actions[non_ego_agent.id] = np.array([0.5,0.5])
    else:
        for non_ego_agent in world.agents[1:]:
            actions[non_ego_agent.id] = np.array([0,0]) # doesn't move past 60 frames so it stays on road

    return actions







if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run VISTA with multiple cars')
    parser.add_argument('--trace-paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Path to the traces to use for simulation')
    parser.add_argument('--out-path',
                        type=str,
                        help='Path to save the video',
                        default=None)
    parser.add_argument('--mesh-dir',
                        type=str,
                        default=None,
                        help='Directory of meshes for virtual agents')
    parser.add_argument('--use-display',
                        action='store_true',
                        default=False,
                        help='Use VISTA default display')
    parser.add_argument('--visualize-privileged-info',
                        action='store_true',
                        default=False,
                        help='Visualize privileged information')      
    args = parser.parse_args()

    main(args)