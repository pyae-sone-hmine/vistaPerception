import argparse
import numpy as np
import os
import cv2

import vista
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature

def main(args):
    # CREATE A VISTA WORLD FOR OUR SIMULATION
    world = vista.World(args.trace_path, trace_config={'road_width': 4}) 
    # POPULATE THE WORLD WITH A CAR WITH PERHAPS OTHER CONFIGS
    car = world.spawn_agent(
        config={
            'length': 5.,
            'width': 2.,
            'wheel_base': 2.78,
            'steering_ratio': 14.7,
            'lookahead_road': True
        })


    # PERHAPS GET RID OF THE CAMERA? 
    camera_front = car.spawn_camera(config={
        'size': (200, 320),
    }) # another problem here, since I will likely need a camera object

    # no need to display, but will be there in case we want to run locally
    display = vista.Display(world)
    

    # for analysis after the fact
    has_video_writer = args.out_path is not None
    if has_video_writer:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        # fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.out_path, fourcc, 30, (960, 320))


    # this is giving us issues, but is necessary to spawn agents and update traces
    world.reset()
    display.reset()


    # actually running simulated movement of agent
    frame_idx = 0
    while not car.done and frame_idx <= 300: # can change this for longer simuls
        try:
            print(f"Frame Index: {frame_idx}")


            ### PYAE'S TODO 

            """
            will have action = (output_of_BarrierNet, given Hector's perception input and BarrierNet processing, 
            will need segmentation and depth), 
            syntax for action is 2-tuple with (angle, velocity), so make sure to format it correctly
            how would you fetch the correct format of the BarrierNet output (formatted as (angle, velocity))
            """


            ### END PYAE'S TODO
            action = None # REMOVE THIS AFTER
            car.step_dynamics(action)
            car.step_sensors()

            # this is for the irrelevant display, we can ignore but keep it 
            vis_img = display.render()
            if has_video_writer:
                video_writer.write(vis_img[:, :, ::-1])
            else:
                cv2.imshow('Visualize RGB', vis_img[:, :, ::-1])
                cv2.waitKey(20) # re-indent this when done TODO
            
            frame_idx += 1
        except KeyboardInterrupt:
            break

    if has_video_writer:
        video_writer.release()
        # os.system(f"ffmpeg -i {args.out_path} -vcodec libx264 {args.out_path}")


def follow_human_trajectory(agent):
    print(f"agent is {agent}")
    print(f"agent.trace is {agent.trace}")
    action = np.array([
        agent.trace.f_curvature(agent.timestamp),
        agent.trace.f_speed(agent.timestamp)
    ])
    return action


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument('--trace-path',
                        type=str,
                        nargs='+',
                        help='Path to the traces to use for simulation')
    parser.add_argument('--out-path',
                        type=str,
                        help='Path to save the video',
                        default=None)
    args = parser.parse_args()

    main(args)