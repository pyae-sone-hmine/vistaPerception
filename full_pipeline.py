import argparse
import numpy as np
import os
import cv2

import vista
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature

def main(args):
    world = vista.World(args.trace_path, trace_config={'road_width': 4})
    car = world.spawn_agent(
        config={
            'length': 5.,
            'width': 2.,
            'wheel_base': 2.78,
            'steering_ratio': 14.7,
            'lookahead_road': True
        })

    camera_front = car.spawn_camera(config={
        'size': (200, 320),
    }) # another problem here, since I will likely need a camera object
    display = vista.Display(world)
    
    has_video_writer = args.out_path is not None
    if has_video_writer:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        # fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.out_path, fourcc, 30, (960, 320))
    
    world.reset()
    display.reset()

    frame_idx = 0
    while not car.done and frame_idx <= 300:
        try:
            print(f"Frame Index: {frame_idx}")
            # action = follow_human_trajectory(car) # FIXME changed this
            action = np.array([0,10]) # FIXME this is a hard-coded statespace controller
            car.step_dynamics(action)
            car.step_sensors()

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