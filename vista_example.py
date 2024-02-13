# import vista

# trace_path = ["/home/gridsan/hmartinez/fm-driving-with-barriernet/20220221-170847_lexus_devens_outerloop"]

# world = vista.World(trace_path,
#                     trace_config={'road_width': 4})
# car = world.spawn_agent(config={'length': 5.,
#                                 'width': 2.,
#                                 'wheel_base': 2.78,
#                                 'steering_ratio': 14.7,
#                                 'lookahead_road': True})
# display = vista.Display(world)

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

    # Adding another car to see if DINOv2 can run segmentation on this.
    obstacle_car = world.spawn_agent(
        config = {
            'length': 5.,
            'width': 2.,
            'wheel_base': 2.78,
            'steering_ratio': 14.7,
            'lookahead_road': False # don't need to keep road data cached, since it doesn't have a camera attached
        }
    )

    camera_front = car.spawn_camera(config={
        'size': (200, 320),
    })
    print(f"view synthesis is {camera_front.view_synthesis}")
    display = vista.Display(world)
    
    has_video_writer = args.out_path is not None
    if has_video_writer:
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        # fourcc = cv2.VideoWriter_fourcc('X','V','I','D') # FIXME COMMENT THIS OUT
        # fourcc = cv2.VideoWriter_fourcc(*'XVID') # FIXME COMMENT THIS OUT
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # FIXME UNCOMMENT THIS
        # video_writer = cv2.VideoWriter(args.out_path, fourcc, 30, (960, 320))  # changing the way we write out visual outputs
    
    world.reset()
    display.reset()

    frame_idx = 0
    while not car.done and frame_idx <= 300:
        try:
            print(f"Frame Index: {frame_idx}")
            action = follow_human_trajectory(car) # FIXME changed this
            # action = np.array([0,10]) # FIXME this is a hard-coded statespace controller
            print(f"Action that ego car followed was {action}")
            car.step_dynamics(action)
            
            # now add actions to obstacle_car
            # obstacle_action = np.array([15,100]) #FIXME This is just a hardcoded path the obstacle car will follow
            obstacle_action = np.array([action[0], action[1]*1.5]) # follows the same path but faster
            obstacle_car.step_dynamics(obstacle_action)

            car.step_sensors()


            # this is to save the image, will save every third frame
            if frame_idx%3 ==0 and has_video_writer:
                vis_img = display.render()
                filename = f"frame{frame_idx}.jpg"
                # define where to save it
                os.chdir(args.out_path) # goes to outpath directory placed in flag
                cv2.imwrite(filename, vis_img)

            # vis_img = display.render() # this defines the image


            # # does this every third frame for space reasons
            # if has_video_writer and frame_idx%3 == 0:
            #     video_writer.write(vis_img[:, :, ::-1])
            # else:
            #     # cv2.imshow('Visualize RGB', vis_img[:, :, ::-1])
            #     # cv2.waitKey(20) # re-indent this when done TODO
            #     print("we all good, no output path specified, not writing output")
            
            frame_idx += 1
        except KeyboardInterrupt:
            break

    # if has_video_writer:
    #     video_writer.release()
        # os.system(f"ffmpeg -i {args.out_path} -vcodec libx264 {args.out_path}")


def follow_human_trajectory(agent):
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