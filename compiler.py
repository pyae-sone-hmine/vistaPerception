import cv2
import os
import argparse

def images_to_video(folder_path, output_video_path, fps=30):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(x.replace('frame', '').replace('.jpg', '')))

    first_image_path = os.path.join(folder_path, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in image_files:
        image_path = os.path.join(folder_path, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    print(f"Video saved as {output_video_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a folder of images into a video.')
    parser.add_argument('--folder_path', type=str, help='Path to the folder containing images')
    parser.add_argument('--output_video_path', type=str, help='Path to save the output video file')

    args = parser.parse_args()

    # Assuming that both arguments are mandatory. If not, check for None before calling.
    if args.folder_path and args.output_video_path:
        images_to_video(args.folder_path, args.output_video_path)
    else:
        print("Both --folder_path and --output_video_path are required.")