
import os
import pickle
import argparse

import cv2

from egtea_gaze.gaze_data.gaze_io_sample import parse_gtea_gaze


if __name__ == '__main__':
    # ATTENTION
    # Clips have been moved to $SCRATCH
    # Make sure they exist there before running this file
    SCRATCH = '/cluster/scratch/tsiebert'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=f'egtea_gaze')
    parser.add_argument('--video_name', default=f'OP01-R01-PastaSalad')
    parser.add_argument('--clip_name', default=f'clip1')
    parser.add_argument('--start_time', type=float, default=66, help='Time in seconds relative to start of original video')
    args = parser.parse_args()

    dataset = args.dataset
    video_name = args.video_name
    clip_name = args.clip_name
    clip_start = args.start_time

    fps = 24
    resolution = (640, 480)

    clip_path = f'{SCRATCH}/{dataset}/analyzed_clips/{video_name}/{clip_name}.mp4'

    cap = cv2.VideoCapture(clip_path)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(clip_name + '_with_gaze' + path_to_clip[-4:], fourcc, fps, resolution)

    if dataset == 'egtea_gaze':
        gaze = parse_gtea_gaze(os.path.join(
            'egtea_gaze/gaze_data/gaze_data/', 
            video_name + '.txt'
        ))
    gaze_each_frame = []

    frame_num = int(clip_start * fps) - 1

    while cap.isOpened(): 
        ret, frame = cap.read() 
        if ret:
            # Extract gaze labels
            gaze_point = gaze[frame_num]
            gaze_x, gaze_y = int(gaze_point[0]*resolution[0]), int(gaze_point[1]*resolution[1])

            # Overlay gaze on frame
            # cv2.circle(frame, (gaze_x, gaze_y), radius=5, thickness=-1, color=(0,0,255))
            # out.write(frame)
            # cv2.imshow('frame', frame)

            gaze_each_frame.append((gaze_x, gaze_y))
            
            frame_num += 1

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else: 
            break 

    cap.release() 
    # out.release() 
    cv2.destroyAllWindows()

    save_dir = f'{dataset}/results/{video_name}/{clip_name}/gaze/'
    with open(save_dir + 'gaze_data.pkl', 'wb') as gaze_data_file:
        pickle.dump(gaze_each_frame, gaze_data_file)



