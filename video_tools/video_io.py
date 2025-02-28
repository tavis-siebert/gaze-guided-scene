"""
For all things I/O (reading and writing frames) for videos
"""

import cv2

def write_frames(clip, out, res, gaze, frame_offset=0, write_gaze=False):
    """
    Writes all the frames >= frame_offset from start of `clip` to `out`.
    The original intent of this function was to be able to call it multiple times 
    on the same `out` object, stitching together multiple clips into one video, using `frame_offset` to 
    handle inconsitencies such as overlapping frames between clips

    Args:
        clip (str): the path to the video clip to read from
        out (cv2.VideoWriter): the video we're copying + editing frames from clip to
        res (tuple[int]): the resolution (x,y) of the out video
        gaze (ndarray): the gaze for the clip (refer to the gaze_data folder in egtea_gaze for more info)
        frame_offset (int, optional): the number of frames after start to begin reading from
        write_gaze (bool, optional): whether we want the gaze overlayed on the video or not
    """
    cap = cv2.VideoCapture(clip)
    resx, resy = res
    frame_count = 0
    while cap.isOpened(): 
        ret, frame = cap.read() 
        if ret:
            if frame_count >= frame_offset and frame_count < len(gaze):
                if write_gaze:
                    gaze_point = gaze[frame_count]
                    gaze_xy = int(gaze_point[0]*resx), int(gaze_point[1]*resy)
                    cv2.circle(frame, gaze_xy, radius=5, thickness=-1, color=(0,0,255))
                out.write(frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        else: 
            break 
    cap.release()