import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, frames_dir, fps=30):
    video = cv2.VideoCapture(video_path)
    video_fps = int(video.get(cv2.CAP_PROP_FPS))
    
    # Define the frame skip based on original video's fps and required fps
    frame_skip = video_fps // fps
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_save = total_frames // frame_skip

    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    count = 0
    save_count = 0
    with tqdm(total=frames_to_save, desc="Extracting Frames") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if count % frame_skip == 0:
                save_path = os.path.join(frames_dir, f'frame_{save_count}.jpg')
                cv2.imwrite(save_path, frame)
                save_count += 1
                pbar.update(1)
            count += 1

    video.release()
    print("Finished extracting frames.")

