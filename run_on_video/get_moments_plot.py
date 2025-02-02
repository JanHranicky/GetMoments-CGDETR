import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os 
from moviepy.editor import VideoFileClip

def extract_frames_with_timestamps(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps  # total video duration in seconds
    
    time_indices = [2*i for i in range(int(duration // 2))]
    time_indices.append(duration)
    
    frames = []
    for timestamp in time_indices:
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        success, frame = cap.read()
        
        if success:
            # Convert BGR (OpenCV format) to RGB for Matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
            frames.append(frame_rgb)
        else:
            print(f"Warning: Could not read frame at {timestamp:.2f}s")
            
    cap.release()
    return time_indices, frames

def plot_frames_and_curve(frames, timestamps, scores, folder, file):
    fig = plt.figure(figsize=(15, 8))
    
    # Create subplots for the frames
    for idx, frame in enumerate(frames):
        ax = fig.add_subplot(2, len(frames), idx + 1)
        ax.imshow(frame)
        ax.axis('off')  # Turn off axis
    
    # Create a subplot for the timeline curve
    ax_sub = fig.add_subplot(212)
    ax_sub.plot(timestamps, scores, 'ro-', label="Saliency score")
    
    max_score_idx = np.argmax(scores)
    max_score = scores[max_score_idx]
    max_score_time = timestamps[max_score_idx]
    ax_sub.plot(max_score_time, max_score, 'go', markersize=12, label=f"Max score timestamp: {2*max_score_idx:.2f}-{(2*max_score_idx+2):.2f}s")

    ax_sub.set_xlim(min(timestamps), max(timestamps))  
    ax_sub.set_xlabel("Time (seconds)")
    ax_sub.set_yticks([])
    ax_sub.legend()
    
    plt.subplots_adjust(wspace=0, hspace=-0.2)
    
    save_path = os.path.join(folder, f"{file}_saliency.png")
    
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


PREDICTION_DICTS = [
    'get_moments_predictions_empty_query.pkl',
    'get_moments_predictions_ones_query.pkl',
    'get_moments_predictions_zeroes_query.pkl',
]
VIDEO_FOLDER_PATH = "run_on_video/example/"
OUT_FOLDER = "get_moments_out/"

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def extract_video_clip_with_audio(video_path, start_time, end_time, output_path):
    video_clip = VideoFileClip(video_path,target_resolution=(1920,1080))
    
    start_time = max(start_time,0)
    end_time = min(end_time,video_clip.duration)
    
    video_clip = video_clip.subclip(start_time, end_time)
    
    audio_clip = video_clip.audio
    video_clip = video_clip.set_audio(audio_clip)

    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    print(f"Extracted video clip with audio saved to {output_path}")
    
    
def plot_saliencies(folder,dict):
    for file in dict.keys():
        timestamps, frames = extract_frames_with_timestamps(VIDEO_FOLDER_PATH+file)
            
        pred_saliency_scores = dict[file][0]['pred_saliency_scores']
        bias = 0 - min(pred_saliency_scores)
        pred_saliency_scores = [score + bias for score in pred_saliency_scores]
            
        if len(pred_saliency_scores) != len(timestamps):
            timestamps.pop()
            
        plot_frames_and_curve(frames, timestamps,pred_saliency_scores,folder,file.split(".")[0])
    

def get_saliance_timestamp(file,dict):
    pred_saliency_scores = dict[file][0]['pred_saliency_scores']
    bias = 0 - min(pred_saliency_scores)
    pred_saliency_scores = [score + bias for score in pred_saliency_scores]

    start = 2*np.argmax(pred_saliency_scores)
    return start, start+2

def extract_max_saliance_clips(folder_path):
    for file in pred_dict.keys():
        start,end = get_saliance_timestamp(file,pred_dict)

        file_without_ext = file.split(".")[0]
        clip_path = os.path.join(folder_path, f"{file_without_ext}_max_saliency.mp4")
        extract_video_clip_with_audio(pred_dict[file][0]["vid"],start,end,clip_path)

def retrieve_predicted_moments(folder_path):
    for file in pred_dict.keys():
        max_rel_window = pred_dict[file][0]['pred_relevant_windows'][0] #first element has the max score
        
        file_without_ext = file.split(".")[0]
        clip_path = os.path.join(folder_path, f"{file_without_ext}_retrieved_moment.mp4")
        
        print(max_rel_window[0])
        print(max_rel_window[1])
        print(clip_path)
        extract_video_clip_with_audio(pred_dict[file][0]["vid"],max_rel_window[0],max_rel_window[1],clip_path)
  
if __name__ == "__main__":
    for dict in PREDICTION_DICTS:
        pred_dict = load_pickle(dict)
        
        folder_path = dict.split('.')[0]
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        plot_saliencies(folder_path,pred_dict)
        extract_max_saliance_clips(folder_path)
        retrieve_predicted_moments(folder_path)