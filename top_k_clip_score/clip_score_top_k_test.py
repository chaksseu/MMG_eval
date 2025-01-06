import os
import re
import json
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Use Agg backend for environments without display
import matplotlib.pyplot as plt
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
import argparse

# Excluded word patterns
excluded_words = ['vggsound', 'sparse', 'test', 'batch', 'proc', 'sample', 'audio', 'video']
pattern_words = re.compile(r'^(?:' + '|'.join(excluded_words) + r')\d*$', re.IGNORECASE)
pattern_numbers = re.compile(r'^\d+$')

def clean_sentence(filename):
    """Remove excluded words/numbers from filename and convert to a readable sentence."""
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    
    sentence = filename.replace('_', ' ').replace('.mp4', '')
    words = sentence.split()
    filtered_words = [
        word for word in words 
        if not pattern_words.match(word) and not pattern_numbers.match(word)
    ]
    cleaned_sentence = ' '.join(filtered_words)
    return cleaned_sentence

def load_video_frames(video_path, num_frames, frame_size=(224, 224)):
    """Load a certain number of frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((frame_size[0], frame_size[1])),
        transforms.CenterCrop((frame_size[0], frame_size[1])),
        transforms.ToTensor()
    ])
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = transform(frame)
        frames.append(tensor_frame)
    cap.release()
    
    if frames:
        return torch.stack(frames)
    else:
        # Return a tensor of zeros if no frames are loaded
        return torch.zeros((0, 3, frame_size[0], frame_size[1]))

def load_videos_with_caps(folder_path, num_frames):
    """Load videos and captions from a folder, return stacked tensor and caption list."""
    videos_tensor_list = []
    caps = []
    
    vid_fnames = sorted(os.listdir(folder_path))
    
    if not vid_fnames:
        print(f"No video files found in {folder_path}.")
        return torch.empty((0, num_frames, 3, 224, 224)), caps
    
    print(f"Loading videos from {folder_path}...")
    for video_name in tqdm(vid_fnames, desc="Loading Videos", unit="video"):
        fname, ext = os.path.splitext(video_name)
        caps.append(clean_sentence(fname))
        
        video_path = os.path.join(folder_path, video_name)
        video_tensor = load_video_frames(video_path, num_frames)
        
        if video_tensor.shape[0] < num_frames:
            # If fewer frames are loaded, pad with zeros
            padding = num_frames - video_tensor.shape[0]
            if padding > 0:
                pad_tensor = torch.zeros((padding, 3, 224, 224))
                video_tensor = torch.cat((video_tensor, pad_tensor), dim=0)
        
        videos_tensor_list.append(video_tensor)
    
    if videos_tensor_list:
        return torch.stack(videos_tensor_list), caps
    else:
        return torch.empty((0, num_frames, 3, 224, 224)), caps

def calculate_clip_topk(
    videos, 
    caps, 
    clip_model="openai/clip-vit-base-patch32", 
    device="cuda", 
    chunk_size=40
):
    """
    Compute per-frame CLIP scores and then get top-1~top-n averages.
    Returns a dict of:
      - clip_topk_scores: list of lists (top-k per video)
      - clip_per_frame_scores: list of lists (per-frame scores)
      - clip_video_setting: shape of the first video
    """
    print("Calculating CLIP Score (per-frame) and top-k averages...")
    metric = CLIPScore(model_name_or_path=clip_model).to(device)

    B, T, C, H, W = videos.shape
    all_videos_topk = []
    all_videos_per_frame = []

    # Iterate over each video with a progress bar
    for video_idx in tqdm(range(B), desc="Processing Videos", unit="video"):
        video_tensor = videos[video_idx] * 255  # Scale to 0-255
        text = caps[video_idx]

        frame_scores = []
        # Iterate over chunks of frames with a nested progress bar
        for start_idx in tqdm(range(0, T, chunk_size), desc=f"Video {video_idx+1}/{B}", leave=False, unit="chunk"):
            end_idx = start_idx + chunk_size
            frames_chunk = video_tensor[start_idx:end_idx].to(device)  # (chunk, C, H, W)

            # Iterate over each frame in the chunk
            for i in range(frames_chunk.shape[0]):
                single_frame = frames_chunk[i].unsqueeze(0)  # (1, C, H, W)
                single_text = [text]
                with torch.no_grad():
                    score_tensor = metric(single_frame, single_text)
                frame_scores.append(score_tensor.item())

            del frames_chunk
            torch.cuda.empty_cache()

        if frame_scores:
            sorted_scores = sorted(frame_scores, reverse=True)
            topk_list = [np.mean(sorted_scores[:k]) for k in range(1, len(sorted_scores) + 1)]
        else:
            # If no frames were scored, fill with zeros
            topk_list = [0.0] * T

        all_videos_topk.append(topk_list)
        all_videos_per_frame.append(frame_scores)

    return {
        "clip_topk_scores": all_videos_topk,
        "clip_per_frame_scores": all_videos_per_frame,
        "clip_video_setting": (T, C, H, W)
    }

def main(args):
    # 1. Load videos and captions
    clip_videos, caps = load_videos_with_caps(args.preds_folder, args.num_frames)
    
    if clip_videos.shape[0] == 0:
        print("No videos loaded. Exiting.")
        return
    
    # 2. Calculate top-k CLIP Scores
    topk_results = calculate_clip_topk(
        videos=clip_videos,
        caps=caps,
        clip_model=args.clip_model,
        device=args.device,
        chunk_size=args.chunk_size
    )

    # 3. Print example (first video)
    if topk_results["clip_topk_scores"]:
        print("=== Top-k CLIP Scores for the first video ===")
        print(topk_results["clip_topk_scores"][0])
    else:
        print("No CLIP scores computed.")
    
    # 4. Compute average over all videos for each k
    all_videos_topk = topk_results["clip_topk_scores"]
    B = len(all_videos_topk)
    if B == 0:
        print("No CLIP scores found. Exiting.")
        return
    
    n = len(all_videos_topk[0])  # frames per video
    avg_topk_scores = []
    for k in range(n):
        kth_scores = [all_videos_topk[v_idx][k] for v_idx in range(B)]
        avg_topk = np.mean(kth_scores)
        avg_topk_scores.append(avg_topk)

    # 5. Save results to JSON
    json_output = {
        "clip_topk_scores": topk_results["clip_topk_scores"],
        "clip_per_frame_scores": topk_results["clip_per_frame_scores"],
        "clip_video_setting": topk_results["clip_video_setting"],
        "avg_topk_scores": avg_topk_scores
    }

    output_json_path = args.output_json_path
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=4)
    print(f"Saved JSON results to: {output_json_path}")
    
    # 6. Plot and save figure
    x_vals = list(range(1, n + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, avg_topk_scores, marker='o', color='blue', label='Average Top-k CLIP Score')
    plt.title("Average Top-k CLIP Scores Across All Videos")
    plt.xlabel("k")
    plt.ylabel("CLIP Score")
    plt.legend()
    plt.grid(True)

    output_fig_path = args.output_fig_path
    plt.savefig(output_fig_path, dpi=150)
    plt.close()
    print(f"Saved figure to: {output_fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Top-k CLIP Scores for Videos")
    parser.add_argument("--preds_folder", type=str, default="/workspace/dataset/vggsound_sparse_test_6s", help="Path to folder containing videos")
    parser.add_argument("--clip_model", type=str, default='openai/clip-vit-base-patch16', help="CLIP model name")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device: cuda or cpu")
    parser.add_argument("--output_json_path", type=str, default="vggsound_sparse_test_clip_6s_topk_results.json", help="Output JSON file path")
    parser.add_argument("--output_fig_path", type=str, default="vggsound_sparse_test_6s_topk_scores.png", help="Output figure file path")
    parser.add_argument("--num_frames", type=int, default=180, help="Number of frames to load from each video")
    parser.add_argument("--chunk_size", type=int, default=180, help="Number of frames to process in each chunk")
    args = parser.parse_args()

    main(args)
