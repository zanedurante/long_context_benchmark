import transformers
import torch
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import os
import sys
import decord
from glob import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

decord.bridge.set_bridge('torch')

# import CLIP models from transformers
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the processor and model
model_ckpt = "openai/clip-vit-base-patch16"

processor = CLIPProcessor.from_pretrained(model_ckpt)
model = CLIPModel.from_pretrained(model_ckpt).to(device)

def sample_frames_from_video_path(video_path, num_frames=32):
    # Load the video
    vr = decord.VideoReader(video_path)
    # Sample frames
    frame_indices = np.linspace(0, len(vr)-1, num_frames).astype(int)
    frames = vr.get_batch(frame_indices)
    return frames

def get_average_clip_features(frames):
    # Preprocess the frames
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Get the features
    with torch.no_grad():
        outputs = model.vision_model(**inputs, return_dict=True)
    # Average the features
    features = outputs.pooler_output.mean(dim=0)
    return features

def get_clip_features(video_path, num_frames=32):
    feature_path = video_path.replace(".mp4", ".npy")
    if os.path.exists(feature_path):
        features = np.load(feature_path)
        return features
    frames = sample_frames_from_video_path(video_path, num_frames=num_frames)
    features = get_average_clip_features(frames)

    # save features and return
    np.save(feature_path, features.cpu().numpy())
    return features.cpu().numpy()

def get_video_paths_from_dir(video_dir):
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    return video_paths

def get_all_video_paths(parent_dir):
    video_paths = []
    for video_dir in sorted(os.listdir(parent_dir)):
        video_paths.extend(get_video_paths_from_dir(os.path.join(parent_dir, video_dir)))
    return video_paths

if __name__ == "__main__":
    VIDS_PER_CLUSTER = 20
    if len(sys.argv) > 1:
        VIDS_PER_CLUSTER = int(sys.argv[1])
    video_paths = get_all_video_paths("video_files_10")
    features = []
    for video_path in tqdm(video_paths):
        features.append(get_clip_features(video_path))
    
    features = np.stack(features)
    print("Features shape:", features.shape)

    num_clusters = len(video_paths) // VIDS_PER_CLUSTER 
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)

    # get the cluster assignments
    cluster_assignments = kmeans.labels_

    # save the cluster assignments for later use
    np.save(f"visualizations/cluster_assignments_{VIDS_PER_CLUSTER}.npy", cluster_assignments)
    cluster_assignments = np.load(f"visualizations/cluster_assignments_{VIDS_PER_CLUSTER}.npy")

    # visualize the distribution of number of labels per cluster
    id2num = {}
    for i in cluster_assignments:
        if i in id2num:
            id2num[i] += 1
        else:
            id2num[i] = 1

    num_vals = sorted(id2num.values())

    val2freq = {}
    for val in num_vals:
        if val in val2freq:
            val2freq[val] += 1
        else:
            val2freq[val] = 1

    # plot the distribution (sorted)
    plt.figure(figsize=(10, 6))
    plt.bar(val2freq.keys(), val2freq.values())
    plt.xlabel("Number of Videos in Cluster")
    plt.ylabel("Number of Clusters")
    plt.title("Distribution of Videos per Cluster")
    
    # save the plot
    plt.savefig(f"visualizations/cluster_distribution_cluster={VIDS_PER_CLUSTER}.png")

    # get the paths to the videos in clusters 0-10
    cluster_idxs = [0,1,2,3,4,5,6,7,8,9]
    for idx in cluster_idxs:
        print(f"Cluster {idx}:")
        video_cluster_paths = []
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id == idx:
                video_cluster_paths.append(video_paths[i])
        
        # copy all videos to a new dir in visualizations/{cluster_idx}
        os.makedirs(f"visualizations/{VIDS_PER_CLUSTER}/cluster_{idx}", exist_ok=True)
        # clear the dir
        os.system(f"rm -rf visualizations/{VIDS_PER_CLUSTER}/cluster_{idx}/*")

        for video_path in video_cluster_paths:
            os.system(f"cp {video_path} visualizations/{VIDS_PER_CLUSTER}/cluster_{idx}")
        print("\n")
