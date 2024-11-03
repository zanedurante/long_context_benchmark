import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from cluster_videos import *
import random
import sys

def merge_and_split_clusters(X, initial_assignments, initial_centers, N):
    """
    Merge or split clusters from initial K-means results so each cluster has exactly N points.
    
    Parameters:
    - X: Data points (array of shape (num_points, num_features))
    - initial_assignments: Initial cluster assignments from vanilla K-means
    - initial_centers: Initial cluster centers from vanilla K-means
    - N: Desired number of points per cluster
    
    Returns:
    - new_assignments: Cluster assignments with each cluster having exactly N points
    - new_centers: Final cluster centers
    """
    # Step 1: Group points by initial clusters
    clusters = {}
    for i, cluster_id in enumerate(initial_assignments):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(i)
    
    # Step 2: Merge small clusters
    cluster_ids = list(clusters.keys())
    centers = np.array([initial_centers[cluster_id] for cluster_id in cluster_ids])
    merged_clusters = []

    while cluster_ids:
        cluster_id = cluster_ids.pop(0)
        points = clusters[cluster_id]
        
        # If the cluster has fewer than N points, find the nearest cluster to merge with
        if len(points) < N and cluster_ids:
            # Find closest cluster to merge with
            remaining_centers = np.array([centers[i] for i in cluster_ids])
            distances = list(cdist([centers[cluster_id]], remaining_centers).flatten())  # TODO converting np.array to python list

            while len(points) < N and cluster_ids:  # TODO use while instead of if here..
                nearest_idx = np.argmin(distances)
                nearest_cluster_id = cluster_ids[nearest_idx]

                # Merge clusters
                points.extend(clusters[nearest_cluster_id])
                cluster_ids.remove(nearest_cluster_id)
                distances.pop(nearest_idx)

        merged_clusters.append(points)

    # Step 3: Split clusters with more than N points into sub-clusters
    new_assignments = np.full(len(X), -1, dtype=int)
    new_centers = []
    cluster_id = 0

    for points in merged_clusters:
        if len(points) > 2*N:  # TODO changed this to 2N (break down only if cluster size is too large!)
            # Split into sub-clusters
            sub_clusters, sub_centers = split_into_subclusters(X[points], N)
            for sub_cluster, sub_center in zip(sub_clusters, sub_centers):
                for idx in sub_cluster:
                    new_assignments[points[idx]] = cluster_id
                new_centers.append(sub_center)
                cluster_id += 1
        else:
            # Assign this cluster as is
            for idx in points:
                new_assignments[idx] = cluster_id
            new_centers.append(np.mean(X[points], axis=0))
            cluster_id += 1

    return new_assignments, np.array(new_centers)

def split_into_subclusters(X_subset, N):
    """
    Split a cluster into multiple sub-clusters with exactly N points each using constrained K-means.
    
    Parameters:
    - X_subset: Data points in the cluster to split
    - N: Desired number of points per sub-cluster
    
    Returns:
    - sub_clusters: List of sub-clusters, each with exactly N points
    - sub_centers: Centers of each sub-cluster
    """
    
    num_points = X_subset.shape[0]
    num_subclusters = num_points // N
    
    kmeans = KMeans(n_clusters=num_subclusters, random_state=0).fit(X_subset)

    sub_clusters = [[] for _ in range(num_subclusters)]
    for i, label in enumerate(kmeans.labels_):
        sub_clusters[label].append(i)
    
    sub_centers = kmeans.cluster_centers_
    return sub_clusters, sub_centers


if __name__=="__main__":
    data = np.load('vid_features.npy')

    N = 4000
    data = data[:N,...]  # taking the first 4000 samples 
    print(data.shape)

    VIDS_PER_CLUSTER = 20
    num_clusters = 200

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(data)

    # get the cluster assignments
    cluster_assignments = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # now merge and split the clusters 
    new_assignments, new_centers = merge_and_split_clusters(data, cluster_assignments, cluster_centers, VIDS_PER_CLUSTER)
    print(min(new_assignments), max(new_assignments))

    # visualize the distribution of number of labels per cluster
    id2num = {}
    for i in new_assignments:
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
    plt.savefig(f"visualizations_merge_and_split_2/cluster_distribution_cluster={VIDS_PER_CLUSTER}.png")

    
    video_paths = get_all_video_paths("video_files_10")
    print('total #videos:', len(video_paths))
    cluster_idxs = [random.randrange(num_clusters) for _ in range(10)] # 10 random cluster indices # [0,1,2,3,4,5,6,7,8,9]
    for idx in cluster_idxs:
        print(f"Cluster {idx}:")
        video_cluster_paths = []
        for i, cluster_id in enumerate(new_assignments):
            if cluster_id == idx:
                video_cluster_paths.append(video_paths[i])
        
        # copy all videos to a new dir in visualizations/{cluster_idx}
        os.makedirs(f"visualizations_merge_and_split_2/{VIDS_PER_CLUSTER}/cluster_{idx}", exist_ok=True)
        # clear the dir
        os.system(f"rm -rf visualizations_merge_and_split_2/{VIDS_PER_CLUSTER}/cluster_{idx}/*")

        for video_path in video_cluster_paths:
            os.system(f"cp {video_path} visualizations_merge_and_split_2/{VIDS_PER_CLUSTER}/cluster_{idx}")
        print("\n")
