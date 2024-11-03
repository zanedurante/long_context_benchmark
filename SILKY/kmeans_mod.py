import numpy as np
from sklearn.datasets import make_blobs
import random
from cluster_videos import *

def k_means_fixed_n(X, k, N, max_iter=100, tol=1e-4):
    """
    K-means algorithm where each cluster has exactly N points.

    Parameters:
    - X: Data points (array of shape (num_points, num_features))
    - k: Number of clusters
    - N: Number of points per cluster
    - max_iter: Maximum number of iterations
    - tol: Convergence tolerance
    
    Returns:
    - assignments: Cluster assignment for each data point
    - centers: Final cluster centers
    """
    n_samples, n_features = X.shape
    assert k * N == n_samples, "Total number of points must equal k * N"
    
    # Step 1: Initialize cluster centers
    rng = np.random.default_rng()
    centers = X[rng.choice(n_samples, k, replace=False)]
    assignments = np.full(n_samples, -1, dtype=int)  # -1 indicates unassigned

    # Loop for the maximum number of iterations
    for _ in range(max_iter):
        # Step 2: Compute distances from each point to each center
        distances = np.array([[np.linalg.norm(x - center) for center in centers] for x in X])

        # Step 3: Assign points to clusters with N-point constraint
        clusters = [[] for _ in range(k)]
        sorted_indices = np.argsort(distances, axis=1)  # Indices of closest centers

        for i, closest_clusters in enumerate(sorted_indices):
            for cluster in closest_clusters:
                if len(clusters[cluster]) < N:
                    clusters[cluster].append(i)
                    assignments[i] = cluster
                    break

        # Check if each cluster has exactly N points
        if all(len(cluster) == N for cluster in clusters):
            break

        # Step 4: Update centers
        new_centers = np.array([X[cluster].mean(axis=0) for cluster in clusters])
        if np.all(np.linalg.norm(new_centers - centers, axis=1) < tol):
            break
        centers = new_centers

    return assignments, centers


if __name__=="__main__":
    data = np.load('vid_features.npy')

    N = 4000
    data = data[:N,...]
    print(data.shape)

    VIDS_PER_CLUSTER = 20
    num_clusters = 200

    ### REMOVE THIS LATER
    cluster_assignments = np.load(f"visualizations_kmod/cluster_assignments_{VIDS_PER_CLUSTER}.npy")

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
    
    # # save the plot
    plt.savefig(f"visualizations_kmod/cluster_distribution_cluster={VIDS_PER_CLUSTER}.png")

    sys.exit(0)
    ### 



    assignments, centers = k_means_fixed_n(data, 200, VIDS_PER_CLUSTER)

    print(min(assignments), max(assignments))
    np.save(f"visualizations/cluster_assignments_{VIDS_PER_CLUSTER}.npy", assignments)
    cluster_assignments = np.load(f"visualizations/cluster_assignments_{VIDS_PER_CLUSTER}.npy")

    video_paths = get_all_video_paths("video_files_10")
    cluster_idxs = [random.randrange(num_clusters) for _ in range(10)] # 10 random cluster indices # [0,1,2,3,4,5,6,7,8,9]
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

    # print(type(assignments), len(assignments), assignments[0], assignments[0].shape)
    # print(type(centers))
    # print(centers.shape)
    # np.save("centers.npy", centers)
