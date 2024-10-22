import subprocess
import os

def get_user_input(prompt, default=None):
    user_input = input(f"{prompt} [{default}]: ").strip() if default else input(f"{prompt}: ").strip()
    return user_input if user_input else default

def run_clustering_test():
    project_dir = "/Users/akhileshnevatia/Desktop/SE_491/LatentConceptAnalysis/NeuroX/NeuroXCode_Test/src/clustering/test_directory/CodeConceptNet/clusters/java_test/activations"
    
    layer = get_user_input("Layer Number?", "1")
    clusters = get_user_input("Number of Clusters?", "5")
    
    use_agglomerative = get_user_input("Use Agglomerative Clustering? (yes/no)", "yes").lower() == 'yes'
    use_kmeans = get_user_input("Use KMeans Clustering? (yes/no)", "yes").lower() == 'yes'
    use_leaders = get_user_input("Use Leaders Clustering? (yes/no)", "yes").lower() == 'yes'
    
    tau = "0.5"
    if use_leaders:
        tau = get_user_input("Tau value for Leaders Clustering?", "0.5")

    command = [
        "neuroxcode", "run_clustering",
        project_dir,
        layer,
        clusters
    ]

    if use_agglomerative:
        command.append("--agglomerative")
    if use_kmeans:
        command.append("--kmeans")
    if use_leaders:
        command.append("--leaders")
        command.extend(["-t", tau])

    print("\nExecuting command:")
    print(" ".join(command))

    try:
        subprocess.run(command, check=True)
        print("\nClustering test completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred while running clustering: {e}")

if __name__ == "__main__":
    run_clustering_test()
