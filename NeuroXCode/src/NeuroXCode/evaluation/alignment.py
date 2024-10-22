import argparse
import csv
import json
import time
import os
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

def load_sentences_and_labels(sentence_file, label_file):
    sentences = []
    labels = []

    with open(sentence_file, 'r') as f_sentences, open(label_file, 'r') as f_labels:
        for sentence, label_line in zip(f_sentences, f_labels):
            sentences.append(sentence.strip())
            labels.append(label_line.strip())

    return sentences, labels

def load_clusters(cluster_file):
    clusters = []

    with open(cluster_file, 'r') as f_clusters:
        for line in f_clusters:
            parts = line.strip().split("|||")
            word = parts[0]
            word_frequency = parts[1]
            sentence_index = int(parts[2])
            word_index = int(parts[3])
            cluster_id = parts[4].split()[-1]
            clusters.append((word, word_frequency, sentence_index, word_index, cluster_id))

    return clusters

def load_activations(activation_file):
    with open(activation_file, 'r') as f:
        activations = [json.loads(line) for line in f]
    return activations

def process_activations(activations, sentences):
    processed_activations = []
    skipped_count = 0
    for act, sent in zip(activations, sentences):
        tokens = sent.split()
        features = act['features']
        if len(tokens) != len(features):
            print(f"Warning: Mismatch in token count for sentence: {sent}")
            skipped_count += 1
            continue
        processed_activations.append([feature['layers'][0]['values'] for feature in features])
    print(f"Total sentences skipped due to token mismatch: {skipped_count}")
    return processed_activations

def filter_label_map(label_map):
    filtered_label_map = {}
    unique_labels = set()

    for tag, word_list in label_map.items():
        if len(word_list) >= 6:
            filtered_label_map[tag] = set(word_list)
            unique_labels.add(tag)

    return filtered_label_map, unique_labels

def create_label_map_2(sentences, labels):
    label_map = {}
    unique_labels = set()

    for sentence_index, label_line in enumerate(labels):
        label_tokens = label_line.split()
        word_tokens = sentences[sentence_index].split()

        for word_index, label in enumerate(label_tokens):
            cluster_words = []

            if label in label_map:
                cluster_words = label_map[label]

            cluster_words.append(word_tokens[word_index])
            label_map[label] = cluster_words
            unique_labels.add(label)

    return filter_label_map(label_map)

def extract_words_items(cluster_words):
    word_items = [item[0] for item in cluster_words]
    return word_items

def assign_labels_to_clusters_2(label_map, clusters, threshold, activations):
    assigned_clusters = {}
    g_c = group_clusters(clusters)

    for cluster_id, cluster_words in g_c:
        word_items = extract_words_items(cluster_words)
        best_match = 0
        best_label = None

        for label_id, label_words in label_map.items():
            x = [value for value in word_items if value in label_words]
            match = len(x) / len(word_items) if word_items else 0

            if match > best_match:
                best_match = match
                best_label = label_id

        if best_match >= threshold:
            assigned_clusters[cluster_id] = best_label
        else:
            assigned_clusters[cluster_id] = "NONE"

    return assigned_clusters, len(assigned_clusters)

def create_label_map(sentences, labels):
    label_map = {}
    unique_labels = set()

    for sentence_index, label_line in enumerate(labels):
        label_tokens = label_line.split()
        word_tokens = sentences[sentence_index].split()

        for word_index, label in enumerate(label_tokens):
            label_map[(word_index, word_tokens[word_index])] = label
            unique_labels.add(label)

    return label_map, unique_labels

def assign_labels_to_clusters(label_map, clusters, threshold, activations):
    assigned_clusters = {}
    assigned_cluster_count = 0

    for cluster_id, cluster_words in group_clusters(clusters):
        cluster_label_counts = {}

        for word, _, sentence_index, word_index, _ in cluster_words:
            if (word_index, word) in label_map:
                label = label_map[(word_index, word)]

                if label in cluster_label_counts:
                    cluster_label_counts[label] += 1
                else:
                    cluster_label_counts[label] = 1

        total_words = len(cluster_words)
        if total_words > 0:
            max_label = max(cluster_label_counts, key=cluster_label_counts.get, default=None)
            if max_label is not None and (cluster_label_counts[max_label] / total_words) >= threshold:
                assigned_clusters[cluster_id] = max_label
                assigned_cluster_count += 1
            else:
                assigned_clusters[cluster_id] = "NONE"
        else:
            assigned_clusters[cluster_id] = "NONE"

    return assigned_clusters, len(assigned_clusters)

def group_clusters(clusters):
    cluster_groups = {}

    for cluster in clusters:
        _, _, _, _, cluster_id = cluster
        if cluster_id in cluster_groups:
            cluster_groups[cluster_id].append(cluster)
        else:
            cluster_groups[cluster_id] = [cluster]

    return cluster_groups.items()

def analyze_clusters(dictionary, unique_labels, assigned_cluster_count, threshold, method):
    key_count = Counter(dictionary.values())
    num_clusters = len(dictionary)
    num_tags_covered = len(key_count) - (1 if 'NONE' in key_count else 0)
    num_unique_tags = len(unique_labels)
    
    # Calculate the proportion of non-NONE labels
    non_none_labels = num_clusters - key_count.get('NONE', 0)
    cluster_label_ratio = non_none_labels / num_clusters
    
    # Calculate tag coverage
    tag_coverage_ratio = num_tags_covered / num_unique_tags
    
    # Calculate overall alignment score
    overall_alignment = (cluster_label_ratio + tag_coverage_ratio) / 2

    return {
        'method': method,
        'threshold': threshold,
        'clusters_labeled': f"{non_none_labels} out of {num_clusters}",
        'tag_coverage': f"{tag_coverage_ratio * 100:.2f}%",
        'overall_alignment': f"{overall_alignment:.4f}",
        'major_labels': ', '.join(f"{label} ({count})" for label, count in key_count.most_common(2) if label != 'NONE'),
        'all_labels': ', '.join(f"{label} ({count})" for label, count in sorted(key_count.items(), key=lambda x: x[1], reverse=True) if label != 'NONE'),
        'none_labels': key_count.get('NONE', 0),
        'unique_tags': num_tags_covered
    }

def generate_final_report(results):
    report = {
        'Metric': ['Threshold', 'Clusters Labeled', 'Tag Coverage', 'Overall Alignment Score', 'Major Labels', 'All Labels', 'None Labels', 'Unique Tags Identified'],
    }
    
    for result in results:
        method = result['method']
        threshold = result['threshold']
        column_name = f"{method}_{threshold}"
        report[column_name] = [
            threshold,
            result['clusters_labeled'],
            result['tag_coverage'],
            result['overall_alignment'],
            result['major_labels'],
            result['all_labels'],
            result['none_labels'],
            result['unique_tags']
        ]

    return report

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Assign labels to clusters based on word labels in sentences.")
    parser.add_argument("--sentence-file", required=True, help="Path to the sentence file")
    parser.add_argument("--label-file", required=True, help="Path to the label file")
    parser.add_argument("--cluster-file", required=True, help="Path to the cluster file")
    parser.add_argument("--activation-dir", required=True, help="Directory containing activation files for different layers")
    parser.add_argument("--thresholds", nargs='+', required=True, help="Alignment thresholds (e.g., 50 60 70)")
    parser.add_argument("--methods", nargs='+', required=True, help="Methods to compare (e.g., M1 M2)")
    parser.add_argument("--output-dir", required=True, help="Directory to save output files")

    args = parser.parse_args()

    sentences, labels = load_sentences_and_labels(args.sentence_file, args.label_file)
    clusters = load_clusters(args.cluster_file)

    # Get all layer directories
    layer_dirs = [d for d in os.listdir(args.activation_dir) if d.startswith("layer")]
    
    for layer_dir in sorted(layer_dirs, key=lambda x: int(x[5:])):
        layer = layer_dir[5:]  # Extract layer number
        print(f"\nProcessing {layer_dir}")
        
        activation_file = os.path.join(args.activation_dir, layer_dir, f"activations-{layer_dir}.json")
        if not os.path.exists(activation_file):
            print(f"No activation file found for {layer_dir}. Skipping.")
            continue

        activations = load_activations(activation_file)
        processed_activations = process_activations(activations, sentences)

        results = []
        all_results = {}

        for method in args.methods:
            for threshold in args.thresholds:
                print(f"\nRunning with method: {method}, threshold: {threshold}")
                if method == "M1":
                    label_map, unique_labels = create_label_map(sentences, labels)
                    assigned_clusters, assigned_cluster_count = assign_labels_to_clusters(label_map, clusters, int(threshold) / 100, processed_activations)
                elif method == "M2":
                    label_map, unique_labels = create_label_map_2(sentences, labels)
                    assigned_clusters, assigned_cluster_count = assign_labels_to_clusters_2(label_map, clusters, int(threshold) / 100, processed_activations)
                
                all_results[f"{method}_{threshold}"] = assigned_clusters
                
                non_none_clusters = sum(1 for label in assigned_clusters.values() if label != "NONE")
                print(f"Number of non-NONE clusters: {non_none_clusters}")
                print(f"Number of clusters successfully assigned a label including NONE: {assigned_cluster_count}")
                results.append(analyze_clusters(assigned_clusters, unique_labels, assigned_cluster_count, threshold, method))

        # Write results for this layer
        output_dir = os.path.join(args.output_dir, layer_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        assigned_labels_file = os.path.join(output_dir, f"assigned_labels_all_methods_thresholds.json")
        with open(assigned_labels_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        final_report = generate_final_report(results)

        # Write the final report to a CSV file
        analysis_results_file = os.path.join(output_dir, f"analysis_results.csv")
        with open(analysis_results_file, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=final_report.keys())
            writer.writeheader()
            for i in range(len(final_report['Metric'])):
                row = {key: final_report[key][i] for key in final_report.keys()}
                writer.writerow(row)

        print(f"Final report for {layer_dir} has been generated and saved to '{analysis_results_file}'")
        print(f"Assigned labels for all methods and thresholds for {layer_dir} have been saved to '{assigned_labels_file}'")

    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()