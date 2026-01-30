import json
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

# Set font to Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Color palette
COLORS = [
    (155/255, 161/255, 205/255),  # R/G/B: 155/161/205
    (174/255, 198/255, 164/255),  # R/G/B: 174/198/164
    (43/255, 136/255, 200/255),   # R/G/B: 43/136/200
    (104/255, 192/255, 200/255),  # R/G/B: 104/192/200
    (215/255, 234/255, 205/255),  # R/G/B: 215/234/205
]

# Gradient color palettes for chunk size distribution
# Green gradient (darker = higher frequency)
GREEN_GRADIENT = [
    '#376439',  # Darkest
    '#4d7e54',
    '#669877',
    '#81b095',
    '#a4cbb7',
    '#cfeadf'   # Lightest
]

# Blue gradient (darker = higher frequency)
BLUE_GRADIENT = [
    '#313772',  # Darkest
    '#2c4ca0',
    '#326db6',
    '#478ecc',
    '#75b5dc',
    '#a8dde1'   # Lightest
]

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def create_gradient_colormap(gradient_colors, name='custom'):
    """Create a continuous colormap from gradient colors."""
    from matplotlib.colors import LinearSegmentedColormap
    
    # Convert hex colors to RGB tuples
    colors_rgb = [hex_to_rgb(c) for c in gradient_colors]
    
    # Create colormap
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list(name, colors_rgb, N=n_bins)
    return cmap


def load_clusters_from_json(json_file: str) -> Tuple[List[Dict], int, int]:
    """
    Load clusters information from JSON file.
    
    Returns:
        (clusters, num_mapped, num_unmapped)
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    clusters = data.get('clusters', [])
    
    # Count mapped and unmapped clusters
    num_mapped = sum(1 for c in clusters if c.get('is_mapped', False))
    num_unmapped = sum(1 for c in clusters if not c.get('is_mapped', False))
    
    return clusters, num_mapped, num_unmapped


def analyze_all_files(input_dir: str) -> Tuple[List[int], List[Dict]]:
    """
    Analyze all JSON files to collect statistics.
    
    Returns:
        (chunk_sizes, file_stats)
    """
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.json')]
    json_files.sort()
    
    chunk_sizes = []
    file_stats = []
    
    print(f"Analyzing {len(json_files)} JSON files...")
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            clusters, num_mapped, num_unmapped = load_clusters_from_json(json_file)
            
            if len(clusters) == 0:
                continue
            
            # Collect chunk sizes for this file
            file_chunk_sizes = [len(c.get('chunks', [])) for c in clusters]
            
            # Collect chunk sizes for overall statistics
            for num_chunks in file_chunk_sizes:
                chunk_sizes.append(num_chunks)
            
            # Calculate detailed chunk size statistics for this file
            if file_chunk_sizes:
                file_chunk_stats = {
                    'mean': float(np.mean(file_chunk_sizes)),
                    'median': float(np.median(file_chunk_sizes)),
                    'std': float(np.std(file_chunk_sizes)),
                    'min': int(np.min(file_chunk_sizes)),
                    'max': int(np.max(file_chunk_sizes)),
                    'percentiles': {
                        '25th': float(np.percentile(file_chunk_sizes, 25)),
                        '50th': float(np.percentile(file_chunk_sizes, 50)),
                        '75th': float(np.percentile(file_chunk_sizes, 75)),
                        '90th': float(np.percentile(file_chunk_sizes, 90)),
                        '95th': float(np.percentile(file_chunk_sizes, 95)),
                        '99th': float(np.percentile(file_chunk_sizes, 99))
                    }
                }
            else:
                file_chunk_stats = {
                    'mean': 0.0,
                    'median': 0.0,
                    'std': 0.0,
                    'min': 0,
                    'max': 0,
                    'percentiles': {
                        '25th': 0.0,
                        '50th': 0.0,
                        '75th': 0.0,
                        '90th': 0.0,
                        '95th': 0.0,
                        '99th': 0.0
                    }
                }
            
            # Calculate combination count
            total_clusters = len(clusters)
            if num_unmapped > 0 and num_unmapped < total_clusters:
                try:
                    combination_count = math.comb(total_clusters, num_unmapped)
                    combination_log10 = math.log10(combination_count) if combination_count > 0 else 0
                except (OverflowError, ValueError):
                    # Estimate using Stirling's approximation for very large numbers
                    # log10(C(n,k)) ≈ log10(n!) - log10(k!) - log10((n-k)!)
                    # Using approximation: log10(n!) ≈ n*log10(n) - n*log10(e) + 0.5*log10(2*pi*n)
                    n = total_clusters
                    k = num_unmapped
                    if n > 0 and k > 0 and n - k > 0:
                        # Rough approximation
                        combination_log10 = k * math.log10(n) if n > k else 0
                    else:
                        combination_log10 = 0
            else:
                combination_count = 0
                combination_log10 = 0
            
            file_stats.append({
                'file': os.path.basename(json_file),
                'total_clusters': total_clusters,
                'num_mapped': num_mapped,
                'num_unmapped': num_unmapped,
                'combination_count': combination_count if 'combination_count' in locals() else 0,
                'combination_log10': combination_log10,
                'chunk_size_stats': file_chunk_stats,
                # Keep backward compatibility fields
                'avg_chunks_per_cluster': file_chunk_stats['mean'],
                'max_chunks_per_cluster': file_chunk_stats['max'],
                'min_chunks_per_cluster': file_chunk_stats['min']
            })
            
        except Exception as e:
            print(f"Error processing {os.path.basename(json_file)}: {e}")
            continue
    
    return chunk_sizes, file_stats


def plot_chunk_size_distribution(chunk_sizes_list: List[List[int]], labels: List[str], output_path: str, 
                                 min_percentage: float = 1.0):
    """
    Plot the distribution of chunk sizes across all clusters using pie charts.
    Uses continuous gradient colors where darker colors represent higher frequencies.
    
    Args:
        chunk_sizes_list: List of chunk size lists (one for each dataset)
        labels: List of dataset labels
        output_path: Output file path
        min_percentage: Minimum percentage to show individually (others merged as "Others")
    """
    from matplotlib.colors import Normalize
    from collections import Counter
    
    if len(chunk_sizes_list) != 2 or len(labels) != 2:
        raise ValueError("Need exactly 2 datasets and 2 labels")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Create continuous colormaps
    green_cmap = create_gradient_colormap(GREEN_GRADIENT, 'green_gradient')
    blue_cmap = create_gradient_colormap(BLUE_GRADIENT, 'blue_gradient')
    colormaps = [green_cmap, blue_cmap]
    
    for idx, (chunk_sizes, label, cmap) in enumerate(zip(chunk_sizes_list, labels, colormaps)):
        ax = axes[idx]
        
        # Count frequencies for each chunk size
        chunk_counter = Counter(chunk_sizes)
        sorted_items = sorted(chunk_counter.items())
        chunk_values = [item[0] for item in sorted_items]
        frequencies = [item[1] for item in sorted_items]
        
        # Calculate percentages
        total = sum(frequencies)
        percentages = [f / total * 100 for f in frequencies]
        
        # Merge small values into "Others"
        main_values = []
        main_frequencies = []
        others_freq = 0
        others_count = 0
        
        for val, freq, pct in zip(chunk_values, frequencies, percentages):
            if pct >= min_percentage:
                main_values.append(str(val))
                main_frequencies.append(freq)
            else:
                others_freq += freq
                others_count += 1
        
        # Add "Others" if there are small values
        if others_freq > 0:
            main_values.append(f"Others ({others_count})")
            main_frequencies.append(others_freq)
        
        # Color each wedge based on frequency (higher frequency = darker color)
        if main_frequencies:
            max_freq = max(main_frequencies)
            min_freq = min(main_frequencies)
            norm = Normalize(vmin=min_freq, vmax=max_freq)
            
            colors = []
            for freq in main_frequencies:
                normalized = 1.0 - norm(freq)  # Reverse: higher freq -> darker color
                colors.append(cmap(normalized))
            
            # Calculate percentages for autopct function
            total_freq = sum(main_frequencies)
            percentages = [f / total_freq * 100 for f in main_frequencies]
            
            # Custom autopct function: only show percentage for slices >= 3%
            def autopct_func(pct):
                return f'{pct:.1f}%' if pct >= 3.0 else ''
            
            # Create pie chart with labeldistance and pctdistance to move labels outward
            wedges, texts, autotexts = ax.pie(main_frequencies, labels=main_values, colors=colors,
                                              autopct=autopct_func, startangle=90,
                                              textprops={'fontsize': 8, 'fontfamily': 'Arial'},
                                              labeldistance=1.15,  # Move labels further out
                                              pctdistance=0.85)  # Move percentage text further out
            
            # Make percentage text dark color for better visibility, especially on small wedges
            for autotext in autotexts:
                if autotext.get_text():  # Only style non-empty text
                    autotext.set_color('#000000')  # Black color
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(10)  # Slightly larger font
            
            # Also adjust label text for better visibility
            for text in texts:
                text.set_fontsize(9)
                text.set_fontfamily('Arial')
        
        ax.set_title(f'{label}', fontsize=14, fontweight='bold', fontfamily='Arial')
        
        # Add statistics
        mean_chunks = np.mean(chunk_sizes)
        median_chunks = np.median(chunk_sizes)
        std_chunks = np.std(chunk_sizes)
        stats_text = f'Mean: {mean_chunks:.2f}\nMedian: {median_chunks:.2f}\nStd: {std_chunks:.2f}\nTotal: {len(chunk_sizes):,}'
        ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
                 verticalalignment='center', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=10, fontfamily='Arial')
    
    plt.tight_layout()
    # Save PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chunk size distribution plot saved to: {output_path}")
    # Also save PDF version
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Chunk size distribution plot (PDF) saved to: {pdf_path}")
    plt.close()


def plot_combination_magnitude(file_stats: List[Dict], output_path: str):
    """
    Plot the magnitude (log10) of combination counts for each file.
    Only shows one subplot: histogram of combination counts in ascending order.
    """
    # Extract data
    combination_log10s = [s['combination_log10'] for s in file_stats if s['combination_log10'] > 0]
    
    if not combination_log10s:
        print("No valid combination counts to plot")
        return
    
    # Sort in ascending order
    sorted_log10s = sorted(combination_log10s)
    
    # Create single subplot figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Create histogram: x-axis is counts (log10), y-axis is frequency
    # Use bins to create histogram
    bins = np.linspace(min(sorted_log10s), max(sorted_log10s), 30)
    counts, bin_edges = np.histogram(sorted_log10s, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot histogram
    ax.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]) * 0.8, 
           alpha=0.7, edgecolor='black', color=COLORS[2])
    
    ax.set_xlabel('log10(Number of Combinations)', fontsize=12, fontfamily='Arial')
    ax.set_ylabel('Frequency', fontsize=12, fontfamily='Arial')
    ax.set_title('Distribution of Combination Counts (log10 scale)', fontsize=14, fontweight='bold', fontfamily='Arial')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_log10 = np.mean(sorted_log10s)
    max_log10 = np.max(sorted_log10s)
    min_log10 = np.min(sorted_log10s)
    median_log10 = np.median(sorted_log10s)
    stats_text = f'Mean: {mean_log10:.2f}\nMedian: {median_log10:.2f}\nMax: {max_log10:.2f}\nMin: {min_log10:.2f}\nTotal files: {len(sorted_log10s)}'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, fontfamily='Arial')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combination magnitude plot saved to: {output_path}")
    plt.close()


def main():
    # Directories containing JSON files
    input_dirs = [
        "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/Mind2Web2/gemini/after_update",
        "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/Tianyu_sampled/gemini/after_update"
    ]
    labels = ["Mind2Web2", "ReportEval"]
    output_dir = "/data/zyh/DeepResearch/HalluBench_backup_0828/HalluDetector/results/Mind2Web2/gemini"
    
    # Analyze all files from both directories
    chunk_sizes_list = []
    all_file_stats = []
    
    for input_dir, label in zip(input_dirs, labels):
        print(f"\nAnalyzing {label}...")
        chunk_sizes, file_stats = analyze_all_files(input_dir)
        chunk_sizes_list.append(chunk_sizes)
        all_file_stats.extend(file_stats)
        print(f"  Found {len(chunk_sizes)} chunks from {len(file_stats)} files")
    
    if not any(chunk_sizes_list):
        print("No data found to plot")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Chunk size distribution (both datasets side by side)
    chunk_dist_path = os.path.join(output_dir, "chunk_size_distribution_combined.png")
    plot_chunk_size_distribution(chunk_sizes_list, labels, chunk_dist_path, min_percentage=1.0)
    
    # Plot 2: Combination magnitude
    combination_mag_path = os.path.join(output_dir, "combination_magnitude.png")
    plot_combination_magnitude(all_file_stats, combination_mag_path)
    
    # Save statistics to JSON (aggregated)
    all_chunk_sizes = []
    for cs in chunk_sizes_list:
        all_chunk_sizes.extend(cs)
    
    stats_path = os.path.join(output_dir, "combination_statistics.json")
    stats_summary = {
        'chunk_size_stats': {
            'total_clusters': len(all_chunk_sizes),
            'mean': float(np.mean(all_chunk_sizes)),
            'median': float(np.median(all_chunk_sizes)),
            'std': float(np.std(all_chunk_sizes)),
            'min': int(np.min(all_chunk_sizes)),
            'max': int(np.max(all_chunk_sizes)),
            'percentiles': {
                '25th': float(np.percentile(all_chunk_sizes, 25)),
                '50th': float(np.percentile(all_chunk_sizes, 50)),
                '75th': float(np.percentile(all_chunk_sizes, 75)),
                '90th': float(np.percentile(all_chunk_sizes, 90)),
                '95th': float(np.percentile(all_chunk_sizes, 95)),
                '99th': float(np.percentile(all_chunk_sizes, 99))
            }
        },
        'combination_stats': {
            'total_files': len(all_file_stats),
            'files_with_valid_combinations': len([s for s in all_file_stats if s['combination_log10'] > 0]),
            'mean_log10': float(np.mean([s['combination_log10'] for s in all_file_stats if s['combination_log10'] > 0])) if any(s['combination_log10'] > 0 for s in all_file_stats) else 0,
            'max_log10': float(np.max([s['combination_log10'] for s in all_file_stats if s['combination_log10'] > 0])) if any(s['combination_log10'] > 0 for s in all_file_stats) else 0,
            'min_log10': float(np.min([s['combination_log10'] for s in all_file_stats if s['combination_log10'] > 0])) if any(s['combination_log10'] > 0 for s in all_file_stats) else 0
        },
        'file_details': all_file_stats
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nStatistics saved to: {stats_path}")
    print(f"\nSummary:")
    print(f"  Total clusters analyzed: {len(all_chunk_sizes):,}")
    print(f"  Mean chunks per cluster: {np.mean(all_chunk_sizes):.2f}")
    print(f"  Median chunks per cluster: {np.median(all_chunk_sizes):.2f}")
    print(f"  Files with valid combinations: {stats_summary['combination_stats']['files_with_valid_combinations']}")
    if stats_summary['combination_stats']['mean_log10'] > 0:
        print(f"  Mean log₁₀(combinations): {stats_summary['combination_stats']['mean_log10']:.2f}")
        print(f"  Max log₁₀(combinations): {stats_summary['combination_stats']['max_log10']:.2f}")


if __name__ == "__main__":
    main()

