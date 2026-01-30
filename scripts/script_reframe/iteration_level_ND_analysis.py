import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

# Set Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10

# Define paths
analysis_dir = '/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/nd_analysis_results/updated_claim_cache_json'
cache_dir = '/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/updated_cache'
combined_dir = '/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/misattribute_rejudge/json/updated_whole_results'
output_dir = '/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/nd_analysis_results/iteration_level/plot'

def divide_iterations(total_iterations):
    """Divide iterations into three phases: early, middle, end"""
    phase_size = total_iterations // 3
    remainder = total_iterations % 3
    
    # Distribute remainder to make divisions as equal as possible
    early_end = phase_size + (1 if remainder > 0 else 0)
    middle_end = early_end + phase_size + (1 if remainder > 1 else 0)
    
    return early_end, middle_end

def calculate_iteration_scores(cache_data):
    """Calculate score for each iteration based on chunk scores"""
    
    chunk_scores = cache_data.get('chunk_score', {})
    iterations = cache_data.get('iterations', [])
    
    # Step 1: Build url_index to url mapping from all search_lists
    url_index_to_url = {}
    url_to_iteration = {}
    
    for iteration_idx, iteration in enumerate(iterations):
        search_list_key = f'search_list_{iteration_idx + 1}'
        if search_list_key in iteration:
            search_list = iteration[search_list_key]
            for url in search_list:
                # Track which iteration this url belongs to
                if url not in url_to_iteration:
                    url_to_iteration[url] = []
                url_to_iteration[url].append(iteration_idx + 1)  # 1-based
    
    # Step 2: Calculate average score for each chunk
    chunk_avg_scores = {}
    for chunk_id, chunk_data in chunk_scores.items():
        scores = chunk_data.get('scores', {})
        if scores:
            # Calculate mean of all query combined scores
            combined_scores = [s.get('combined', 0) for s in scores.values()]
            avg_score = np.mean(combined_scores) if combined_scores else 0
            chunk_avg_scores[chunk_id] = avg_score
    
    # Step 3: Group chunks by url and find max score for each url
    url_scores = {}
    for chunk_id, avg_score in chunk_avg_scores.items():
        # Extract url_index from chunk_id (format: "url_index-chunk_Y")
        try:
            url_index_str = chunk_id.split('-')[0]
            url_index = int(url_index_str)
            
            if url_index not in url_scores:
                url_scores[url_index] = []
            url_scores[url_index].append(avg_score)
        except (ValueError, IndexError):
            continue
    
    # Get max score for each url
    url_max_scores = {}
    for url_index, scores in url_scores.items():
        url_max_scores[url_index] = max(scores) if scores else 0
    
    # Step 4: Build url_index mapping by traversing all search_lists in order
    all_urls = []
    for iteration_idx, iteration in enumerate(iterations):
        search_list_key = f'search_list_{iteration_idx + 1}'
        if search_list_key in iteration:
            for url in iteration[search_list_key]:
                if url not in all_urls:
                    all_urls.append(url)
    
    # Create mapping: url -> url_index
    url_to_url_index = {url: idx for idx, url in enumerate(all_urls)}
    
    # Step 5: Calculate average score for each iteration
    iteration_scores = {}
    for iteration_idx, iteration in enumerate(iterations):
        search_list_key = f'search_list_{iteration_idx + 1}'
        if search_list_key in iteration:
            search_list = iteration[search_list_key]
            iter_url_scores = []
            
            for url in search_list:
                if url in url_to_url_index:
                    url_idx = url_to_url_index[url]
                    if url_idx in url_max_scores:
                        iter_url_scores.append(url_max_scores[url_idx])
            
            if iter_url_scores:
                iteration_scores[iteration_idx + 1] = np.mean(iter_url_scores)
            else:
                iteration_scores[iteration_idx + 1] = 0
    
    return iteration_scores

def get_not_support_actions(file_id):
    """Get all NotSupport actions from the combined JSON file"""
    combined_path = os.path.join(combined_dir, f'{file_id}_combined.json')
    if not os.path.exists(combined_path):
        return []
    
    with open(combined_path, 'r') as f:
        data = json.load(f)
    
    not_support_actions = []
    for action in data.get('hallucinated_actions', {}).get('results', []):
        if action['judgment']['label'] == 'NotSupport':
            not_support_actions.append({
                'action_index': action['action_index'],
                'action': action['action']
            })
    
    return not_support_actions

def map_actions_to_iterations(file_id, not_support_actions):
    """Map NotSupport actions to their iteration indices"""
    cache_path = os.path.join(cache_dir, f'cache_{file_id}.json')
    if not os.path.exists(cache_path):
        return {}
    
    with open(cache_path, 'r') as f:
        cache_data = json.load(f)
    
    iterations = cache_data.get('iterations', [])
    
    # Map action text to iteration index
    action_to_iteration = {}
    for iteration_idx, iteration in enumerate(iterations):
        action_list_key = f'action_list_{iteration_idx + 1}'
        if action_list_key in iteration:
            for action_text in iteration[action_list_key]:
                action_to_iteration[action_text] = iteration_idx + 1  # 1-based
    
    # Map NotSupport actions to iterations
    iteration_ha_count = {}
    for ns_action in not_support_actions:
        action_text = ns_action['action']
        if action_text in action_to_iteration:
            iter_idx = action_to_iteration[action_text]
            iteration_ha_count[iter_idx] = iteration_ha_count.get(iter_idx, 0) + 1
    
    return iteration_ha_count

def process_files():
    """Process all analysis and cache files to extract iteration-level data"""
    
    # Get all analysis files
    analysis_files = sorted([f for f in os.listdir(analysis_dir) 
                           if f.startswith('analysis_') and f.endswith('.json')])
    
    results = {}
    
    for analysis_file in analysis_files:
        # Extract file_id from filename
        file_id = analysis_file.replace('analysis_', '').replace('.json', '')
        
        # Read analysis file
        analysis_path = os.path.join(analysis_dir, analysis_file)
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        
        # Check if iteration_level_nd_analysis exists
        if 'iteration_level_nd_analysis' not in analysis_data:
            print(f"Skipping {file_id}: no iteration_level_nd_analysis")
            continue
        
        iteration_results = analysis_data['iteration_level_nd_analysis'].get('iteration_results', {})
        
        # Read cache file to get total iterations
        cache_path = os.path.join(cache_dir, f'cache_{file_id}.json')
        if not os.path.exists(cache_path):
            print(f"Skipping {file_id}: cache file not found")
            continue
        
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        total_iterations = len(cache_data.get('iterations', []))
        
        if total_iterations == 0:
            print(f"Skipping {file_id}: no iterations in cache")
            continue
        
        # Divide iterations into phases
        early_end, middle_end = divide_iterations(total_iterations)
        
        # Collect hallucination scores by phase
        phase_scores = {
            'early': [],
            'middle': [],
            'end': []
        }
        
        all_iteration_scores = []
        
        for iter_key, iter_data in iteration_results.items():
            iteration_index = iter_data.get('iteration_index')
            hallucination_score = iter_data.get('hallucination_score')
            
            if iteration_index is None or hallucination_score is None:
                continue
            
            all_iteration_scores.append(hallucination_score)
            
            # Determine phase (iteration_index is 1-based)
            if iteration_index <= early_end:
                phase_scores['early'].append(hallucination_score)
            elif iteration_index <= middle_end:
                phase_scores['middle'].append(hallucination_score)
            else:
                phase_scores['end'].append(hallucination_score)
        
        # Calculate average scores for each phase
        early_avg = np.mean(phase_scores['early']) if phase_scores['early'] else 0
        middle_avg = np.mean(phase_scores['middle']) if phase_scores['middle'] else 0
        end_avg = np.mean(phase_scores['end']) if phase_scores['end'] else 0
        
        # Get file overall hallucination score
        file_overall_score = analysis_data.get('hallucination_score', 0)
        
        # Calculate mean of all iteration scores
        iteration_mean_score = np.mean(all_iteration_scores) if all_iteration_scores else 0
        
        # Normalize phase scores to [0, 1] as relative proportions
        total_phase_score = early_avg + middle_avg + end_avg
        if total_phase_score > 0:
            early_norm = early_avg / total_phase_score
            middle_norm = middle_avg / total_phase_score
            end_norm = end_avg / total_phase_score
        else:
            early_norm = middle_norm = end_norm = 0
        
        results[file_id] = {
            'early': early_avg,
            'middle': middle_avg,
            'end': end_avg,
            'early_norm': early_norm,
            'middle_norm': middle_norm,
            'end_norm': end_norm,
            'file_overall_score': file_overall_score,
            'iteration_mean_score': iteration_mean_score,
            'total_iterations': total_iterations
        }
        
        print(f"Processed {file_id}: {total_iterations} iterations, "
              f"Early={early_avg:.4f}({early_norm:.3f}), "
              f"Middle={middle_avg:.4f}({middle_norm:.3f}), "
              f"End={end_avg:.4f}({end_norm:.3f}), "
              f"File={file_overall_score:.4f}, Iter_Mean={iteration_mean_score:.4f}")
    
    return results

def process_all_files_for_detail():
    """Process all files and collect data for detailed plotting with HA highlighting"""
    analysis_files = sorted([f for f in os.listdir(analysis_dir) 
                           if f.startswith('analysis_') and f.endswith('.json')])
    
    all_data = {}
    
    for analysis_file in analysis_files:
        file_id = analysis_file.replace('analysis_', '').replace('.json', '')
        
        # Read analysis file
        analysis_path = os.path.join(analysis_dir, analysis_file)
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
        
        if 'iteration_level_nd_analysis' not in analysis_data:
            continue
        
        iteration_results = analysis_data['iteration_level_nd_analysis'].get('iteration_results', {})
        
        # Extract iteration scores
        iteration_hallucination_scores = {}
        for iter_key, iter_data in iteration_results.items():
            iteration_index = iter_data.get('iteration_index')
            hallucination_score = iter_data.get('hallucination_score')
            if iteration_index is not None and hallucination_score is not None:
                iteration_hallucination_scores[iteration_index] = hallucination_score
        
        if not iteration_hallucination_scores:
            continue
        
        # Read cache file
        cache_path = os.path.join(cache_dir, f'cache_{file_id}.json')
        if not os.path.exists(cache_path):
            continue
        
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)
        
        # Calculate iteration scores based on chunk scores
        iteration_chunk_scores = calculate_iteration_scores(cache_data)
        
        # Get NotSupport actions
        not_support_actions = get_not_support_actions(file_id)
        
        # Map to iterations
        iteration_ha_count = map_actions_to_iterations(file_id, not_support_actions)
        
        all_data[file_id] = {
            'iteration_hallucination_scores': iteration_hallucination_scores,
            'iteration_chunk_scores': iteration_chunk_scores,
            'iteration_ha_count': iteration_ha_count,
            'total_ha': len(not_support_actions)
        }
        
        print(f"Processed {file_id}: {len(iteration_hallucination_scores)} iterations, "
              f"{len(not_support_actions)} NotSupport actions")
    
    return all_data

def plot_results(results):
    """Create 4 subplots showing correlations between phase proportions and overall scores"""
    
    # Sort by file_id for consistent ordering
    file_ids = sorted(results.keys())
    
    # Extract normalized data for each phase (relative proportions in [0, 1])
    early_norm = np.array([results[fid]['early_norm'] for fid in file_ids])
    middle_norm = np.array([results[fid]['middle_norm'] for fid in file_ids])
    end_norm = np.array([results[fid]['end_norm'] for fid in file_ids])
    
    # Extract reference scores
    file_overall_scores = np.array([results[fid]['file_overall_score'] for fid in file_ids])
    iteration_mean_scores = np.array([results[fid]['iteration_mean_score'] for fid in file_ids])
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Use professional academic colors
    colors = {
        'early': '#2E86AB',    # Deep blue
        'middle': '#A23B72',   # Purple-red
        'end': '#F18F01',      # Orange
        'iter_mean': '#E65100' # Dark orange
    }
    
    # Data for each subplot: (x_data, y_data, x_label, title, color)
    subplot_data = [
        (early_norm, file_overall_scores, 'Early Phase (Normalized)', 
         'Early Phase vs. Overall Score', colors['early']),
        (middle_norm, file_overall_scores, 'Middle Phase (Normalized)', 
         'Middle Phase vs. Overall Score', colors['middle']),
        (end_norm, file_overall_scores, 'End Phase (Normalized)', 
         'End Phase vs. Overall Score', colors['end']),
        (iteration_mean_scores, file_overall_scores, 'Iteration Mean Score', 
         'Iteration Mean vs. Overall Score', colors['iter_mean'])
    ]
    
    # Create each subplot
    for idx, (x_data, y_data, x_label, title, color) in enumerate(subplot_data):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(x_data, y_data, alpha=0.6, s=80, color=color, edgecolors='black', linewidth=0.5)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(x_data, y_data)[0, 1]
        
        # Fit linear regression line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='gray', linewidth=2, alpha=0.8, label='Fit Line')
        
        # Add correlation coefficient text in upper right
        ax.text(0.95, 0.95, f'r = {correlation:.3f}',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Labels and title
        ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
        ax.set_ylabel('Overall Hallucination Score', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set reasonable limits
        ax.set_xlim(x_data.min() - 0.05, x_data.max() + 0.05)
        ax.set_ylim(y_data.min() - 0.02, y_data.max() + 0.02)
    
    # Overall title
    fig.suptitle('Correlation Analysis: Phase Proportions vs. Overall Hallucination Score',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    output_path = os.path.join(output_dir, 'iteration_phase_hallucination_scores.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.close()

def plot_detailed_analysis(all_data):
    """Create subplot grid showing hallucination scores and chunk scores with HA highlighting"""
    
    n_files = len(all_data)
    if n_files == 0:
        print("No data to plot")
        return
    
    # Calculate grid dimensions (try to make it roughly square)
    n_cols = math.ceil(math.sqrt(n_files))
    n_rows = math.ceil(n_files / n_cols)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    if n_files == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    file_ids = sorted(all_data.keys())
    
    # Define color map for HA intensity (white to deep red)
    colors_map = ['#FFFFFF', '#FFE5E5', '#FFCCCC', '#FF9999', '#FF6666', '#FF3333', '#FF0000', '#CC0000', '#990000']
    
    for idx, file_id in enumerate(file_ids):
        ax = axes[idx]
        data = all_data[file_id]
        
        iteration_hallucination_scores = data['iteration_hallucination_scores']
        iteration_chunk_scores = data['iteration_chunk_scores']
        iteration_ha_count = data['iteration_ha_count']
        
        # Sort by iteration index
        iterations = sorted(iteration_hallucination_scores.keys())
        hallucination_scores = [iteration_hallucination_scores[i] for i in iterations]
        chunk_scores = [iteration_chunk_scores.get(i, 0) for i in iterations]
        
        # Determine colors based on HA count
        max_ha = max(iteration_ha_count.values()) if iteration_ha_count else 0
        
        # Set up bar positions for grouped bars
        x = np.arange(len(iterations))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, hallucination_scores, width, 
                      label='Hallucination Score', alpha=0.8, color='#2E86AB', 
                      edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, chunk_scores, width, 
                      label='Chunk Score', alpha=0.8, color='#A23B72', 
                      edgecolor='black', linewidth=0.5)
        
        # Highlight bars with HA
        for i, iter_idx in enumerate(iterations):
            if iter_idx in iteration_ha_count:
                ha_count = iteration_ha_count[iter_idx]
                # Highlight both bars with red edge
                bars1[i].set_edgecolor('darkred')
                bars1[i].set_linewidth(2.5)
                bars2[i].set_edgecolor('darkred')
                bars2[i].set_linewidth(2.5)
                
                # Add text annotation for HA count above the taller bar
                max_height = max(bars1[i].get_height(), bars2[i].get_height())
                ax.text(x[i], max_height,
                       f'HA:{ha_count}',
                       ha='center', va='bottom', fontsize=7, color='darkred', 
                       fontweight='bold')
        
        # Set labels
        ax.set_xlabel('Iteration Index', fontsize=9)
        ax.set_ylabel('Score', fontsize=9)
        ax.set_title(f'{file_id[:8]}... (HA:{data["total_ha"]})', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        
        # Set x-axis to show integer ticks
        ax.set_xticks(x)
        ax.set_xticklabels(iterations)
        ax.tick_params(axis='both', labelsize=8)
        
        # Set y-axis limit
        all_scores = hallucination_scores + chunk_scores
        if all_scores:
            ax.set_ylim(0, max(all_scores) * 1.25)
        
        # Add legend for the first subplot
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_files, len(axes)):
        axes[idx].axis('off')
    
    # Add overall title
    fig.suptitle('Iteration-Level Analysis: Hallucination Score & Chunk Score with HA Highlighting',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    output_path = os.path.join(output_dir, 'iteration_detail_ha_analysis_merged.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDetailed plot saved to: {output_path}")
    
    plt.close()

def main():
    print("Starting iteration-level ND analysis...")
    print(f"Analysis directory: {analysis_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Combined directory: {combined_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Process files for phase analysis
    print("=" * 60)
    print("PHASE 1: Processing files for phase-level analysis")
    print("=" * 60)
    results = process_files()
    
    if not results:
        print("No valid results found for phase analysis.")
    else:
        print(f"\nProcessed {len(results)} files successfully for phase analysis.")
        
        # Create phase plot
        plot_results(results)
        
        # Save results to JSON for reference
        output_dir_json = '/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/nd_analysis_results/iteration_level/json'
        os.makedirs(output_dir_json, exist_ok=True)
        output_json = os.path.join(output_dir_json, 'iteration_phase_results.json')
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Phase results JSON saved to: {output_json}")
    
    # Process files for detailed iteration analysis
    print("\n" + "=" * 60)
    print("PHASE 2: Processing files for detailed iteration analysis")
    print("=" * 60)
    all_data = process_all_files_for_detail()
    
    if not all_data:
        print("No valid data found for detailed analysis.")
    else:
        print(f"\nProcessed {len(all_data)} files successfully for detailed analysis.")
        
        # Create detailed plot
        plot_detailed_analysis(all_data)
        
        # Save data to JSON
        output_dir_json = '/data/zyh/DeepResearch/HalluBench_backup_0828/evaluation_and_analysis/nd_analysis_results/iteration_level/json'
        os.makedirs(output_dir_json, exist_ok=True)
        output_json = os.path.join(output_dir_json, 'iteration_detail_ha_data.json')
        with open(output_json, 'w') as f:
            json.dump(all_data, f, indent=2)
        print(f"Detailed data JSON saved to: {output_json}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()

