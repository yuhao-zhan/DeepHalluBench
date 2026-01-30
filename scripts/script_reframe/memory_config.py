#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory management configuration for HalluDetector evaluation pipeline.
Adjust these settings based on your available GPU memory and system resources.
"""

# GPU Memory Management
GPU_MEMORY_LIMITS = {
    'max_gpu_batch_size': 500,       # REDUCED: Maximum pairs per GPU batch for RTX 3090
    'max_query_batch_size': 5,       # REDUCED: Maximum queries processed simultaneously
    'max_documents': 100,            # Maximum documents to process
    'max_chunks': 5000,              # Maximum chunks to process
    'gpu_memory_threshold': 0.7,     # REDUCED: GPU memory usage threshold (70% for safety)
    'cleanup_frequency': 10,         # REDUCED: Cleanup every N paragraphs
    'max_gpu_memory_gb': 20,         # NEW: Maximum GPU memory to use (leave 4GB buffer)
    'min_gpu_memory_buffer_gb': 4,   # NEW: Minimum buffer to keep free
}

# System Memory Management
SYSTEM_MEMORY_LIMITS = {
    'max_system_memory_percent': 80,  # REDUCED: Maximum system memory usage
    'cleanup_after_iterations': True, # Cleanup after each iteration
    'cleanup_after_paragraphs': True, # Cleanup after N paragraphs
    'monitor_memory_continuously': True, # Continuous memory monitoring
}

# Processing Limits
PROCESSING_LIMITS = {
    'max_iterations_per_batch': 3,   # REDUCED: Maximum iterations processed in one batch
    'max_paragraphs_per_batch': 30,  # REDUCED: Maximum paragraphs processed in one batch
    'enable_streaming': True,        # Enable streaming processing
    'enable_chunking': True,         # Enable chunk-based processing
    'max_concurrent_gpu_workers': 2, # NEW: Maximum concurrent GPU workers
}

# Memory Cleanup Settings
CLEANUP_SETTINGS = {
    'force_gc_after_batch': True,    # Force garbage collection after each batch
    'clear_gpu_cache_after_batch': True,  # Clear GPU cache after each batch
    'clear_torch_cache': True,       # Clear PyTorch cache
    'clear_cuda_cache': True,        # Clear CUDA cache
    'cleanup_after_each_chunk': True, # NEW: Cleanup after each chunk
    'force_memory_cleanup_threshold': 0.8, # NEW: Force cleanup when memory > 80%
}

# Adaptive Settings
ADAPTIVE_SETTINGS = {
    'auto_adjust_batch_sizes': True, # Automatically adjust batch sizes based on memory
    'memory_based_throttling': True, # Throttle processing based on memory usage
    'dynamic_chunk_sizing': True,    # Dynamically adjust chunk sizes
    'load_balance_across_gpus': True, # NEW: Ensure even distribution across GPUs
    'prevent_gpu_0_overload': True,  # NEW: Special protection for GPU 0
}

def get_memory_config():
    """Get the complete memory configuration."""
    return {
        'gpu_limits': GPU_MEMORY_LIMITS,
        'system_limits': SYSTEM_MEMORY_LIMITS,
        'processing_limits': PROCESSING_LIMITS,
        'cleanup_settings': CLEANUP_SETTINGS,
        'adaptive_settings': ADAPTIVE_SETTINGS,
    }

def get_gpu_batch_size(available_gpu_memory_gb: float) -> int:
    """Dynamically calculate GPU batch size based on available memory."""
    base_size = GPU_MEMORY_LIMITS['max_gpu_batch_size']
    
    if available_gpu_memory_gb < 8:
        return min(base_size, 500)  # Very limited GPU memory
    elif available_gpu_memory_gb < 16:
        return min(base_size, 1000)  # Limited GPU memory
    elif available_gpu_memory_gb < 24:
        return min(base_size, 2000)  # Moderate GPU memory
    else:
        return base_size  # Plenty of GPU memory

def get_query_batch_size(available_system_memory_gb: float) -> int:
    """Dynamically calculate query batch size based on available system memory."""
    base_size = GPU_MEMORY_LIMITS['max_query_batch_size']
    
    if available_system_memory_gb < 16:
        return min(base_size, 5)  # Very limited system memory
    elif available_system_memory_gb < 32:
        return min(base_size, 10)  # Limited system memory
    elif available_system_memory_gb < 64:
        return min(base_size, 20)  # Moderate system memory
    else:
        return base_size  # Plenty of system memory

def should_cleanup_memory(current_gpu_usage: float, current_system_usage: float) -> bool:
    """Determine if memory cleanup is needed."""
    gpu_threshold = GPU_MEMORY_LIMITS['gpu_memory_threshold']
    system_threshold = SYSTEM_MEMORY_LIMITS['max_system_memory_percent'] / 100
    
    return (current_gpu_usage > gpu_threshold or 
            current_system_usage > system_threshold)

def get_optimal_gpu_batch_size(gpu_id: int, available_memory_gb: float) -> int:
    """Get optimal batch size for a specific GPU based on available memory."""
    base_size = GPU_MEMORY_LIMITS['max_gpu_batch_size']
    buffer_gb = GPU_MEMORY_LIMITS['min_gpu_memory_buffer_gb']
    
    # Ensure we leave enough buffer memory
    usable_memory = max(0, available_memory_gb - buffer_gb)
    
    if usable_memory < 2:
        return 100  # Very limited memory
    elif usable_memory < 4:
        return 200  # Limited memory
    elif usable_memory < 8:
        return 300  # Moderate memory
    elif usable_memory < 16:
        return 400  # Good memory
    else:
        return min(base_size, 500)  # Plenty of memory, but capped

def should_prevent_gpu_0_overload(gpu_0_memory_usage: float, other_gpus_avg_usage: float) -> bool:
    """Check if GPU 0 is overloaded compared to other GPUs."""
    if not ADAPTIVE_SETTINGS['prevent_gpu_0_overload']:
        return False
    
    # If GPU 0 usage is significantly higher than other GPUs, prevent overload
    return gpu_0_memory_usage > (other_gpus_avg_usage * 1.5)

def get_load_balanced_batch_distribution(num_tasks: int, gpu_memory_info: list) -> list:
    """Get load-balanced batch distribution across GPUs."""
    if not ADAPTIVE_SETTINGS['load_balance_across_gpus']:
        return None
    
    num_gpus = len(gpu_memory_info)
    if num_gpus == 0:
        return None
    
    # Sort GPUs by available memory (descending)
    sorted_gpus = sorted(gpu_memory_info, key=lambda x: x[2], reverse=True)
    
    # Calculate proportional distribution
    total_available = sum(available for _, _, available, _ in sorted_gpus)
    if total_available == 0:
        return None
    
    distribution = []
    for gpu_idx, (_, gpu_id, available, _) in enumerate(sorted_gpus):
        # Calculate tasks proportionally to available memory
        gpu_tasks = int((available / total_available) * num_tasks)
        distribution.append((gpu_id, gpu_tasks))
    
    return distribution

def get_memory_efficient_chunk_size(gpu_memory_gb: float) -> int:
    """Get memory-efficient chunk size for processing."""
    if gpu_memory_gb < 8:
        return 100   # Very small chunks for limited memory
    elif gpu_memory_gb < 16:
        return 200   # Small chunks for limited memory
    elif gpu_memory_gb < 24:
        return 300   # Medium chunks for moderate memory
    else:
        return 500   # Larger chunks for plenty of memory

if __name__ == "__main__":
    # Print current configuration
    config = get_memory_config()
    print("🔧 HalluDetector Memory Configuration:")
    for category, settings in config.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
