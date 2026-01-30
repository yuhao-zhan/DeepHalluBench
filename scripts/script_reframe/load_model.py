import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import threading


# Global NLI models for multi-GPU processing
_nli_models = {}
_nli_tokenizers = {}
_nli_devices = {}
_nli_locks = {}
_nli_initialized = False

def initialize_nli_models_once(num_gpus: int = 4):
    """Initialize NLI models on multiple GPUs only once at the beginning."""
    global _nli_models, _nli_tokenizers, _nli_devices, _nli_locks, _nli_initialized
    
    if _nli_initialized:
        print("✅ NLI models already initialized, skipping...")
        return
    
    print(f"🔧 Initializing NLI models on {num_gpus} GPUs (ONLY ONCE)...")
    
    available_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
    model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    
    if available_gpus > 0:
        # Load models on multiple GPUs
        for gpu_id in range(available_gpus):
            try:
                torch.cuda.set_device(gpu_id)
                
                # Check GPU memory
                allocated = torch.cuda.memory_allocated(gpu_id)
                total = torch.cuda.get_device_properties(gpu_id).total_memory
                available = total - allocated
                
                if available > 4 * 1024**3:  # 4GB minimum
                    print(f"🎮 Loading NLI model on GPU {gpu_id} with {available/1024**3:.2f}GB available")
                    
                    # Load tokenizer and model
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    
                    # Optimize model for inference
                    model.eval()  # Set to evaluation mode
                    model = model.half()  # Use half precision for speed and memory efficiency
                    
                    # Move model to GPU
                    model = model.cuda(gpu_id)
                    device = torch.device(f"cuda:{gpu_id}")
                    
                    # Store in global dictionaries
                    _nli_models[gpu_id] = model
                    _nli_tokenizers[gpu_id] = tokenizer
                    _nli_devices[gpu_id] = device
                    _nli_locks[gpu_id] = threading.Lock()
                    
                    print(f"✅ NLI model loaded on GPU {gpu_id} with half precision")
                else:
                    print(f"⚠️ Insufficient memory on GPU {gpu_id} ({available/1024**3:.2f}GB), skipping")
                    
            except Exception as e:
                print(f"❌ Failed to load NLI model on GPU {gpu_id}: {e}")
                continue
    
    # Fallback to CPU if no GPUs available
    if not _nli_models:
        print("🖥️ No GPUs available, loading NLI model on CPU")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()  # Set to evaluation mode
            model = model.cpu()
            device = torch.device("cpu")
            
            _nli_models[-1] = model  # Use -1 to indicate CPU
            _nli_tokenizers[-1] = tokenizer
            _nli_devices[-1] = device
            _nli_locks[-1] = threading.Lock()
            
            print("✅ NLI model loaded on CPU")
        except Exception as e:
            print(f"❌ Failed to load NLI model on CPU: {e}")
            return
    
    _nli_initialized = True
    print(f"✅ NLI models initialized on {len(_nli_models)} devices")


def nli_score_batch_parallel(claim_chunk_pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
    """
    Score multiple claim-chunk pairs in parallel across multiple GPUs.
    This is the main NLI scoring function that should be used for batch processing.
    
    Args:
        claim_chunk_pairs: List of (claim, chunk) tuples to score
        
    Returns:
        List of dictionaries with entailment, neutral, and contradiction scores
    """
    if not _nli_initialized:
        initialize_nli_models_once()
    
    if not _nli_models:
        print("❌ No NLI models available")
        return [{'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0} for _ in claim_chunk_pairs]
    
    print(f"🧠 Processing {len(claim_chunk_pairs)} claim-chunk pairs across {len(_nli_models)} GPUs in batch")
    
    # Distribute pairs evenly across available GPUs
    num_gpus = len(_nli_models)
    distributed_pairs = _distribute_data_evenly(claim_chunk_pairs, num_gpus)
    
    all_results = [None] * len(claim_chunk_pairs)
    threads = []
    
    for gpu_id, gpu_pairs in enumerate(distributed_pairs):
        if gpu_id in _nli_models and gpu_pairs:
            start_idx = sum(len(batch) for batch in distributed_pairs[:gpu_id])
            
            print(f"GPU {gpu_id}: Processing {len(gpu_pairs)} pairs (indices {start_idx}-{start_idx + len(gpu_pairs) - 1})")
            
            thread = threading.Thread(
                target=_process_nli_on_gpu,
                args=(gpu_id, gpu_pairs, start_idx, all_results)
            )
            threads.append(thread)
            thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check for any None results (failed processing)
    if any(result is None for result in all_results):
        print("❌ Some NLI scoring failed")
        # Fill None results with neutral scores
        for i, result in enumerate(all_results):
            if result is None:
                all_results[i] = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}
    
    print(f"✅ Batch NLI scoring completed for {len(claim_chunk_pairs)} pairs")
    
    # Memory cleanup after NLI processing
    print("🧹 Cleaning up GPU memory after NLI batch processing")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("✅ Memory cleanup completed after NLI processing")
    except Exception as e:
        print(f"⚠️ Memory cleanup after NLI processing failed: {e}")
    
    return all_results


def _distribute_data_evenly(data: List, num_gpus: int) -> List[List]:
    """Distribute data perfectly evenly across GPUs."""
    if num_gpus <= 1:
        return [data]
    
    # Calculate exact distribution
    total_items = len(data)
    base_size = total_items // num_gpus
    remainder = total_items % num_gpus
    
    distributed = []
    start_idx = 0
    
    for gpu_id in range(num_gpus):
        # First 'remainder' GPUs get one extra item
        current_size = base_size + (1 if gpu_id < remainder else 0)
        end_idx = start_idx + current_size
        
        distributed.append(data[start_idx:end_idx])
        start_idx = end_idx
    
    return distributed


def _process_nli_on_gpu(gpu_id: int, pairs: List[Tuple[str, str]], start_idx: int, results: List):
    """Process NLI scoring on a specific GPU using optimized batching."""
    try:
        with _nli_locks[gpu_id]:
            # Set device context
            if gpu_id >= 0:
                torch.cuda.set_device(gpu_id)
            
            model = _nli_models[gpu_id]
            tokenizer = _nli_tokenizers[gpu_id]
            device = _nli_devices[gpu_id]
            
            label_names = ["entailment", "neutral", "contradiction"]
            
            # Optimized batch processing
            batch_size = _get_optimal_batch_size(gpu_id, len(pairs))
            print(f"GPU {gpu_id}: Using batch size {batch_size} for {len(pairs)} pairs")
            
            # Process in batches
            for batch_start in range(0, len(pairs), batch_size):
                batch_end = min(batch_start + batch_size, len(pairs))
                batch_pairs = pairs[batch_start:batch_end]
                
                try:
                    # Prepare batch inputs
                    batch_claims = [pair[0] for pair in batch_pairs]
                    batch_chunks = [pair[1] for pair in batch_pairs]
                    
                    # Tokenize batch with padding
                    batch_inputs = tokenizer(
                        batch_chunks, 
                        batch_claims, 
                        truncation=True, 
                        padding=True, 
                        return_tensors="pt",
                        max_length=512  # Optimize for speed
                    )
                    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                    
                    # Run batch inference
                    with torch.no_grad():
                        # Convert inputs to half precision if model is half precision
                        if model.dtype == torch.float16:
                            batch_inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in batch_inputs.items()}
                        
                        output = model(**batch_inputs)
                        predictions = torch.softmax(output["logits"], -1)
                    
                    # Process batch results
                    for i, prediction in enumerate(predictions):
                        scores = {name: float(pred) for pred, name in zip(prediction.tolist(), label_names)}
                        results[start_idx + batch_start + i] = scores
                    
                    # Clear batch from memory
                    del batch_inputs, output, predictions
                    
                    # Force memory cleanup every few batches
                    if batch_start % (batch_size * 4) == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"❌ Error processing batch {batch_start}-{batch_end} on GPU {gpu_id}: {e}")
                    # Fill failed batch with neutral scores
                    for i in range(batch_start, batch_end):
                        results[start_idx + i] = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}
            
            print(f"GPU {gpu_id}: Completed {len(pairs)} NLI scoring tasks with optimized batching")
            
    except Exception as e:
        print(f"❌ Error on GPU {gpu_id}: {e}")
        # Mark failed results as neutral
        for i in range(len(pairs)):
            results[start_idx + i] = {'entailment': 0.0, 'neutral': 1.0, 'contradiction': 0.0}


def _get_optimal_batch_size(gpu_id: int, total_pairs: int) -> int:
    """Determine optimal batch size based on GPU memory and total pairs."""
    if gpu_id < 0:  # CPU
        return min(64, total_pairs)
    
    try:
        # Get GPU memory info
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        free_memory = total_memory - allocated_memory
        
        # More aggressive memory estimation for half precision DeBERTa-large
        # With half precision and optimized batching, we can use more memory
        memory_per_sample = 1024 * 1024 * 25  # 25MB per sample (half precision + optimized)
        
        # Calculate optimal batch size - be more aggressive
        max_batch_size = max(1, free_memory // memory_per_sample)
        
        # More aggressive limits for better GPU utilization
        max_batch_size = min(max_batch_size, 256)  # Increased from 128 to 256
        max_batch_size = min(max_batch_size, total_pairs)  # Don't exceed total pairs
        
        # Ensure minimum batch size for efficiency - increased minimum
        optimal_batch_size = max(32, max_batch_size)  # Increased from 16 to 32
        
        print(f"GPU {gpu_id}: Free memory: {free_memory/1024**3:.2f}GB, Optimal batch size: {optimal_batch_size}")
        
        return optimal_batch_size
        
    except Exception as e:
        print(f"⚠️ Could not determine optimal batch size for GPU {gpu_id}: {e}")
        return min(64, total_pairs)  # Increased fallback
