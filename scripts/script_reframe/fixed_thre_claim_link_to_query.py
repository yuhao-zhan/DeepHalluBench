#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import os
import logging
import torch
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Any

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable progress bars and verbose output
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ClaimQueryReranker:
    """Reranker for computing scores between claims and queries."""
    
    def __init__(self, num_gpus: int = 4):
        print(f"Initializing Claim-Query Reranker...")
        self.num_gpus = num_gpus
        self.available_gpus = min(num_gpus, torch.cuda.device_count()) if torch.cuda.is_available() else 0
        
        if self.available_gpus == 0:
            print("No GPUs available, using CPU")
            self.available_gpus = 0
        
        print(f"Using {'GPU' if self.available_gpus > 0 else 'CPU'} for processing")
        
        # Initialize reranker
        from FlagEmbedding import FlagReranker
        if self.available_gpus > 0:
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        else:
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=False)
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags and extra whitespace."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove noise characters and patterns
        text = re.sub(r'=+\s*', ' ', text)  # Remove repeated equals signs
        text = re.sub(r'-+\s*', ' ', text)   # Remove repeated dashes
        text = re.sub(r'_+\s*', ' ', text)   # Remove repeated underscores
        text = re.sub(r'\*+\s*', ' ', text)  # Remove repeated asterisks
        text = re.sub(r'\++\s*', ' ', text)  # Remove repeated plus signs
        text = re.sub(r'~+\s*', ' ', text)   # Remove repeated tildes
        
        # Clean up whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _score_claim_query_pairs(self, claims: List[str], queries: List[str]) -> Dict[str, Dict[str, float]]:
        """Score all claim-query pairs using BGE Reranker."""
        logger.info(f"Scoring {len(claims)} claims against {len(queries)} queries...")
        
        # Clean all texts
        cleaned_claims = [self._clean_text(claim) for claim in claims]
        cleaned_queries = [self._clean_text(query) for query in queries]
        
        # Create all claim-query pairs
        claim_query_pairs = []
        pair_mapping = []  # To track which claim-query pair corresponds to which score
        
        for i, claim in enumerate(cleaned_claims):
            for j, query in enumerate(cleaned_queries):
                claim_query_pairs.append([claim, query])
                pair_mapping.append((i, j))
        
        logger.info(f"Created {len(claim_query_pairs)} claim-query pairs for scoring")
        
        # Score all pairs using BGE Reranker
        scores = self.reranker.compute_score(claim_query_pairs, normalize=True)
        
        # Map scores back to claim-query pairs with rounding and filtering
        # Round to 3 decimal places and only keep scores > 0.000
        claim_query_scores = {}
        for claim_idx in range(len(claims)):
            claim_query_scores[claim_idx] = {}
            for query_idx in range(len(queries)):
                claim_query_scores[claim_idx][query_idx] = 0.0
        
        for pair_idx, (claim_idx, query_idx) in enumerate(pair_mapping):
            raw_score = float(scores[pair_idx])
            # Round to 2 decimal places
            rounded_score = round(raw_score, 4)
            # Only keep scores greater than 0.00
            if rounded_score > 0.0000:
                claim_query_scores[claim_idx][query_idx] = rounded_score
            else:
                claim_query_scores[claim_idx][query_idx] = 0.0
        
        return claim_query_scores
    

    def find_relevant_queries_for_claims(self, claims: List[str], queries: List[str], 
                                       fixed_threshold: float = 0.001) -> Dict[str, Any]:
        """Find queries that are relevant to each claim based on fixed threshold."""
        logger.info("Starting claim-query relevance analysis with fixed threshold...")
        
        # Score all claim-query pairs
        claim_query_scores = self._score_claim_query_pairs(claims, queries)

        logger.info(f"Using fixed threshold of {fixed_threshold} for all claims")

        # Find relevant queries for each claim using claim-specific thresholds
        claim_relevant_queries = {}
        
        for claim_idx, claim in enumerate(claims):
            claim_threshold = fixed_threshold
            
            claim_relevant_queries[claim_idx] = {
                'claim': claim,
                'relevant_queries': [],
                'all_scores': {},
                'threshold_used': fixed_threshold,
                'threshold_method': 'fixed',
                'fixed_threshold': fixed_threshold
            }
            
            for query_idx, query in enumerate(queries):
                score = claim_query_scores[claim_idx][query_idx]
                claim_relevant_queries[claim_idx]['all_scores'][query_idx] = {
                    'query': query,
                    'score': score,
                    'is_relevant': score >= claim_threshold
                }
                
                if score >= claim_threshold:
                    claim_relevant_queries[claim_idx]['relevant_queries'].append({
                        'query_idx': query_idx,
                        'query': query,
                        'score': score
                    })
            
            # Sort relevant queries by score (highest first)
            claim_relevant_queries[claim_idx]['relevant_queries'].sort(
                key=lambda x: x['score'], reverse=True
            )
        
        
        # Return only the selected queries for each claim
        selected_queries = {}
        for claim_idx, claim_info in claim_relevant_queries.items():
            selected_queries[claim_idx] = {
                'claim': claim_info['claim'],
                'relevant_queries': claim_info['relevant_queries']
            }
        
        return selected_queries

# Standalone function for easy import
def find_relevant_queries_for_claims(claims: List[str], queries: List[str], fixed_threshold: float = 0.001) -> Dict[str, Any]:
    """
    Standalone function to find relevant queries for claims using ClaimQueryReranker.
    
    Args:
        claims: List of claims
        queries: List of queries
        percentiles: List of percentiles for threshold detection
        
    Returns:
        Dictionary containing relevant queries for each claim
    """
    # Create reranker instance only when function is called
    reranker = ClaimQueryReranker(num_gpus=4)
    return reranker.find_relevant_queries_for_claims(claims, queries, fixed_threshold)


# Usage example:
# 
# from claim_link_to_query import ClaimQueryReranker
# 
# # Initialize reranker
# reranker = ClaimQueryReranker(num_gpus=4)
# 
# # Define your claims and queries
# claims = ["Claim 1", "Claim 2", ...]
# queries = ["Query 1", "Query 2", ...]
# 
# # Find relevant queries for claims
# selected_queries = reranker.find_relevant_queries_for_claims(claims, queries)
# 
# # Access results - selected_queries is a dict with claim_idx as keys
# for claim_idx, claim_info in selected_queries.items():
#     print(f"Claim {claim_idx}: {claim_info['claim']}")
#     print("Relevant queries:")
#     for query_info in claim_info['relevant_queries']:
#         print(f"  - {query_info['query']} (Score: {query_info['score']:.4f})")