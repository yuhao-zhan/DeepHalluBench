import json
import re
import os
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Tuple, Any, Optional
import argparse
from pathlib import Path
import logging
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from utils import split_report_into_paragraphs
# Import from local utils module - use absolute path to avoid multiprocessing conflicts
import os
import sys
import importlib.util
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import with explicit module path to avoid conflicts
spec = importlib.util.spec_from_file_location("local_utils", os.path.join(current_dir, "utils.py"))
local_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_utils)

# Now import the function we need
is_url = local_utils.is_url

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class URLClaimMapper:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def extract_urls_and_split_paragraph(self, paragraph_text: str) -> Tuple[List[str], List[Optional[Any]]]:
        """
        Extract URLs from paragraph and split into segments.
        Each segment ends with a URL (except the last segment).
        Returns: (segments, urls_per_segment)
        urls_per_segment can be a single URL string, a list of URLs (for grouped URLs), or None
        """
        # Pattern to match markdown links: [text](url)
        url_pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)'

        # Pattern to match URLs: @https://timesofindia.indiatimes.com/...
        # Updated to match URLs that may be followed by another @url (with optional whitespace/comma)
        # URL can contain various characters including % for encoding, /, ?, =, etc.
        # Match until whitespace or @ (start of next URL) - this is safer than excluding punctuation
        # since URLs can contain periods (e.g., .com, .org) and other punctuation
        url_pattern_2 = r'@(https?://[^\s@]+)'
        
        # Find all URLs with their positions
        url_matches = list(re.finditer(url_pattern, paragraph_text))
        url_matches_2 = list(re.finditer(url_pattern_2, paragraph_text))
        
        if not url_matches and not url_matches_2:
            # No URLs found, return the whole paragraph as one segment
            return [paragraph_text.strip()], [None]
        
        segments = []
        urls = []
        last_end = 0
        
        # Combine and sort all matches by position
        all_matches = []
        for match in url_matches:
            all_matches.append((match.start(), match.end(), match, 'pattern1'))
        for match in url_matches_2:
            all_matches.append((match.start(), match.end(), match, 'pattern2'))
        
        # Sort by start position
        all_matches.sort(key=lambda x: x[0])
        
        # Group consecutive URLs that are adjacent (with only whitespace/comma/punctuation between them)
        i = 0
        while i < len(all_matches):
            start, end, match, pattern_type = all_matches[i]
            
            # Collect consecutive URLs that form a group
            url_group = []
            group_end = end
            j = i
            
            # Check if this URL is followed by another URL (within reasonable distance)
            while j < len(all_matches):
                current_start, current_end, current_match, current_pattern = all_matches[j]
                
                # Extract URL from current match
                if current_pattern == 'pattern1':
                    current_url = current_match.group(2)
                else:
                    current_url = current_match.group(1)
                url_group.append(current_url)
                
                # Check if there's another URL nearby (within 200 chars, with only whitespace/punctuation)
                if j + 1 < len(all_matches):
                    next_start, next_end, next_match, next_pattern = all_matches[j + 1]
                    # Check the text between current URL end and next URL start
                    between_text = paragraph_text[current_end:next_start]
                    # If it's mostly whitespace, comma, dash, or similar separators, consider them grouped
                    if len(between_text.strip()) < 50 and re.match(r'^[\s,\-–—]*$', between_text):
                        group_end = next_end
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Add text from last position up to and including this URL group as a segment
            segment_text = paragraph_text[last_end:group_end].strip()
            if segment_text:
                segments.append(segment_text)
                # If multiple URLs in group, store as list; otherwise store as single string
                if len(url_group) > 1:
                    urls.append(url_group)
                else:
                    urls.append(url_group[0] if url_group else None)
            
            last_end = group_end
            i = j + 1
        
        # Add remaining text after the last URL
        if last_end < len(paragraph_text):
            remaining_text = paragraph_text[last_end:].strip()
            if remaining_text:
                segments.append(remaining_text)
                urls.append(None)  # No URL for the final segment
        
        return segments, urls
    
    def preprocess_text_for_bm25(self, text: str) -> List[str]:
        """
        Preprocess text for BM25 matching.
        """
        # Remove URLs and markdown formatting
        text = re.sub(r'\[([^\]]+)\]\((https?://[^\)]+)\)', r'\1', text)
        
        # Remove other markdown formatting
        text = re.sub(r'[#*_`]', '', text)
        
        # Tokenize and clean
        tokens = word_tokenize(text.lower())
        
        # Remove punctuation and stopwords
        tokens = [token for token in tokens 
                 if token not in self.stop_words and token not in string.punctuation and len(token) > 1]
        
        return tokens
    
    def match_claims_to_segments(self, claims: List[str], segments: List[str]) -> Dict[int, List[int]]:
        """
        Use BM25 to match claims to paragraph segments.
        Returns: {segment_index: [claim_indices]}
        """
        if not segments or not claims:
            return {}
        
        # Preprocess segments for BM25
        processed_segments = [self.preprocess_text_for_bm25(segment) for segment in segments]
        
        # Handle empty segments
        processed_segments = [seg if seg else ['empty'] for seg in processed_segments]
        
        # Create BM25 index
        bm25 = BM25Okapi(processed_segments)
        
        # Match each claim to best segment
        segment_to_claims = {}
        
        for claim_idx, claim in enumerate(claims):
            processed_claim = self.preprocess_text_for_bm25(claim)
            
            if not processed_claim:
                continue
                
            # Get BM25 scores for all segments
            scores = bm25.get_scores(processed_claim)
            
            # Find best matching segment
            best_segment_idx = int(scores.argmax())
            
            if best_segment_idx not in segment_to_claims:
                segment_to_claims[best_segment_idx] = []
            
            segment_to_claims[best_segment_idx].append(claim_idx)
        
        return segment_to_claims
    
    def add_urls_to_claims(self, atomic_claims: List[str], segments: List[str], 
                          urls: List[Optional[Any]], segment_to_claims: Dict[int, List[int]]) -> List[str]:
        """
        Add URLs to the beginning of claim groups that belong to the same segment.
        If multiple consecutive claims belong to the same URL, the URL is only inserted once at the beginning.
        If a segment has multiple grouped URLs (list), all URLs are added.
        """
        updated_claims = []
        processed_claims = set()
        
        # Process segments in order
        sorted_segments = sorted(segment_to_claims.keys())
        current_url_group = None
        
        for segment_idx in sorted_segments:
            claim_indices = segment_to_claims[segment_idx]
            if not claim_indices:
                continue
                
            # Get URL for this segment (can be a string, list of strings, or None)
            segment_url = urls[segment_idx] if segment_idx < len(urls) else None
            
            # Normalize URL representation for comparison
            def normalize_url(url_val):
                """Normalize URL to a comparable format."""
                if url_val is None:
                    return None
                if isinstance(url_val, list):
                    return tuple(sorted(url_val))  # Use tuple for comparison
                return url_val
            
            # Add URL(s) only if different from the previous one
            normalized_segment_url = normalize_url(segment_url)
            if segment_url and normalized_segment_url != current_url_group:
                # If it's a list of URLs, add all of them
                if isinstance(segment_url, list):
                    for url in segment_url:
                        updated_claims.append(url)
                else:
                    updated_claims.append(segment_url)
                current_url_group = normalized_segment_url
            
            # Add all claims for this segment
            sorted_claim_indices = sorted(claim_indices)
            for claim_idx in sorted_claim_indices:
                if claim_idx < len(atomic_claims) and claim_idx not in processed_claims:
                    updated_claims.append(atomic_claims[claim_idx])
                    processed_claims.add(claim_idx)
        
        # Add any remaining claims that weren't matched to any segment
        for i, claim in enumerate(atomic_claims):
            if i not in processed_claims:
                updated_claims.append(claim)
        
        return updated_claims
    
    def load_original_report(self, original_file_path: str) -> Optional[str]:
        """
        Load the original report from the json_with_citation file.
        """
        try:
            with open(original_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('final_report', '')
        except Exception as e:
            logger.error(f"Error loading original report from {original_file_path}: {str(e)}")
            return None
    

    
    def get_paragraph_by_index(self, report_text: str, paragraph_index: int) -> str:
        """
        Get paragraph by index from the report text (split by \n\n or Markdown headers).
        """
        paragraphs = split_report_into_paragraphs(report_text)
        # Debug: print paragraph count and indices
        # print(f"Total paragraphs in report: {len(paragraphs)}")
        # print(f"Looking for paragraph index: {paragraph_index}")
        
        if 1 <= paragraph_index <= len(paragraphs):
            target_paragraph = paragraphs[paragraph_index - 1].strip()
            # print(f"Found paragraph {paragraph_index}: length={len(target_paragraph)}")
            return target_paragraph
        else:
            # print(f"Paragraph index {paragraph_index} out of range [1, {len(paragraphs)}]")
            return ""
    
    def process_paragraph(self, paragraph_data: Dict[str, Any], original_report: str) -> Dict[str, Any]:
        """
        Process a single paragraph to fix URL placement.
        """
        try:
            paragraph_index = paragraph_data.get('paragraph_index', 0)
            atomic_claims = paragraph_data.get('atomic_claims', [])
            paragraph_text = paragraph_data.get('paragraph_text', '')
            print(f'Processing paragraph_index: {paragraph_index}')
            
            if not atomic_claims or paragraph_index == 0:
                return paragraph_data
            
            # First try to find by content (more reliable than index)
            original_paragraph = None
            found_index = 0
            
            if paragraph_text:
                # print(f"Trying to find paragraph by content for index {paragraph_index}")
                # Search for paragraphs that contain some of the same words
                paragraphs = split_report_into_paragraphs(original_report)
                best_match = None
                best_score = 0
                
                for i, para in enumerate(paragraphs):
                    para_clean = re.sub(r'\s+', ' ', para.strip())
                    text_clean = re.sub(r'\s+', ' ', paragraph_text.strip())
                    
                    # Skip very short paragraphs (likely headers)
                    if len(para_clean) < 50:
                        continue
                    
                    # Calculate word overlap similarity
                    para_words = set(para_clean.lower().split())
                    text_words = set(text_clean.lower().split())
                    
                    if len(text_words) > 0:
                        # Word overlap ratio
                        overlap = len(para_words.intersection(text_words)) / len(text_words)
                        
                        if overlap > best_score and overlap > 0.1:  # Reasonable threshold
                            best_score = overlap
                            best_match = (i + 1, para.strip())
                
                if best_match:
                    found_index, original_paragraph = best_match
                    # print(f"Found paragraph by content similarity at index {found_index} (score: {best_score:.3f})")
            
            # If content-based search failed, fall back to index-based search
            if not original_paragraph:
                # print(f"Content-based search failed, trying index-based search")
                original_paragraph = self.get_paragraph_by_index(original_report, paragraph_index)
                if original_paragraph:
                    print(f"Found paragraph by index at index {paragraph_index}")
            
            if not original_paragraph:
                print(f"❌ WARNING: Could not find paragraph {paragraph_index} in original report!")
                print(f"   Cache paragraph text: {paragraph_text[:100]}...")
                
                # Debug: show how many paragraphs are in original report
                paragraphs = split_report_into_paragraphs(original_report)
                print(f"   Original report has {len(paragraphs)} paragraphs")
                print(f"   Cache file has paragraph_index: {paragraph_index}")
                
                # Show first few paragraphs of original report
                # print("   First 3 paragraphs of original report:")
                # for i, para in enumerate(paragraphs[:3], 1):
                #     print(f"     {i}: {para[:100]}...")
                
                return paragraph_data
            
            print(f"✅ Found original paragraph {found_index if found_index > 0 else paragraph_index}")
            
            # Extract URLs and split into segments
            segments, urls = self.extract_urls_and_split_paragraph(original_paragraph)
            # Count URLs (including those in URL groups)
            url_count = 0
            for u in urls:
                if u:
                    if isinstance(u, list):
                        url_count += len(u)
                    else:
                        url_count += 1
            print(f"  Split into {len(segments)} segments with {url_count} URLs")
            
            if not urls or all(url is None for url in urls):
                print(f"  No URLs found in paragraph {found_index if found_index > 0 else paragraph_index}")
                return paragraph_data
            
            # Remove existing URLs from claims (they shouldn't be there, but just in case)
            clean_claims = []
            for claim in atomic_claims:
                if not is_url(claim):
                    clean_claims.append(claim)
                else:
                    print(f"  Removed existing URL claim: {claim[:50]}...")
            
            if not clean_claims:
                print(f"  No non-URL claims found after cleaning")
                return paragraph_data
            
            print(f"  Cleaned claims: {len(atomic_claims)} -> {len(clean_claims)}")
            
            # Match claims to segments using BM25
            segment_to_claims = self.match_claims_to_segments(clean_claims, segments)
            print(f"  Mapped claims to segments: {segment_to_claims}")
            
            # Add URLs to claims
            updated_claims = self.add_urls_to_claims(clean_claims, segments, urls, segment_to_claims)
            
            # Update the paragraph data
            updated_paragraph = paragraph_data.copy()
            updated_paragraph['atomic_claims'] = updated_claims
            updated_paragraph['urls_added'] = True
            # Count URLs (including those in URL groups)
            num_urls = 0
            for u in urls:
                if u:
                    if isinstance(u, list):
                        num_urls += len(u)
                    else:
                        num_urls += 1
            updated_paragraph['num_urls_added'] = num_urls
            updated_paragraph['matched_original_index'] = found_index if found_index > 0 else paragraph_index
            
            print(f"  ✅ Added URLs to {len(updated_claims)} claims")
            return updated_paragraph
            
        except Exception as e:
            print(f"❌ Error processing paragraph {paragraph_index}: {str(e)}")
            return paragraph_data

def process_single_file(args_tuple: Tuple[str, str]) -> bool:
    """
    Process a single JSON cache file.
    """
    cache_file_path, original_dir = args_tuple
    
    try:
        logger.info(f"Processing file: {cache_file_path}")
        
        # Load cache file
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Determine original file path
        cache_filename = os.path.basename(cache_file_path)
        original_filename = cache_filename.replace('cache_', '')
        original_file_path = os.path.join(original_dir, original_filename)
        
        if not os.path.exists(original_file_path):
            logger.error(f"Original file not found: {original_file_path}")
            return False
        
        mapper = URLClaimMapper()
        
        # Load original report
        original_report = mapper.load_original_report(original_file_path)
        if not original_report:
            logger.error(f"Could not load original report from {original_file_path}")
            return False
        
        # Process only the 'report' section
        if 'report' in cache_data and isinstance(cache_data['report'], list):
            for i, paragraph_data in enumerate(cache_data['report']):
                # print(f'processing paragraph {i}')
                # print(f'paragraph_data type: {type(paragraph_data)}')
                # print(f'paragraph_data keys: {paragraph_data.keys() if isinstance(paragraph_data, dict) else "N/A"}')
                
                if isinstance(paragraph_data, dict) and 'atomic_claims' in paragraph_data:
                    cache_data['report'][i] = mapper.process_paragraph(paragraph_data, original_report)
        
        # Write back to file
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully processed: {cache_file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing file {cache_file_path}: {str(e)}")
        return False

def process_cache_directory(cache_dir: str, original_dir: str, num_processes: int = None) -> None:
    """
    Process all JSON files in the cache directory in parallel.
    """
    cache_path = Path(cache_dir)
    original_path = Path(original_dir)
    
    if not cache_path.exists():
        logger.error(f"Cache directory does not exist: {cache_dir}")
        return
        
    if not original_path.exists():
        logger.error(f"Original directory does not exist: {original_dir}")
        return
    
    # Find all JSON files
    json_files = list(cache_path.glob("cache_*.json"))
    
    if not json_files:
        logger.warning(f"No cache JSON files found in: {cache_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), len(json_files))
    
    logger.info(f"Using {num_processes} processes")
    
    # Prepare arguments for parallel processing
    args_list = [(str(f), original_dir) for f in json_files]
    
    # Process files in parallel
    with Pool(num_processes) as pool:
        results = pool.map(process_single_file, args_list)
    
    # Report results
    successful = sum(results)
    failed = len(results) - successful
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")

def main():
    parser = argparse.ArgumentParser(description='Fix URL placement in cached JSON files')
    parser.add_argument('--cache_dir', type=str, 
                       default='/data2/yuhaoz/DeepResearch/HalluBench/HalluDetector/json_cache/train_gemini',
                       help='Path to cache directory')
    parser.add_argument('--original_dir', type=str,
                       default='/data2/yuhaoz/DeepResearch/HalluBench/data/train/close-source/gemini/temp',
                       help='Path to original files directory')
    parser.add_argument('--processes', type=int, default=None,
                       help='Number of parallel processes (default: auto)')
    
    args = parser.parse_args()
    
    process_cache_directory(args.cache_dir, args.original_dir, args.processes)

if __name__ == "__main__":
    main()
