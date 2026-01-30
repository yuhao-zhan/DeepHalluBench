import re
import os
import logging
import aiohttp
import asyncio
from typing import List, Dict, Set, Any, Tuple
import logging
import numpy as np
import math
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoModelWithLMHead, AutoTokenizer
import spacy
from urllib.parse import urljoin
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

# Stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 
    'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 
    'the', 'to', 'was', 'were', 'will', 'with'
}

# Add the missing fetch_pages_async function
async def fetch_pages_async(urls: List[str]) -> Dict[str, str]:
    """Async function to fetch web pages content."""
    # This is a placeholder - we don't actually need this for similarity filtering
    # since we're reading from cached web content files
    return {url: "" for url in urls}

# OptimizedContextLocator is used to extract sentences and chunks from a document
class OptimizedContextLocator:
    """Fast context locator with 10-sentence chunks with 2-sentence overlap."""
    
    def __init__(self):
        self.url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        self.markdown_pattern = re.compile(r'[#*\[\](){}|`_~]+')
    
    def _clean_text(self, text: str) -> str:
        """Simplified text cleaning with URL preservation."""
        # First, protect URLs by temporarily replacing them with placeholders
        urls = self.url_pattern.findall(text)
        url_placeholders = {}
        
        for i, url in enumerate(urls):
            # Use a unique placeholder that won't be affected by any cleaning operations
            placeholder = f"URLSAFE{i}URLSAFE"
            url_placeholders[placeholder] = url
            text = text.replace(url, placeholder)
        
        # Remove markdown characters but preserve newlines for sentence boundaries
        # First, replace newlines with a special marker
        text = text.replace('\n', ' @NEWLINE@ ')
        
        # Remove other markdown characters
        text = self.markdown_pattern.sub(' ', text)
        
        # Remove noise characters and patterns
        # Remove repeated equals signs (markdown headers)
        text = re.sub(r'=+\s*', ' ', text)
        # Remove repeated dashes (markdown separators)
        text = re.sub(r'-+\s*', ' ', text)
        # Remove repeated underscores (markdown separators)
        text = re.sub(r'_+\s*', ' ', text)
        # Remove repeated asterisks (markdown separators)
        text = re.sub(r'\*+\s*', ' ', text)
        # Remove repeated plus signs
        text = re.sub(r'\++\s*', ' ', text)
        # Remove repeated tildes
        text = re.sub(r'~+\s*', ' ', text)
        
        # Restore newlines and clean up whitespace
        text = text.replace(' @NEWLINE@ ', '\n')
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n\s*\n', '\n', text)  # Replace multiple newlines with single newline
        
        # Restore the original URLs
        for placeholder, original_url in url_placeholders.items():
            text = text.replace(placeholder, original_url)
        
        return text.strip()
    
    def _extract_all_sentences(self, document: str) -> List[Dict[str, Any]]:
        """Extract all complete sentences from document."""
        document = self._clean_text(document)
        sentences = []
        
        # First, protect URLs in the document to prevent splitting at dots within URLs
        urls = self.url_pattern.findall(document)
        url_placeholders = {}
        
        for i, url in enumerate(urls):
            placeholder = f"URLSAFE{i}URLSAFE"
            url_placeholders[placeholder] = url
            document = document.replace(url, placeholder)
        
        # Split by sentence endings (but not dots within URL placeholders and number like 4.4)
        # First, protect numbers with decimals to prevent splitting
        decimal_pattern = re.compile(r'\d+\.\d+')
        decimal_placeholders = {}
        
        for i, decimal in enumerate(decimal_pattern.findall(document)):
            placeholder = f"DECIMAL{i}DECIMAL"
            decimal_placeholders[placeholder] = decimal
            document = document.replace(decimal, placeholder)
        
        sentence_pattern = re.compile(r'[^.!?\n]+[.!?\n]+')
        matches = sentence_pattern.finditer(document)
        
        for i, match in enumerate(matches):
            sentence_text = match.group().strip()
            
            # Restore URLs in this sentence
            for placeholder, original_url in url_placeholders.items():
                sentence_text = sentence_text.replace(placeholder, original_url)
            
            # Restore decimal numbers
            for placeholder, original_decimal in decimal_placeholders.items():
                sentence_text = sentence_text.replace(placeholder, original_decimal)
            
            # Filter out very short sentences and sentences that are just whitespace
            if len(sentence_text) > 1 and not sentence_text.isspace():
                # Limit sentence length to prevent overly long sentences
                if len(sentence_text) > 500:
                    # Find a good breaking point
                    words = sentence_text.split()
                    if len(words) > 50:
                        # Try to break at sentence boundaries or word boundaries
                        truncated = ' '.join(words[:50])
                        # Find the last complete sentence or word
                        last_period = truncated.rfind('.')
                        last_exclamation = truncated.rfind('!')
                        last_question = truncated.rfind('?')
                        break_point = max(last_period, last_exclamation, last_question)
                        if break_point > 0:
                            sentence_text = truncated[:break_point + 1]
                        else:
                            sentence_text = truncated
                
                sentences.append({
                    'sentence_id': f"sentence_{i}_{hash(sentence_text[:100])}",
                    'sentence_text': sentence_text,
                    'position': match.start(),
                    'length': len(sentence_text),
                    'sentence_index': i
                })
        
        return sentences
    
    def _fix_url_parentheses(self, text: str) -> str:
        """
        Fix URL parentheses issues by adding opening parenthesis before URLs that have closing parenthesis.
        This handles cases like "San Francisco )" -> "San Francisco ("
        """
        # Pattern to find URLs followed by closing parenthesis
        url_close_pattern = re.compile(r'(https?://[^\s<>"{}|\\^`\[\]]+)\)')
        
        def replace_url_parentheses(match):
            url = match.group(1)
            return f"({url})"
        
        # Replace URLs followed by ) with (URL)
        text = url_close_pattern.sub(replace_url_parentheses, text)
        
        return text
    
    def _create_sentence_chunks_with_overlap(self, sentences: List[Dict[str, Any]], chunk_size: int = 10, overlap_size: int = 2) -> List[Dict[str, Any]]:
        """Create chunks of 10 sentences with 2 sentence overlap."""
        if len(sentences) == 0:
            return []
        
        # Sequential chunk index to ensure stable, hash-free chunk IDs
        chunk_index = 0

        if len(sentences) <= chunk_size:
            # If we have fewer sentences than chunk size, create single chunk
            chunk_text = ' '.join([sent['sentence_text'] for sent in sentences])
            
            # Fix URL parentheses issues
            chunk_text = self._fix_url_parentheses(chunk_text)
            
            return [{
                'chunk_id': f"chunk_{chunk_index}",
                'chunk_text': chunk_text,
                'position': sentences[0]['position'] if sentences else 0,
                'length': len(chunk_text),
                'sentence_count': len(sentences),
                'sentence_indices': [sent['sentence_index'] for sent in sentences]
            }]
        
        chunks = []
        step_size = chunk_size - overlap_size  # How many new sentences to add per chunk
        
        for i in range(0, len(sentences) - chunk_size + 1, step_size):
            # Get sentences for this chunk
            chunk_sentences = sentences[i:i + chunk_size]
            chunk_text = ' '.join([sent['sentence_text'] for sent in chunk_sentences])
            
            # Calculate position and length
            start_position = chunk_sentences[0]['position']
            end_position = chunk_sentences[-1]['position'] + chunk_sentences[-1]['length']
            total_length = end_position - start_position
            
            # Fix URL parentheses issues
            chunk_text = self._fix_url_parentheses(chunk_text)
            
            chunks.append({
                'chunk_id': f"chunk_{chunk_index}",
                'chunk_text': chunk_text,
                'position': start_position,
                'length': len(chunk_text),
                'sentence_count': len(chunk_sentences),
                'sentence_indices': [sent['sentence_index'] for sent in chunk_sentences]
            })
            chunk_index += 1
        
        # Handle the last chunk if there are remaining sentences
        if len(sentences) % step_size != 0:
            remaining_start = len(sentences) - chunk_size
            if remaining_start >= 0:
                last_chunk_sentences = sentences[remaining_start:]
                last_chunk_text = ' '.join([sent['sentence_text'] for sent in last_chunk_sentences])
                
                start_position = last_chunk_sentences[0]['position']
                end_position = last_chunk_sentences[-1]['position'] + last_chunk_sentences[-1]['length']
                total_length = end_position - start_position
                
                # Fix URL parentheses issues
                last_chunk_text = self._fix_url_parentheses(last_chunk_text)
                
                chunks.append({
                'chunk_id': f"chunk_{chunk_index}",
                    'chunk_text': last_chunk_text,
                    'position': start_position,
                    'length': len(last_chunk_text),
                    'sentence_count': len(last_chunk_sentences),
                    'sentence_indices': [sent['sentence_index'] for sent in last_chunk_sentences]
                })
                chunk_index += 1
        
        return chunks
    
    def extract_sentences(self, document: str) -> List[Dict[str, Any]]:
        """Extract chunks of 4 sentences with 1 sentence overlap."""
        # First extract all sentences
        all_sentences = self._extract_all_sentences(document)
        
        # Then create chunks with overlap
        return self._create_sentence_chunks_with_overlap(all_sentences, chunk_size=15, overlap_size=3)


def is_url(text: str) -> bool:
    """
    Check if a text string is a URL.
    
    Args:
        text: The text to check
        
    Returns:
        True if the text appears to be a URL, False otherwise
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check if it starts with common URL schemes
    url_schemes = ['http://', 'https://', 'www.']
    if any(scheme in text.lower() for scheme in url_schemes):
        # Extract the first URL from the text and return it
        return True
    else:
        return False

def _min_max_norm(values: List[float]) -> Tuple[List[float], float, float]:
    """Normalize values using min-max normalization."""
    if not values:
        return [], 0.0, 0.0
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [0.0 for _ in values], vmin, vmax
    return [float((v - vmin) / (vmax - vmin)) for v in values], vmin, vmax


def extract_url_from_claim(claim: str) -> str:
    """Extract the URL from a claim."""
    # Extract the first URL from the claim
    extracted_url = ""
    match = re.search(r'\[.*?\]\((https?://[^\s\)]+)\)', claim)
    if match:
        extracted_url = match.group(1)
    else:
        # Fallback to original pattern for non-markdown URLs
        match = re.search(r'(https?://[^\s]+)', claim)
        if match:
            extracted_url = match.group(1)

    return extracted_url

def find_target_url(claim: str, atomic_claims: List[str]) -> List[str]:
    """Find the target URL for a claim.
    If multiple URLs are adjacent (consecutive URL claims), all of them are included.
    """
    target_urls = []
    # Find the last url-format claim in atomic_claims before the current claim as target_url
    # Locate the claim in atomic_claims
    claim_index = atomic_claims.index(claim)
    
    # Find the first URL before the claim
    first_url_index = None
    for j in range(claim_index - 1, -1, -1):
        if is_url(atomic_claims[j]):
            first_url_index = j
            break
    
    if first_url_index is not None:
        # Collect all consecutive URLs starting from first_url_index (going backwards and forwards)
        # First, go backwards to find the start of the URL group
        url_group_start = first_url_index
        for j in range(first_url_index - 1, -1, -1):
            if is_url(atomic_claims[j]):
                url_group_start = j
            else:
                break
        
        # Then, collect all URLs in the group (from start to end)
        for j in range(url_group_start, claim_index):
            if is_url(atomic_claims[j]):
                # Extract the URL from the claim and remove the trailing slash
                extracted_url = extract_url_from_claim(atomic_claims[j])
                if extracted_url.endswith('/'):
                    extracted_url = extracted_url[:-1]
                target_urls.append(extracted_url)

    return target_urls


def split_report_into_paragraphs(report: str) -> List[str]:
    """
    Split report into paragraphs using double newlines or Markdown headers.
    - If report contains '\n\n', split by '\n\n'
    - Otherwise, split by Markdown headers (#, ##, ###, ####)
      Handles both headers at line start and headers within a single line
    
    Args:
        report: The full report text to split
        
    Returns:
        List of paragraph strings
    """
    # Check if report contains double newlines
    if '\n\n' in report:
        # Split by double newlines
        paragraphs = [p.strip() for p in report.split('\n\n') if p.strip()]
        return paragraphs if paragraphs else [report.strip()] if report.strip() else []
    
    # If no '\n\n', split by Markdown headers
    # Pattern to match Markdown headers: #, ##, ###, #### followed by space and text
    # Match headers that may appear at start of line or within a line
    # Look for patterns like "## Title" or " ## Title" (with optional leading space)
    header_pattern = r'(?:^|\s)(#{1,4}\s+[^\n#]+?)(?=\s+#{1,4}\s+|$)'
    
    # Find all header positions in the text
    # First, try to find headers that start at the beginning of a line
    lines = report.split('\n')
    
    # If we have multiple lines, process line by line
    if len(lines) > 1:
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            # Check if this line starts with a Markdown header
            if re.match(r'^#{1,4}\s+.+', line.strip()):
                # If we have accumulated content, save it as a paragraph
                if current_paragraph:
                    para_text = '\n'.join(current_paragraph).strip()
                    if para_text:
                        paragraphs.append(para_text)
                    current_paragraph = []
                # Start new paragraph with the header
                current_paragraph.append(line)
            else:
                # Add line to current paragraph
                current_paragraph.append(line)
        
        # Don't forget the last paragraph
        if current_paragraph:
            para_text = '\n'.join(current_paragraph).strip()
            if para_text:
                paragraphs.append(para_text)
        
        if len(paragraphs) > 1:
            return paragraphs
    
    # If single line or line-by-line didn't work, try splitting within the line
    # Find all header positions (including those within a single line)
    # Pattern: match ## Title - find the start position of each header
    # Match headers that may have leading space (for headers in middle of line)
    header_pattern_inline = r'(?:^|\s)(#{1,4}\s+)'
    
    # Find all header start positions
    matches = list(re.finditer(header_pattern_inline, report))
    
    if len(matches) > 1:
        # Reconstruct paragraphs with headers
        final_paragraphs = []
        for i, match in enumerate(matches):
            # Find where this header's content ends (start of next header or end of text)
            if i + 1 < len(matches):
                content_end = matches[i + 1].start()
            else:
                content_end = len(report)
            
            # Get the full paragraph including header
            # match.start() might include leading space, so we need to handle that
            start_pos = match.start()
            if start_pos > 0 and report[start_pos] == ' ':
                # If match started with space, include it in the paragraph
                start_pos = match.start()
            else:
                # If match started at beginning or with #, use match.start() or match.end(1)
                start_pos = match.start() if match.start() == 0 else match.start()
            
            para_text = report[start_pos:content_end].strip()
            if para_text:
                final_paragraphs.append(para_text)
        
        if len(final_paragraphs) > 1:
            return final_paragraphs
    
    # Fallback: if we still only have 1 paragraph, return it as is
    return [report.strip()] if report.strip() else []