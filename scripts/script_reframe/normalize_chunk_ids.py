#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import sys
from typing import Dict, Any


BASE_ID_RE = re.compile(r'^(?:\d+-)?(chunk_(?:\d+|final))(?:_.*)?$')


def get_base_chunk_id(raw_id: str) -> str:
    if not isinstance(raw_id, str):
        return str(raw_id)
    m = BASE_ID_RE.match(raw_id)
    if m:
        return m.group(1)
    if raw_id.startswith('chunk_'):
        parts = raw_id.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            return f"chunk_{parts[1]}"
    return raw_id


def build_url_to_index_mapping(chunk_score: Dict[str, Any]) -> Dict[str, str]:
    url_to_idx: Dict[str, str] = {}
    for key, value in chunk_score.items():
        try:
            prefix, rest = key.split('-', 1)
            if prefix.isdigit():
                url = value.get('url') or value.get('source_url') or ''
                if url and url not in url_to_idx:
                    url_to_idx[url] = prefix
        except Exception:
            continue
    return url_to_idx


def normalize_cache_json(path: str) -> Dict[str, str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1) Prepare URL -> index mapping from existing chunk_score keys
    chunk_score = data.get('chunk_score', {}) or {}
    url_to_idx = build_url_to_index_mapping(chunk_score)

    # 2) Rewrite chunk_score keys to remove postfix and keep stable prefix
    if chunk_score:
        new_chunk_score: Dict[str, Any] = {}
        for key, val in chunk_score.items():
            # Determine index prefix
            idx = None
            if '-' in key:
                pfx, _ = key.split('-', 1)
                if pfx.isdigit():
                    idx = pfx
            if idx is None:
                # fallback from url
                url = val.get('url') or val.get('source_url') or ''
                idx = url_to_idx.get(url, '0')

            # Base id from key or stored original id
            base_from_key = get_base_chunk_id(key)
            base_from_val = get_base_chunk_id(val.get('chunk_id_original', ''))
            base_id = base_from_val if base_from_val.startswith('chunk_') else base_from_key

            # Update nested fields if present
            if 'chunk_id_original' in val:
                val['chunk_id_original'] = base_id
            if 'chunk_id' in val:
                val['chunk_id'] = f"{idx}-{base_id}"

            new_key = f"{idx}-{base_id}"
            new_chunk_score[new_key] = val

        data['chunk_score'] = new_chunk_score

    # 3) Rewrite top_k_chunks chunk_id to include prefix and remove postfix
    top_k = data.get('top_k_chunks', {}) or {}
    if top_k:
        for claim, chunks in top_k.items():
            if isinstance(chunks, list):
                for ch in chunks:
                    if isinstance(ch, dict):
                        raw_id = ch.get('chunk_id', '')
                        base_id = get_base_chunk_id(raw_id)
                        url = ch.get('source_url', '')
                        idx = url_to_idx.get(url)
                        if idx is not None:
                            ch['chunk_id'] = f"{idx}-{base_id}"
                        else:
                            # Fallback: keep base id only if index is unknown
                            ch['chunk_id'] = base_id

    # 4) Also sanitize any nested fields named 'chunk_id' and 'chunk_id_original' elsewhere
    def sanitize_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            if 'chunk_id' in obj:
                raw = obj['chunk_id']
                base = get_base_chunk_id(raw)
                url = obj.get('source_url', '')
                idx = url_to_idx.get(url)
                obj['chunk_id'] = f"{idx}-{base}" if idx is not None else base
            if 'chunk_id_original' in obj:
                obj['chunk_id_original'] = get_base_chunk_id(obj['chunk_id_original'])
            for k, v in list(obj.items()):
                obj[k] = sanitize_obj(v)
        elif isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = sanitize_obj(obj[i])
        return obj

    # Apply sanitization conservatively to sections likely to include chunk structures
    for section in ['related_query', 'iterations', 'report']:
        if section in data:
            data[section] = sanitize_obj(data[section])

    # Write back
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return url_to_idx


def _infer_cache_path_from_results(results_path: str) -> str:
    # Convert .../results/.../results_XXXX.json -> .../json_cache/.../cache_XXXX.json
    # Be conservative and only replace the last segment names
    import os
    d, fname = os.path.split(results_path)
    dd, parent = os.path.split(d)
    # parent may be 'reframe' or other
    # Replace 'results' with 'json_cache' in dd
    ddd, grand = os.path.split(dd)
    if grand == 'results':
        dd_converted = os.path.join(ddd, 'json_cache', parent)
    else:
        # fallback: try replacing first occurrence of '/results/' in path
        dd_converted = dd.replace('/results/', '/json_cache/')
    if fname.startswith('results_'):
        cache_fname = 'cache_' + fname[len('results_'):]
    else:
        cache_fname = fname.replace('results_', 'cache_')
    return os.path.join(dd_converted, cache_fname)


def _sanitize_obj_with_mapping(obj: Any, url_to_idx: Dict[str, str]) -> Any:
    if isinstance(obj, dict):
        if 'chunk_id' in obj:
            raw = obj['chunk_id']
            base = get_base_chunk_id(raw)
            url = obj.get('source_url', '')
            idx = url_to_idx.get(url)
            obj['chunk_id'] = f"{idx}-{base}" if idx is not None else base
        if 'chunk_id_original' in obj:
            obj['chunk_id_original'] = get_base_chunk_id(obj['chunk_id_original'])
        for k, v in list(obj.items()):
            obj[k] = _sanitize_obj_with_mapping(v, url_to_idx)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _sanitize_obj_with_mapping(obj[i], url_to_idx)
    return obj


def normalize_results_json(results_path: str, cache_path: str = None) -> None:
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Build URL -> idx mapping from cache if available
    url_to_idx: Dict[str, str] = {}
    if cache_path:
        try:
            url_to_idx = normalize_cache_json(cache_path)
        except Exception:
            url_to_idx = {}
    if not url_to_idx and not cache_path:
        # Try to infer cache path
        inferred = _infer_cache_path_from_results(results_path)
        try:
            url_to_idx = normalize_cache_json(inferred)
        except Exception:
            url_to_idx = {}

    # Sanitize known sections
    for section in ['chain_of_research_results', 'report_results']:
        if section in data and isinstance(data[section], list):
            for item in data[section]:
                if isinstance(item, dict) and 'claim_results' in item:
                    cr = item['claim_results']
                    item['claim_results'] = _sanitize_obj_with_mapping(cr, url_to_idx)

    # Write back
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    # Usage:
    #  - normalize_chunk_ids.py <cache_json_or_results_json>
    #    If results json detected, will infer cache path and normalize both.
    #  - normalize_chunk_ids.py <cache_json> <results_json>
    #    Normalize cache first (build mapping) then results using mapping.
    if len(sys.argv) == 2:
        target = sys.argv[1]
        # Heuristics: if file name starts with 'results_', treat as results json
        import os
        base = os.path.basename(target)
        if base.startswith('results_'):
            normalize_results_json(target)
            print(f"✅ Normalized chunk IDs in results: {target}")
        else:
            normalize_cache_json(target)
            print(f"✅ Normalized chunk IDs in cache: {target}")
    elif len(sys.argv) == 3:
        cache_path, results_path = sys.argv[1], sys.argv[2]
        normalize_cache_json(cache_path)
        normalize_results_json(results_path, cache_path)
        print(f"✅ Normalized chunk IDs in cache: {cache_path}")
        print(f"✅ Normalized chunk IDs in results: {results_path}")
    else:
        print("Usage: normalize_chunk_ids.py <cache_json_or_results_json> | <cache_json> <results_json>")
        sys.exit(1)


if __name__ == '__main__':
    main()


