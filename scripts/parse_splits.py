#!/usr/bin/env python3
"""
Parse EGTEA Gaze+ dataset split files and generate structured CSV
containing both IDs and translated names of actions, verbs, and nouns.
Also parses timestamps and adds formatted time columns.
"""

import csv
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import timedelta

def load_mapping(file_path: str) -> Dict[int, str]:
    """Load index to name mapping from file."""
    mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Extract name and index from line
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                name, idx_str = parts
                try:
                    # Convert to 0-indexed
                    idx = int(idx_str) - 1
                    mapping[idx] = name
                except ValueError:
                    print(f"Warning: Invalid line format in {file_path}: {line}")
    return mapping

def format_ms_to_mmss(ms: int) -> str:
    """Convert milliseconds to mm:ss.ms format."""
    seconds, ms = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}.{ms:03d}"

def parse_split_file(file_path: str, verb_mapping: Dict[int, str], 
                     action_mapping: Dict[int, str], noun_mapping: Dict[int, str]) -> List[dict]:
    """Parse split file and return structured data with IDs and names."""
    results = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Parse the line: {clipname}-dd-dd-F{startframe}-F{endframe} {verb-id} {action_id} [{nound-id}+]
            parts = line.split(' ')
            if len(parts) < 3:
                print(f"Warning: Invalid line format: {line}")
                continue
                
            clip_info = parts[0]
            # Extract clip name, timestamps, and frame information
            # Format: {clipname}-start_time-end_time-Fstart_frame-Fend_frame
            clip_match = re.match(r'(.+)-(\d+)-(\d+)-F(\d+)-F(\d+)', clip_info)
            if not clip_match:
                print(f"Warning: Invalid clip format: {clip_info}")
                continue
                
            clip_name, start_time, end_time, start_frame, end_frame = clip_match.groups()
            
            # Convert to integers
            start_time = int(start_time)
            end_time = int(end_time)
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            
            # Format time as mm:ss.ms
            start_time_fmt = format_ms_to_mmss(start_time)
            end_time_fmt = format_ms_to_mmss(end_time)
            
            # Parse IDs (0-indexed in the file)
            try:
                verb_id = int(parts[1]) - 1  # Convert to 0-indexed
                action_id = int(parts[2]) - 1
                noun_ids = [int(n) - 1 for n in parts[3:]]  # Convert to 0-indexed
            except ValueError:
                print(f"Warning: Invalid ID format: {line}")
                continue
            
            # Get names from mappings
            verb_name = verb_mapping.get(verb_id, f"Unknown verb {verb_id}")
            action_name = action_mapping.get(action_id, f"Unknown action {action_id}")
            noun_names = [noun_mapping.get(n_id, f"Unknown noun {n_id}") for n_id in noun_ids]
            
            # Create entry
            entry = {
                'clip_name': clip_name,
                'start_time_ms': start_time,
                'end_time_ms': end_time,
                'start_time_fmt': start_time_fmt,
                'end_time_fmt': end_time_fmt,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'verb_id': verb_id + 1,  # Store as 1-indexed for consistency with original files
                'verb_name': verb_name,
                'action_id': action_id + 1,  # Store as 1-indexed
                'action_name': action_name,
                'noun_ids': [n_id + 1 for n_id in noun_ids],  # Store as 1-indexed
                'noun_names': noun_names
            }
            results.append(entry)
            
    # Sort by clip_name and start_frame
    results.sort(key=lambda x: (x['clip_name'], x['start_frame']))
    return results

def write_csv(data: List[dict], output_path: str) -> None:
    """Write the structured data to a CSV file."""
    if not data:
        print("No data to write.")
        return
        
    # Define CSV fields
    fields = [
        'clip_name', 
        'start_time_ms', 'end_time_ms',
        'start_time_fmt', 'end_time_fmt',
        'start_frame', 'end_frame', 
        'verb_id', 'verb_name', 
        'action_id', 'action_name', 
        'noun_ids', 'noun_names'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for entry in data:
            # Convert lists to strings for CSV storage
            entry_copy = entry.copy()
            entry_copy['noun_ids'] = ','.join(map(str, entry['noun_ids']))
            entry_copy['noun_names'] = ','.join(entry['noun_names'])
            writer.writerow(entry_copy)
    
    print(f"CSV file created: {output_path}")

def main(split_file: str) -> None:
    """Main function to parse split file and generate CSV."""
    # Get base directory
    base_dir = Path('egtea_gaze/action_annotation')
    
    # Load mappings
    verb_map = load_mapping(base_dir / 'verb_idx.txt')
    noun_map = load_mapping(base_dir / 'noun_idx.txt')
    action_map = load_mapping(base_dir / 'action_idx.txt')
    
    # Parse split file
    split_path = base_dir / split_file
    if not split_path.exists():
        print(f"Error: Split file not found: {split_path}")
        sys.exit(1)
        
    data = parse_split_file(split_path, verb_map, action_map, noun_map)
    
    # Generate output path in the same directory as the input file
    split_name = os.path.splitext(os.path.basename(split_file))[0]
    output_path = base_dir / f"{split_name}_parsed.csv"
    
    # Write CSV
    write_csv(data, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_splits.py <split_file>")
        print("Example: python parse_splits.py train_split1.txt")
        sys.exit(1)
        
    main(sys.argv[1]) 