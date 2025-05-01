#!/usr/bin/env python3
"""
Extends the ego-topo-action-mapping.csv file with verb and noun names
by using verb_idx.txt and noun_idx.txt lookup tables.
"""

import os
import csv
import argparse
from pathlib import Path


def load_index_file(file_path):
    """Load name-to-index mapping from index file."""
    idx_to_name = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ')
            if len(parts) >= 2:
                # Format is "Name IndexNumber"
                idx = int(parts[-1]) - 1  # Convert from 1-indexed to 0-indexed
                name = ' '.join(parts[:-1])
                idx_to_name[idx] = name
    return idx_to_name


def extend_action_mapping(mapping_file, verb_idx_file, noun_idx_file, output_file):
    """Extend action mapping with verb and noun names."""
    
    # Load lookup tables
    verb_dict = load_index_file(verb_idx_file)
    noun_dict = load_index_file(noun_idx_file)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process the mapping file
    with open(mapping_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read header
        header = next(reader)
        # Add new columns
        extended_header = header[:2] + ['verb_name', 'noun_name', 'action_description'] + header[2:]
        writer.writerow(extended_header)
        
        # Process rows
        for row in reader:
            if len(row) >= 4:  # Ensure row has expected format
                noun_id = int(row[0])
                verb_id = int(row[1])
                
                # Look up names (with fallback for missing indices)
                verb_name = verb_dict.get(verb_id, f"Unknown-{verb_id}")
                noun_name = noun_dict.get(noun_id, f"Unknown-{noun_id}")
                
                # Create action description
                action_description = f"{verb_name} {noun_name}"
                
                # Create extended row
                extended_row = row[:2] + [verb_name, noun_name, action_description] + row[2:]
                writer.writerow(extended_row)


def main():
    parser = argparse.ArgumentParser(description='Extend action mapping CSV with verb and noun names')
    parser.add_argument('--mapping', default='egtea_gaze/action_annotation/ego-topo-action-mapping.csv',
                        help='Path to the action mapping CSV file')
    parser.add_argument('--verb-idx', default='egtea_gaze/action_annotation/verb_idx.txt',
                        help='Path to the verb index file')
    parser.add_argument('--noun-idx', default='egtea_gaze/action_annotation/noun_idx.txt',
                        help='Path to the noun index file')
    parser.add_argument('--output', default=None,
                        help='Path to save the extended CSV file (default: {original}_extended.csv)')
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        base_name = os.path.splitext(args.mapping)[0]
        args.output = f"{base_name}_extended.csv"
    
    extend_action_mapping(args.mapping, args.verb_idx, args.noun_idx, args.output)
    print(f"Extended action mapping saved to {args.output}")


if __name__ == "__main__":
    main() 