#!/usr/bin/env python3
"""
Extends the ego-topo-action-mapping.csv file with verb and noun names
by using verb_idx.txt and noun_idx.txt lookup tables and adds egtea_action_id
by matching against action_idx.txt.
"""

import os
import csv
import argparse
import json
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


def load_action_index_file(file_path):
    """Load action descriptions to index mapping from action index file."""
    action_to_idx = {}
    verb_noun_to_idx = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ')
            if len(parts) >= 2:
                # Format is "Action Description IndexNumber"
                idx = int(parts[-1]) - 1  # Convert from 1-indexed to 0-indexed
                action_desc = ' '.join(parts[:-1])
                
                # Store the full action description
                action_to_idx[action_desc.lower()] = idx
                
                # Parse the action into verb and noun parts
                action_parts = action_desc.split(' ')
                if len(action_parts) >= 2:
                    verb = action_parts[0].lower()
                    # The noun might contain multiple parts (e.g., "condiment,bread,eating_utensil")
                    noun_parts = ' '.join(action_parts[1:]).lower()
                    
                    # Some nouns have comma-separated values
                    nouns = [n.strip() for n in noun_parts.split(',')]
                    
                    # Store each verb-noun combination
                    for noun in nouns:
                        verb_noun_to_idx[(verb, noun)] = idx
    
    return action_to_idx, verb_noun_to_idx


def extend_action_mapping(mapping_file, verb_idx_file, noun_idx_file, action_idx_file, output_file, debug=False):
    """Extend action mapping with verb and noun names and egtea action IDs."""
    
    # Load lookup tables
    verb_dict = load_index_file(verb_idx_file)
    noun_dict = load_index_file(noun_idx_file)
    action_dict, verb_noun_to_idx = load_action_index_file(action_idx_file)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # For debug/analysis
    matching_stats = {
        "exact_matches": 0,
        "verb_noun_matches": 0,
        "flexible_matches": 0,
        "unmatched": 0,
        "match_details": []
    }
    
    # Process the mapping file
    with open(mapping_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Read header
        header = next(reader)
        # Add new columns
        extended_header = header[:2] + ['verb_name', 'noun_name', 'action_description', 'egtea_action_id'] + header[2:]
        writer.writerow(extended_header)
        
        # Process rows
        for row_idx, row in enumerate(reader, 1):
            if len(row) >= 4:  # Ensure row has expected format
                # The first column is noun_id, the second is verb_id
                noun_id = int(row[0])
                verb_id = int(row[1])
                
                # Look up names (with fallback for missing indices)
                verb_name = verb_dict.get(verb_id, f"Unknown-{verb_id}")
                noun_name = noun_dict.get(noun_id, f"Unknown-{noun_id}")
                
                # Create action description
                action_description = f"{verb_name} {noun_name}"
                
                # Look up EGTEA action ID
                egtea_action_id = -1  # Default value if not found
                match_type = "unmatched"
                matched_action = None
                
                # Try exact match first
                action_key = action_description.lower()
                if action_key in action_dict:
                    egtea_action_id = action_dict[action_key]
                    match_type = "exact_match"
                    matched_action = action_key
                    matching_stats["exact_matches"] += 1
                else:
                    # Try matching by verb and noun combination
                    verb_key = verb_name.lower()
                    noun_key = noun_name.lower()
                    
                    # Try exact verb-noun pair
                    if (verb_key, noun_key) in verb_noun_to_idx:
                        egtea_action_id = verb_noun_to_idx[(verb_key, noun_key)]
                        match_type = "verb_noun_match"
                        for act, idx in action_dict.items():
                            if idx == egtea_action_id:
                                matched_action = act
                                break
                        matching_stats["verb_noun_matches"] += 1
                    else:
                        # Try finding actions where both verb and noun appear
                        for act_desc, idx in action_dict.items():
                            if verb_key in act_desc and noun_key in act_desc:
                                egtea_action_id = idx
                                match_type = "flexible_match"
                                matched_action = act_desc
                                matching_stats["flexible_matches"] += 1
                                break
                        
                        if egtea_action_id == -1:
                            matching_stats["unmatched"] += 1
                
                # Store match details for debugging
                if debug:
                    matching_stats["match_details"].append({
                        "row": row_idx,
                        "ego_topo_action_id": int(row[2]),
                        "verb_id": verb_id,
                        "noun_id": noun_id,
                        "verb_name": verb_name,
                        "noun_name": noun_name,
                        "action_description": action_description,
                        "egtea_action_id": egtea_action_id,
                        "match_type": match_type,
                        "matched_action": matched_action
                    })
                
                # Create extended row
                extended_row = row[:2] + [verb_name, noun_name, action_description, str(egtea_action_id)] + row[2:]
                writer.writerow(extended_row)
    
    # Print matching statistics
    print(f"Action matching statistics:")
    print(f"  - Exact matches: {matching_stats['exact_matches']}")
    print(f"  - Verb-noun matches: {matching_stats['verb_noun_matches']}")
    print(f"  - Flexible matches: {matching_stats['flexible_matches']}")
    print(f"  - Unmatched: {matching_stats['unmatched']}")
    print(f"  - Total processed: {matching_stats['exact_matches'] + matching_stats['verb_noun_matches'] + matching_stats['flexible_matches'] + matching_stats['unmatched']}")
    
    # Save debug information if requested
    if debug:
        debug_file = f"{os.path.splitext(output_file)[0]}_debug.json"
        with open(debug_file, 'w') as f:
            json.dump(matching_stats, f, indent=2)
        print(f"Debug information saved to {debug_file}")
    
    return matching_stats


def main():
    parser = argparse.ArgumentParser(description='Extend action mapping CSV with verb and noun names')
    parser.add_argument('--mapping', default='egtea_gaze/action_annotation/ego-topo-action-mapping.csv',
                        help='Path to the action mapping CSV file')
    parser.add_argument('--verb-idx', default='egtea_gaze/action_annotation/verb_idx.txt',
                        help='Path to the verb index file')
    parser.add_argument('--noun-idx', default='egtea_gaze/action_annotation/noun_idx.txt',
                        help='Path to the noun index file')
    parser.add_argument('--action-idx', default='egtea_gaze/action_annotation/action_idx.txt',
                        help='Path to the action index file')
    parser.add_argument('--output', default=None,
                        help='Path to save the extended CSV file (default: {original}_extended.csv)')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug information about matching')
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        base_name = os.path.splitext(args.mapping)[0]
        args.output = f"{base_name}_extended.csv"
    
    extend_action_mapping(args.mapping, args.verb_idx, args.noun_idx, args.action_idx, args.output, args.debug)
    print(f"Extended action mapping saved to {args.output}")


if __name__ == "__main__":
    main() 