#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import argparse
from pathlib import Path


def filter_contained_bboxes(bboxes):
    """
    Filters a list of bounding boxes by removing any box that is fully contained within another.

    Args:
        bboxes (list of dict): A list of bounding boxes. Each bbox is expected to be a
                               dictionary with 'column_min', 'row_min', 'column_max', 'row_max'.

    Returns:
        list of dict: A new list of bounding boxes with the smaller, contained boxes removed.
    """
    # A set to store the indices of bboxes to be removed.
    # Using a set avoids duplicates and provides fast membership checking.
    indices_to_remove = set()

    # Compare every bbox with every other bbox.
    for i in range(len(bboxes)):
        for j in range(len(bboxes)):
            # Don't compare a bbox to itself.
            if i == j:
                continue

            bbox_a = bboxes[i]
            bbox_b = bboxes[j]

            # Check if bbox_a contains bbox_b
            # This is true if bbox_b's corners are all within bbox_a's corners.
            # Note: If they are identical, one will contain the other.
            is_contained = (bbox_a['column_min'] <= bbox_b['column_min'] and
                            bbox_a['row_min'] <= bbox_b['row_min'] and
                            bbox_a['column_max'] >= bbox_b['column_max'] and
                            bbox_a['row_max'] >= bbox_b['row_max'])

            if is_contained:
                # If bbox_a and bbox_b are identical, this logic will flag one for removal
                # based on its index, which is acceptable for de-duplication.
                # If bbox_a is strictly larger, bbox_b is the smaller one and should be removed.
                indices_to_remove.add(j)

    # Create a new list containing only the bboxes that were not marked for removal.
    filtered_bboxes = [bbox for i, bbox in enumerate(bboxes) if i not in indices_to_remove]

    return filtered_bboxes


def main():
    parser = argparse.ArgumentParser(
        description="Filter bounding boxes in a JSON file, removing any box that is contained within another.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('input_file', type=Path,
                        help="Path to the input JSON file containing a list of bounding boxes.\n"
                             "Example: public/assets/demo1_output/ip/demo1.json")
    parser.add_argument('output_file', type=Path,
                        help="Path to save the filtered JSON file.\n"
                             "Example: public/assets/demo1_output/ip/demo1_filtered.json")
    args = parser.parse_args()

    # --- Read Input ---
    if not args.input_file.exists():
        print(f"Error: Input file not found at {args.input_file}")
        return

    print(f"Reading bounding boxes from: {args.input_file}")
    with open(args.input_file, 'r') as f:
        # The JSON format from compo_detection is a dictionary with a 'compos' key.
        data = json.load(f)
        if isinstance(data, dict) and 'compos' in data:
            initial_bboxes = data['compos']
        elif isinstance(data, list):
            initial_bboxes = data
        else:
            print(f"Error: Unexpected JSON format in {args.input_file}")
            return

    print(f"Found {len(initial_bboxes)} bounding boxes.")

    # --- Filter Bboxes ---
    print("Filtering contained bounding boxes...")
    filtered_bboxes = filter_contained_bboxes(initial_bboxes)
    print(f"{len(filtered_bboxes)} bounding boxes remaining after filtering.")

    # --- Write Output ---
    # Create the output directory if it doesn't exist.
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Wrap the result in the same structure as the input
    if isinstance(data, dict) and 'compos' in data:
        output_data = data.copy()
        output_data['compos'] = filtered_bboxes
    else:
        output_data = filtered_bboxes

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Successfully saved filtered bounding boxes to: {args.output_file}")


if __name__ == '__main__':
    main() 