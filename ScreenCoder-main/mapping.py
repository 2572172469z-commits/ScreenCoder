"""
python script/mapping.py --gray /Users/jimmyzhengyz/Documents/Research/ui2code_demo/public/assets/debug/bboxes.json --uied /Users/jimmyzhengyz/Documents/Research/ui2code_demo/public/assets/demo1_output/ip/demo1_filtered.json --debug overlay.png --debug-src public/assets/demo1.png
"""
import json, argparse, numpy as np, cv2
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from sklearn.linear_model import RANSACRegressor
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import sys

CIOU_STRICT = -0.9      # Min CIoU score for a valid one-to-one mapping
FILTER_MIN_WH = 10     # UIED filter: ignore boxes smaller than this

# Tools
def ciou(a, b):
    """
    Calculate Complete IoU (CIoU) between two bounding boxes.
    `a`, `b`: bounding boxes in format (x, y, w, h).
    Returns a value between -1 and 1. Higher is better.
    """
    # Epsilon to prevent division by zero
    epsilon = 1e-7

    # Standard IoU
    xa, ya, wa, ha = a
    xb, yb, wb, hb = b
    x1, y1 = max(xa, xb), max(ya, yb)
    x2, y2 = min(xa + wa, xb + wb), min(ya + ha, yb + hb)
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (wa * ha) + (wb * hb) - intersection_area
    iou_val = intersection_area / (union_area + epsilon)

    # Center points distance
    center_a = center(a)
    center_b = center(b)
    center_distance_sq = np.sum((center_a - center_b) ** 2)

    # Enclosing box diagonal
    enclose_x1 = min(xa, xb)
    enclose_y1 = min(ya, yb)
    enclose_x2 = max(xa + wa, xb + wb)
    enclose_y2 = max(ya + ha, yb + hb)
    enclose_diag_sq = ((enclose_x2 - enclose_x1) ** 2) + ((enclose_y2 - enclose_y1) ** 2)
    
    distance_penalty = center_distance_sq / (enclose_diag_sq + epsilon)

    # Aspect ratio consistency
    arctan_a = np.arctan(wa / (ha + epsilon))
    arctan_b = np.arctan(wb / (hb + epsilon))
    v = (4 / (np.pi ** 2)) * ((arctan_a - arctan_b) ** 2)
    
    # Trade-off parameter alpha
    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = v / (1 - iou_val + v + epsilon)
        alpha = 0 if np.isnan(alpha) else alpha # if iou=1 and v=0, alpha is nan.
    
    aspect_ratio_penalty = alpha * v
    
    # CIOU
    ciou_val = iou_val - distance_penalty - aspect_ratio_penalty
    return ciou_val

def center(box):
    x, y, w, h = box
    return np.array([x + w / 2, y + h / 2])

def load_regions_and_placeholders(p: Path, W_img, H_img):
    """
    Loads region and placeholder data from the specified JSON file.
    The file is expected to have 'regions' and 'placeholders' keys with
    proportional bbox values, which are converted to absolute pixel values.
    """
    data = json.loads(p.read_text())
    
    def to_pixels(b):
        return (b['x']*W_img, b['y']*H_img, b['w']*W_img, b['h']*H_img)

    regions = [{**d, "bbox": to_pixels(d)} for d in data.get("regions", [])]
    placeholders = [{**d, "bbox": to_pixels(d)} for d in data.get("placeholders", [])]
    
    if not regions or not placeholders:
        print(f"Warning: JSON file {p} does not contain 'regions' or 'placeholders' keys.")
        
    return regions, placeholders

def load_uied_boxes(p: Path):
    """
    Loads UIED component detection data.
    The JSON file is expected to contain the shape of the image that was
    processed, which is crucial for calculating scaling factors later.
    """
    data = json.loads(p.read_text())
    compos = data.get("compos", [])
    shape = data.get("img_shape")  # e.g., [800, 571, 3]

    items = []
    for d in compos:
        w, h = d.get("width", 0), d.get("height", 0)
        if w < FILTER_MIN_WH or h < FILTER_MIN_WH: continue
        items.append({"id": d["id"],
                      "bbox": (d["column_min"], d["row_min"], w, h)})
        # print(d["id"], d["column_min"], d["row_min"], w, h)
    return items, shape

def estimate_global_transform(pixel_placeholders, uied_boxes, uied_shape, W_orig, H_orig):
    """
    Estimates a global affine transform from the UIED coordinate space to the
    original screenshot's coordinate space. This is used for rough alignment.
    """
    # 1. Calculate base scaling from image dimension ratios
    H_proc, W_proc, _ = uied_shape
    scale_x = W_orig / W_proc
    scale_y = H_orig / H_proc
    
    # 2. Apply this scaling to all UIED boxes
    uied_scaled = [{**u, "bbox": (u["bbox"][0]*scale_x, u["bbox"][1]*scale_y, u["bbox"][2]*scale_x, u["bbox"][3]*scale_y)} for u in uied_boxes]

    # 3. Estimate residual translation (dx, dy) by matching centers
    if not pixel_placeholders or not uied_scaled:
        return scale_x, scale_y, 0, 0

    ph_centers = np.array([center(p["bbox"]) for p in pixel_placeholders])
    uied_scaled_centers = np.array([center(u["bbox"]) for u in uied_scaled])
    
    indices = cdist(ph_centers, uied_scaled_centers).argmin(axis=1)
    translations = ph_centers - uied_scaled_centers[indices]
    dx, dy = np.median(translations, axis=0)
    
    return scale_x, scale_y, dx, dy

def apply_affine_transform(box, scale_x, scale_y, dx, dy):
    x, y, w, h = box
    return (x * scale_x + dx, y * scale_y + dy, w * scale_x, h * scale_y)

# Mapping Function
def find_local_mapping_and_transform(placeholders, uied_boxes, uied_shape, W_orig, H_orig):
    """
    Finds the optimal one-to-one mapping and the local affine transform for a given
    subset of placeholders and UIED boxes.
    """
    if not placeholders or not uied_boxes:
        return {}, (1, 1, 0, 0)
    
    # 1. Estimate local affine transform
    # 1a. Calculate base scaling from image dimension ratios
    H_proc, W_proc, _ = uied_shape
    scale_x = W_orig / W_proc
    scale_y = H_orig / H_proc

    # 1b. Apply this scaling to UIED boxes
    uied_scaled = [{**u, "bbox": (u["bbox"][0]*scale_x, u["bbox"][1]*scale_y, u["bbox"][2]*scale_x, u["bbox"][3]*scale_y)} for u in uied_boxes]

    # 1c. Estimate residual translation (dx, dy) by matching centers
    ph_centers = np.array([center(p["bbox"]) for p in placeholders])
    uied_scaled_centers = np.array([center(u["bbox"]) for u in uied_scaled])
    
    indices = cdist(ph_centers, uied_scaled_centers).argmin(axis=1)
    translations = ph_centers - uied_scaled_centers[indices]
    dx, dy = np.median(translations, axis=0)

    transform = (scale_x, scale_y, dx, dy)
    
    # 2. Apply the final, full transformation to all UIED boxes in this subset
    uied_tf = [{**u, "bbox_tf": apply_affine_transform(u["bbox"], scale_x, scale_y, dx, dy)} for u in uied_boxes]
    
    # 3. Create a cost matrix and find optimal assignment
    num_gray = len(placeholders)
    num_uied = len(uied_tf)
    cost_matrix = np.zeros((num_gray, num_uied))

    for i in range(num_gray):
        for j in range(num_uied):
            cost_matrix[i, j] = -ciou(placeholders[i]["bbox"], uied_tf[j]["bbox_tf"])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. Create the one-to-one mapping
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        score = -cost_matrix[r, c]
        if score >= CIOU_STRICT:
            g_id = placeholders[r]["id"]
            u_id = uied_tf[c]["id"]
            mapping[g_id] = u_id
            
    return mapping, transform


def generate_debug_overlay(img_path, all_uied_boxes, region_results, uied_shape, out_png):
    """
    Generates a debug image by drawing the mapped UIED boxes on the original screenshot.
    This version uses a simple scaling based on image dimensions, without any translation.
    """
    canvas = cv2.imread(str(img_path))
    if canvas is None:
        print(f"Error: Could not read debug source image at {img_path}.")
        return

    # Use a fixed red color for all bounding boxes for consistency
    color = (0, 0, 255) # Red in BGR

    # 1. Calculate simple scaling factors from the provided image shapes.
    H_proc, W_proc, _ = uied_shape
    H_orig, W_orig, _ = canvas.shape
    scale_x = W_orig / W_proc
    scale_y = H_orig / H_proc

    # 2. Draw all mapped UIED boxes using only this simple scaling.
    for region_id, result in region_results.items():
        mapping = result.get("mapping", {})
        for g_id, uid in mapping.items():
            u_box = next((box for box in all_uied_boxes if box["id"] == uid), None)
            if u_box is None: continue

            # Apply simple scaling directly, without any translation offset.
            x_proc, y_proc, w_proc, h_proc = u_box["bbox"]
            x = x_proc * scale_x
            y = y_proc * scale_y
            w = w_proc * scale_x
            h = h_proc * scale_y
            
            cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            cv2.putText(canvas, f"uied_{uid}", (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(str(out_png), canvas)


def main(args):
    # 1. Load the original screenshot to get its absolute dimensions
    if not args.debug_src or not args.debug_src.exists():
        sys.exit("Error: A valid --debug-src image path must be provided for coordinate conversion.")
    
    orig_img = cv2.imread(str(args.debug_src))
    if orig_img is None:
        sys.exit(f"Error: Could not read debug source image at {args.debug_src}.")
    H_orig, W_orig, _ = orig_img.shape

    # 2. Load proportional data and convert to absolute pixel coordinates
    pixel_regions, pixel_placeholders = load_regions_and_placeholders(args.gray, W_orig, H_orig)
    
    # 3. Load UIED data
    all_uied_boxes, uied_shape = load_uied_boxes(args.uied)
    
    if not pixel_placeholders or not all_uied_boxes:
        print("Error: Could not proceed without placeholder and UIED data.")
        return

    # 4. Estimate a GLOBAL transform for rough, initial alignment of all UIED boxes
    g_scale_x, g_scale_y, g_dx, g_dy = estimate_global_transform(pixel_placeholders, all_uied_boxes, uied_shape, W_orig, H_orig)
    print(f"Estimated Global Transform: scale_x={g_scale_x:.3f}, scale_y={g_scale_y:.3f}, dx={g_dx:.1f}, dy={g_dy:.1f}")
    
    # Apply the global transform to all UIED boxes to get them into the main coordinate space
    uied_tf_global = [{**u, "bbox_tf": apply_affine_transform(u["bbox"], g_scale_x, g_scale_y, g_dx, g_dy)} for u in all_uied_boxes]

    # 5. Loop through regions and perform LOCALIZED matching and transform estimation
    final_results = {}
    total_placeholders_count = len(pixel_placeholders)
    total_mappings_count = 0

    for region in pixel_regions:
        # Filter placeholders for the current region
        region_placeholders = [p for p in pixel_placeholders if p.get("region_id") == region["id"]]
        if not region_placeholders:
            continue

        # Filter UIED boxes for the current region using the globally transformed coordinates
        rx, ry, rw, rh = region["bbox"]
        region_uied_ids = {
            u['id'] for u in uied_tf_global 
            if rx <= center(u["bbox_tf"])[0] <= rx + rw and ry <= center(u["bbox_tf"])[1] <= ry + rh
        }
        # Get the original uied boxes that correspond to this region
        region_uied_boxes = [u for u in all_uied_boxes if u['id'] in region_uied_ids]
        
        if not region_uied_boxes:
            print(f"Warning: No UIED boxes found in region {region['id']} after global alignment.")
            continue

        # Find the precise LOCAL mapping and transform for this region
        region_mapping, region_transform = find_local_mapping_and_transform(
            region_placeholders, region_uied_boxes, uied_shape, W_orig, H_orig
        )
        
        if region_mapping:
            total_mappings_count += len(region_mapping)
            l_scale_x, l_scale_y, l_dx, l_dy = region_transform
            final_results[region["id"]] = {
                "transform": { "scale_x": l_scale_x, "scale_y": l_scale_y, "dx": l_dx, "dy": l_dy },
                "mapping": region_mapping
            }

    # 6. Report and save results
    print(f"Successfully created {total_mappings_count} one-to-one mappings out of {total_placeholders_count} placeholders.")

    args.out.write_text(json.dumps(final_results, indent=2, ensure_ascii=False))
    print(f"Mapping data written to {args.out}")
    
    if args.debug:
        if not args.debug_src or not args.debug_src.exists():
            print("Error: A valid --debug-src image path must be provided when using --debug.")
            return
        generate_debug_overlay(args.debug_src, all_uied_boxes, final_results, uied_shape, args.debug)
        print(f"Debug image written to {args.debug}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gray", type=Path, default=Path("data/tmp/test1_bboxes.json"), help="Path to the JSON file with gray placeholder boxes.")
    ap.add_argument("--uied", type=Path, default=Path("data/tmp/ip/test1.json"), help="Path to the JSON file with UIED detected boxes.")
    ap.add_argument("--out", default=Path("data/tmp/mapping_full_test1.json"), type=Path, help="Output path for the mapping JSON file.")
    ap.add_argument("--debug", type=Path, default=Path("data/tmp/overlay_test_test1.png"), help="Output path for the debug overlay PNG.")
    ap.add_argument("--debug-src", type=Path, default=Path("data/input/test1.png"), help="Path to the original screenshot for the debug overlay background.")
    main(ap.parse_args())
