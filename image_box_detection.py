import argparse, asyncio, cv2, json, os, sys
from pathlib import Path
import numpy as np
from playwright.async_api import async_playwright
import sys
import io

# 设置标准输出和错误流的编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
# ---------- Main logic ----------
async def extract_bboxes_from_html(html_path: Path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        page = await ctx.new_page()
        await page.goto(html_path.resolve().as_uri())

        metrics = await page.evaluate("""
            () => {
                // 1. Find and store region containers and their bboxes
                const region_containers = Array.from(document.querySelectorAll('.box[id]'));
                const region_bboxes = region_containers.map(el => {
                    const rect = el.getBoundingClientRect();
                    return { id: el.id, x: rect.x, y: rect.y, w: rect.width, h: rect.height };
                });

                // 2. Find all potential placeholders on the page
                const placeholder_bboxes = [];
                let ph_id_counter = 0;
                //精准检测
                const all_potential_placeholders = document.querySelectorAll('.bg-gray-400');

                for (const el of all_potential_placeholders) {
                    // Apply the same filters as before
                    if (el.tagName === 'SVG') continue;
                    if (el.innerText && el.innerText.trim() !== '') continue;
                    
                    const el_rect = el.getBoundingClientRect();
                    const el_center = { x: el_rect.left + el_rect.width / 2, y: el_rect.top + el_rect.height / 2 };
                    
                    // Find which region this placeholder is inside
                    let containing_region_id = null;
                    for (const region_el of region_containers) {
                        const region_rect = region_el.getBoundingClientRect();
                        if (el_center.x >= region_rect.left && el_center.x <= region_rect.right &&
                            el_center.y >= region_rect.top && el_center.y <= region_rect.bottom) {
                            containing_region_id = region_el.id;
                            break; // Assume non-overlapping regions
                        }
                    }
                    
                    // Only include placeholders that are inside a detected region
                    if (containing_region_id) {
                        placeholder_bboxes.push({
                            id: 'ph' + ph_id_counter++,
                            x: el_rect.x,
                            y: el_rect.y,
                            w: el_rect.width,
                            h: el_rect.height,
                            region_id: containing_region_id
                        });
                    }
                }

                const layout_rect = document.documentElement.getBoundingClientRect();
                return { 
                    region_bboxes, 
                    placeholder_bboxes, 
                    layout_width: layout_rect.width, 
                    layout_height: layout_rect.height 
                };
            }
        """)
        await browser.close()
    return metrics['region_bboxes'], metrics['placeholder_bboxes'], metrics['layout_width'], metrics['layout_height']


def draw_bboxes_on_image(img, region_bboxes, placeholder_bboxes):
    """Draw region (green) and placeholder (red) boxes with labels on img."""
    boxed = img.copy()
    H, W = img.shape[:2]
    
    # --- Helper to draw a single box with label ---
    def draw_box_with_label(b, color, label_text):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        # Boundary correction
        x_draw, y_draw = max(0, x), max(0, y)
        w_draw, h_draw = min(w, W - x_draw), min(h, H - y_draw)
        cv2.rectangle(boxed, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), color, 3) # Thicker lines
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_color = (255, 255, 255)

        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Position for the label background. Put it just above the box.
        label_y_start = y - text_height - baseline - 5
        if label_y_start < 0: # Adjust if the label goes off the top of the image
            label_y_start = y + 5
        
        label_x_start = x
        label_y_end = label_y_start + text_height + baseline
        
        cv2.rectangle(boxed, (label_x_start, label_y_start), (label_x_start + text_width, label_y_end), color, cv2.FILLED)
        cv2.putText(boxed, label_text, (label_x_start + 2, label_y_start + text_height), font, font_scale, text_color, font_thickness)

    # --- Draw Regions (Green) ---
    for b in region_bboxes:
        draw_box_with_label(b, color=(0, 255, 0), label_text=f'Area_{b.get("id", "")}')

    # --- Draw Placeholders (Red) ---
    for b in placeholder_bboxes:
        draw_box_with_label(b, color=(0, 0, 255), label_text=f'{b.get("region_id")}_{b.get("id")}')
        
    return boxed


def main(args):
    # Read original screenshot
    img = cv2.imread(str(args.screenshot))
    if img is None:
        sys.exit(f"Error: Cannot read image {args.screenshot}")
    if img.std() < 5:
        print("Warning: The screenshot is almost pure color, it may not be the original screenshot with real thumbnails.")

    H, W = img.shape[:2]

    # Parse HTML → Get bboxes
    region_bboxes, placeholder_bboxes, layout_width, layout_height = asyncio.run(
        extract_bboxes_from_html(args.html)
    )
    if not placeholder_bboxes:
        sys.exit("Error: No gray placeholder blocks found!")

    # Calculate separate scale factors for X and Y to handle aspect ratio differences
    scale_x = W / layout_width if layout_width > 0 else 1
    scale_y = H / layout_height if layout_height > 0 else 1
    
    if abs(scale_x - scale_y) > 0.05:
        print(f"[*] Detected different X/Y scales. X: {scale_x:.2f}, Y: {scale_y:.2f}")
    elif abs(scale_x - 1.0) > 0.05:
        print(f"[*] Detected uniform scale: {scale_x:.2f}")


    # Scale all bboxes to the original image coordinate system
    scaled_regions = []
    for b in region_bboxes:
        scaled_regions.append({
            **b,
            "x": int(b['x'] * scale_x), "y": int(b['y'] * scale_y),
            "w": int(b['w'] * scale_x), "h": int(b['h'] * scale_y)
        })

    scaled_placeholders = []
    for b in placeholder_bboxes:
        scaled_placeholders.append({
            **b,
            "x": int(b['x'] * scale_x), "y": int(b['y'] * scale_y),
            "w": int(b['w'] * scale_x), "h": int(b['h'] * scale_y)
        })

    # Draw boxes using the now-scaled data
    overlay = draw_bboxes_on_image(img, scaled_regions, scaled_placeholders)

    # Save debug image
    out_png = args.out / "debug_gray_bboxes_test1.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), overlay)
    print(f"Success: BBox overlay saved to {out_png}")


    # Convert absolute pixel coordinates to proportions for the final JSON output
    proportional_regions = []
    for b in scaled_regions:
        proportional_regions.append({
            **b,
            "x": b["x"] / W, "y": b["y"] / H,
            "w": b["w"] / W, "h": b["h"] / H
        })
        
    proportional_placeholders = []
    for b in scaled_placeholders:
        proportional_placeholders.append({
            **b,
            "x": b["x"] / W, "y": b["y"] / H,
            "w": b["w"] / W, "h": b["h"] / H
        })

    # Print/save bbox array
    print("\n=== BBox (proportional to image dimensions) ===")
    output_data = {
        "regions": proportional_regions,
        "placeholders": proportional_placeholders
    }
    output_json = json.dumps(output_data, indent=2, ensure_ascii=False)
    print(output_json)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output_json)
        print(f"Success: BBox list saved to {args.json}")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw BBoxes parsed from HTML on the original screenshot"
    )
    parser.add_argument("--html", required=False, type=Path, default=Path("data/output/test1_layout.html"),
                        help="Generated HTML file (with gray placeholder)")
    parser.add_argument("--screenshot", required=False, type=Path, default=Path("data/input/test1.png"),
                        help="Original UI screenshot (with real thumbnails)")
    parser.add_argument("--out", default=Path("data/tmp"), type=Path,
                        help="Output directory (save debug_gray_bboxes_test1.png)")
    parser.add_argument("--json", type=Path, default=Path("data/tmp/test1_bboxes.json"),
                        help="If provided, write BBox list to JSON file")
    args = parser.parse_args()
    main(args)
