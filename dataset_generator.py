import argparse
import os
import csv
import numpy as np
import cv2

def generate_pattern(arch_type, size=(64, 64)):
    canvas = np.full(size, 180, dtype=np.uint8)
    if arch_type.lower() == 'dram':
        # Hexagonal/grid contact hole array
        for r in range(8, size[0] - 8, 12):
            for c in range(8, size[1] - 8, 12):
                cv2.circle(canvas, (c, r), 3, (50,), -1)
    else:  # FinFET
        # Dense parallel vertical fins with gate crossings
        for c in range(6, size[1] - 6, 8):
            cv2.line(canvas, (c, 4), (c, size[0] - 4), (60,), 2)
        cv2.line(canvas, (4, size[0] // 2), (size[1] - 4, size[0] // 2), (100,), 4)
    return canvas

def add_sem_noise(img):
    # Poisson shot noise + Gaussian thermal noise
    noisy = img.astype(np.float32)
    poisson = np.random.poisson(noisy).astype(np.float32)
    gauss = np.random.normal(0, 12, img.shape).astype(np.float32)
    combined = np.clip(poisson + gauss, 0, 255).astype(np.uint8)
    return combined

def main():
    parser = argparse.ArgumentParser(description="Synthetic Semiconductor Pattern Generator")
    parser.add_argument("--arch", type=str, choices=["DRAM", "FinFET"], default="DRAM", help="Architecture pattern style")
    parser.add_argument("--num_pairs", type=int, default=20, help="Number of image pairs to generate")
    parser.add_argument("--output_dir", type=str, default="synthetic_dataset", help="Output directory path")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "search"), exist_ok=True)
    
    gt_file = os.path.join(args.output_dir, "ground_truth.csv")
    
    with open(gt_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "arch", "ref_file", "search_file", "gt_center_x", "gt_center_y"])

        for i in range(args.num_pairs):
            ref_pat = generate_pattern(args.arch, size=(64, 64))
            ref_noisy = add_sem_noise(ref_pat)
            
            # Large search scene (128x128)
            search_scene = np.full((128, 128), 200, dtype=np.uint8)
            # Add background noise textures
            search_scene = add_sem_noise(search_scene)
            
            # Embed reference pattern at a random location
            max_x = 128 - 64
            max_y = 128 - 64
            top_x = np.random.randint(0, max_x + 1)
            top_y = np.random.randint(0, max_y + 1)
            
            search_scene[top_y:top_y+64, top_x:top_x+64] = ref_noisy
            
            center_x = top_x + 32
            center_y = top_y + 32
            
            ref_name = f"ref_{args.arch.lower()}_{i:04d}.png"
            search_name = f"search_{args.arch.lower()}_{i:04d}.png"
            
            cv2.imwrite(os.path.join(args.output_dir, "reference", ref_name), ref_noisy)
            cv2.imwrite(os.path.join(args.output_dir, "search", search_name), search_scene)
            
            writer.writerow([i, args.arch, ref_name, search_name, center_x, center_y])

    print(f"Generated {args.num_pairs} pairs for {args.arch} in '{args.output_dir}'")
    print(f"Ground truth written to '{gt_file}'")

if __name__ == "__main__":
    main()