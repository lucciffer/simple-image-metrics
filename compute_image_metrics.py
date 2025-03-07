import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import argparse

# compute RMSE
def calculate_rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))

# compute MSE
def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

# compute MAE
def calculate_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

# compute PSNR
def calculate_psnr(img1, img2):
    return psnr(img1, img2, data_range=img1.max() - img1.min())

# compute SSIM
def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min(), channel_axis=2)

def compute_metrics(gt_folder, pred_folder, metrics_to_compute, output_csv):
    results = []

    gt_images = sorted(os.listdir(gt_folder))
    pred_images = sorted(os.listdir(pred_folder))

    if len(gt_images) != len(pred_images):
        print("Warning: Number of images in ground truth and predicted folders do not match.")

    for gt_img_name, pred_img_name in zip(gt_images, pred_images):
        if gt_img_name != pred_img_name:
            print(f"Warning: Image name mismatch ({gt_img_name} vs {pred_img_name}). Skipping...")
            continue

        gt_img_path = os.path.join(gt_folder, gt_img_name)
        pred_img_path = os.path.join(pred_folder, pred_img_name)

        gt_img = cv2.imread(gt_img_path)
        pred_img = cv2.imread(pred_img_path)

        if gt_img is None or pred_img is None:
            print(f"Error reading {gt_img_name}, skipping...")
            continue

        if gt_img.shape != pred_img.shape:
            print(f"Shape mismatch for {gt_img_name}, skipping...")
            continue

        img_metrics = {"image_name": gt_img_name}

        if "RMSE" in metrics_to_compute:
            img_metrics["RMSE"] = calculate_rmse(gt_img, pred_img)

        if "MSE" in metrics_to_compute:
            img_metrics["MSE"] = calculate_mse(gt_img, pred_img)

        if "MAE" in metrics_to_compute:
            img_metrics["MAE"] = calculate_mae(gt_img, pred_img)

        if "PSNR" in metrics_to_compute:
            img_metrics["PSNR"] = calculate_psnr(gt_img, pred_img)

        if "SSIM" in metrics_to_compute:
            img_metrics["SSIM"] = calculate_ssim(gt_img, pred_img)

        results.append(img_metrics)

    # log all computed metrics to CSV file
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nMetrics saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Compute image similarity metrics between ground truth and predicted images.")
    parser.add_argument("--gt_folder", required=True, help="Path to ground truth images folder")
    parser.add_argument("--pred_folder", required=True, help="Path to predicted images folder")
    parser.add_argument("--metrics", nargs='+', required=True, choices=["RMSE", "MSE", "MAE", "PSNR", "SSIM"],
                        help="Metrics to compute (space-separated). Example: --metrics RMSE PSNR SSIM")
    parser.add_argument("--output_csv", required=True, help="Path to save output CSV file")

    args = parser.parse_args()

    compute_metrics(args.gt_folder, args.pred_folder, args.metrics, args.output_csv)

if __name__ == "__main__":
    main()
