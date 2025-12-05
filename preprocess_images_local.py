"""
Image Preprocessing Script for Plant Pathology 2021 Dataset
============================================================
Run this script on your MacBook M4 Pro to preprocess all images.
After preprocessing, upload the 'preprocessed_images' folder to Google Drive.

Usage:
    python preprocess_images_local.py

Requirements:
    pip install pillow pandas tqdm numpy
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial

# ============================================================================
# CONFIGURATION - ADJUST THESE PATHS
# ============================================================================

# Path to your dataset on MacBook
BASE_PATH = 'Plant_Pathology_2021'  # CHANGE THIS
TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, 'train_images')
PREPROCESSED_PATH = os.path.join(BASE_PATH, 'preprocessed_images')

# Image size for preprocessing
IMG_SIZE = 224  # Change to 384 if you want higher resolution

# Number of CPU cores to use (M4 Pro has excellent multi-core performance)
NUM_PROCESSES = 8  # Adjust based on your M4 Pro (it has 14 cores, so 8-12 is good)

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_single_image(args):
    """
    Preprocess a single image: resize and save
    
    Args:
        args: tuple of (img_name, source_dir, target_dir, img_size)
    
    Returns:
        tuple: (success: bool, img_name: str, error_msg: str or None)
    """
    img_name, source_dir, target_dir, img_size = args
    
    source_path = os.path.join(source_dir, img_name)
    target_path = os.path.join(target_dir, img_name)
    
    try:
        # Load image
        img = Image.open(source_path).convert('RGB')
        
        # Resize image using high-quality LANCZOS resampling
        img_resized = img.resize((img_size, img_size), Image.LANCZOS)
        
        # Save preprocessed image with high quality
        img_resized.save(target_path, 'JPEG', quality=95)
        
        return (True, img_name, None)
        
    except Exception as e:
        return (False, img_name, str(e))


def preprocess_images_parallel(df, source_dir, target_dir, img_size=224, num_processes=8):
    """
    Preprocess all images using parallel processing for faster execution on M4 Pro
    
    Args:
        df: DataFrame containing image information
        source_dir: Source directory with original images
        target_dir: Target directory for preprocessed images
        img_size: Target image size (default: 224)
        num_processes: Number of parallel processes (default: 8)
    """
    print("=" * 80)
    print("IMAGE PREPROCESSING FOR PLANT PATHOLOGY 2021 DATASET")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Source directory: {source_dir}")
    print(f"  Target directory: {target_dir}")
    print(f"  Target size: {img_size}x{img_size}")
    print(f"  Total images: {len(df)}")
    print(f"  CPU cores to use: {num_processes}")
    print(f"  Running on: MacBook with M4 Pro chip")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Check which images need processing
    existing_files = set(os.listdir(target_dir)) if os.path.exists(target_dir) else set()
    images_to_process = [row['image'] for _, row in df.iterrows() 
                         if row['image'] not in existing_files]
    
    if len(images_to_process) == 0:
        print("\n[INFO] All images already preprocessed. Nothing to do.")
        return
    
    already_done = len(df) - len(images_to_process)
    if already_done > 0:
        print(f"\n[INFO] {already_done} images already preprocessed, skipping...")
    
    print(f"\n[INFO] Processing {len(images_to_process)} images...")
    print(f"[INFO] Starting parallel processing with {num_processes} workers...\n")
    
    # Prepare arguments for parallel processing
    process_args = [(img_name, source_dir, target_dir, img_size) 
                    for img_name in images_to_process]
    
    # Start timer
    start_time = time.time()
    
    # Process images in parallel with progress bar
    success_count = 0
    error_count = 0
    errors = []
    
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(preprocess_single_image, process_args),
            total=len(process_args),
            desc="Preprocessing images",
            unit="img"
        ))
    
    # Count results
    for success, img_name, error_msg in results:
        if success:
            success_count += 1
        else:
            error_count += 1
            errors.append((img_name, error_msg))
    
    # End timer
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  Successfully processed: {success_count} images")
    print(f"  Errors: {error_count} images")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Average speed: {success_count/elapsed_time:.2f} images/second")
    print(f"  Total preprocessed: {already_done + success_count}/{len(df)} images")
    
    if errors:
        print(f"\n[WARNING] {len(errors)} images failed to process:")
        for img_name, error_msg in errors[:10]:  # Show first 10 errors
            print(f"  - {img_name}: {error_msg}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    print(f"\n[SUCCESS] Preprocessed images saved to: {target_dir}")
    print(f"\n[NEXT STEP] Upload the '{os.path.basename(target_dir)}' folder to Google Drive")


def verify_preprocessing(df, preprocessed_dir, img_size=224):
    """
    Verify that all images were preprocessed correctly
    
    Args:
        df: DataFrame containing image information
        preprocessed_dir: Directory with preprocessed images
        img_size: Expected image size
    """
    print("\n" + "=" * 80)
    print("VERIFYING PREPROCESSED IMAGES")
    print("=" * 80)
    
    if not os.path.exists(preprocessed_dir):
        print(f"\n[ERROR] Preprocessed directory not found: {preprocessed_dir}")
        return False
    
    preprocessed_files = set(os.listdir(preprocessed_dir))
    expected_files = set(df['image'].tolist())
    
    missing_files = expected_files - preprocessed_files
    extra_files = preprocessed_files - expected_files
    
    print(f"\nVerification results:")
    print(f"  Expected images: {len(expected_files)}")
    print(f"  Found images: {len(preprocessed_files)}")
    print(f"  Missing images: {len(missing_files)}")
    print(f"  Extra images: {len(extra_files)}")
    
    if missing_files:
        print(f"\n[WARNING] {len(missing_files)} images are missing:")
        for img in list(missing_files)[:10]:
            print(f"  - {img}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    # Check image dimensions for a sample
    print(f"\nChecking dimensions of sample images...")
    sample_size = min(10, len(preprocessed_files))
    sample_files = list(preprocessed_files)[:sample_size]
    
    dimension_ok = 0
    dimension_errors = []
    
    for img_name in sample_files:
        img_path = os.path.join(preprocessed_dir, img_name)
        try:
            img = Image.open(img_path)
            if img.size == (img_size, img_size):
                dimension_ok += 1
            else:
                dimension_errors.append((img_name, img.size))
        except Exception as e:
            dimension_errors.append((img_name, f"Error: {str(e)}"))
    
    print(f"  Checked {sample_size} sample images")
    print(f"  Correct dimensions ({img_size}x{img_size}): {dimension_ok}/{sample_size}")
    
    if dimension_errors:
        print(f"\n[WARNING] Dimension issues found:")
        for img_name, issue in dimension_errors:
            print(f"  - {img_name}: {issue}")
    
    # Final verdict
    if len(missing_files) == 0 and len(dimension_errors) == 0:
        print("\n[SUCCESS] All images preprocessed correctly!")
        return True
    else:
        print("\n[WARNING] Some issues found. Please review above.")
        return False


def print_instructions():
    """Print instructions for next steps"""
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Upload preprocessed images to Google Drive:")
    print("   - Locate the 'preprocessed_images' folder on your Mac")
    print("   - Upload it to your Google Drive in the same directory as train_images")
    print("   - Path structure should be:")
    print("     /MyDrive/Plant_Pathology_2021/")
    print("       ├── train.csv")
    print("       ├── train_images/")
    print("       └── preprocessed_images/  <-- Upload this folder")
    
    print("\n2. Run the modified Colab notebook:")
    print("   - The notebook will automatically detect preprocessed images")
    print("   - Training will be much faster without resize operations")
    
    print("\n3. Tips for uploading to Google Drive:")
    print("   - Use Google Drive desktop app for faster upload")
    print("   - Or compress the folder to .zip and upload via web")
    print("   - Make sure all files are uploaded before running Colab")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("PLANT PATHOLOGY 2021 - IMAGE PREPROCESSING SCRIPT")
    print("Optimized for MacBook M4 Pro")
    print("=" * 80)
    
    # Verify paths exist
    print(f"\nVerifying paths...")
    if not os.path.exists(BASE_PATH):
        print(f"\n[ERROR] BASE_PATH not found: {BASE_PATH}")
        print(f"[ACTION] Please update BASE_PATH in the script to your dataset location")
        return
    
    if not os.path.exists(TRAIN_CSV):
        print(f"\n[ERROR] train.csv not found: {TRAIN_CSV}")
        print(f"[ACTION] Please check your dataset location")
        return
    
    if not os.path.exists(TRAIN_IMAGES_PATH):
        print(f"\n[ERROR] train_images folder not found: {TRAIN_IMAGES_PATH}")
        print(f"[ACTION] Please check your dataset location")
        return
    
    print(f"[OK] All paths verified")
    
    # Load CSV
    print(f"\nLoading train.csv...")
    try:
        df = pd.read_csv(TRAIN_CSV)
        print(f"[OK] Loaded {len(df)} image records")
    except Exception as e:
        print(f"\n[ERROR] Failed to load train.csv: {str(e)}")
        return
    
    # Start preprocessing
    print(f"\nStarting preprocessing...")
    print(f"This will take advantage of your M4 Pro's multi-core performance!")
    
    try:
        preprocess_images_parallel(
            df=df,
            source_dir=TRAIN_IMAGES_PATH,
            target_dir=PREPROCESSED_PATH,
            img_size=IMG_SIZE,
            num_processes=NUM_PROCESSES
        )
    except Exception as e:
        print(f"\n[ERROR] Preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Verify preprocessing
    verify_preprocessing(df, PREPROCESSED_PATH, IMG_SIZE)
    
    # Print next steps
    print_instructions()


if __name__ == "__main__":
    # Set start method for macOS compatibility
    mp.set_start_method('spawn', force=True)
    main()
