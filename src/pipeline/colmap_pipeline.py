from pathlib import Path
import subprocess
import shutil
import json
import time

import open3d as o3d

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Experiment
    "experiment_id": 2,
    "experiment_name": "25fps_baseline",
    # Paths
    "colmap_path": r"C:\Users\sejal\Downloads\colmap-x64-windows-cuda\bin\colmap.exe",
    "frames_path": "../data/frames",
    # COLMAP settings
    "use_gpu": True,
    "gpu_index": 0,
    # Matching
    # "matcher": "sequential",
    "matcher": "exhaustive",
    # "overlap": 20,
    # Dense reconstruction
    "geom_consistency": True,
    # Visualization
    "visualize": True,
}


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FRAMES_PATH = PROJECT_ROOT / CONFIG["frames_path"]

RESULTS_ROOT = PROJECT_ROOT / "results" / "colmap"

EXP_ID = CONFIG["experiment_id"]
EXP_NAME = CONFIG["experiment_name"]

EXPERIMENT_PATH = RESULTS_ROOT / f"exp_{EXP_ID:02d}_{EXP_NAME}"

WORKSPACE_PATH = EXPERIMENT_PATH / "workspace"

DATABASE_PATH = WORKSPACE_PATH / "database.db"
SPARSE_PATH = WORKSPACE_PATH / "sparse"
DENSE_PATH = WORKSPACE_PATH / "dense"

FINAL_SPARSE_PLY = EXPERIMENT_PATH / "sparse.ply"
FINAL_DENSE_PLY = EXPERIMENT_PATH / "dense.ply"

METADATA_PATH = EXPERIMENT_PATH / "metadata.json"


COLMAP = Path(CONFIG["colmap_path"])


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def run_command(command, description):
    """Run a COLMAP command and stop if it fails."""

    print()
    print("=" * 70)
    print(description)
    print("=" * 70)

    print("Command:")
    print(" ".join(f'"{x}"' if " " in str(x) else str(x) for x in command))

    start = time.time()

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    elapsed = time.time() - start

    print(result.stdout)

    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed " f"with exit code {result.returncode}"
        )

    print(f"\nCompleted in {elapsed:.2f} seconds.")

    return result.stdout


def count_frames():
    """Count input images."""

    frames = sorted(FRAMES_PATH.glob("*.jpg"))

    if not frames:
        raise RuntimeError(f"No JPG frames found in {FRAMES_PATH}")

    return len(frames)


# ============================================================
# STEP 1: FEATURE EXTRACTION
# ============================================================


def feature_extraction():

    command = [
        str(COLMAP),
        "feature_extractor",
        "--database_path",
        str(DATABASE_PATH),
        "--image_path",
        str(FRAMES_PATH),
        "--FeatureExtraction.use_gpu",
        "1" if CONFIG["use_gpu"] else "0",
        "--FeatureExtraction.gpu_index",
        str(CONFIG["gpu_index"]),
    ]

    run_command(command, "STEP 1 — Feature Extraction")


# ============================================================
# STEP 2: FEATURE MATCHING
# ============================================================


def feature_matching():

    command = [
        str(COLMAP),
        f"{CONFIG['matcher']}_matcher",
        "--database_path",
        str(DATABASE_PATH),
        "--FeatureMatching.use_gpu",
        "1" if CONFIG["use_gpu"] else "0",
        "--FeatureMatching.gpu_index",
        str(CONFIG["gpu_index"]),
    ]

    # Sequential matcher specific option
    if CONFIG["matcher"] == "sequential":

        command += [
            "--SequentialMatching.overlap",
            str(CONFIG["overlap"]),
        ]

    run_command(command, "STEP 2 — Feature Matching")


# ============================================================
# STEP 3: SPARSE RECONSTRUCTION
# ============================================================


def sparse_reconstruction():

    SPARSE_PATH.mkdir(parents=True, exist_ok=True)

    command = [
        str(COLMAP),
        "mapper",
        "--database_path",
        str(DATABASE_PATH),
        "--image_path",
        str(FRAMES_PATH),
        "--output_path",
        str(SPARSE_PATH),
    ]

    run_command(command, "STEP 3 — Sparse Reconstruction / SfM")


# ============================================================
# STEP 4: CONVERT MODEL TO TEXT
# ============================================================


def get_model_stats(model_id):

    model_path = SPARSE_PATH / str(model_id)

    txt_path = WORKSPACE_PATH / "sparse_txt" / str(model_id)

    txt_path.mkdir(parents=True, exist_ok=True)

    command = [
        str(COLMAP),
        "model_converter",
        "--input_path",
        str(model_path),
        "--output_path",
        str(txt_path),
        "--output_type",
        "TXT",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        return None

    images_file = txt_path / "images.txt"
    points_file = txt_path / "points3D.txt"

    if not images_file.exists():
        return None

    # images.txt stores:
    #
    # image information
    # 2D point information
    #
    # for every registered image.

    with open(images_file, "r") as f:
        image_lines = [line for line in f if line.strip() and not line.startswith("#")]

    registered_images = len(image_lines) // 2

    num_points = 0

    if points_file.exists():

        with open(points_file, "r") as f:
            point_lines = [
                line for line in f if line.strip() and not line.startswith("#")
            ]

        num_points = len(point_lines)

    return {
        "model_id": model_id,
        "registered_images": registered_images,
        "points": num_points,
    }


# ============================================================
# STEP 5: SELECT BEST MODEL
# ============================================================


def select_best_model():

    models = []

    for model_path in sorted(SPARSE_PATH.iterdir()):

        if not model_path.is_dir():
            continue

        if not model_path.name.isdigit():
            continue

        model_id = int(model_path.name)

        stats = get_model_stats(model_id)

        if stats:
            models.append(stats)

    if not models:
        raise RuntimeError("COLMAP did not produce any valid sparse models.")

    print()
    print("=" * 70)
    print("SPARSE MODEL COMPARISON")
    print("=" * 70)

    for model in models:

        print(
            f"Model {model['model_id']}: "
            f"{model['registered_images']} images, "
            f"{model['points']} 3D points"
        )

    # Primary criterion:
    # number of registered images.
    #
    # Secondary criterion:
    # number of 3D points.

    best = max(models, key=lambda x: (x["registered_images"], x["points"]))

    print()
    print(f"BEST MODEL: {best['model_id']}")

    print(f"Registered images: " f"{best['registered_images']}")

    print(f"3D points: " f"{best['points']}")

    return best


# ============================================================
# STEP 6: EXPORT SPARSE PLY
# ============================================================


def export_sparse_ply(best_model_id):

    model_path = SPARSE_PATH / str(best_model_id)

    command = [
        str(COLMAP),
        "model_converter",
        "--input_path",
        str(model_path),
        "--output_path",
        str(FINAL_SPARSE_PLY),
        "--output_type",
        "PLY",
    ]

    run_command(command, "STEP 4 — Export Sparse Point Cloud")


# ============================================================
# STEP 7: IMAGE UNDISTORTION
# ============================================================


def image_undistortion(best_model_id):

    model_path = SPARSE_PATH / str(best_model_id)

    DENSE_PATH.mkdir(parents=True, exist_ok=True)

    command = [
        str(COLMAP),
        "image_undistorter",
        "--image_path",
        str(FRAMES_PATH),
        "--input_path",
        str(model_path),
        "--output_path",
        str(DENSE_PATH),
        "--output_type",
        "COLMAP",
    ]

    run_command(command, "STEP 5 — Image Undistortion")


# ============================================================
# STEP 8: PATCHMATCH STEREO
# ============================================================


def patch_match_stereo():

    command = [
        str(COLMAP),
        "patch_match_stereo",
        "--workspace_path",
        str(DENSE_PATH),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.gpu_index",
        str(CONFIG["gpu_index"]),
        "--PatchMatchStereo.geom_consistency",
        "true" if CONFIG["geom_consistency"] else "false",
    ]

    run_command(command, "STEP 6 — PatchMatch Stereo")


# ============================================================
# STEP 9: STEREO FUSION
# ============================================================


def stereo_fusion():

    command = [
        str(COLMAP),
        "stereo_fusion",
        "--workspace_path",
        str(DENSE_PATH),
        "--workspace_format",
        "COLMAP",
        "--output_path",
        str(FINAL_DENSE_PLY),
    ]

    run_command(command, "STEP 7 — Stereo Fusion")


# ============================================================
# STEP 10: SAVE METADATA
# ============================================================


def save_metadata(
    frame_count,
    best_model,
    elapsed_time,
):

    metadata = {
        "experiment_id": EXP_ID,
        "experiment_name": EXP_NAME,
        "method": "COLMAP",
        "frames": frame_count,
        "matcher": CONFIG["matcher"],
        "use_gpu": CONFIG["use_gpu"],
        "gpu_index": CONFIG["gpu_index"],
        "geom_consistency": CONFIG["geom_consistency"],
        "best_model_id": best_model["model_id"],
        "registered_images": best_model["registered_images"],
        "sparse_points": best_model["points"],
        "sparse_ply": str(FINAL_SPARSE_PLY),
        "dense_ply": str(FINAL_DENSE_PLY),
        "runtime_seconds": round(elapsed_time, 2),
    }
    # "sequential_overlap": CONFIG["overlap"],

    if "overlap" in CONFIG.keys():
        metadata["sequential_overlap"] = CONFIG["overlap"]

    with open(METADATA_PATH, "w") as f:

        json.dump(metadata, f, indent=4)

    print()
    print("Metadata saved:")
    print(METADATA_PATH)


# ============================================================
# STEP 11: VISUALIZE
# ============================================================


def visualize_dense():

    if not FINAL_DENSE_PLY.exists():

        print("Dense PLY does not exist. " "Skipping visualization.")

        return

    print()
    print("=" * 70)
    print("OPEN3D VISUALIZATION")
    print("=" * 70)

    pcd = o3d.io.read_point_cloud(str(FINAL_DENSE_PLY))

    print(pcd)

    print("Dense points:", len(pcd.points))

    if len(pcd.points) == 0:

        print("Point cloud is empty.")

        return

    o3d.visualization.draw_geometries([pcd], window_name=f"COLMAP - {EXP_NAME}")


# ============================================================
# MAIN PIPELINE
# ============================================================


def main():

    start_time = time.time()

    print()
    print("=" * 70)
    print("COLMAP 3D RECONSTRUCTION")
    print("=" * 70)

    print("Experiment:", EXP_NAME)
    print("Frames:", FRAMES_PATH)
    print("Results:", EXPERIMENT_PATH)

    # --------------------------------------------------------
    # Validate COLMAP
    # --------------------------------------------------------

    if not COLMAP.exists():

        raise FileNotFoundError(f"COLMAP executable not found:\n{COLMAP}")

    # --------------------------------------------------------
    # Validate frames
    # --------------------------------------------------------

    frame_count = count_frames()

    print(f"Input frames: {frame_count}")

    # --------------------------------------------------------
    # Create experiment directory
    # --------------------------------------------------------

    EXPERIMENT_PATH.mkdir(parents=True, exist_ok=True)

    WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    if DATABASE_PATH.exists():

        raise RuntimeError(
            f"Experiment already exists:\n"
            f"{EXPERIMENT_PATH}\n\n"
            f"Choose a new experiment_id/name."
        )

    # --------------------------------------------------------
    # Run COLMAP
    # --------------------------------------------------------

    feature_extraction()

    feature_matching()

    sparse_reconstruction()

    # --------------------------------------------------------
    # Select best sparse model
    # --------------------------------------------------------

    best_model = select_best_model()

    best_model_id = best_model["model_id"]

    # --------------------------------------------------------
    # Export sparse point cloud
    # --------------------------------------------------------

    export_sparse_ply(best_model_id)

    # --------------------------------------------------------
    # Dense reconstruction
    # --------------------------------------------------------

    image_undistortion(best_model_id)

    patch_match_stereo()

    stereo_fusion()

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    elapsed_time = time.time() - start_time

    save_metadata(frame_count, best_model, elapsed_time)

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RECONSTRUCTION COMPLETED")
    print("=" * 70)

    print("Sparse:")
    print(FINAL_SPARSE_PLY)

    print()

    print("Dense:")
    print(FINAL_DENSE_PLY)

    print()

    print(f"Dense PLY exists: " f"{FINAL_DENSE_PLY.exists()}")

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    if CONFIG["visualize"]:

        visualize_dense()


if __name__ == "__main__":
    main()
