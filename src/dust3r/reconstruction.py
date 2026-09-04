from pathlib import Path

import torch

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


def reconstruct(
    frames_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    scene_graph:str="swin-2",
    niter:int=300,
    lr:float=0.01
):
    frames_dir = Path(frames_dir)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Frames:", frames_dir)
    print("Checkpoint:", checkpoint_path)

    # --------------------------------------------------
    # 1. Load DUSt3R model
    # --------------------------------------------------

    from dust3r.model import AsymmetricCroCo3DStereo

    print("\nLoading DUSt3R model...")

    model = AsymmetricCroCo3DStereo.from_pretrained(
        str(checkpoint_path)
    ).to(device)

    print("Model loaded.")

    # --------------------------------------------------
    # 2. Load images
    # --------------------------------------------------

    image_paths = sorted(frames_dir.glob("*.jpg"))

    if len(image_paths) < 2:
        raise RuntimeError("At least 2 images are required.")

    print(f"\nLoading {len(image_paths)} images...")

    images = load_images(
        [str(path) for path in image_paths],
        size=512,
    )

    print("Images loaded.")

    # --------------------------------------------------
    # 3. Create image pairs
    # --------------------------------------------------

    print("\nCreating image pairs...")

    pairs = make_pairs(
        images,
        scene_graph=scene_graph,
        prefilter=None,
        symmetrize=True,
    )

    print(f"Number of pairs: {len(pairs)}")

    # --------------------------------------------------
    # 4. DUSt3R inference
    # --------------------------------------------------

    print("\nRunning DUSt3R inference...")

    output = inference(
        pairs,
        model,
        device,
        batch_size=1,
    )

    print("Inference completed.")

    # --------------------------------------------------
    # 5. Global alignment
    # --------------------------------------------------

    print("\nRunning global alignment...")

    scene = global_aligner(
        output,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
    )

    loss = scene.compute_global_alignment(
        init="mst",
        niter=niter,
        schedule="cosine",
        lr=lr,
    )

    print("Global alignment completed.")
    print("Final loss:", loss)

    # --------------------------------------------------
    # 6. Get point cloud
    # --------------------------------------------------

    print("\nExtracting point cloud...")

    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # --------------------------------------------------
    # 7. Save point cloud
    # --------------------------------------------------

    import numpy as np
    import open3d as o3d

    points = []
    colors = []

    for i, (pts, mask) in enumerate(zip(pts3d, confidence_masks)):

        pts = pts.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        valid_points = pts[mask]

        # DUSt3R images contain RGB information
        image = images[i]["img"]

        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()

        image = image.squeeze(0)

        # CHW -> HWC
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        image = np.clip(image, 0, 1)

        valid_colors = image[mask]

        points.append(valid_points)
        colors.append(valid_colors)

    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)

    print("Point count:", len(points))

    point_cloud = o3d.geometry.PointCloud()

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    output_path = output_dir / "point_cloud.ply"

    o3d.io.write_point_cloud(
        str(output_path),
        point_cloud,
    )

    print("\nPoint cloud saved:")
    print(output_path)

    return output_path