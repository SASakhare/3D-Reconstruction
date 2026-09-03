from pathlib import Path
import urllib.request
import time


MODELS = {
    "512_dpt": {
        "filename": "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "url": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
    },

    "512_linear": {
        "filename": "DUSt3R_ViTLarge_BaseDecoder_512_linear.pth",
        "url": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth",
    },

    "224_linear": {
        "filename": "DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
        "url": "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth",
    },
}


def download_progress(block_num, block_size, total_size):
    """
    Display download progress.
    """

    if total_size <= 0:
        return

    downloaded = min(
        block_num * block_size,
        total_size,
    )

    percent = (
        downloaded / total_size
    ) * 100

    downloaded_mb = (
        downloaded / (1024 * 1024)
    )

    total_mb = (
        total_size / (1024 * 1024)
    )

    # First callback
    if block_num == 0:
        download_progress.start_time = time.time()

    elapsed = (
        time.time()
        - download_progress.start_time
    )

    if elapsed > 0:
        speed = downloaded / elapsed
    else:
        speed = 0

    speed_mb = (
        speed / (1024 * 1024)
    )

    if speed > 0:
        remaining = (
            total_size - downloaded
        )

        eta_seconds = (
            remaining / speed
        )
    else:
        eta_seconds = 0

    # Progress bar
    bar_length = 30

    filled = int(
        bar_length * percent / 100
    )

    bar = (
        "█" * filled
        + "░" * (bar_length - filled)
    )

    # ETA
    eta_seconds = int(eta_seconds)

    minutes, seconds = divmod(
        eta_seconds,
        60
    )

    if minutes > 0:
        eta = f"{minutes}m {seconds}s"
    else:
        eta = f"{seconds}s"

    print(
        f"\r{bar} "
        f"{percent:6.2f}% | "
        f"{downloaded_mb:.1f}/{total_mb:.1f} MB | "
        f"{speed_mb:.2f} MB/s | "
        f"ETA: {eta}",
        end="",
        flush=True,
    )

    if downloaded >= total_size:
        print()


def get_checkpoint(
    model_name: str,
    checkpoint_dir: str | Path,
) -> Path:

    # --------------------------------------------------
    # Validate model
    # --------------------------------------------------

    if model_name not in MODELS:

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODELS.keys())}"
        )

    # --------------------------------------------------
    # Create checkpoint directory
    # --------------------------------------------------

    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = MODELS[model_name]

    checkpoint_path = (
        checkpoint_dir
        / model["filename"]
    )

    # --------------------------------------------------
    # Check existing checkpoint
    # --------------------------------------------------

    if checkpoint_path.exists():

        print(
            "Checkpoint already exists:"
        )

        print(checkpoint_path)

        return checkpoint_path

    # --------------------------------------------------
    # Download
    # --------------------------------------------------

    print()
    print(
        f"Downloading DUSt3R model: "
        f"{model_name}"
    )

    print(
        f"Destination: {checkpoint_path}"
    )

    print()

    urllib.request.urlretrieve(
        model["url"],
        str(checkpoint_path),
        reporthook=download_progress,
    )

    print()
    print("Download completed.")
    print(
        f"Checkpoint: {checkpoint_path}"
    )

    return checkpoint_path