from pathlib import Path
import cv2
import json


def extract_uniform_frames(
    video_path: str,
    output_dir: str,
    num_frames: int,
):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Could not determine video frame count.")

    if num_frames > total_frames:
        num_frames = total_frames

    frame_indices = [
        round(i * (total_frames - 1) / (num_frames - 1))
        for i in range(num_frames)
    ]

    saved_frames = []

    for output_index, frame_index in enumerate(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if not success:
            print(f"Warning: could not read frame {frame_index}")
            continue

        frame_name = f"frame_{output_index:04d}.jpg"
        frame_path = output_dir / frame_name

        cv2.imwrite(str(frame_path), frame)

        saved_frames.append({
            "output_frame": frame_name,
            "source_frame_index": frame_index,
            "timestamp_seconds": frame_index / fps if fps > 0 else None
        })

    cap.release()

    dataset_info = {
        "source_video": str(video_path),
        "original_frame_count": total_frames,
        "original_fps": fps,
        "selected_frame_count": len(saved_frames),
        "selection_method": "uniform",
        "frames": saved_frames,
    }

    info_path = output_dir.parent / "dataset_info.json"

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=4)

    print(f"Video: {video_path}")
    print(f"Original frames: {total_frames}")
    print(f"Original FPS: {fps:.2f}")
    print(f"Selected frames: {len(saved_frames)}")
    print(f"Saved to: {output_dir}")
    print(f"Metadata: {info_path}")