import cv2
from pathlib import Path


def extract_frames(video_path, save_location, fps):
    """
    Extract frames from a video at a specified FPS.

    Args:
        video_path: Path to the input video.
        save_location: Folder where extracted frames will be saved.
        fps: Number of frames to extract per second.
    """

    video_path = Path(video_path)
    save_location = Path(save_location)

    save_location.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > original_fps:
        raise ValueError(
            f"fps must be between 0 and {original_fps:.2f}"
        )

    frame_interval = original_fps / fps

    frame_number = 0
    saved_frame_number = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # Save frame at the required FPS
        if frame_number >= saved_frame_number * frame_interval:

            frame_path = save_location / f"frame_{saved_frame_number:06d}.jpg"

            cv2.imwrite(str(frame_path), frame)

            saved_frame_number += 1

        frame_number += 1

    cap.release()

    print(f"Original FPS : {original_fps:.2f}")
    print(f"Extraction FPS: {fps}")
    print(f"Frames saved : {saved_frame_number}")
    print(f"Saved to     : {save_location}")