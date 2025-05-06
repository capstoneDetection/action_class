import os
import sys

# ----------------------------------------------------------------------------
# 1) Suppress OpenCV’s internal logs at the Python level
# ----------------------------------------------------------------------------
import cv2
if hasattr(cv2, 'utils') and hasattr(cv2.utils, 'logging'):
    # Available since OpenCV-4.4+
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
else:
    # Older OpenCV: try the global cv2.setLogLevel
    try:
        cv2.setLogLevel(0)  # 0 == QUIET
    except:
        pass

# ----------------------------------------------------------------------------
# 2) (Optional) also tell OpenCV at import-time via an environment variable
# ----------------------------------------------------------------------------
# Put this *before* importing cv2 if you prefer -- shown here for clarity
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')

# ----------------------------------------------------------------------------
# Video validation
# ----------------------------------------------------------------------------


def validate_video(filepath):
    """
    Tries to open + read one frame. Returns True if successful.
    All ffmpeg/OpenCV warnings are silenced.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        cap.release()
        return False

    ret, _ = cap.read()
    cap.release()
    return bool(ret)


def validate_videos_in_folder(root_folder):
    """
    Walks folder recursively, validating each video file.
    Returns (total, ok, corrupted).
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    total = ok = bad = 0

    for dirpath, _, filenames in os.walk(root_folder):
        for fn in filenames:
            if any(fn.lower().endswith(ext) for ext in video_extensions):
                total += 1
                path = os.path.join(dirpath, fn)
                if validate_video(path):
                    ok += 1
                else:
                    print(f"- Invalid or could not be opened: {path}")
                    bad += 1

    return total, ok, bad


if __name__ == "__main__":
    folder = input(
        "Enter the path to the folder containing the videos: ").strip()
    if not os.path.isdir(folder):
        print(f"Error: Folder '{folder}' does not exist.")
        sys.exit(1)

    t, o, b = validate_videos_in_folder(folder)
    print(f"Total videos:     {t}")
    print(f"OK videos:        {o}")
    print(f"Corrupted/missing:{b}")
