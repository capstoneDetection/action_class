import os
import cv2


def validate_video(filepath):
    """
    Validates if a video file can be opened using OpenCV.

    Args:
        filepath (str): The full path to the video file.

    Returns:
        bool: True if the video can be opened, False otherwise.
    """
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return False
        # Optionally, you can read a frame to further check validity
        ret, frame = cap.read()
        cap.release()
        return ret  # Returns True if a frame could be read, False otherwise
    except Exception as e:
        print(f"Error opening video file '{filepath}': {e}")
        return False


def validate_videos_in_folder(root_folder):
    """
    Recursively validates all video files found within a folder and its subfolders.

    Args:
        root_folder (str): The path to the root folder to start the validation from.
    """
    video_extensions = ['.mp4', '.avi', '.mov',
                        '.mkv', '.wmv', '.flv']  # Add more if needed
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if any(filename.lower().endswith(ext) for ext in video_extensions):
                filepath = os.path.join(dirpath, filename)
                print(f"Validating: {filepath}")
                if validate_video(filepath):
                    # print(f"  - Valid")
                    pass
                else:
                    print(f"  - Invalid or could not be opened")


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing the videos: ")
    if os.path.isdir(folder_path):
        validate_videos_in_folder(folder_path)
    else:
        print(f"Error: Folder '{folder_path}' does not exist.")
