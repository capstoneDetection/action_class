import os
import cv2
import random


def generate_nofall_clips(root_dir, output_dir, num_clips=1):
    """
    Generate MP4 clips of 30 frames from videos, taken only from before a fall occurs.

    Args:
        root_dir (str): Root directory of the dataset (e.g., 'archive').
        output_dir (str): Directory to save the output clips (e.g., 'processed/nofall/video').
        num_clips (int): Number of 30-frame clips to generate per video (default=1).

    The function traverses the dataset recursively, finds subfolders with 'Annotation_files' and 'Videos',
    and saves clips directly in output_dir without subfolders.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse the root directory to find folders with both 'Annotation_files' and 'Videos'
    for root, dirs, files in os.walk(root_dir):
        # Handle case sensitivity and common naming variations
        annotation_dir_name = 'Annotation_files' if 'Annotation_files' in dirs else 'Annotations_files' if 'Annotations_files' in dirs else None
        videos_dir_name = 'Videos' if 'Videos' in dirs else None

        if annotation_dir_name and videos_dir_name:
            annotation_folder = os.path.join(root, annotation_dir_name)
            videos_folder = os.path.join(root, videos_dir_name)
            location_name = os.path.basename(root)  # e.g., 'Coffee_room_01'

            try:
                # Process each video file in the Videos folder
                for video_file in os.listdir(videos_folder):
                    if video_file.endswith('.avi'):
                        video_name = os.path.splitext(
                            video_file)[0]  # e.g., 'video (1)'
                        try:
                            # Extract video number, e.g., '1' from 'video (1)'
                            video_number = video_name.split(
                                ' (')[1].split(')')[0]
                        except IndexError:
                            print(
                                f"Skipping invalid video file name: {video_file}")
                            continue

                        # Corresponding annotation file
                        annotation_file = os.path.join(
                            annotation_folder, f"{video_name}.txt")

                        # Skip if annotation file is missing
                        if not os.path.exists(annotation_file):
                            print(
                                f"Skipping {video_file}: Annotation file not found")
                            continue

                        # Read start frame from annotation
                        try:
                            with open(annotation_file, 'r', encoding='ISO-8859-1') as f:
                                lines = f.readlines()
                                if len(lines) < 1:
                                    print(
                                        f"Skipping {video_file}: Invalid annotation file")
                                    continue
                                # Start of fall
                                start_frame = int(lines[0].strip())
                        except (ValueError, Exception) as e:
                            print(
                                f"Skipping {video_file}: Error reading annotation ({e})")
                            continue

                        # Skip if fall starts before frame 30
                        if start_frame < 30:
                            print(
                                f"Skipping {video_file}: Fall starts too early (frame {start_frame})")
                            continue

                        video_path = os.path.join(videos_folder, video_file)
                        cap = cv2.VideoCapture(video_path)

                        # Check if video opened successfully
                        if not cap.isOpened():
                            print(
                                f"Skipping {video_file}: Could not open video")
                            continue

                        # Get video properties
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # Skip if video is too short
                        if total_frames < start_frame:
                            print(
                                f"Skipping {video_file}: Not enough frames (total={total_frames}, start={start_frame})")
                            cap.release()
                            continue

                        # Generate specified number of clips
                        for clip_num in range(num_clips):
                            # Randomly select start frame for a 30-frame clip (before the fall)
                            random_start = random.randint(0, start_frame - 30)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, random_start)

                            # Define output video file with unique name
                            output_video_name = f"{location_name}_video_{video_number}_nofall_clip{clip_num+1}.mp4"
                            output_video_path = os.path.join(
                                output_dir, output_video_name)
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(
                                output_video_path, fourcc, fps, (width, height))

                            # Write 30 frames to output
                            for _ in range(30):
                                ret, frame = cap.read()
                                if ret:
                                    writer.write(frame)
                                else:
                                    print(
                                        f"Warning: Could not read 30 frames from {video_file} for clip {clip_num+1}")
                                    break
                            writer.release()
                            print(f"Saved clip: {output_video_path}")

                        cap.release()

            except Exception as e:
                print(f"Error processing {location_name}: {e}")


# Example usage
if __name__ == "__main__":
    root_dir = "../archive"
    output_dir = "../processed/nofall/video"
    num_clips = 1
    generate_nofall_clips(root_dir, output_dir, num_clips)
    print("Processing complete!")
