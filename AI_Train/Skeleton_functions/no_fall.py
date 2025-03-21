import os
import cv2
import random


def generate_nofall_clips(root_dir, output_dir, num_clips=1):
    """
    Generate MP4 clips of 30 frames from videos, taken only from before a fall occurs.

    :param root_dir: Root directory of the dataset (e.g., "FallDataset")
    :param output_dir: Directory to save the output clips (e.g., "NoFallClips")
    :param num_clips: Number of 30-frame clips to generate per video (default=1)
    """
    # Create output directory if it doesn’t exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Traverse each location folder in root_dir
    for location in os.listdir(root_dir):
        location_path = os.path.join(root_dir, location)
        if os.path.isdir(location_path):
            videos_path = os.path.join(location_path, "Videos")
            annotations_path = os.path.join(location_path, "Annotation_files")

            # Create corresponding output subfolder (same structure as input)
            output_location_path = os.path.join(output_dir, location)
            if not os.path.exists(output_location_path):
                os.makedirs(output_location_path)
            try:
                # Process each video file
                for video_file in os.listdir(videos_path):
                    if video_file.endswith(".avi"):
                        video_name = os.path.splitext(
                            video_file)[0]  # e.g., "video (1)"
                        annotation_file = os.path.join(
                            annotations_path, video_name + ".txt")

                        # Skip if annotation file is missing
                        if not os.path.exists(annotation_file):
                            print(
                                f"Skipping {video_file}: Annotation file not found")
                            continue

                        # Read start frame from annotation
                        with open(annotation_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) < 1:
                                print(
                                    f"Skipping {video_file}: Invalid annotation file")
                                continue
                            try:
                                # Start of fall
                                start_frame = int(lines[0].strip())
                            except ValueError:
                                print(
                                    f"Skipping {video_file}: Invalid frame number in annotation")
                                continue

                        # Skip if fall starts before frame 30
                        if start_frame < 30:
                            print(
                                f"Skipping {video_file}: Fall starts too early (frame {start_frame})")
                            continue

                        video_path = os.path.join(videos_path, video_file)
                        cap = cv2.VideoCapture(video_path)

                        # Get video properties
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        # Skip if video is too short
                        if total_frames < start_frame:
                            print(
                                f"Skipping {video_file}: Not enough frames (total={total_frames}, start={start_frame})")
                            continue

                        # Generate specified number of clips
                        for clip_num in range(num_clips):
                            # Randomly select start frame for a 30-frame clip (before the fall)
                            random_start = random.randint(0, start_frame - 30)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, random_start)

                            # Define output video file with the same base name plus a clip number
                            output_video_name = f"{video_name}_nofall_clip{clip_num+1}.mp4"
                            output_video_path = os.path.join(
                                output_location_path, output_video_name)
                            fourcc = cv2.VideoWriter_fourcc(
                                *'mp4v')  # Use 'mp4v' codec
                            writer = cv2.VideoWriter(
                                output_video_path, fourcc, fps, (width, height))

                            # Write 30 frames to output (only from before the fall)
                            for _ in range(30):
                                ret, frame = cap.read()
                                if ret:
                                    writer.write(frame)
                                else:
                                    print(
                                        f"Warning: Could not read 30 frames from {video_file} for clip {clip_num+1}")
                                    break
                            writer.release()

                        cap.release()
            except Exception as e:
                print(f"Error processing {location}: {e}")
