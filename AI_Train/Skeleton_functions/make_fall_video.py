import os
import cv2

def generate_fall_clips(root_dir, output_dir, num_clips_per_vid=1):
    """
    Generate fall clips from videos in the dataset based on annotation files.

    Args:
        root_dir (str): Root directory of the dataset (e.g., 'archive').
        output_dir (str): Directory where extracted clips will be saved (e.g., 'processed/fall/video').
        num_clips_per_vid (int): Number of clips to extract per video. Currently supports only 1.

    The function traverses the dataset, finds subfolders with 'Annotation_files' and 'Videos',
    and extracts video segments based on start and end frames from annotation files.
    """
    # Check if num_clips_per_vid > 1 is requested (not supported yet)
    if num_clips_per_vid != 1:
        print("Warning: num_clips_per_vid > 1 is not supported yet. Extracting one clip per video.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Traverse the root directory to find folders with both 'Annotation_files' and 'Videos'
    for root, dirs, files in os.walk(root_dir):
        # Handle case sensitivity and common naming variations
        annotation_dir_name = 'Annotation_files' if 'Annotation_files' in dirs else 'Annotations_files' if 'Annotations_files' in dirs else None
        videos_dir_name = 'Videos' if 'Videos' in dirs else None

        if annotation_dir_name and videos_dir_name:
            annotation_folder = os.path.join(root, annotation_dir_name)
            video_folder = os.path.join(root, videos_dir_name)
            location_name = os.path.basename(root)  # e.g., 'Coffee_room_01'

            # List all .txt files in the annotation folder
            annotation_files = [f for f in os.listdir(annotation_folder) if f.endswith('.txt')]

            for annotation_file in annotation_files:
                # Extract video number from annotation filename, e.g., '1' from 'video (1).txt'
                try:
                    video_number = annotation_file.split(' (')[1].split(').txt')[0]
                except IndexError:
                    print(f"Skipping invalid annotation file: {annotation_file}")
                    continue

                # Construct corresponding video filename
                video_file_name = f"video ({video_number}).avi"
                video_path = os.path.join(video_folder, video_file_name)
                groundtruth_file = os.path.join(annotation_folder, annotation_file)

                # Check if the video file exists
                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue

                # Define output filename with location and video number for uniqueness
                output_file_name = f"{location_name}_video_{video_number}.mp4"
                output_video_path = os.path.join(output_dir, output_file_name)

                # Open the video file
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error opening video file: {video_path}")
                    continue

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Read start and end frames from the annotation file
                try:
                    with open(groundtruth_file, 'r', encoding='ISO-8859-1') as file:
                        lines = file.readlines()
                        start_frame = int(lines[0].strip())
                        end_frame = int(lines[1].strip())
                except Exception as e:
                    print(f"Error reading groundtruth file {groundtruth_file}: {e}")
                    cap.release()
                    continue

                # Define the segment to extract (30 frames starting 15 frames before the fall)
                segment_start = max(0, start_frame - 15)
                segment_end = min(total_frames, segment_start + 30)

                # Create video writer for the output clip
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

                # Extract and write the frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start)
                for i in range(segment_start, segment_end):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error reading frame {i} from {video_path}")
                        break
                    out.write(frame)

                # Release resources
                cap.release()
                out.release()
                print(f"Video segment saved to {output_video_path}")

# Example usage
if __name__ == "__main__":
    generate_fall_clips(root_dir='../archive', output_dir='../processed/fall/video', num_clips_per_vid=1)