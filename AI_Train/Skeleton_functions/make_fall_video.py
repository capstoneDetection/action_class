import os
import cv2
from tqdm import tqdm


def generate_fall_clips(root_dir, output_dir, num_clips_per_vid=1):
    if num_clips_per_vid != 1:
        print("Warning: num_clips_per_vid > 1 is not supported yet. Extracting one clip per video.")

    os.makedirs(output_dir, exist_ok=True)

    for root, dirs, files in os.walk(root_dir):
        annotation_dir_name = next(
            (d for d in ['Annotation_files', 'Annotations_files'] if d in dirs), None)
        videos_dir_name = 'Videos' if 'Videos' in dirs else None

        if annotation_dir_name and videos_dir_name:
            annotation_folder = os.path.join(root, annotation_dir_name)
            video_folder = os.path.join(root, videos_dir_name)
            location_name = os.path.basename(root)

            annotation_files = [f for f in os.listdir(
                annotation_folder) if f.endswith('.txt')]

            for annotation_file in tqdm(annotation_files, desc=f"Processing {location_name}"):
                try:
                    video_number = annotation_file.split(
                        ' (')[1].split(').txt')[0]
                except IndexError:
                    print(
                        f"Skipping invalid annotation file: {annotation_file}")
                    continue

                video_file_name = f"video ({video_number}).avi"
                video_path = os.path.join(video_folder, video_file_name)
                groundtruth_file = os.path.join(
                    annotation_folder, annotation_file)

                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error opening video file: {video_path}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                try:
                    with open(groundtruth_file, 'r', encoding='ISO-8859-1') as file:
                        lines = file.readlines()
                        start_frame = int(lines[0].strip())
                        end_frame = int(lines[1].strip())
                except Exception as e:
                    print(
                        f"Error reading groundtruth file {groundtruth_file}: {e}")
                    cap.release()
                    continue

                if start_frame >= end_frame:
                    # print(f"Skipping {groundtruth_file} since start_frame >= end_frame")
                    cap.release()
                    continue

                # Calculate middle frame
                middle = (start_frame + end_frame) // 2

                # Determine segment boundaries
                segment_start = middle - 15
                segment_end = segment_start + 30

                # Clamp to video boundaries
                if segment_start < 0:
                    segment_start = 0
                    segment_end = 30
                if segment_end > total_frames:
                    segment_end = total_frames
                    segment_start = segment_end - 30
                    if segment_start < 0:
                        segment_start = 0

                # Ensure exactly 30 frames
                if (segment_end - segment_start) != 30:
                    print(
                        f"Skipping {groundtruth_file} due to insufficient frames for 30-frame clip")
                    cap.release()
                    continue

                # Proceed to write the clip
                output_file_name = f"{location_name}_video_{video_number}.mp4"
                output_video_path = os.path.join(output_dir, output_file_name)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(
                    output_video_path, fourcc, fps, (frame_width, frame_height))

                cap.set(cv2.CAP_PROP_POS_FRAMES, segment_start)
                for _ in range(segment_end - segment_start):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error reading frames for {video_path}")
                        break
                    out.write(frame)

                cap.release()
                out.release()
                # print(f"Video segment saved to {output_video_path}")


if __name__ == "__main__":
    generate_fall_clips(root_dir='../archive',
                        output_dir='../processed/fall/video', num_clips_per_vid=1)
