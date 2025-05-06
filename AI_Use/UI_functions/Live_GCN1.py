import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# Set device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the ST-GCN model class (unchanged)


class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_joints=18, num_classes=2):
        super(STGCN, self).__init__()
        self.num_joints = num_joints
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, num_joints))
        self.fc = nn.Linear(128 * num_joints, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the pre-trained model (unchanged)
# model_path = "../AI_Train/Models/stgcn_fall_detection.pth"


model_path = "../../AI_Train/Models/stgcn_fall_detection.pth"
try:
    model = STGCN(in_channels=3, num_joints=18, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize MediaPipe Pose (unchanged)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the 18 landmarks in the same order as used during training (unchanged)
landmark_order = [
    "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
    "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip", "Right Knee",
    "Right Ankle", "Left Hip", "Left Knee", "Left Ankle", "Right Eye",
    "Left Eye", "Right Ear", "Left Ear"
]

landmark_indices = {
    "Nose": 0, "Right Shoulder": 12, "Right Elbow": 14, "Right Wrist": 16,
    "Left Shoulder": 11, "Left Elbow": 13, "Left Wrist": 15, "Right Hip": 24,
    "Right Knee": 26, "Right Ankle": 28, "Left Hip": 23, "Left Knee": 25,
    "Left Ankle": 27, "Right Eye": 5, "Left Eye": 2, "Right Ear": 8, "Left Ear": 7
}

# Initialize buffer to store skeleton data and probabilities for graph
buffer = []
prob_buffer = []  # Store fall probabilities for plotting
window_size = 30  # Number of frames per window
step_size = 1     # Step size for sliding window
threshold = 0.8   # Initial threshold for fall detection
frame_count = 0   # Track frames for time calculation
fps = 30          # Assumed FPS for time axis (adjust if known)

# Graph settings
graph_height = 100  # Height of graph area in pixels
graph_width = 400   # Width of graph area in pixels
max_points = 100    # Number of probability points to display (adjustable)
camera_id = 1
# Start capturing video from the default camera
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Get frame dimensions (to position graph at bottom)
_, frame = cap.read()
if frame is not None:
    frame_height, frame_width = frame.shape[:2]
else:
    frame_height, frame_width = 480, 640  # Fallback dimensions
graph_y_offset = frame_height - graph_height - 10  # 10-pixel margin from bottom

# Main loop for real-time processing
while True:
    success, image = cap.read()
    if not success:
        print("Error: Unable to read frame from camera.")
        break

    frame_count += 1  # Increment frame counter
    # Convert the frame to RGB for MediaPipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # Initialize skeleton frame with zeros (18 joints, 3 coordinates: x, y, z)
    skeleton_frame = np.zeros((18, 3))

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Compute reference point: average of left and right hip
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        ref_x = (left_hip.x + right_hip.x) / 2
        ref_y = (left_hip.y + right_hip.y) / 2
        ref_z = (left_hip.z + right_hip.z) / 2

        # Compute neck as the average of left and right shoulders
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        neck_x = (left_shoulder.x + right_shoulder.x) / 2
        neck_y = (left_shoulder.y + right_shoulder.y) / 2
        neck_z = (left_shoulder.z + right_shoulder.z) / 2

        # Extract relative coordinates for the 18 landmarks
        for i, part in enumerate(landmark_order):
            if part == "Neck":
                x = neck_x - ref_x
                y = neck_y - ref_y
                z = neck_z - ref_z
            else:
                lm = landmarks[landmark_indices[part]]
                x = lm.x - ref_x
                y = lm.y - ref_y
                z = lm.z - ref_z
            skeleton_frame[i] = [x, y, z]

        # Draw the detected landmarks on the image for visualization
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Add the current skeleton frame to the buffer
    buffer.append(skeleton_frame)

    # Process sliding windows if enough frames are available
    if len(buffer) >= window_size:
        # Process the most recent window
        start_idx = max(0, len(buffer) - window_size)
        window = buffer[start_idx:start_idx + window_size]

        # Convert window to tensor with shape (1, 3, window_size, 18)
        # Shape: (window_size, 18, 3)
        skeleton_sequence = np.stack(window, axis=0)
        skeleton_sequence = torch.tensor(skeleton_sequence, dtype=torch.float32).permute(
            2, 0, 1).unsqueeze(0)  # Shape: (1, 3, window_size, 18)

        # Perform inference
        with torch.no_grad():
            output = model(skeleton_sequence.to(device))
            probabilities = torch.softmax(output, dim=1)
            fall_prob = probabilities[0, 1].item()
            prediction = 1 if fall_prob >= threshold else 0
            "##########""##########""##########""##########"

        # Store probability for graph
        prob_buffer.append(fall_prob * 100)  # Convert to percentage
        if len(prob_buffer) > max_points:
            prob_buffer.pop(0)  # Keep only the latest max_points

        # Display the prediction and probability on the frame
        if prediction == 1:
            text = f"Fall Detected! ({fall_prob:.2f})"
            color = (0, 0, 255)  # Red
        else:
            text = f"Fall Detected ({fall_prob:.2f})"  # Changed from "No Fall"
            color = (0, 255, 0)  # Green
        cv2.putText(image, text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Display the current threshold and window info
        info_text = f"Threshold: {threshold:.2f} (+/-), Window: {window_size}, Step: {step_size}"
        cv2.putText(image, info_text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Display buffering message until enough frames are collected
        cv2.putText(image, f"Buffering... ({len(buffer)}/{window_size})",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw live graph at the bottom
    graph_img = np.zeros((graph_height, graph_width, 3),
                         dtype=np.uint8)  # Black background
    if prob_buffer:
        # Draw axes
        cv2.line(graph_img, (0, graph_height - 10), (graph_width,
                 graph_height - 10), (255, 255, 255), 1)  # X-axis
        cv2.line(graph_img, (10, 0), (10, graph_height),
                 (255, 255, 255), 1)  # Y-axis

        # Draw axis labels
        cv2.putText(graph_img, "Time (s)", (graph_width - 50, graph_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(graph_img, "Prob (%)", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(graph_img, "100", (2, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(graph_img, "0", (2, graph_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Plot probabilities
        points = []
        for i, prob in enumerate(prob_buffer):
            x = int(i * (graph_width - 20) / max_points) + \
                10  # Scale x to graph width
            y = int((1 - prob / 100) * (graph_height - 20)) + \
                10  # Scale y (0 at bottom, 100 at top)
            points.append((x, y))

        # Draw lines between points
        for i in range(1, len(points)):
            cv2.line(graph_img, points[i-1], points[i], (0, 255, 0), 1)

        # Draw time label for x-axis (approximate seconds)
        time_span = len(prob_buffer) / fps
        cv2.putText(graph_img, f"{time_span:.1f}s", (graph_width - 30,
                    graph_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Place graph at bottom of frame
    image[graph_y_offset:graph_y_offset +
          graph_height, 10:10 + graph_width] = graph_img

    # Show the processed frame
    cv2.imshow("Live Fall Detection", image)

    # Handle key presses to adjust threshold, window size, step size, or exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Exit on 'q'
        break
    elif key == ord('+') or key == ord('='):  # Increase threshold
        threshold = min(1.0, threshold + 0.05)
    elif key == ord('-'):  # Decrease threshold
        threshold = max(0.0, threshold - 0.05)
    elif key == ord('w'):  # Increase window size
        window_size = min(60, window_size + 5)
    elif key == ord('s'):  # Decrease window size
        window_size = max(10, window_size - 5)
    elif key == ord('d'):  # Increase step size
        step_size = min(10, step_size + 1)
    elif key == ord('a'):  # Decrease step size
        step_size = max(1, step_size - 1)

# Release resources
cap.release()
cv2.destroyAllWindows()
pose.close()
