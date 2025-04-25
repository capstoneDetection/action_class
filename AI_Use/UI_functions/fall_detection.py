import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn


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


class FallDetector:
    def __init__(self,
                 model_path,
                 window_size=30,
                 step_size=1,
                 threshold=0.8,
                 fps=30,
                 device=None):
        self.window_size = window_size
        self.step_size = step_size
        self.threshold = threshold
        self.fps = fps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = STGCN(in_channels=3, num_joints=18,
                           num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))
        self.model.eval()

        # MediaPipe init
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

        # Define the 18 landmarks in the same order as used during training (unchanged)
        self.landmark_order = [
            "Nose", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist",
            "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip", "Right Knee",
            "Right Ankle", "Left Hip", "Left Knee", "Left Ankle", "Right Eye",
            "Left Eye", "Right Ear", "Left Ear"
        ]

        self.landmark_indices = {
            "Nose": 0, "Right Shoulder": 12, "Right Elbow": 14, "Right Wrist": 16,
            "Left Shoulder": 11, "Left Elbow": 13, "Left Wrist": 15, "Right Hip": 24,
            "Right Knee": 26, "Right Ankle": 28, "Left Hip": 23, "Left Knee": 25,
            "Left Ankle": 27, "Right Eye": 5, "Left Eye": 2, "Right Ear": 8, "Left Ear": 7
        }
        self.buffer = []

    def process_frame(self, image):
        """
        Process a single image frame, return (fall_detected: bool, fall_probability: float)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        # Extract skeleton
        skeleton_frame = np.zeros((18, 3))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_hip, right_hip = landmarks[23], landmarks[24]
            ref_x = (left_hip.x + right_hip.x) / 2
            ref_y = (left_hip.y + right_hip.y) / 2
            ref_z = (left_hip.z + right_hip.z) / 2
            left_shoulder, right_shoulder = landmarks[11], landmarks[12]
            neck_x = (left_shoulder.x + right_shoulder.x) / 2
            neck_y = (left_shoulder.y + right_shoulder.y) / 2
            neck_z = (left_shoulder.z + right_shoulder.z) / 2
            for i, part in enumerate(self.landmark_order):
                if part == "Neck":
                    x, y, z = neck_x - ref_x, neck_y - ref_y, neck_z - ref_z
                else:
                    lm = landmarks[self.landmark_indices[part]]
                    x, y, z = lm.x - ref_x, lm.y - ref_y, lm.z - ref_z
                skeleton_frame[i] = [x, y, z]

        # Add frame to buffer
        self.buffer.append(skeleton_frame)
        if len(self.buffer) < self.window_size:
            return False, 0.0  # Not enough data yet

        # Use only the latest window for detection
        window = self.buffer[-self.window_size:]
        skeleton_sequence = np.stack(window, axis=0)
        skeleton_sequence = torch.tensor(
            skeleton_sequence, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            output = self.model(skeleton_sequence.to(self.device))
            probabilities = torch.softmax(output, dim=1)
            fall_prob = probabilities[0, 1].item()
            detected = fall_prob >= self.threshold

        return detected, fall_prob

    def close(self):
        self.pose.close()
