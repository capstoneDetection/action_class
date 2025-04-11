import os
import cv2
import numpy as np
import random
import time
from datetime import datetime
from PIL import ImageFont, ImageDraw, Image

# ----------------- Default Settings -----------------
CAMERA = 0
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 1280
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_X = (WINDOW_WIDTH - VIDEO_WIDTH) // 2
VIDEO_Y = 30

LABEL_HEIGHT = 60
LABEL_Y = VIDEO_Y + VIDEO_HEIGHT + 10

MSG_BOX_HEIGHT = 40
MSG_BOX_Y = LABEL_Y + LABEL_HEIGHT + 5

BUTTON_WIDTH = VIDEO_WIDTH
BUTTON_HEIGHT = 80
PAUSE_BUTTON_Y = WINDOW_HEIGHT - 2 * (BUTTON_HEIGHT + 10)
QUIT_BUTTON_Y = WINDOW_HEIGHT - (BUTTON_HEIGHT + 10)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH =  os.path.join(SCRIPT_DIR,"D2Coding-Ver1.3.2-20180524-ligature.ttf")
FONT_SMALL = ImageFont.truetype(FONT_PATH, 20)
FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 28)
FONT_LARGE = ImageFont.truetype(FONT_PATH, 32)

# ----------------- Global State Variables -----------------
paused = False
prev_classification = "NoFall"
fall_detected_time = None
last_frame = None
should_exit = False
pause_clicked = False
quit_clicked = False

last_classification_time = 0
classification_interval = 0.5  # Classification interval in seconds

# ----------------- Dummy Classification Function -----------------
def get_classification(frame):
    return random.choice(["Fall", "NoFall"])

# ----------------- Mouse Callback Function -----------------
def on_mouse(event, x, y, flags, param):
    global paused, should_exit, pause_clicked, quit_clicked

    if event == cv2.EVENT_LBUTTONDOWN:
        if VIDEO_X <= x <= VIDEO_X + BUTTON_WIDTH:
            if PAUSE_BUTTON_Y <= y <= PAUSE_BUTTON_Y + BUTTON_HEIGHT:
                pause_clicked = True
                paused = not paused
            elif QUIT_BUTTON_Y <= y <= QUIT_BUTTON_Y + BUTTON_HEIGHT:
                quit_clicked = True
                should_exit = True

# Get text size using textbbox
def get_text_size(text, font):
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return (width, height)

# Draw text using PIL
def draw_text_pil(img, text, position, font, fill):
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=fill)
    return np.array(pil_img)

# ----------------- Main Function -----------------
def main():
    global last_frame, prev_classification, fall_detected_time
    global pause_clicked, quit_clicked, should_exit
    global last_classification_time

    cap = cv2.VideoCapture(CAMERA)
    if not cap.isOpened():
        print("Cannot open the camera.")
        return
    
    cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Video", on_mouse)

    while True:
        if should_exit:
            break

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        if not paused:
            last_frame = frame.copy()

            current_time = time.time()
            if current_time - last_classification_time >= classification_interval:
                current_classification = get_classification(frame)
                last_classification_time = current_time

                if current_classification == "Fall" and prev_classification == "NoFall":
                    fall_detected_time = datetime.now()

                prev_classification = current_classification
            else:
                current_classification = prev_classification
        else:
            if last_frame is None:
                continue
            frame = last_frame.copy()
            current_classification = prev_classification

        canvas = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255
        canvas[VIDEO_Y:VIDEO_Y + VIDEO_HEIGHT, VIDEO_X:VIDEO_X + VIDEO_WIDTH] = frame

       # Status label
        if current_classification == "Fall" and fall_detected_time is not None:
            label_text = f"Fall detected at {fall_detected_time.strftime('%H:%M:%S')}"
            label_color = (255, 255, 255)
            label_bg = (0, 0, 255)
            show_msg_box = True
        else:
            label_text = "No Fall"
            label_color = (50, 200, 50)
            label_bg = (230, 230, 230)
            show_msg_box = False

        text_size = get_text_size(label_text, FONT_MEDIUM)
        text_x = VIDEO_X + (VIDEO_WIDTH - text_size[0]) // 2
        text_y = LABEL_Y + (LABEL_HEIGHT - text_size[1]) // 2
        cv2.rectangle(canvas, (VIDEO_X, LABEL_Y), (VIDEO_X + VIDEO_WIDTH, LABEL_Y + LABEL_HEIGHT), label_bg, -1)
        canvas = draw_text_pil(canvas, label_text, (text_x, text_y), FONT_MEDIUM, label_color)

        # Message box
        if show_msg_box:
            cv2.rectangle(canvas, (VIDEO_X, MSG_BOX_Y), (VIDEO_X + VIDEO_WIDTH, MSG_BOX_Y + MSG_BOX_HEIGHT), (240, 240, 240), -1)
            msg = "Alert message sent!"
            msg_size = get_text_size(msg, FONT_SMALL)
            msg_x = VIDEO_X + (VIDEO_WIDTH - msg_size[0]) // 2
            msg_y = MSG_BOX_Y + (MSG_BOX_HEIGHT - msg_size[1]) // 2
            canvas = draw_text_pil(canvas, msg, (msg_x, msg_y), FONT_SMALL, (80, 80, 80))

        # Pause button
        pause_color = (180, 180, 180) if pause_clicked else (230, 230, 230)
        cv2.rectangle(canvas, (VIDEO_X, PAUSE_BUTTON_Y), (VIDEO_X + BUTTON_WIDTH, PAUSE_BUTTON_Y + BUTTON_HEIGHT), pause_color, -1)
        pause_text = "RESUME" if paused else "PAUSE"
        pause_text_size = get_text_size(pause_text, FONT_LARGE)
        pause_text_x = VIDEO_X + (VIDEO_WIDTH - pause_text_size[0]) // 2
        pause_text_y = PAUSE_BUTTON_Y + (BUTTON_HEIGHT - pause_text_size[1]) // 2
        canvas = draw_text_pil(canvas, pause_text, (pause_text_x, pause_text_y), FONT_LARGE, (0, 0, 0))

        # Quit button
        quit_color = (180, 180, 180) if quit_clicked else (200, 200, 200)
        cv2.rectangle(canvas, (VIDEO_X, QUIT_BUTTON_Y), (VIDEO_X + BUTTON_WIDTH, QUIT_BUTTON_Y + BUTTON_HEIGHT), quit_color, -1)
        quit_text_size = get_text_size("QUIT", FONT_LARGE)
        quit_text_x = VIDEO_X + (VIDEO_WIDTH - quit_text_size[0]) // 2
        quit_text_y = QUIT_BUTTON_Y + (BUTTON_HEIGHT - quit_text_size[1]) // 2
        canvas = draw_text_pil(canvas, "QUIT", (quit_text_x, quit_text_y), FONT_LARGE, (0, 0, 0))

        pause_clicked = False
        quit_clicked = False

        # Display
        cv2.imshow("Video", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    main()