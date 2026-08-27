import cv2
import numpy as np
import mediapipe as mp
import time
import os

# ============================================================
# AIR WRITING / VIRTUAL DRAWING APP
# OpenCV + MediaPipe Hands
# ============================================================

# -------------------- SETTINGS --------------------

WIDTH = 1280
HEIGHT = 720

BRUSH_THICKNESS = 8
ERASER_THICKNESS = 60

MAX_UNDO = 20

# -------------------- COLORS -----------------------

COLORS = {
    "PURPLE": (255, 0, 255),
    "BLUE": (255, 0, 0),
    "GREEN": (0, 255, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 255, 255),
    "WHITE": (255, 255, 255),
    "BLACK": (0, 0, 0),       # Eraser
}

draw_color_name = "PURPLE"
draw_color = COLORS[draw_color_name]

brush_thickness = BRUSH_THICKNESS
eraser_thickness = ERASER_THICKNESS

# -------------------- MEDIAPIPE --------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -------------------- CANVAS -----------------------

canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# Undo history
undo_stack = []

# Previous drawing point
xp, yp = 0, 0


# ============================================================
# FUNCTIONS
# ============================================================

def save_undo_state():
    """Save current canvas for undo."""
    global undo_stack

    undo_stack.append(canvas.copy())

    if len(undo_stack) > MAX_UNDO:
        undo_stack.pop(0)


def undo():
    """Undo the previous drawing action."""
    global canvas

    if len(undo_stack) > 0:
        canvas[:] = undo_stack.pop()
        print("Undo completed")
    else:
        print("Nothing to undo")


def clear_canvas():
    """Clear the entire drawing."""
    global canvas

    save_undo_state()
    canvas[:] = 0

    print("Canvas cleared")


def save_drawing():
    """Save only the drawing canvas."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filename = f"air_drawing_{timestamp}.png"

    cv2.imwrite(filename, canvas)

    print(f"Drawing saved: {filename}")


def save_final_image(image):
    """Save webcam + drawing combined image."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    filename = f"air_writing_final_{timestamp}.png"

    cv2.imwrite(filename, image)

    print(f"Final image saved: {filename}")


def get_finger_states(hand_landmarks, handedness):
    """
    Return finger states:
    [thumb, index, middle, ring, pinky]
    """

    landmarks = hand_landmarks.landmark

    fingers = []

    # --------------------------------------------------------
    # Thumb
    # --------------------------------------------------------

    if handedness == "Right":
        thumb_up = landmarks[4].x < landmarks[3].x
    else:
        thumb_up = landmarks[4].x > landmarks[3].x

    fingers.append(1 if thumb_up else 0)

    # --------------------------------------------------------
    # Four fingers
    # --------------------------------------------------------

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip, pip in zip(finger_tips, finger_pips):

        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


def draw_toolbar(image):
    """Draw the toolbar at the top."""

    # Toolbar background
    cv2.rectangle(
        image,
        (0, 0),
        (WIDTH, 115),
        (40, 40, 40),
        cv2.FILLED
    )

    # --------------------------------------------------------
    # Color buttons
    # --------------------------------------------------------

    buttons = [
        ("PURPLE", 10, 120),
        ("BLUE", 130, 240),
        ("GREEN", 250, 360),
        ("RED", 370, 480),
        ("YELLOW", 490, 600),
        ("WHITE", 610, 720),
        ("BLACK", 730, 840),
    ]

    for name, x1, x2 in buttons:

        color = COLORS[name]

        cv2.rectangle(
            image,
            (x1, 10),
            (x2, 75),
            color,
            cv2.FILLED
        )

        # Text color
        text_color = (0, 0, 0) if name in ["YELLOW", "WHITE"] else (255, 255, 255)

        cv2.putText(
            image,
            name,
            (x1 + 10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            2
        )

    # --------------------------------------------------------
    # Clear button
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (850, 10),
        (940, 75),
        (100, 100, 100),
        cv2.FILLED
    )

    cv2.putText(
        image,
        "CLEAR",
        (860, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Status information
    # --------------------------------------------------------

    cv2.putText(
        image,
        f"Color: {draw_color_name}",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        f"Brush: {brush_thickness}",
        (200, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        image,
        "S:Save  U:Undo  C:Clear  +/-:Brush  F:Final  ESC:Exit",
        (400, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


def select_toolbar(x, y):
    """
    Select a color/tool based on index finger position.
    """

    global draw_color
    global draw_color_name

    if y > 115:
        return

    # Purple
    if 10 < x < 120:
        draw_color_name = "PURPLE"
        draw_color = COLORS["PURPLE"]

    # Blue
    elif 130 < x < 240:
        draw_color_name = "BLUE"
        draw_color = COLORS["BLUE"]

    # Green
    elif 250 < x < 360:
        draw_color_name = "GREEN"
        draw_color = COLORS["GREEN"]

    # Red
    elif 370 < x < 480:
        draw_color_name = "RED"
        draw_color = COLORS["RED"]

    # Yellow
    elif 490 < x < 600:
        draw_color_name = "YELLOW"
        draw_color = COLORS["YELLOW"]

    # White
    elif 610 < x < 720:
        draw_color_name = "WHITE"
        draw_color = COLORS["WHITE"]

    # Eraser
    elif 730 < x < 840:
        draw_color_name = "BLACK"
        draw_color = COLORS["BLACK"]

    # Clear
    elif 850 < x < 940:
        clear_canvas()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print("========================================")
print("        AIR WRITING STARTED")
print("========================================")
print("Index finger  : Draw")
print("Index + Middle: Select")
print("S             : Save drawing")
print("U             : Undo")
print("C             : Clear")
print("+ / -         : Brush size")
print("F             : Save final image")
print("ESC           : Exit")
print("========================================")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Could not read webcam.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe processing
    results = hands.process(rgb)

    # --------------------------------------------------------
    # HAND DETECTION
    # --------------------------------------------------------

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Get handedness
        handedness = "Right"

        if results.multi_handedness:

            handedness = (
                results.multi_handedness[0]
                .classification[0]
                .label
            )

        # Get landmark coordinates
        landmark_list = []

        for lm in hand_landmarks.landmark:

            x = int(lm.x * WIDTH)
            y = int(lm.y * HEIGHT)

            landmark_list.append((x, y))

        # Index fingertip
        x1, y1 = landmark_list[8]

        # Middle fingertip
        x2, y2 = landmark_list[12]

        # Finger states
        fingers = get_finger_states(
            hand_landmarks,
            handedness
        )

        # ----------------------------------------------------
        # SELECTION MODE
        # Index + Middle finger
        # ----------------------------------------------------

        if fingers[1] and fingers[2]:

            xp, yp = 0, 0

            # Highlight selection area
            cv2.rectangle(
                frame,
                (x1 - 20, y1 - 20),
                (x2 + 20, y2 + 20),
                draw_color,
                2
            )

            select_toolbar(x1, y1)

        # ----------------------------------------------------
        # DRAWING MODE
        # Index finger only
        # ----------------------------------------------------

        elif fingers[1] and not fingers[2]:

            # Draw cursor
            cursor_color = draw_color

            cv2.circle(
                frame,
                (x1, y1),
                max(8, brush_thickness),
                cursor_color,
                cv2.FILLED
            )

            # First point
            if xp == 0 and yp == 0:

                xp, yp = x1, y1

                # Save state before starting a new stroke
                save_undo_state()

            # Eraser
            if draw_color_name == "BLACK":

                cv2.line(
                    frame,
                    (xp, yp),
                    (x1, y1),
                    (0, 0, 0),
                    eraser_thickness
                )

                cv2.line(
                    canvas,
                    (xp, yp),
                    (x1, y1),
                    (0, 0, 0),
                    eraser_thickness
                )

            # Normal brush
            else:

                cv2.line(
                    frame,
                    (xp, yp),
                    (x1, y1),
                    draw_color,
                    brush_thickness
                )

                cv2.line(
                    canvas,
                    (xp, yp),
                    (x1, y1),
                    draw_color,
                    brush_thickness
                )

            xp, yp = x1, y1

        # ----------------------------------------------------
        # No drawing gesture
        # ----------------------------------------------------

        else:

            xp, yp = 0, 0

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

    else:

        # Reset drawing point when hand disappears
        xp, yp = 0, 0

    # ========================================================
    # MERGE CANVAS WITH CAMERA
    # ========================================================

    gray_canvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2GRAY
    )

    _, inverse_canvas = cv2.threshold(
        gray_canvas,
        20,
        255,
        cv2.THRESH_BINARY_INV
    )

    inverse_canvas = cv2.cvtColor(
        inverse_canvas,
        cv2.COLOR_GRAY2BGR
    )

    # Remove black canvas area from frame
    frame_without_canvas = cv2.bitwise_and(
        frame,
        inverse_canvas
    )

    # Add drawing
    final_frame = cv2.bitwise_or(
        frame_without_canvas,
        canvas
    )

    # ========================================================
    # TOOLBAR
    # ========================================================

    draw_toolbar(final_frame)

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Air Writing - OpenCV + MediaPipe",
        final_frame
    )

    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------------------------
    # SAVE DRAWING
    # --------------------------------------------------------

    if key == ord("s"):

        save_drawing()

    # --------------------------------------------------------
    # UNDO
    # --------------------------------------------------------

    elif key == ord("u"):

        undo()

    # --------------------------------------------------------
    # CLEAR
    # --------------------------------------------------------

    elif key == ord("c"):

        clear_canvas()

    # --------------------------------------------------------
    # INCREASE BRUSH
    # --------------------------------------------------------

    elif key in [ord("+"), ord("=")]:

        brush_thickness += 2

        if brush_thickness > 50:
            brush_thickness = 50

        print(f"Brush size: {brush_thickness}")

    # --------------------------------------------------------
    # DECREASE BRUSH
    # --------------------------------------------------------

    elif key in [ord("-"), ord("_")]:

        brush_thickness -= 2

        if brush_thickness < 2:
            brush_thickness = 2

        print(f"Brush size: {brush_thickness}")

    # --------------------------------------------------------
    # SAVE FINAL IMAGE
    # --------------------------------------------------------

    elif key == ord("f"):

        save_final_image(final_frame)

    # --------------------------------------------------------
    # ESCAPE
    # --------------------------------------------------------

    elif key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Air Writing stopped.")