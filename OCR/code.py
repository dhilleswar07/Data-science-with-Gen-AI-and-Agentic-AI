import cv2
import imutils
import pytesseract
import matplotlib.pyplot as plt

# --------------------------------------------------
# Tesseract configuration
# --------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract.exe'
)

# --------------------------------------------------
# 1. Read image
# --------------------------------------------------
image = cv2.imread(
    r"E:\DATASCIENCE WITH GEN AI & AGENTIC AI\VSCODE\OCR\image.jpeg"
)

# Check if image loaded correctly
if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

resized_image = imutils.resize(image)


# --------------------------------------------------
# 2. Convert to grayscale
# --------------------------------------------------
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------
# 3. Smooth image
# --------------------------------------------------
smooth_image = cv2.bilateralFilter(
    gray_image,
    11,
    17,
    17
)


# --------------------------------------------------
# 4. Edge detection
# --------------------------------------------------
edged = cv2.Canny(
    smooth_image,
    30,
    200
)


# --------------------------------------------------
# 5. Find contours
# --------------------------------------------------
cnts, new = cv2.findContours(
    edged.copy(),
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

# Draw all contours
image1 = resized_image.copy()

cv2.drawContours(
    image1,
    cnts,
    -1,
    (0, 255, 0),
    3
)


# --------------------------------------------------
# 6. Select top 30 contours
# --------------------------------------------------
cnts = sorted(
    cnts,
    key=cv2.contourArea,
    reverse=True
)[:30]

image2 = resized_image.copy()

cv2.drawContours(
    image2,
    cnts,
    -1,
    (0, 255, 0),
    3
)


# --------------------------------------------------
# 7. Detect license plate
# --------------------------------------------------
screenCnt = None
new_img = None

for c in cnts:

    perimeter = cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(
        c,
        0.018 * perimeter,
        True
    )

    # License plate normally has 4 corners
    if len(approx) == 4:

        screenCnt = approx

        x, y, w, h = cv2.boundingRect(c)

        new_img = resized_image[
            y:y+h,
            x:x+w
        ]

        cv2.imwrite(
            "./detected_license_plate.png",
            new_img
        )

        break


# --------------------------------------------------
# 8. Draw detected license plate
# --------------------------------------------------
detected_image = resized_image.copy()

if screenCnt is not None:

    cv2.drawContours(
        detected_image,
        [screenCnt],
        -1,
        (0, 255, 0),
        3
    )


# --------------------------------------------------
# 9. Convert images for Matplotlib
# --------------------------------------------------

original_rgb = cv2.cvtColor(
    resized_image,
    cv2.COLOR_BGR2RGB
)

contours_rgb = cv2.cvtColor(
    image1,
    cv2.COLOR_BGR2RGB
)

top30_rgb = cv2.cvtColor(
    image2,
    cv2.COLOR_BGR2RGB
)

detected_rgb = cv2.cvtColor(
    detected_image,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# 10. DISPLAY ALL OUTPUTS SIDE BY SIDE
# --------------------------------------------------

plt.figure(figsize=(20, 10))


# 1. Original
plt.subplot(2, 4, 1)
plt.imshow(original_rgb)
plt.title("1. Original Image", fontsize=14)
plt.axis("off")


# 2. Grayscale
plt.subplot(2, 4, 2)
plt.imshow(gray_image, cmap="gray")
plt.title("2. Grayscale Image", fontsize=14)
plt.axis("off")


# 3. Smoothened
plt.subplot(2, 4, 3)
plt.imshow(smooth_image, cmap="gray")
plt.title("3. Smoothened Image", fontsize=14)
plt.axis("off")


# 4. Edges
plt.subplot(2, 4, 4)
plt.imshow(edged, cmap="gray")
plt.title("4. Edge Detection", fontsize=14)
plt.axis("off")


# 5. All contours
plt.subplot(2, 4, 5)
plt.imshow(contours_rgb)
plt.title("5. All Contours", fontsize=14)
plt.axis("off")


# 6. Top 30 contours
plt.subplot(2, 4, 6)
plt.imshow(top30_rgb)
plt.title("6. Top 30 Contours", fontsize=14)
plt.axis("off")


# 7. Detected license plate
plt.subplot(2, 4, 7)
plt.imshow(detected_rgb)
plt.title("7. Detected License Plate", fontsize=14)
plt.axis("off")


# 8. Cropped license plate
plt.subplot(2, 4, 8)

if new_img is not None:

    cropped_rgb = cv2.cvtColor(
        new_img,
        cv2.COLOR_BGR2RGB
    )

    plt.imshow(cropped_rgb)
    plt.title("8. Cropped License Plate", fontsize=14)

else:

    plt.text(
        0.5,
        0.5,
        "License Plate\nNot Detected",
        ha="center",
        va="center",
        fontsize=14
    )

plt.axis("off")


# --------------------------------------------------
# Final layout
# --------------------------------------------------

plt.tight_layout()
plt.show()