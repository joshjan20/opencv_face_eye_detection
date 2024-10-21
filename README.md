```markdown
# Face and Eye Detection with OpenCV

This Python project demonstrates real-time face and eye detection using OpenCV and Haar Cascades. The program uses the computer's webcam to detect faces and eyes and marks them with rectangles in real time.

## Features

- **Real-Time Detection**: Detects faces and eyes using the webcam.
- **Haar Cascade Classifiers**: Uses pre-trained Haar Cascade models to identify faces and eyes.
- **Visualization**: Draws rectangles around detected faces (in blue) and eyes (in green).

## Requirements

- Python 3.x
- OpenCV

Install OpenCV using `pip`:

```bash
pip install opencv-python
```

## How to Run

1. Clone this repository or download the script.
2. Ensure your computer has a working webcam.
3. Run the script using Python:

```bash
python main.py
```

4. A window will appear showing the live feed from the webcam with detected faces and eyes marked.

5. Press `q` to quit the application and close the webcam feed.

## Code Breakdown

```python
import numpy as np
import cv2

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Load pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()  # Capture frame from webcam

    # Convert the frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Loop through detected faces and draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]  # Region of interest for face in grayscale
        roi_color = frame[y:y+h, x:x+w]  # Region of interest for face in color

        # Detect eyes within the detected face region
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    # Show the frame with detected faces and eyes
    cv2.imshow('frame', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
```

### Explanation

1. **Webcam Initialization**:
   - The script uses OpenCV to capture live video feed from the webcam (`cv2.VideoCapture(0)`).

2. **Face and Eye Detection**:
   - It uses two Haar Cascade classifiers: one for face detection (`haarcascade_frontalface_default.xml`) and another for eye detection (`haarcascade_eye.xml`).
   - The classifiers are loaded using `cv2.CascadeClassifier()`.

3. **Rectangle Drawing**:
   - For each detected face, a blue rectangle is drawn around the face.
   - For each detected eye within the face region, a green rectangle is drawn.

4. **Live Feed Display**:
   - The processed frame with rectangles is displayed in real time.
   - The window can be closed by pressing the `q` key.

### Example Output

The application will display the live video feed from your webcam, highlighting detected faces with a blue rectangle and detected eyes with a green rectangle.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```