import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def main():
    st.title("Simple AR App without MediaPipe")
    
    img_file = st.camera_input("Take a picture")
    
    if img_file is not None:
        # Load the image
        image = Image.open(img_file)
        image_np = np.array(image)

        # Convert to grayscale for face detection
        gray_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw a 3D-like cube on each detected face
        for (x, y, w, h) in faces:
            size = h  # Use the height of the face as the cube size
            cube_points = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                                      [0,0,-1], [0,1,-1], [1,1,-1], [1,0,-1]])

            # Project the cube points onto the face
            projected_points = []
            for point in cube_points:
                px = int(x + point[0] * size)
                py = int(y + point[1] * size)
                projected_points.append((px, py))
            
            # Draw the cube
            pts = np.array(projected_points, np.int32)
            cv2.polylines(image_np, [pts[:4]], True, (0, 255, 0), 2)  # Top face
            cv2.polylines(image_np, [pts[4:]], True, (0, 255, 0), 2)  # Bottom face
            for i in range(4):  # Vertical edges
                cv2.line(image_np, projected_points[i], projected_points[i+4], (0, 255, 0), 2)

        # Display the image with overlays
        st.image(image_np, channels="BGR")

if __name__ == '__main__':
    main()
