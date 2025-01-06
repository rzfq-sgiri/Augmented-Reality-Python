import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

def main():
    st.title("Simple AR App")
    
    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
    
    img_file = st.camera_input("Take a picture")
    
    if img_file is not None:
        image = Image.open(img_file)
        image_np = np.array(image)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image_np.shape
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                
                # Draw 3D cube
                size = height
                cube_points = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                                        [0,0,-1], [0,1,-1], [1,1,-1], [1,0,-1]])
                
                projected_points = []
                for point in cube_points:
                    px = int(x + point[0] * size)
                    py = int(y + point[1] * size)
                    projected_points.append((px, py))
                
                pts = np.array(projected_points, np.int32)
                cv2.polylines(image_np, [pts[:4]], True, (0,255,0), 2)
                cv2.polylines(image_np, [pts[4:]], True, (0,255,0), 2)
                for i in range(4):
                    cv2.line(image_np, projected_points[i], projected_points[i+4], (0,255,0), 2)
        
        st.image(image_np)

if __name__ == '__main__':
    main()
