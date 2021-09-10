#!/usr/bin/env python3

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

NUM_FACE = 1

PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_DIR / 'notebooks'
imgRGB = cv2.imread(str(IMAGE_DIR) + "/head.jpg")

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
mpDrawingStyles = mp.solutions.drawing_styles

# results = faceMesh.process(imgRGB)
ih, iw, ic = imgRGB.shape
cv2.namedWindow("Toms Mug", 0)

with mpFaceMesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        mesh_image = imgRGB.copy()

        for face_landmarks in results.multi_face_landmarks:
            for id, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv2.putText(imgRGB, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            mpDraw.draw_landmarks(image=mesh_image, landmark_list=face_landmarks,
                                  connections=mpFaceMesh.FACEMESH_TESSELATION,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_tesselation_style())
            mpDraw.draw_landmarks(image=mesh_image, landmark_list=face_landmarks,
                                  connections=mpFaceMesh.FACEMESH_CONTOURS,
                                  landmark_drawing_spec=None,
                                  connection_drawing_spec=mpDrawingStyles.get_default_face_mesh_contours_style())

    cv2.imshow("Toms Mug", np.concatenate((mesh_image, imgRGB), axis=1))
    cv2.imwrite(str(IMAGE_DIR) + "/head_with_landmarks.jpg", imgRGB)
    cv2.imwrite(str(IMAGE_DIR) + "/head_with_mesh.jpg", mesh_image)
    cv2.waitKey(-1)
