import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands and Drawing modules
mphands = mp.solutions.hands
mpdraw = mp.solutions.drawing_utils
hands = mphands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.9)
cap = cv2.VideoCapture(0)

# Function to calculate the Euclidean distance in 3D space (x, y, z)
def calculate_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Function to dynamically adjust the threshold based on hand distance
def adjust_threshold(z_wrist):
    # Example: If hand is far, make the threshold smaller, if close, larger
    # z_wrist is the depth (z-coordinate) of the wrist (landmark 0)
    # The farther the hand, the stricter the threshold
    base_threshold = 50
    z_factor = abs(z_wrist * 500)  # Adjust this factor based on camera depth range
    return max(base_threshold - z_factor, 10)  # Ensure the threshold doesn't get too small

while True:
    success, img = cap.read()
    # img = cv2.resize(img, (1800, 900))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            thumb_x, thumb_y, thumb_z = 0, 0, 0
            little_x, little_y, little_z = 0, 0, 0
            wrist_z = 0

            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cz = lm.z  # Capture the z-coordinate for 3D depth

                if id == 4:  # Thumb tip
                    thumb_x, thumb_y, thumb_z = cx, cy, cz
                if id == 20:  # Little finger tip
                    little_x, little_y, little_z = cx, cy, cz
                if id == 0:  # Wrist (can be used to estimate hand distance from camera)
                    wrist_z = cz

            # Adjust the distance threshold based on how far the hand is from the camera
            distance_threshold = adjust_threshold(wrist_z)

            # Calculate the distance between thumb and little finger in 3D
            if thumb_x and thumb_y and little_x and little_y:
                cv2.line(img, (thumb_x, thumb_y), (little_x, little_y), (255, 0, 255), 15)
                distance = calculate_distance(thumb_x, thumb_y, thumb_z, little_x, little_y, little_z)
                
                # Check the 3D distance, including z, and use the dynamic threshold
                if distance < distance_threshold:
                    print(f"OK! Distance: {distance}, Threshold: {distance_threshold}")
                else:
                    print(f"Not OK. Distance: {distance}, Threshold: {distance_threshold}")

            # Draw landmarks on the hand
            mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)

    cv2.imshow("image", img)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp

# # Initialize Mediapipe Face Mesh and Drawing modules
# mp_face_mesh = mp.solutions.face_mesh
# mp_face_mesh = mp.solutions.face_mesh
# mp_draw = mp.solutions.drawing_utils
# face_mesh = mp_face_mesh.FaceMesh()

# # # Open the webcam
# cap = cv2.VideoCapture(0)
# draw_spec_points = mp_draw.DrawingSpec(thickness=1, circle_radius=0, color=(255, 0, 0))  # Smaller points
# draw_spec_connections = mp_draw.DrawingSpec(thickness=1, color=(255, 0, 0))  # Thinner lines

# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(imgRGB)

#     if results.multi_face_landmarks:
#         for facelms in results.multi_face_landmarks:
#             # Draw landmarks and use the tesselation for face mesh
#             mp_draw.draw_landmarks(
#                 img,
#                 facelms,
#                 mp_face_mesh.FACEMESH_TESSELATION,
#                 landmark_drawing_spec=draw_spec_points,  # Adjust landmark point size
#                 connection_drawing_spec=draw_spec_connections  # Adjust connection thickness
                
#             )
#     # Show the image
#     cv2.imshow("Face Mesh", img)

#     # Exit when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# # cv2.destroyAllWindows()
# import cv2
# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2
# import numpy as np

# # Function to draw landmarks on the image
# def draw_landmarks_on_image(rgb_image, detection_result):
#     face_landmarks_list = detection_result.face_landmarks
#     annotated_image = np.copy(rgb_image)

#     for idx in range(len(face_landmarks_list)):
#         face_landmarks = face_landmarks_list[idx]

#         # Draw the face landmarks.
#         face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#         face_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
#         ])

#         mp.solutions.drawing_utils.draw_landmarks(
#             image=annotated_image,
#             landmark_list=face_landmarks_proto,
#             connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

#     return annotated_image

# # Function to calculate the Euclidean distance between two points
# def calculate_distance(landmark1, landmark2):
#     return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2)

# # Set up the face landmarker model
# base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
# options = vision.FaceLandmarkerOptions(base_options=base_options,
#                                        output_face_blendshapes=True,
#                                        output_facial_transformation_matrixes=True,
#                                        num_faces=1)
# detector = vision.FaceLandmarker.create_from_options(options)

# # Start the webcam capture
# cap = cv2.VideoCapture(0)

# while True:
#     success, frame = cap.read()  # Capture frame-by-frame
#     if not success:
#         print("Ignoring empty camera frame.")
#         continue

#     # Convert the frame to RGB as mediapipe requires RGB images
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Create a mediapipe Image object from the RGB frame
#     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

#     # Detect face landmarks from the current frame
#     detection_result = detector.detect(image)

#     # Visualize the face landmarks on the frame
#     if detection_result.face_landmarks:
#         face_landmarks = detection_result.face_landmarks[0]

#         # Get the landmarks for the left eye
#         left_eye_top = face_landmarks[159]  # Top of the left eyelid
#         left_eye_bottom = face_landmarks[145]  # Bottom of the left eyelid

#         # Calculate the vertical distance between the top and bottom of the left eyelid
#         eye_distance = calculate_distance(left_eye_top, left_eye_bottom)

#         # Define a threshold for the eye being closed (this might need adjustment based on your setup)
#         eye_closed_threshold = 0.008  # This is a normalized value; you may need to adjust it

#         # Check if the eye is closed
#         if eye_distance < eye_closed_threshold:
#             print("Left eye closed")
#         else:
#             print("Left eye open")

#         # Annotate and display the landmarks on the frame
#         annotated_frame = draw_landmarks_on_image(frame_rgb, detection_result)
#         frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV display

#         # Show the frame
#         cv2.imshow('Face Landmarks', frame_bgr)

#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()  # Release the capture
# cv2.destroyAllWindows()
