# import cv2
# cap = cv2.VideoCapture('data/videos/hello1p.mp4')
# if cap.isOpened():
#     print("Video opened successfully!")
# else:
#     print("Failed to open video.")
# # Read frames from the video
# ret, frame = cap.read()

# if not ret:
#     print("Error: Could not read frame from video.")
# else:
#     print("Successfully read a frame.")
# model = load_model('model/lip_model.h5', compile=False)
# model.summary()


# import cv2
# import numpy as np

# def show_mouth_frames(video_path, frame_count=75):
#     cap = cv2.VideoCapture(video_path)
#     count = 0

#     if not cap.isOpened():
#         print(f"❌ Error: Cannot open video: {video_path}")
#         return

#     while count < frame_count:
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ End of video or read error.")
#             break

#         h, w, _ = frame.shape

#         # Adjusted ROI: move higher for better lip capture
#         x1, y1 = int(w * 0.35), int(h * 0.45)
#         x2, y2 = int(w * 0.65), int(h * 0.65)

#         # Draw box on original frame for reference
#         frame_with_box = frame.copy()
#         cv2.rectangle(frame_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Crop the mouth region
#         mouth_roi = frame[y1:y2, x1:x2]

#         # Resize for uniformity
#         if mouth_roi.size != 0:
#             mouth_resized = cv2.resize(mouth_roi, (100, 50))
#         else:
#             print(f"⚠️ Empty mouth ROI at frame {count}")
#             count += 1
#             continue

#         # Display original frame and cropped mouth region
#         cv2.imshow("Original Frame with ROI", frame_with_box)
#         cv2.imshow("Mouth Region", mouth_resized)

#         key = cv2.waitKey(100)  # Wait 100 ms between frames
#         if key == 27:  # ESC to quit early
#             break

#         count += 1

#     cap.release()
#     cv2.destroyAllWindows()

# show_mouth_frames("data/newdata/hello2n.mp4")
import cv2
import numpy as np

FRAME_WIDTH = 100
FRAME_HEIGHT = 50

def extract_mouth_frames(video_path, max_frames=75):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return frames

    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ End of video or error at frame {count}")
            break

        h, w, _ = frame.shape
        x1, y1 = int(w * 0.35), int(h * 0.45)
        x2, y2 = int(w * 0.65), int(h * 0.65)

        mouth_roi = frame[y1:y2, x1:x2]

        if mouth_roi.size != 0:
            mouth_resized = cv2.resize(mouth_roi, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append(mouth_resized)
        else:
            print(f"⚠️ Skipping empty ROI at frame {count}")

        count += 1

    cap.release()
    return frames

# test_extract.py
frames = extract_mouth_frames("data/testvideo/sampworld.mp4")
print(f"✅ Total extracted frames: {len(frames)}")
cv2.imshow("Frame Example", frames[12])
cv2.waitKey(0)
cv2.destroyAllWindows()
