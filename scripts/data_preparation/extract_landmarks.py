import os
import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands


dataset_path = "dataset"
output_base = "landmarks_raw"


os.makedirs(output_base, exist_ok=True)


global_total = 0
global_success = 0
global_failed = 0


classes = os.listdir(dataset_path)

for class_name in classes:

    class_path = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_path):
        continue

    print("\n==============================")
    print(f" Class: {class_name}")

    output_file = os.path.join(output_base, f"{class_name}.csv")

    total = 0
    success = 0
    failed = 0

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:

            images = [img for img in os.listdir(class_path)
                      if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

            total = len(images)

            for img_name in images:

                img_path = os.path.join(class_path, img_name)

                image = cv2.imread(img_path)

                if image is None:
                    failed += 1
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:

                    success += 1

                    for handLms in results.multi_hand_landmarks:
                        row = []

                        for lm in handLms.landmark:
                            row.extend([lm.x, lm.y, lm.z])

                        row.append(class_name)
                        writer.writerow(row)

                else:
                    failed += 1

   
    print(f" Total: {total}")
    print(f" Success: {success}")
    print(f" No hand detected: {failed}")

    global_total += total
    global_success += success
    global_failed += failed


print("\n==============================")
print(" FINAL REPORT")
print("==============================")
print(f" Total Images: {global_total}")
print(f" Total Landmarks: {global_success}")
print(f" Total No Hand Detected: {global_failed}")