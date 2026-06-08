import os
import cv2
import csv
import numpy as np
import mediapipe as mp



DATASET_PATH = "dataset"
OUTPUT_PATH = "data/landmarks_raw"

os.makedirs(OUTPUT_PATH, exist_ok=True)



mp_hands = mp.solutions.hands



def normalize_hand(hand_landmarks):
    """
    Wrist Centering + Scale Normalization
    Returns 63 features
    """

    wrist = hand_landmarks.landmark[0]

    wrist_x = wrist.x
    wrist_y = wrist.y
    wrist_z = wrist.z

    middle_mcp = hand_landmarks.landmark[9]

    scale = np.sqrt(
        (middle_mcp.x - wrist_x) ** 2 +
        (middle_mcp.y - wrist_y) ** 2 +
        (middle_mcp.z - wrist_z) ** 2
    )

    if scale < 1e-6:
        scale = 1e-6

    features = []

    for lm in hand_landmarks.landmark:

        x = (lm.x - wrist_x) / scale
        y = (lm.y - wrist_y) / scale
        z = (lm.z - wrist_z) / scale

        features.extend([x, y, z])

    return features




global_total = 0
global_success = 0
global_failed = 0



classes = sorted(os.listdir(DATASET_PATH))

for class_name in classes:

    class_path = os.path.join(DATASET_PATH, class_name)

    if not os.path.isdir(class_path):
        continue

    print("\n" + "=" * 50)
    print("CLASS:", class_name)
    print("=" * 50)

    output_file = os.path.join(
        OUTPUT_PATH,
        f"{class_name}.csv"
    )

    total = 0
    success = 0
    failed = 0

    with open(output_file, "w", newline="") as f:

        writer = csv.writer(f)

        

        header = []

        for hand_id in [1, 2]:

            for i in range(21):

                header.extend([
                    f"h{hand_id}_x{i}",
                    f"h{hand_id}_y{i}",
                    f"h{hand_id}_z{i}"
                ])

        header.append("label")

        writer.writerow(header)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        ) as hands:

            images = [

                img for img in os.listdir(class_path)

                if img.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )

            ]

            total = len(images)

            for img_name in images:

                img_path = os.path.join(
                    class_path,
                    img_name
                )

                image = cv2.imread(img_path)

                if image is None:
                    failed += 1
                    continue

                image_rgb = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2RGB
                )

                results = hands.process(image_rgb)

                if not results.multi_hand_landmarks:

                    failed += 1
                    continue

                row = []

                detected_hands = results.multi_hand_landmarks

                

                row.extend(
                    normalize_hand(
                        detected_hands[0]
                    )
                )

                

                if len(detected_hands) >= 2:

                    row.extend(
                        normalize_hand(
                            detected_hands[1]
                        )
                    )

                else:

                    row.extend([0.0] * 63)

                row.append(class_name)

                writer.writerow(row)

                success += 1

    print(f"Total Images      : {total}")
    print(f"Landmarks Saved   : {success}")
    print(f"No Hands Detected : {failed}")

    global_total += total
    global_success += success
    global_failed += failed



print("\n" + "=" * 60)
print("FINAL REPORT")
print("=" * 60)

print("Total Images      :", global_total)
print("Landmarks Saved   :", global_success)
print("No Hands Detected :", global_failed)

print("\nDataset Features:")
print("2 Hands × 21 Landmarks × 3 Coordinates")
print("= 126 Features")