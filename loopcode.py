import cv2
import math
import mediapipe as mp
import numpy as np
import csv
from csv import writer
import pandas as pd
import os
import re
from pathlib import Path

from feature_extraction import FEATURE_COLUMNS, extract_feature_vector
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

IMAGE_ROOT = Path(__file__).resolve().parent / "super_database"
OUTPUT_CSV = Path(__file__).resolve().parent / "all_features.csv"
IMAGE_PATTERN = re.compile(r"IMG_(\d+)\s*\((\d+)\)", re.IGNORECASE)


def discover_images():
    images = []
    for path in IMAGE_ROOT.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = IMAGE_PATTERN.search(path.stem)
        if match is None:
            continue
        person_id = int(match.group(1))
        image_index = int(match.group(2))
        images.append((person_id, image_index, path))
    images.sort(key=lambda item: (item[0], item[1], item[2].name))
    return images


def main():
    images = discover_images()
    if not images:
        raise FileNotFoundError(f"No hand images found in {IMAGE_ROOT}")

    with open(OUTPUT_CSV, "w", newline="") as csv_file:
        csv_writer = writer(csv_file)
        csv_writer.writerow(FEATURE_COLUMNS + ["Label"])

        processed = 0
        skipped = 0
        for person_id, image_index, image_path in images:
            try:
                features = extract_feature_vector(str(image_path))
                row = [features[column] for column in FEATURE_COLUMNS] + [f"Person {person_id}"]
                csv_writer.writerow(row)
                processed += 1
            except Exception as exc:
                skipped += 1
                print(f"Skipped {image_path.name}: {exc}")

            if processed and processed % 100 == 0:
                print(f"Processed {processed}/{len(images)} images...")

    print(f"Done. Wrote {processed} rows to {OUTPUT_CSV.name}.")
    if skipped:
        print(f"Skipped {skipped} images.")


raise SystemExit(main())

for i in range(100, 168):  # person
    if i == 6 or i == 16 or i == 52 or i==86:
        continue
    for n in range(1, 9):  # hand images
        with mp_hands.Hands(static_image_mode=True,  max_num_hands=1,  min_detection_confidence=0.5) as hands:
            path = r"C:\Users\skssu\OneDrive\Desktop\Working-Directory\projects\Finger_Geometry\super_database\IMG_" + \
                str(i) + "("+str(n)+").JPG"
            image = cv2.imread(path)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            arr = results.multi_hand_landmarks[0]
            # print("The cordinates of different hand landmarks are ")

            # wrist cordinates point 0
            wrist = []
            wrist.append(
                arr.landmark[mp_hands.HandLandmark.WRIST].x * image_width)
            wrist.append(
                arr.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
            # print("wrist",wrist)

            # thumb cmc point 1
            thumb_cmc = []
            thumb_cmc.append(
                arr.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
            thumb_cmc.append(
                arr.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
            # print("thumb cmc",thumb_cmc)

            # thumb mcp point 2
            thumb_mcp = []
            thumb_mcp.append(
                arr.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumb_mcp.append(
                arr.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            # print("thumb mcp",thumb_mcp)

            # thumb ip point 3
            thumb_ip = []
            thumb_ip.append(
                arr.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
            thumb_ip.append(
                arr.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
            # print("thumb ip",thumb_ip)

            # thumb tip point 4
            thumb_tip = []
            thumb_tip.append(
                arr.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
            thumb_tip.append(
                arr.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)
            # print("thumb tip",thumb_tip)

            # index_finger_mcp  point 5
            index_finger_mcp = []
            index_finger_mcp.append(
                arr.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
            index_finger_mcp.append(
                arr.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
            # print("index finger mcp",index_finger_mcp)

            # index_finger_pip point 6
            index_finger_pip = []
            index_finger_pip.append(
                arr.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
            index_finger_pip.append(
                arr.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
            # print("index finger pip",index_finger_pip)

            # index_finger_dip point 7
            index_finger_dip = []
            index_finger_dip.append(
                arr.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
            index_finger_dip.append(
                arr.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
            # print("index finger dip",index_finger_dip)

            # index_finger_tip point 8
            index_finger_tip = []
            index_finger_tip.append(
                arr.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
            index_finger_tip.append(
                arr.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
            # print("index finger tip",index_finger_tip)

            # middle_finger_mcp  point 9
            middle_finger_mcp = []
            middle_finger_mcp.append(
                arr.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
            middle_finger_mcp.append(
                arr.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
            # print("middle finger mcp ",middle_finger_mcp)

            # middle_finger_pip point 10
            middle_finger_pip = []
            middle_finger_pip.append(
                arr.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
            middle_finger_pip.append(
                arr.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
            # print("middle finger pip",middle_finger_pip)

            # middle_finger_dip point 11
            middle_finger_dip = []
            middle_finger_dip.append(
                arr.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
            middle_finger_dip.append(
                arr.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
            # print("middle finger dip",middle_finger_dip)

            # middle_finger_tip point 12
            middle_finger_tip = []
            middle_finger_tip.append(
                arr.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
            middle_finger_tip.append(
                arr.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)
            # print("middle finger tip",middle_finger_tip)

            # ring_finger_mcp  point 13
            ring_finger_mcp = []
            ring_finger_mcp.append(
                arr.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
            ring_finger_mcp.append(
                arr.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
            # print("ring finger mcp",ring_finger_mcp)

            # ring_finger_pip point 14
            ring_finger_pip = []
            ring_finger_pip.append(
                arr.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
            ring_finger_pip.append(
                arr.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
            # print("ring finger pip",ring_finger_pip)

            # ring_finger_dip point 15
            ring_finger_dip = []
            ring_finger_dip.append(
                arr.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
            ring_finger_dip.append(
                arr.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
            # print("ring finger dip",ring_finger_dip)

            # ring_finger_tip point 16
            ring_finger_tip = []
            ring_finger_tip.append(
                arr.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)
            ring_finger_tip.append(
                arr.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)
            # print("ring finger tip",ring_finger_tip)

            # pinky_mcp point 17
            pinky_mcp = []
            pinky_mcp.append(
                arr.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
            pinky_mcp.append(
                arr.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
            # print("pinky finger mcp",pinky_mcp)

            # pinky_mcp point 18
            pinky_pip = []
            pinky_pip.append(
                arr.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
            pinky_pip.append(
                arr.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
            # print("pinky finger pip",pinky_pip)

            # pinky_mcp point 19
            pinky_dip = []
            pinky_dip.append(
                arr.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
            pinky_dip.append(
                arr.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
            # print("pinky finger dip",pinky_dip)

            # pinky_mcp point 20
            pinky_tip = []
            pinky_tip.append(
                arr.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)
            pinky_tip.append(
                arr.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)
            # print("pinky finger tip",pinky_tip,"\n\n")

            # co-ordinates of the top-most point
            highest = []
            highest.append(thumb_tip)
            highest.append(index_finger_tip)
            highest.append(middle_finger_tip)
            highest.append(ring_finger_tip)
            highest.append(pinky_tip)

            # extremeties of the wrist
            d = math.dist(index_finger_mcp, pinky_mcp)
            d = d/2
            left = []
            left.append(wrist[0]-d)
            left.append(wrist[1])
            cv2.circle(image, center=(int(left[0]), int(
                left[1])), radius=2, color=(255, 0, 0), thickness=-1)
            right = []
            right.append(wrist[0]+d-5)
            right.append(wrist[1])
            cv2.circle(image, center=(int(right[0]), int(
                right[1])), radius=2, color=(255, 0, 0), thickness=-1)
            # cv2.imshow("wrist", image)
            # cv2.waitKey(0)
            # print("The cordinates of the extremeties of wrist are :","left---> ",left,"right---> ",right)
            width_of_wrist = math.dist(left, right)
            # print("The cordinates of the highest point is ",sorted(highest, key = lambda x: x[1])[0])
        # print("Length of Thumb finger ----------> ",math.dist(thumb_tip,thumb_mcp))
        # print("Length of Index finger ----------> ",math.dist(index_finger_tip,index_finger_mcp))
        # print("Length of Middle finger ----------> ",math.dist(middle_finger_tip,middle_finger_mcp))
        # print("Length of Ring finger ----------> ",math.dist(ring_finger_tip,ring_finger_mcp))
        # print("Length of Pinky finger ----------> ",math.dist(pinky_tip,pinky_mcp))
        # Property 1.Length of fingers calculation
        length = []
        length.append(math.dist(thumb_tip, thumb_mcp))
        length.append(math.dist(index_finger_tip, index_finger_mcp))
        length.append(math.dist(middle_finger_tip, middle_finger_mcp))
        length.append(math.dist(ring_finger_tip, ring_finger_mcp))
        length.append(math.dist(pinky_tip, pinky_mcp))
        longest = max(length)
        # print(longest)
        # **Property 2.Length of the Palm**
        # print("The diameter of the palm ",math.dist(wrist,middle_finger_mcp))
        mp_drawing.draw_landmarks(
            annotated_image, arr, mp_hands.HAND_CONNECTIONS)
        image2 = cv2.flip(annotated_image, 1)
        # cv2.imshow("marked image", image2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # ########################################3
        a = math.dist(wrist, index_finger_mcp)
        b = math.dist(wrist, pinky_mcp)
        c = math.dist(index_finger_mcp, pinky_mcp)
        s = (a+b+c)/2
        # print(s)
        # print(a)
        # print(b)
        # print(c)
        area_on_palm = math.sqrt(s*(s-a)*(s-b)*(s-c))
        peri_triangle = (a+b+c)
        # print("The area of the triangle formed on the palm = ",area_on_palm)
        ####################################################################

        # for the thumb
        # print("Thumb -----------> ",math.dist(thumb_tip,thumb_ip),",",math.dist(thumb_mcp,thumb_ip))
        # print("Index finger -----------> ",math.dist(index_finger_tip,index_finger_dip),",",math.dist(index_finger_pip,index_finger_dip),",",math.dist(index_finger_pip,index_finger_mcp))
        # print("Middle finger -----------> ",math.dist(middle_finger_tip,middle_finger_dip),",",math.dist(middle_finger_pip,middle_finger_dip),",",math.dist(middle_finger_pip,middle_finger_mcp))
        # print("Ring finger -----------> ",math.dist(ring_finger_tip,ring_finger_dip),",",math.dist(ring_finger_pip,ring_finger_dip),",",math.dist(ring_finger_pip,ring_finger_mcp))
        # print("Pinky finger -----------> ",math.dist(pinky_tip,pinky_dip),",",math.dist(pinky_pip,pinky_dip),",",math.dist(pinky_pip,pinky_mcp))
        ########################################################################
        # print("The distance is ",math.dist(highest[0],wrist))
        # print("Thumb-Index",math.dist(thumb_mcp,index_finger_mcp))
        # print("Index-Middle",math.dist(index_finger_mcp,middle_finger_mcp))
        # print("Middle-Ring",math.dist(middle_finger_mcp,ring_finger_mcp))
        # print("Ring-Pinky",math.dist(ring_finger_mcp,pinky_mcp))

        ###############################################################################

        a = math.dist(wrist, thumb_mcp)
        b = math.dist(thumb_mcp, index_finger_mcp)
        c = math.dist(index_finger_mcp, middle_finger_mcp)
        d = math.dist(middle_finger_mcp, ring_finger_mcp)
        e = math.dist(ring_finger_mcp, pinky_mcp)
        f = math.dist(pinky_mcp, wrist)
        # print("The perimeter of the polygon is ",(a+b+c+d+e+f))
        poly_peri = a+b+c+d+e+f

        ##############################################################################

        # print("The width of the wrist is ",math.dist(left,right))
        # print(math.dist(index_finger_mcp,pinky_mcp))
        #############################################################################

        a = math.dist(left, highest[0])
        b = math.dist(right, highest[0])
        c = math.dist(left, right)
        s = (a+b+c)/2
        area_on_hand = math.sqrt(s*(s-a)*(s-b)*(s-c))
        # print("The area of the triangle formed on the palm = ",area_on_hand)
        #############################################################################

        a = math.dist(left, thumb_tip)
        b = math.dist(index_finger_tip, thumb_tip)
        c = math.dist(index_finger_tip, middle_finger_tip)
        d = math.dist(middle_finger_tip, ring_finger_tip)
        e = math.dist(ring_finger_tip, pinky_tip)
        f = math.dist(pinky_tip, right)
        g = a = math.dist(left, right)
        # print("The perimeter of the polygon is ",(a+b+c+d+e+f+g))

        x1 = index_finger_pip[0]
        x2 = middle_finger_pip[0]
        x3 = ring_finger_pip[0]
        y1 = index_finger_pip[1]
        y2 = middle_finger_pip[1]
        y3 = ring_finger_pip[1]

        # print(x2,y2)

        x12 = x1 - x2
        x13 = x1 - x3

        y12 = y1 - y2
        y13 = y1 - y3

        y31 = y3 - y1
        y21 = y2 - y1

        x31 = x3 - x1
        x21 = x2 - x1

        # x1^2 - x3^2
        sx13 = pow(x1, 2) - pow(x3, 2)

        # y1^2 - y3^2
        sy13 = pow(y1, 2) - pow(y3, 2)

        sx21 = pow(x2, 2) - pow(x1, 2)
        sy21 = pow(y2, 2) - pow(y1, 2)

        f = (((sx13) * (x12) + (sy13) *
              (x12) + (sx21) * (x13) +
              (sy21) * (x13)) // (2 *
                                  ((y31) * (x12) - (y21) * (x13))))

        g = (((sx13) * (y12) + (sy13) * (y12) +
              (sx21) * (y13) + (sy21) * (y13)) //
             (2 * ((x31) * (y12) - (x21) * (y13))))

        c = (-pow(x1, 2) - pow(y1, 2) -
             2 * g * x1 - 2 * f * y1)

        # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
        # where centre is (h = -g, k = -f) and
        # radius r as r^2 = h^2 + k^2 - c
        h = -g
        k = -f
        sqr_of_r = h * h + k * k - c

        # r is the radius
        r = round(math.sqrt(sqr_of_r), 5)

        # print("Centre = (", h, ", ", k, ")")
        # print("Radius = ", r)
        ################################################################################
        # print("Thumb : Longest finger ",math.dist(thumb_tip,thumb_mcp)/longest)
        # print("Index finger : Longest finger ",math.dist(index_finger_tip,index_finger_mcp)/longest)
        # print("Middle finger : Longest finger ",math.dist(middle_finger_tip,middle_finger_mcp)/longest)
        # print("Ring finger : Longest finger ",math.dist(ring_finger_tip,ring_finger_mcp)/longest)
        # print("Pinky finger : Longest finger ",math.dist(pinky_tip,pinky_mcp)/longest)
        a = math.dist(thumb_tip, thumb_mcp)/longest
        b = math.dist(index_finger_tip, index_finger_mcp)/longest
        c = math.dist(middle_finger_tip, middle_finger_mcp)/longest
        d = math.dist(ring_finger_tip, ring_finger_mcp)/longest
        e = math.dist(pinky_tip, pinky_mcp)/longest

        #########################################################################################

        # print(area_on_palm/(width_of_wrist*math.dist(wrist,middle_finger_mcp)))
        f2 = area_on_palm/(width_of_wrist*math.dist(wrist, middle_finger_mcp))
        ##################################################################################

        # print(math.dist(wrist,middle_finger_mcp)/longest)
        f3 = math.dist(wrist, middle_finger_mcp)/longest

        ####################################################################################

        # print(poly_peri/peri_triangle)
        f4 = poly_peri/peri_triangle

        #####################################################################################

        # print(area_on_palm/area_on_hand)
        f5 = area_on_palm/area_on_hand

        ####################################################################################
        # print(math.dist(index_finger_tip,wrist)/width_of_wrist)
        f6 = math.dist(index_finger_tip, wrist)/width_of_wrist

        ###################################################################################
        # List that we want to add as a new row
        List = [a, b, c, d, e, f2, f3, f4, f5, f6, "Person "+str(i)]

        # Open our existing CSV file in append mode
        # Create a file object for this file
        with open('all_features.csv', 'a', newline='') as f_object:

            #     # Pass this file object to csv.writer()
            #     # and get a writer object
            writer_object = writer(f_object)

        #     # Pass the list as an argument into
        #     # the writerow()
            writer_object.writerow(List)

        #     # Close the file object
            f_object.close()
