# from __future__ import print_function
import cv2
import argparse
import math
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                  OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.',
                        default='test_videos\\long1.mp4')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()
    if args.algo == 'MOG2':
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        backSub = cv2.createBackgroundSubtractorKNN()

    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))
    if not capture.isOpened:
        print('Unable to open: ' + args.input)
        exit(0)
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break


def dist(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def findCorners(cnt):
    maxX = 0
    minX = 99999
    maxY = 0
    minY = 99999

    pairmaxX = []
    pairminX = []
    pairmaxY = []
    pairminY = []

    for tcour in cnt:
        cour = tcour[0]
        # maxX=max(maxX,cour[0])
        if cour[0] > maxX:
            maxX = cour[0]
            pairmaxX = cour
        # minX=min(minX,cour[0])
        if cour[0] < minX:
            minX = cour[0]
            pairminX = cour
        # maxY=max(maxY,cour[1])
        if cour[1] > maxY:
            maxY = cour[1]
            pairmaxY = cour
        # minY=min(minY,cour[1])
        if cour[1] < minY:
            minY = cour[1]
            pairminY = cour
    return pairminX, pairminY, pairmaxX, pairmaxY


def findCenter(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


def check_if_close(A, B, d=10):
    if type(A) == type({'d':0}):
        if dist(A['cp'], B['cp']) <= d:
            return True
        else:
            return False
        # Ax, Ay = A['cp']
        # Bx, By = B['cp']
    else:
        if dist(A, B) <= d:
            return True
        else:
            return False
    #     Ax, Ay = A
    #     Bx, By = B
    # if Bx - dist < Ax < Bx + dist and By - dist < Ay < By + dist:
    #     return True
    # else:
    #     return False

def make_cnt_datadict(contour):
    x, y, w, h = cv2.boundingRect(contour)
    cp = findCenter(contour)
    return {'cp': cp, 'x': x, 'y': y, 'w': w, 'h': h , 'life':3}

def app(camera_device=0):
    # camera_device = args.camera
    # camera_device = 0
    # -- 2. Read the video stream
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    first_frame = None
    nonmoving_cnts = []
    prev_contours = None
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 1.1)
        if first_frame is None:
            first_frame = gray
            height, width = frame.shape[:2]
        delta_frame = cv2.absdiff(first_frame, gray)
        # delta_frame = cv2.GaussianBlur(delta_frame, (11, 11), 1.8)
        # print(delta_frame.dtype)
        # delta_frame = (delta_frame * 2.5).astype(delta_frame.dtype)
        thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # _SIMPLE

        hier = cnts[1]
        cnts = cnts[0]
        if not prev_contours:
            prev_contours = []
            for i, contour in enumerate(cnts):
                if hier[0][i][2] == -1 and 50 < cv2.contourArea(contour) < 2600:
                    # minX, minY, maxX, maxY = findCorners(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    cp = findCenter(contour)
                    if not ((dist((x, y), (x, y + h)) > 150 and dist((x, y), (x + w, y)) < 15) or (
                            dist((x, y), (x, y + h)) < 15 and dist((x, y), (x + w, y)) > 150)):
                        prev_contours.append(make_cnt_datadict(contour))
            continue
        curent_non_moving_contours = []
        distances = []
        to_append = []
        indexes = []
        for i, contour in enumerate(cnts):
            # print('edw')

            # print('hier')
            # print(i)
            # print(hier[0][i])
            if hier[0][i][2] == -1 and 100 < cv2.contourArea(contour) < 2600:
                # minX, minY, maxX, maxY = findCorners(contour)
                x, y, w, h = cv2.boundingRect(contour)
                cp= findCenter(contour)
                if not ((dist((x, y), (x, y + h)) > 200 and dist((x, y), (x + w, y)) < 15) or (
                        dist((x, y), (x, y + h)) < 25 and dist((x, y), (x + w, y)) > 200) or
                        (dist((x, y), (x, y + h)) > 200 and dist((x, y), (x + w, y)) > 200)):
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.drawContours(frame, contour, -1, (0, 255, 0), 2)
                    cur_elem = make_cnt_datadict(contour)
                    for elem in nonmoving_cnts:
                        print(len(nonmoving_cnts))
                        if check_if_close(elem, cur_elem, 25):
                            elem['life'] += 1
                            indexes.append(nonmoving_cnts.index(elem))
                            # equals.append(elem)
                            # curent_non_moving_contours.remove(cur_elem)
                            break
                    else:
                        to_append.append(cur_elem)
        to_delete = []
        for i in range(len(nonmoving_cnts)):
            if nonmoving_cnts[i] in indexes:
                nonmoving_cnts[i]['life'] += 1
            else:
                nonmoving_cnts[i]['life'] -= 1
            if nonmoving_cnts[i]['life'] <= 0:
                to_delete.append(nonmoving_cnts[i])
        for elem in to_delete:
            nonmoving_cnts.remove(elem)
        nonmoving_cnts += to_append.copy()

        cv2.imshow('original', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
        cv2.imshow('delta', cv2.resize(delta_frame, (0, 0), fx=0.5, fy=0.5))
        cv2.imshow('thresh', cv2.resize(thresh_frame, (0, 0), fx=0.5, fy=0.5))
        pressedKey = cv2.waitKey(1) & 0xFF

        # input()
        if pressedKey == ord('q'):
            cv2.destroyAllWindows()
            break

    print('=================================================')
    print(len(nonmoving_cnts))
    for elem in nonmoving_cnts:
        cv2.rectangle(first_frame, (elem['x'], elem['y']), (elem['x'] + elem['w'], elem['y'] + elem['h']), (0, 255, 0), 2)
    cv2.imshow('original', cv2.resize(first_frame, (0, 0), fx=0.5, fy=0.5))
    pressedKey = cv2.waitKey(0) & 0xFF
    if pressedKey == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app('test_videos\\0001.mp4')
