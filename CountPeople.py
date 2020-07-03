from __future__ import print_function
import cv2 as cv
import argparse

# parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
# parser.add_argument('--body_cascade', help='Path to face cascade.',
#                     default='blpcvenv/Lib/site-packages/cv2/data/haarcascade_upperbody.xml')
# # parser.add_argument('--eyes_cascade', help='Path to eyes cascade.',
# #                     default='blpcvenv/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml')
# parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
# args = parser.parse_args()
# body_cascade_name = args.body_cascade
# eyes_cascade_name = args.eyes_cascade
# body_cascade = cv.CascadeClassifier()
body_cascade = cv.CascadeClassifier('blpcvenv/Lib/site-packages/cv2/data/haarcascade_upperbody.xml')
# body_cascade = cv.CascadeClassifier('blpcvenv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# eyes_cascade = cv.CascadeClassifier()
# -- 1. Load the cascades
# if not body_cascade.load(cv.samples.findFile(body_cascade_name)):
#     print('--(!)Error loading face cascade')
#     exit(0)
# if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
#     print('--(!)Error loading eyes cascade')
#     exit(0)


def detectAndDisplay(frame, j=1.04, i=12, threh=127):
    # frameArea = frame.shape[0] * frame.shape[1]
    # areaTH = frameArea / 250
    frame = cv.resize(frame, (0, 0), fx=0.7, fy=0.7)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # ret, frame_gray = cv.threshold(frame_gray, threh, 255, cv.THRESH_BINARY)
    #
    # cv.imshow('Capture - body detection', frame_gray)
    #
    # if cv.waitKey(0) & 0xFF == ord('q'):
    #     cv.destroyAllWindows()
    # -- Detect faces
    people = body_cascade.detectMultiScale(frame_gray, j, i)
    print(people)
    for (x, y, w, h) in people:
        center = (x + w // 2, y + h // 2)
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)
        # faceROI = frame_gray[y:y + h, x:x + w]
        # -- In each face, detect eyes
        # eyes = eyes_cascade.detectMultiScale(faceROI)
        # for (x2, y2, w2, h2) in eyes:
        #     eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
        #     radius = int(round((w2 + h2) * 0.25))
        #     frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)
    cv.imshow('Capture - body detection', frame)





def app(camera_device = 0):
    # camera_device = args.camera
    # camera_device = 0
    # -- 2. Read the video stream
    cap = cv.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        # frame = cv.imread('sample0.jpg')
        detectAndDisplay(frame)
        pressedKey = cv.waitKey(1) & 0xFF
        if pressedKey == ord('q'):
            break


if __name__ == '__main__':
    app(camera_device='test_videos/0001.mp4')
    # im = cv.imread('sample2.jpg')
    # while True:
    #     j = float(input('1== '))
    #     i = int(input('2== '))
    #     thresh = int(input('threh== '))
    #     detectAndDisplay(im, j, i, thresh)
    #     # gray_frame = (cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY))
    #     # cv.imshow('Capture - Face detection', im)
    #     key = cv.waitKey(0) & 0xFF
    #     if key == ord('c'):
    #         cv.destroyAllWindows()
    #     if key == ord('q'):
    #         cv.destroyAllWindows()
    #         break
