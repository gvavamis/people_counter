import numpy as np
# import cv2 as cv
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import math
import dlib
import cv2
import os
from datetime import datetime
import time
import Person


def dist(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def calc_r(rect, div=2):
    xlen = int((rect[2] - rect[0]) / div)
    ylen = int((rect[3] - rect[1]) / div)
    return rect[0] - xlen, rect[1] - ylen, rect[2] + xlen, rect[3] + ylen


class PeopleCounter():

    def __init__(self, cap=0, in_direction='D', logfile_path='', people_in=0, upper_padding=1):
        self.mtp = 2  # epi poso tha einai pio megalo to kathe tetragono wste na ginei upoparathiro
        WORKING_DIR = os.path.abspath(__file__)
        self.WORKING_DIR = WORKING_DIR[:len(WORKING_DIR) - len(WORKING_DIR.split('\\')[-1])]
        self.IN_DIRECTION = in_direction
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        self.person_class_ID = CLASSES.index('person')

        prototxt = 'mobilenet_ssd\\MobileNetSSD_deploy.prototxt'
        model = 'mobilenet_ssd\\MobileNetSSD_deploy.caffemodel'
        self.confidence = 0.1
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        if self.IN_DIRECTION == 'D':
            self.OUT_DIRECTION = 'U'
        elif self.IN_DIRECTION == 'U':
            self.OUT_DIRECTION = 'D'
        else:
            print('DIRECTION INPUT ERROR')
            exit(IndexError)
        self.LOG_FILE_TYPE = '.txt'

        self.CAP = cv2.VideoCapture(cap)
        if not self.CAP.isOpened:
            print('--(!)Error opening video capture')
            exit(0)

        self.count_up = 0
        self.count_down = 0

        self.total_in = 0
        self.total_out = 0

        self.people_in = people_in

        self.upper_padding = upper_padding

        self.h_resize = 1
        self.w_resize = 1

        self.Vheight = int(self.CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))  # *self.upper_padding)
        self.Vwidth = int(self.CAP.get(cv2.CAP_PROP_FRAME_WIDTH))

        # if self.Vwidth > 500:
        #     self.Vwidth = 500

        if self.upper_padding == 1:
            self.ofset = 0
        else:
            self.ofset = int(self.Vheight * self.upper_padding)

        frameArea = self.Vheight * self.Vwidth
        self.THRESSHOLD_AREA = frameArea * 250 / 307200
        self.dist_thress = 30

        print('THRESSHOLD_AREA THRESSHOLD_AREA THRESSHOLD_AREA')
        print(self.THRESSHOLD_AREA)

        # self.line_up = int(5 * (self.Vheight / 11))  # +self.ofset
        # self.line_down = int(6 * (self.Vheight / 11))  # +self.ofset
        # self.up_limit = int(4 * (self.Vheight / 11))  # +self.ofset
        # self.down_limit = int(7 * (self.Vheight / 11))  # +self.ofset

        self.up_limit = int(3.5 * (self.Vheight / 10))  # +self.ofset
        self.middle_line = int(5 * (self.Vheight / 10))  # +self.ofset
        self.down_limit = int(6.5 * (self.Vheight / 10))  # +self.ofset

        # print(self.Vheight)
        # print('---------------')
        # print(self.line_up)
        # print(self.line_down)
        # print(self.up_limit)
        # print(self.down_limit)
        # input()
        #
        self.line_limit_down_color = (0, 255, 0)
        self.line_limit_up_color = (0, 0, 255)
        self.line_middle_color = (255, 255, 255)

        if logfile_path:
            try:
                self.LOG = open(logfile_path, "w+")
                self.LOG.write('TIME,IN/OUT,TOTAL_IN,TOTAL_OUT,CURRENT_IN;\n')
                # log = open('log' + self.LOG_FILE_TYPE, "w")
            except:
                print("file-not-found")
        else:
            log_file_name = 'People_counter_at_' + datetime.now().strftime(
                '%d-%m-%y__%I_%M%p') + self.LOG_FILE_TYPE
            # print('log_file_name')
            # print(log_file_name)
            # input()
            self.LOG = open(log_file_name, 'w+')
            self.LOG.write('TIME,IN/OUT,TOTAL_IN,TOTAL_OUT,CURRENT_IN;\n')

        self.monitors = [True, True]

    def set_resize(self, frame, w, h):
        # TODO den einai teleiomenh
        self.h_resize = h
        self.w_resize = w
        self.Vheight = int(self.Vheight * self.h_resize)
        self.Vwidth = int(self.Vwidth * self.w_resize)
        frameArea = self.Vheight * self.Vwidth
        self.THRESSHOLD_AREA = frameArea / 250 / 307200

        self.up_limit = int(4 * (self.Vheight / 10))  # +self.ofset
        self.middle_line = int(5 * (self.Vheight / 10))  # +self.ofset
        self.down_limit = int(6 * (self.Vheight / 10))  # +self.ofset
        return cv2.resize(frame, (0, 0), fx=self.w_resize, fy=self.h_resize)

    def update_Log(self, this_person):
        direction = this_person.getDir()
        # self.total_in, self.total_out = (self.count_up, self.count_down) if self.IN_DIRECTION == 'up' else (
        # self.count_down, self.count_up)
        this_person.set_counted()
        res = ''
        if direction == self.IN_DIRECTION:
            self.total_in += 1
            self.people_in += 1
            res = 'IN'
        elif direction == self.OUT_DIRECTION:
            self.total_out += 1
            self.people_in -= 1
            res = 'OUT'
        if res:
            print("ID:", this_person.getId(), 'Gets', res, time.strftime("%H:%M:%S"))
            self.LOG.write(
                time.strftime("%H:%M:%S") + ',' + res + ',' + str(self.total_in) + ',' + str(
                    self.total_out) + ',' + str(
                    self.people_in) + ';\n')

    def open_live_monitor(self, frame, people, show_details=True, show_lines=True, scale=0.6):
        image = frame.copy()
        pt3 = [0, self.middle_line]
        pt4 = [self.Vwidth, self.middle_line]
        pts_L2 = np.array([pt3, pt4], np.int32)
        pts_L2 = pts_L2.reshape((-1, 1, 2))

        pt5 = [0, self.up_limit]
        pt6 = [self.Vwidth, self.up_limit]
        pts_L3 = np.array([pt5, pt6], np.int32)
        pts_L3 = pts_L3.reshape((-1, 1, 2))

        pt7 = [0, self.down_limit]
        pt8 = [self.Vwidth, self.down_limit]
        pts_L4 = np.array([pt7, pt8], np.int32)
        pts_L4 = pts_L4.reshape((-1, 1, 2))

        font = cv2.FONT_HERSHEY_SIMPLEX

        if show_details:
            total_in_str = 'People entered: ' + str(self.total_in)
            total_out_str = 'People came out: ' + str(self.total_out)
            people_in_now_str = 'now in: ' + str(self.people_in)

            cv2.putText(image, total_in_str, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, total_in_str, (10, 40), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(image, total_out_str, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, total_out_str, (10, 90), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, people_in_now_str, (10, 140), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, people_in_now_str, (10, 140), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if show_lines:
            image = cv2.polylines(image, [pts_L2], False, self.line_middle_color, thickness=2)
            image = cv2.polylines(image, [pts_L3], False, self.line_limit_up_color, thickness=1)
            image = cv2.polylines(image, [pts_L4], False, self.line_limit_down_color, thickness=1)

        # image = cv2.resize(image, (0, 0), fx=0.6, fy=0.6)
        cv2.imshow('LIVE', image)

    def find_center(self, rectangle):
        center = int((rectangle[2] - rectangle[0]) / 2) + rectangle[0], int((rectangle[3] - rectangle[1]) / 2) + \
                 rectangle[1]
        return center

    def run(self):
        background = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        kernelOp = np.ones((3, 3), np.uint8)
        kernelOp2 = np.ones((5, 5), np.uint8)
        kernelCl = np.ones((11, 11), np.uint8)

        people = []
        max_p_age = 5
        pid = 1
        frame_id = 0
        while self.CAP.isOpened():
            # print('%%%%%%%%%%%%%%%%%')
            # print('frame: ', frame_id)
            to_remove = []
            ret, frame = self.CAP.read()
            if frame is None:
                break
            # frame = self.set_resize(frame, 0.5, 0.5)

            # frame = imutils.resize(frame, width=self.Vwidth)
            # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_frame = frame.copy()
            # frame = cv2.resize(frame, (0, 0), fx=self.w_resize, fy=self.h_resize)

            # for i in persons:
            #     i.age_one()  # age every person one frame

            fgmask = background.apply(frame)
            fgmask2 = background.apply(frame)

            try:
                ret, imBin = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
                ret, imBin2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)

                mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
                mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)

                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)

            except:
                # TODO na stelnei se sunarthsh gia end
                print('EOF')
                print('UP:', self.count_up)
                print('DOWN:', self.count_down)
                break

            # ARXH TOU ALGORUTHMOU

            temp = people.copy()
            # print('people', len(people))
            to_append = []
            while len(temp):
                this_person = temp.pop()
                to_append = []
                # print(this_person.is_done())
                # if this_person.is_done() == 'delete':
                #     continue
                # for this_person in people:

                startX, startY, endX, endY = this_person.getRect()
                cntr = self.find_center((startX, startY, endX, endY))
                Xlen = (endX - startX) * self.mtp
                Ylen = (endY - startY) * self.mtp
                # Xlen = w * mtp
                # Ylen = h * mtp
                xExtra = int(Xlen / 2)
                yExtra = int(Ylen / 2)
                # print('=========')
                A = startY - yExtra
                B = endY + yExtra
                C = startX - xExtra
                D = endX + xExtra
                if A < 0:
                    A = 0
                if B > frame.shape[0]:
                    B = frame.shape[0]
                if C < 0:
                    C = 0
                elif D > frame.shape[1]:
                    D = frame.shape[1]

                # print(A, B, C, D)
                # img[y:y + h, x:x + w]
                # temp_rgb = rgb[A: B, C: D]

                temp_mask = mask2[A: B, C: D]
                temp_frame = frame[A: B, C: D]

                # cv2.imshow('temp_frame', temp_frame)
                # k = cv2.waitKey(30) & 0xff
                # if k == ord('c'):
                #     cv2.destroyWindow(temp_frame)

                output_frame = cv2.rectangle(output_frame, (C, A), (D, B), (0, 255, 255), 2)

                blob = cv2.dnn.blobFromImage(temp_frame, 0.007843, (D - C, B - A), 127.5)
                # blob = cv2.dnn.blobFromImage(temp_frame, 0.007843, (D-C, B-A), 127.5)
                self.net.setInput(blob)
                detections = self.net.forward()

                # print('detection')
                # print(detections)

                rect_detections = []
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > self.confidence:
                        idx = int(detections[0, 0, i, 1])

                        if idx != self.person_class_ID:
                            continue

                        box = detections[0, 0, i, 3:7] * np.array([D - C, B - A, D - C, B - A])

                        # tmp_rect = box.astype("int") + np.array([startX, startY, startX, startY])
                        tmp_rect = box.astype("int") + np.array([C, A, C, A])

                        rect_detections.append(tmp_rect)
                        del tmp_rect
                updated = False
                # print('len(rect_detections)', len(rect_detections))
                for i, r in enumerate(rect_detections):
                    c = self.find_center(r)
                    r0 = calc_r(this_person.getRect())
                    output_frame = cv2.rectangle(output_frame, (r0[0], r0[1]), (r0[2], r0[3]), (255, 0, 0), 2)
                    # print('rect', this_person.getRect())
                    # print('r0', r0)
                    # print('c', c)
                    # input()
                    closer_dist = min([dist(c, (_.getX(), _.getY())) for _ in temp]) if len(temp) else None
                    if c[0] in range(r0[0], r0[2]) and c[1] in range(r0[1], r0[3]):
                        if len(rect_detections) == 1:
                            this_person.updateCoords(c[0], c[1], r)
                            print('UPDATE_0 : ', this_person.getId(), 'age: ', this_person.get_age(),
                                  'status: ', this_person.get_status())
                            updated = True
                        else:
                            this_person_dist = dist(c, (this_person.getX(), this_person.getY()))
                            if closer_dist and closer_dist < this_person_dist:
                                canditade_person = None
                                for temp_person in temp:
                                    if dist(c, (temp_person.getX(), temp_person.getY())) < this_person_dist:
                                        canditade_person = temp_person
                                        break
                                if canditade_person is not None:
                                    if this_person.get_tracks_len() > canditade_person.get_tracks_len():
                                        this_person.updateCoords(c[0], c[1], r)
                                        print('UPDATE_1 : ', this_person.getId(), 'age:', this_person.get_age(),
                                              'status: ', this_person.get_status())
                                        updated = True
                                    else:
                                        canditade_person.updateCoords(c[0], c[1], r)
                        # del rect_detections[i]
                    else:
                        for temp_person in temp:
                            r0 = calc_r(temp_person.getRect())
                            if c[0] in range(r0[0], r0[2]) and c[1] in range(r0[1], r0[3]):
                                temp_person.updateCoords(c[0], c[1], r)
                                print('UPDATE : ', this_person.getId(), ' DELETED: ', temp_person.getId(), )
                                # del rect_detections[i]
                                temp.remove(temp_person)
                                break
                        else:  # else ths for (dld an den brei kanena allo na tairiazei
                            if closer_dist:
                                if closer_dist > self.dist_thress:
                                    p = Person.MyPerson(pid, c[0], c[1], self.middle_line, self.down_limit,
                                                        self.up_limit,
                                                        r)
                                    to_append.append(p)
                                    pid += 1

                    tmp_m = temp_mask[r[1] - A:r[2] - A, r[0] - C:r[3] - C]
                    contours0, hierarchy = cv2.findContours(tmp_m, cv2.RETR_EXTERNAL,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours0:
                        tmp_m = cv2.fillPoly(tmp_m, [cnt], (0, 0, 0))
                    temp_mask[r[1] - A:r[2] - A, r[0] - C:r[3] - C] = tmp_m

                if not updated:
                    this_person.forced_move()
                    # print('FORCED_MOVE   ', this_person.getId())

                mask2[A: B, C: D] = temp_mask
            people += to_append.copy()
            to_append = []

            for this_person in people:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(output_frame, str(this_person.getId()),
                            (this_person.getRect()[0], this_person.getRect()[1]),
                            font, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(output_frame, (this_person.getX(), this_person.getY()), 5, (0, 0, 255), -1)
                # result = this_person.get_status()
                result = this_person.is_done()
                if result == 'count':
                    self.update_Log(this_person)
                    people.remove(this_person)
                elif result == 'delete':
                    people.remove(this_person)
                else:
                    R = this_person.getRect()
                    if this_person.getDir() == self.IN_DIRECTION:
                        output_frame = cv2.rectangle(output_frame, (R[0], R[1]), (R[2], R[3]), (0, 255, 0), 2)
                    elif this_person.getDir() == self.OUT_DIRECTION:
                        output_frame = cv2.rectangle(output_frame, (R[0], R[1]), (R[2], R[3]), (0, 0, 255), 2)
                    else:
                        output_frame = cv2.rectangle(output_frame, (R[0], R[1]), (R[2], R[3]), (255, 0, 0), 2)

                # if this_person.is_done():
                #     if not this_person.is_counted():
                #         self.update_Log(this_person)
                # if frame_id > this_person.get_frame_id() + this_person.getMaxAge():
                #     people.remove(this_person)
                # else:
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     cv2.circle(output_frame, (this_person.getX(), this_person.getY()), 3, (0, 0, 255), -1)
                #     cv2.putText(output_frame, str(this_person.getId()), (this_person.getX(), this_person.getY()),
                #                 font, 1, this_person.getRGB(), 1, cv2.LINE_AA)
                #     R = this_person.getRect()
                #     if this_person.getDir() == self.IN_DIRECTION:
                #         output_frame = cv2.rectangle(output_frame, (R[0], R[1]), (R[2], R[3]), (0, 255, 0), 2)
                #     else:
                #         output_frame = cv2.rectangle(output_frame, (R[0], R[1]), (R[2], R[3]), (0, 0, 255), 2)

            # ----------------------------------------------------------------------------------------------------------
            contours0, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours0:
                cnt_area = cv2.contourArea(cnt)
                if cnt_area > self.THRESSHOLD_AREA:
                    x, y, w, h = cv2.boundingRect(cnt)
                    startX, startY, endX, endY = x, y, x + w, y + h
                    cntr = self.find_center((startX, startY, endX, endY))
                    if self.up_limit < cntr[1] < self.down_limit:
                        # print(cntr)
                        Xlen = (endX - startX) * 1.1  # self.mtp
                        Ylen = (endY - startY) * 1.1  # self.mtp
                        # Xlen = w * mtp
                        # Ylen = h * mtp
                        xExtra = int(Xlen / 2)
                        yExtra = int(Ylen / 2)
                        # print('=========')
                        A = startY - yExtra
                        B = endY + yExtra
                        C = startX - xExtra
                        D = endX + xExtra
                        if A < 0:
                            A = 0
                        if B > frame.shape[0]:
                            B = frame.shape[0]
                        if C < 0:
                            C = 0
                        elif D > frame.shape[1]:
                            D = frame.shape[1]

                        temp_mask = mask2[A: B, C: D]
                        temp_frame = frame[A: B, C: D]
                        blob = cv2.dnn.blobFromImage(temp_frame, 0.007843, (D - C, B - A), 127.5)
                        self.net.setInput(blob)

                        detections = self.net.forward()

                        rect_detections = []

                        for i in np.arange(0, detections.shape[2]):
                            confidence = detections[0, 0, i, 2]

                            if confidence > self.confidence:
                                idx = int(detections[0, 0, i, 1])

                                if idx != self.person_class_ID:
                                    continue

                                box = detections[0, 0, i, 3:7] * np.array([D - C, B - A, D - C, B - A])

                                # tmp_rect = box.astype("int") + np.array([startX, startY, startX, startY])
                                tmp_rect = box.astype("int") + np.array([C, A, C, A])

                                rect_detections.append(tmp_rect)
                                del tmp_rect

                        for r in rect_detections:
                            c = self.find_center(r)
                            if not any([c[0] in range(calc_r(p.getRect())[0], calc_r(p.getRect())[2]) and c[1] in
                                        range(calc_r(p.getRect())[1], calc_r(p.getRect())[3]) for p in
                                        people + to_append]):
                                p = Person.MyPerson(pid, c[0], c[1], self.middle_line, self.down_limit, self.up_limit,
                                                    r)
                                to_append.append(p)
                                pid += 1

            people += to_append

            self.open_live_monitor(output_frame, people)
            # cv.imshow('Mask', mask)

            frame_id += 1
            k = cv2.waitKey(30) & 0xff
            if k == ord('q'):
                break
        print('____________________________________________')
        print(frame_id)
        for p in people:
            print(p.frame_id)
        cv2.destroyAllWindows()
        self.CAP.release()
        # self.LOG.flush()
        self.LOG.close()
        print('FINISH')


if __name__ == '__main__':
    # people_counter = PeopleCounter(cap='test_videos\\0001.mp4', upper_padding=0.5)
    # people_counter = PeopleCounter(cap='test_videos\\TestVideo.avi', upper_padding=0.5)
    people_counter = PeopleCounter(cap='test_videos\\TestVideo.avi', upper_padding=0.5)
    # people_counter = PeopleCounter(cap='test_videos\\0002.mp4', upper_padding=0.5)
    people_counter.run()
