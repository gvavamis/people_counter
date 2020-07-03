import numpy as np
# import cv2 as cv
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import dlib
import cv2
import os
from datetime import datetime
import time
import Person


class PeopleCounter():

    def __init__(self, cap=0, in_direction='DOWN', logfile_path='', people_in=0, upper_padding=1):
        WORKING_DIR = os.path.abspath(__file__)
        self.WORKING_DIR = WORKING_DIR[:len(WORKING_DIR) - len(WORKING_DIR.split('\\')[-1])]
        self.IN_DIRECTION = in_direction.lower()
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        self.personID = CLASSES.index('person')

        prototxt = 'mobilenet_ssd\\MobileNetSSD_deploy.prototxt'
        model = 'mobilenet_ssd\\MobileNetSSD_deploy.caffemodel'
        self.confidence = 0.3
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        if self.IN_DIRECTION == 'down':
             self.OUT_DIRECTION = 'up'
        elif self.IN_DIRECTION == 'up':
            self.OUT_DIRECTION = 'down'
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
        self.Vwidth = 500 #int(self.CAP.get(cv2.CAP_PROP_FRAME_WIDTH))

        if self.upper_padding == 1:
            self.ofset = 0
        else:
            self.ofset = int(self.Vheight * self.upper_padding)

        frameArea = self.Vheight * self.Vwidth
        self.THRESSHOLD_AREA = frameArea * 250 / 307200
        print('THRESSHOLD_AREA THRESSHOLD_AREA THRESSHOLD_AREA')
        print(self.THRESSHOLD_AREA)

        self.line_up = int(7 * (self.Vheight / 10))  # +self.ofset
        self.line_down = int(8 * (self.Vheight / 10))  # +self.ofset
        self.up_limit = int(6 * (self.Vheight / 10))  # +self.ofset
        self.down_limit = int(9 * (self.Vheight / 10))  # +self.ofset

        # print(self.Vheight)
        # print('---------------')
        # print(self.line_up)
        # print(self.line_down)
        # print(self.up_limit)
        # print(self.down_limit)

        self.line_down_color = (255, 0, 0)
        self.line_up_color = (0, 0, 255)

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

    def set_resize(self, w, h):
        # TODO den einai teleiomenh
        self.h_resize = h * self.upper_padding
        self.w_resize = w
        self.Vheight = int(self.Vheight * self.h_resize)
        self.Vwidth = int(self.Vwidth * self.w_resize)
        frameArea = self.Vheight * self.Vwidth
        self.THRESSHOLD_AREA = frameArea / 250

        self.line_up = int(2 * (self.Vheight / 5)) + self.ofset
        self.line_down = int(3 * (self.Vheight / 5)) + self.ofset
        self.up_limit = int(1 * (self.Vheight / 5)) + self.ofset
        self.down_limit = int(4 * (self.Vheight / 5)) + self.ofset

    def update_Log(self, this_person):
        direction = this_person.getDir()
        # self.total_in, self.total_out = (self.count_up, self.count_down) if self.IN_DIRECTION == 'up' else (
        # self.count_down, self.count_up)

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

    def open_live_monitor(self, frame, people, show_ids=True, show_details=True, show_lines=True, scale=0.6):
        image = frame.copy()
        pt1 = [0, self.line_down]
        pt2 = [self.Vwidth, self.line_down]
        pts_L1 = np.array([pt1, pt2], np.int32)
        pts_L1 = pts_L1.reshape((-1, 1, 2))
        pt3 = [0, self.line_up]
        pt4 = [self.Vwidth, self.line_up]
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

        if show_ids:
            for this_person in people:
                if this_person.getDir():
                    cv2.putText(image, str(this_person.getId()), (this_person.getX(), this_person.getY()),
                                font, 1, this_person.getRGB(), 1, cv2.LINE_AA)

        if show_details:
            total_in_str = 'People entered: ' + str(self.total_in)
            total_out_str = 'People came out: ' + str(self.total_out)
            people_in_now_str = 'now in: ' + str(self.people_in)

            cv2.putText(image, total_in_str, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, total_in_str, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, total_out_str, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, total_out_str, (10, 90), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, people_in_now_str, (10, 140), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, people_in_now_str, (10, 140), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if show_lines:
            image = cv2.polylines(image, [pts_L1], False, self.line_down_color, thickness=2)
            image = cv2.polylines(image, [pts_L2], False, self.line_up_color, thickness=2)
            image = cv2.polylines(image, [pts_L3], False, (255, 255, 255), thickness=1)
            image = cv2.polylines(image, [pts_L4], False, (255, 255, 255), thickness=1)

        # image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        cv2.imshow('LIVE', image)

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
            print('%%%%%%%%%%%%%%%%%')
            print('frame: ', frame_id)
            to_remove = []
            ret, frame = self.CAP.read()
            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

            output_frame = frame.copy()
            # ARXH TOU ALGORUTHMOU


            temp = people.copy()
            while len(temp):
                this_person = temp.pop()
                # for this_person in people:
                mtp = 1
                x, y, w, h = this_person.getRect()
                # Xlen= (w-x)*mtp
                # Ylen= (h-y)*mtp
                Xlen = w * mtp
                Ylen = h * mtp
                xExtra = int(Xlen / 2)
                yExtra = int(Ylen / 2)
                print('=========')
                A = y - yExtra
                B = y + h + yExtra
                C = x - xExtra
                D = x + w + xExtra
                if A < 0:
                    A = 0
                if B > frame.shape[0]:
                    B = frame.shape[0]
                if C < 0:
                    C = 0
                elif D > frame.shape[1]:
                    D = frame.shape[1]
                # img[y:y + h, x:x + w]
                print(A, B, C, D)
                temp_im = mask2[A: B, C: D]
                window = frame.copy()

                # print(temp_im)
                # print(xExtra, yExtra)
                # print(x, y, w, h)
                # print(y - yExtra, qy + yExtra, x - xExtra, x + xExtra)

                contours0, hierarchy = cv2.findContours(temp_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print('contours')
                # print(contours0)
                # print(len(contours0))
                if contours0:
                    max_area = max([cv2.contourArea(cnt) for cnt in contours0])
                else:
                    max_area = 0

                for cnt in contours0:
                    if max_area == cv2.contourArea(cnt) and max_area:
                        # print(max_area)
                        # print(cnt)
                        M = cv2.moments(cnt)
                        # print(M)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        xi, yi, wi, hi = cv2.boundingRect(cnt)
                        globalX = C + cy
                        globalY = A + cx
                        if ((xi > 0 and wi < temp_im.shape[1] and yi > 0 and hi < temp_im.shape[0]) and
                                (wi * hi > this_person.getRect()[2] * this_person.getRect()[3])):
                            this_person.updateCoords(globalX, globalY, (xi+C, yi+A, wi, hi), frame_id=frame_id)
                        else:
                            this_person.updateCoords(globalX, globalY, (x+C, y+C, w, h), frame_id=frame_id)

                        cv2.circle(window, (globalX, globalY), 5, (255, 0, 0), -1)
                        cv2.circle(window, (cx, cy), 5, (0, 250, 0), -1)

                        cv2.imshow('cnt', temp_im)
                        # print('-------------')
                        temp_im = cv2.fillPoly(temp_im, [cnt], (0, 0, 0))
                        cv2.imshow('cnt_filled', temp_im)
                        mask2[A: B, C: D] = temp_im
                        window = cv2.rectangle(window, (C, A), (D, B), (255, 0, 255), 2)

                        cv2.imshow('window', window)
                        cv2.imshow('cnt', temp_im)
                        if cv2.waitKey(0) & 0xff == ord('c'):
                            cv2.destroyWindow('window')
                            cv2.destroyWindow('cnt')

                        del temp_im
                        # if cv2.waitKey(0) & 0xff:
                        #     cv2.destroyWindow('cnt')
                        #     cv2.destroyWindow('cnt_filled')
                    else:
                        # if frame_id > this_person.getframe_id() + this_person.getMaxAge():
                        #     to_remove.append(this_person)
                        #     print('CONTURE TIMED_OUT')
                        # else:
                        #     this_person.forced_move()
                        this_person.forced_move()
                        window = cv2.rectangle(window, (this_person.getRect()[0], this_person.getRect()[1]), (this_person.getRect()[2]+this_person.getRect()[0], this_person.getRect()[3]+this_person.getRect()[1]), (0, 255, 255), 2)
                        cv2.circle(window, (this_person.getX(), this_person.getY()), 5, (0, 255, 255), -1)

                cv2.circle(output_frame, (this_person.getX(), this_person.getY()), 5, (0, 0, 255), -1)
                # cv2.circle(output_frame, (this_person.getX(), this_person.getY()), 5, (0, 0, 255), -1)

                R = this_person.getRect()
                if this_person.getDir() == self.IN_DIRECTION:
                    output_frame = cv2.rectangle(output_frame, (R[0], R[1]), (R[0] + R[2], R[1] + R[3]), (0, 255, 0), 2)
                else:
                    output_frame = cv2.rectangle(output_frame, (R[0], R[1]), (R[0] + R[2], R[1] + R[3]), (0, 0, 255), 2)
                del R

                if this_person.timedOut():
                    self.update_Log(this_person)
                    to_remove.append(this_person)
                    print('A CONTURE FINISHED')
                if frame_id > this_person.getframe_id() + this_person.getMaxAge():
                    to_remove.append(this_person)
                    print('CONTURE TIMED_OUT')


            print('people: ', len(people))
            cv2.imshow('mask', mask2)
            contours0, hierarchy = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            count_new = 0
            for cnt in contours0:
                cnt_area = cv2.contourArea(cnt)
                if cnt_area > self.THRESSHOLD_AREA:

                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    x, y, w, h = cv2.boundingRect(cnt)
                    rect = x, y, w, h
                    # curr_dir = ''
                    # new_person = True

                    if cy in range(self.up_limit, self.down_limit):
                        if not any([p.getRect()[0] < cx < p.getRect()[0] + p.getRect()[2] and p.getRect()[1] < cy <
                                    p.getRect()[1] + p.getRect()[3] for p in people]):
                            p = Person.MyPerson(pid, cx, cy, self.line_down, self.line_up, self.down_limit,
                                                self.up_limit, rect, frame_id=frame_id)
                            # p.getId()
                            people.append(p)
                            pid += 1
                            # curr_dir = None
                            count_new += 1
                            cv2.circle(output_frame, (cx, cy), 5, (0, 0, 255), -1)
                # print('curr_dir == self.IN_DIRECTION')
                # print(self.IN_DIRECTION)
                # print(curr_dir)
                #     if curr_dir == None:
                #         # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                #         pass

                # cv.drawContours(frame, cnt, -1, (0,255,0), 3)

            self.open_live_monitor(output_frame, people)
            # cv.imshow('Mask', mask)

            print('people added : ', count_new)
            print('to remove \n ------------------')

            while len(to_remove):
                print(len(to_remove))
                print(len(people))
                people.remove(to_remove.pop())
            print('------------------')
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
    people_counter = PeopleCounter(cap='test_videos\\TestVideo.avi', upper_padding=0.5)
    people_counter.run()
