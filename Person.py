from random import randint
import time
from statistics import mean
import math

class MyPerson:
    tracks = []

    def __init__(self, i, xi, yi, line_down, line_up, down_limit, up_limit, rect=None, frame_id=None, max_age=50):
        if frame_id:
            self.frame_id = frame_id
        if rect is None:
            self.rect = []
        else:
            self.rect = rect

        self.down_limit = down_limit
        self.up_limit = up_limit
        self.line_down = line_down
        self.line_up = line_up
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
        self.counted = False
        self.ready_to_delete = False
        self.age = 0
        self.max_age = max_age
        self.dir = None
    def getRGB(self):
        return (self.R, self.G, self.B)

    def getTracks(self):
        return self.tracks

    def getId(self):
        return self.i

    def getMaxAge(self):
        return self.max_age

    def is_done(self):
        return self.done

    def is_counted(self):
        return self.counted

    def set_counted(self):
        self.counted = True

    # def getIfReadyToDelete(self):
    #     return self.ready_to_delete


    def getDir(self):
        return self.dir

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getRect(self):
        return self.rect

    def get_frame_id(self):
        return self.frame_id

    def updateCoords(self, xn, yn, rect, frame_id=None):
        print('update: ', self.getId())
        if frame_id:
            self.frame_id = frame_id

        self.rect = rect

        # self.age = 0

        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

        if len(self.tracks) >= 2:
            if self.tracks[-1][1] < self.line_down <= self.tracks[-2][1]:
                self.dir = 'U'

            elif self.tracks[-1][1] > self.line_up >= self.tracks[-2][1]:
                self.dir = 'D'

        # print(self.getId())
        # print(self.dir)
        # print(self.rect)
        # print('top', self.rect[1])
        # print('bottom', self.rect[1] + self.rect[3])
        # if self.rect[1] >= self.down_limit:
        #     print('DONE with dir : ',self.dir,'down_limit')
        #     print(self.rect[1], self.down_limit)
        #     self.done = True
        # elif self.rect[1] + self.rect[3] <= self.up_limit:
        #     print('DONE with dir : ', self.dir, 'up_limit')
        #     print(self.rect[1] + self.rect[3], self.up_limit)
        #     self.done = True
        if len(self.tracks)>15:
            # TODO na elegthei an to proto tous center htan prin apo tis grammes
            if self.y >= self.down_limit:
                # print('DONE with dir : ',self.dir,'down_limit')
                # print(self.rect[1], self.down_limit)
                self.done = True
            elif self.y <= self.up_limit:
                # print('DONE with dir : ', self.dir, 'up_limit')
                # print(self.rect[1] + self.rect[3], self.up_limit)
                self.done = True



    def age_one(self):
        self.age += 1
        # if self.age > self.max_age:
            # self.done = True
        return

    # def dist(self, A, B):
    #     return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def forced_move(self):
        print('FORCED_MOVE   ', self.getId())
        if len(self.tracks) >=  2:
            tmp = [(tracks0, tracks1) for tracks0, tracks1 in zip(self.tracks[-10:], self.tracks[-9:])]
            # print(tmp)
            tmpdis = [(_[1][0] - _[0][0], _[1][1] - _[0][1]) for _ in tmp]
            # print(tmpdis)
            stepX = int(mean([_[0] for _ in tmpdis]))
            stepY = int(mean([_[0] for _ in tmpdis]))
        else:
            stepX = 0
            stepY = 0
        new_rect = self.rect
        if stepX and stepY:
            if stepX > 0:
                if stepY > 0:
                    new_rect = self.rect[0]+abs(stepX), self.rect[1]+abs(stepY), self.rect[2]+abs(stepX), self.rect[3]+abs(stepY)
                elif stepY < 0:
                    new_rect = self.rect[0]+abs(stepX), self.rect[1]-abs(stepY), self.rect[2]+abs(stepX), self.rect[3]-abs(stepY)
            elif stepX < 0:
                if stepY > 0:
                    new_rect = self.rect[0]-abs(stepX), self.rect[1]+abs(stepY), self.rect[2]-abs(stepX), self.rect[3]+abs(stepY)
                elif stepY < 0:
                    new_rect = self.rect[0]-abs(stepX), self.rect[1]-abs(stepY), self.rect[2]-abs(stepX), self.rect[3]-abs(stepY)
        # else:
        #     new_rect = self.rect
        self.updateCoords(self.x+stepX, self.y+stepY, new_rect)


class MultiPerson:
    def __init__(self, persons, xi, yi):
        self.persons = persons
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
