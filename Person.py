from random import randint
import time
from statistics import mean
import math

class MyPerson:
    tracks = []

    def __init__(self, i, xi, yi, middle_line, down_limit, up_limit, rect=None, frame_id=None, max_age=10):
        if frame_id:
            self.frame_id = frame_id
        if rect is None:
            self.rect = []
        else:
            self.rect = rect

        self.down_limit = down_limit
        self.up_limit = up_limit
        self.middle_line = middle_line
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

    # def is_done(self):
    #     return self.done

    def is_counted(self):
        return self.counted

    def set_counted(self):
        self.counted = True

    def get_tracks_len(self):
        return len(self.tracks)

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

    def is_done(self):
        if self.age > self.max_age:
            self.done = True
            if len(self.tracks) > 2:
                if self.tracks[-1][1] <= self.middle_line < self.tracks[0][1]:
                    self.dir = 'U'
                    return 'count'
                elif self.tracks[-1][1] >= self.middle_line > self.tracks[0][1]:
                    self.dir = 'D'
                    return 'count'
                else:
                    return 'delete'
            else:
                return 'delete'
        elif len(self.tracks) > 2 and self.getY() >= self.down_limit:
            if self.dir == 'D':
                return 'count'
            else:
                return 'delete'
        elif len(self.tracks) > 2 and self.getY() <= self.up_limit:
            if self.dir == 'U':
                return 'count'
            else:
                return 'delete'
        else:
            return None

    def inc_age(self):
        self.age += 1
        return self.is_done()

    # def getIfReadyToDelete(self):
    #     return self.ready_to_delete


    def updateCoords(self, xn, yn, rect, frame_id=None):
        # print('update: ', self.getId())
        if frame_id:
            self.frame_id = frame_id

        self.rect = rect

        # self.age = 0

        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

        if len(self.tracks) >= 2:
            if self.tracks[-1][1] - self.tracks[0][1] > 0:
                self.dir = 'D'
            elif self.tracks[-1][1] - self.tracks[0][1] < 0:
                self.dir = 'U'
            # if self.tracks[-1][1] <= self.middle_line < self.tracks[0][1]:
            #     self.dir = 'U'
            #
            # elif self.tracks[-1][1] >= self.middle_line > self.tracks[0][1]:
            #     self.dir = 'D'

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


    # def dist(self, A, B):
    #     return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

    def forced_move(self):
        if len(self.tracks) >= 2:
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
        print('FORCED_MOVE   ', self.getId(), 'old: ', self.x, self.y, 'setpX: ', stepX,'setpX: ',stepY, ' tracksLen: ', self.get_tracks_len())
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
