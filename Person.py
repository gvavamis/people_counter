from random import randint
import time
from statistics import mean
import math

class MyPerson:
    tracks = []

    def __init__(self, i, xi, yi, middle_line, down_limit, up_limit, rect=None, max_age=40):
        self.frame_id = None
        if rect is None:
            self.rect = []
        else:
            self.rect = rect
        self.status = None
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
        self.count_age = None
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

    def get_age(self):
        return self.age

    def get_status(self):
        return self.status

    def is_done(self):
        self.inc_age()
        if self.age > self.max_age:
            if len(self.tracks) > 2:
                if self.tracks[-1][1] <= self.middle_line < self.tracks[0][1]:
                    self.dir = 'U'
                    if self.age == self.max_age + 1:
                        self.count_age = self.age
                        self.status = 'count'
                        return self.status
                    elif self.max_age + 1 < self.age - self.count_age <= int(self.max_age * 0.2):
                        self.status = 'counted'
                        return self.status
                    elif self.age - self.count_age > int(self.max_age * 0.2):
                        self.done = True
                        self.status = 'delete'
                        return self.status
                elif self.tracks[-1][1] >= self.middle_line > self.tracks[0][1]:
                    self.dir = 'D'
                    if self.age == self.max_age + 1:
                        self.count_age = self.age
                        self.status = 'count'
                        return self.status
                    elif self.max_age + 1 < self.age - self.count_age <= int(self.max_age * 0.2):
                        self.status = 'counted'
                        return self.status
                    elif self.age - self.count_age > int(self.max_age * 0.2):
                        self.done = True
                        self.status = 'delete'
                        return self.status
                else:
                    self.status = 'delete'
                    return self.status
            else:
                self.status = 'delete'
                return self.status
        elif len(self.tracks) > 2 and self.getY() >= self.down_limit > self.tracks[0][1]:
            if self.dir == 'D':
                if self.count_age is not 'not_counted':
                    self.count_age = self.age
                if self.age - self.count_age == 0:
                    self.status = 'count'
                    return self.status
                elif self.max_age + 1 < self.age - self.count_age <= int(self.max_age * 0.2):
                    self.status = 'counted'
                    return self.status
                elif self.age - self.count_age > int(self.max_age * 0.2):
                    self.done = True
                    self.status = 'delete'
                    return self.status
            else:
                self.status = 'delete'
                return self.status
        elif len(self.tracks) > 2 and self.getY() <= self.up_limit < self.tracks[0][1]:
            if self.dir == 'U':
                if self.count_age is not 'not_counted':
                    self.count_age = self.age
                if self.age - self.count_age == 0:
                    self.status = 'count'
                    return self.status
                elif self.max_age + 1 < self.age - self.count_age <= int(self.max_age * 0.2):
                    self.status = 'counted'
                    return self.status
                elif self.age - self.count_age > int(self.max_age * 0.2):
                    self.done = True
                    self.status = 'delete'
                    return self.status
            else:
                self.status = 'delete'
                return self.status
        else:
            if self.getY() < self.up_limit or self.getY() > self.down_limit:
                self.status = 'delete'
                return self.status
            else:
                self.status = 'not_counted'
                return self.status

    def inc_age(self):
        self.age += 1
        # return self.is_done()

    # def getIfReadyToDelete(self):
    #     return self.ready_to_delete


    def updateCoords(self, xn, yn, rect):

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
        print(len(self.tracks))
        if len(self.tracks) >= 2:
            tmp = [(tracks0, tracks1) for tracks0, tracks1 in zip(self.tracks[-5:], self.tracks[-4:])]
            # print(tmp)
            tmpdis = [(_[1][0] - _[0][0], _[1][1] - _[0][1]) for _ in tmp]
            # print(tmpdis)
            stepX = int(mean([_[0] for _ in tmpdis]))
            stepY = int(mean([_[1] for _ in tmpdis]))
            print(tmp)
            print([_[0] for _ in tmpdis])
            print([_[1] for _ in tmpdis])
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
        print('FORCED_MOVE   ', self.getId(), 'old: ', self.x, self.y, 'setpX: ', stepX,'setpX: ',stepY,
              ' tracksLen: ', self.get_tracks_len(), 'status: ',self.status)
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
