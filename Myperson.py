import cv2


class Person():
    footage = []
    # direction 4 = left, 6= right, 8=top, 2=bottom
    direction = 0

    def __init__(self, Xi, Yi, left, right, top, bottom, vertical=True):
        self.xi, self.yi = Xi, Yi
        self.left, self.right, self.top, self.bottom = left, right, top, bottom
        self.vertical = vertical
        self.active = True

    def findDirection(self):
        if self.footage:
            if abs(self.footage[0][0] - self.xi) > abs(self.footage[0][0] < self.xi):
                #     X axriko > X twra = left
                if self.footage[0][0] > self.xi:
                    self.direction = 4
                #     X axriko < X twra = right
                elif self.footage[0][0] < self.xi:
                    self.direction = 6
            else:
                #     Y axriko < Y twra = top
                if self.footage[0][1] > self.yi:
                    self.direction = 8
                #     Y axriko < Y twra = bottom
                elif self.footage[0][1] < self.yi:
                    self.direction = 2
        return self.direction

    def UpdateCoordinantes(self, xi, yi):
        self.footage.append((self.xi, self.yi))
        self.xi, self.yi = xi, yi
        if self.xi > self.right:
            self.active = False
            return self.findDirection()

        if self.xi < self.left:
            self.active = False
            return self.findDirection()

        if self.xi > self.right:
            self.active = False
            return self.findDirection()

        if self.xi > self.right:
            self.active = False
            return self.findDirection()

