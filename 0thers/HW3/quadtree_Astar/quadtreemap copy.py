# reference:
#   https://en.wikipedia.org/wiki/Quadtree
#   https://github.com/volkerp/quadtree_Astar/blob/master/quadtree.py

from utils import png_to_ogm
import numpy as np
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def disOf2Points(point1, point2):
        return math.sqrt((point1.x-point2.x)**2+(point1.y-point2.y)**2)

class BoundingBox:
    def __init__(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height

    def containsPoint(self, point: Point):
        return (self.x0 < point.x < self.x0+self.width) and (self.y0 < point.y < self.y0+self.height)

    def intersectsBB(self, other):
        return (self.x0 < other.x0+other.width and self.y0 < other.y0+other.height) and \
               (self.x0+self.width > other.x0 and self.y0+self.height > other.y0)

    def center(self):
        return Point(self.x0+self.width/2, self.y0+self.height/2)

class Tile:
    def __init__(self, x0, y0, width, height, tile_capacity=20):
        self.tile_capacity = tile_capacity
        self.boundary = BoundingBox(x0, y0, width, height)
        self.tile_points = []
        self.tile_points_len = 0
        self.topleft = None
        self.topright = None
        self.botleft = None
        self.botright = None

    def __lt__(self, other):
        return self.tile_points_len < other.tile_points_len

    def __eq__(self, other):
        return (id(self) == id(other))

    def __hash__(self):
        return id(self)

    def insert(self, point):
        if not self.boundary.containsPoint(point):
            return False
        if self.tile_points_len < self.tile_capacity and not self.topleft:
            self.tile_points.append(point)
            self.tile_points_len += 1
            return True
        if not self.topleft:
            self.__subdivide()
        if self.topleft.insert(point):
            return True
        if self.topright.insert(point):
            return True
        if self.botleft.insert(point):
            return True
        if self.botright.insert(point):
            return True
        return False

    def __subdivide(self):
        self.topleft = Tile(self.boundary.x0, self.boundary.y0, self.boundary.width/2, self.boundary.height/2, self.tile_capacity)
        self.topright = Tile(self.boundary.x0+self.boundary.width/2, self.boundary.y0, self.boundary.width/2, self.boundary.height/2, self.tile_capacity)
        self.botleft = Tile(self.boundary.x0, self.boundary.y0+self.boundary.height/2, self.boundary.width/2, self.boundary.height/2, self.tile_capacity)
        self.botright = Tile(self.boundary.x0+self.boundary.width/2, self.boundary.y0+self.boundary.height/2, self.boundary.width/2, self.boundary.height/2, self.tile_capacity)
        for point in self.tile_points:
            self.topleft.insert(point)
            self.topright.insert(point)
            self.botleft.insert(point)
            self.botright.insert(point)
        self.tile_points = []
        self.tile_points_len = 0

    def getCenter(self):
        return self.boundary.center()

    def searchTileByIdx(self, point):
        if not self.boundary.containsPoint(point):
            return None
        if not self.topleft:
            return self
        if self.topleft.boundary.containsPoint(point):
            return self.topleft.searchTileByIdx(point)
        if self.topright.boundary.containsPoint(point):
            return self.topright.searchTileByIdx(point)
        if self.botleft.boundary.containsPoint(point):
            return self.botleft.searchTileByIdx(point)
        if self.botright.boundary.containsPoint(point):
            return self.botright.searchTileByIdx(point)
        return None

    def tileIntersect(self, otherBB: BoundingBox) -> list:
        intescList = []
        if not self.boundary.intersectsBB(otherBB):
            return intescList
        if not self.topleft:
            intescList.append(self)
            return intescList
        else:
            # print("all childrens: {}\t{}\t{}\t{}".format(self.topleft, self.topright, self.botleft, self.botright))
            # print("topleft: ", self.topleft.tileIntersect(otherBB))
            intescList += self.topleft.tileIntersect(otherBB)
            # print("topright: ", self.topright.tileIntersect(otherBB))
            intescList += self.topright.tileIntersect(otherBB)
            # print("botleft: ", self.botleft.tileIntersect(otherBB))
            intescList += self.botleft.tileIntersect(otherBB)
            # print("botright: ", self.botright.tileIntersect(otherBB))
            intescList += self.botright.tileIntersect(otherBB)
        return intescList

    def queryRange(self, otherBB: BoundingBox):
        pointsInRange = []
        if not self.boundary.intersectsBB(otherBB):
            return pointsInRange
        if not self.topleft:
            for point in self.tile_points:
                if otherBB.containsPoint(point):
                    pointsInRange.append(point)
            return pointsInRange
        pointsInRange += self.topleft.queryRange(otherBB)
        pointsInRange += self.topright.queryRange(otherBB)
        pointsInRange += self.botleft.queryRange(otherBB)
        pointsInRange += self.botright.queryRange(otherBB)
        return pointsInRange

    def drawTileByCanvas(self, cv, canvas_height, color="gray", width=2):
        cv.create_rectangle(self.boundary.x0, canvas_height-self.boundary.y0, 
                            self.boundary.x0+self.boundary.width, canvas_height-self.boundary.y0-self.boundary.height,
                            outline=color, fill=None, width=width)
        if self.topleft:
            self.topleft.drawTileByCanvas(cv, canvas_height, color, width)
        if self.topright:
            self.topright.drawTileByCanvas(cv, canvas_height, color, width)
        if self.botleft:
            self.botleft.drawTileByCanvas(cv, canvas_height, color, width)
        if self.botright:
            self.botright.drawTileByCanvas(cv, canvas_height, color, width)

class QuadTreeMap:
    def __init__(self, qtm, cell_size=1, occupancy_threshold=0.8, tile_capacity=20):
        self.dim_cells = qtm.shape
        print(self.dim_cells)
        self.dim_meters = (self.dim_cells[0] * cell_size, self.dim_cells[1] * cell_size)
        self.quadtree = Tile(0, 0, self.dim_cells[1], self.dim_cells[0], tile_capacity)
        self.cell_size = cell_size
        self.occupancy_threshold = occupancy_threshold
        indices1, indices2 = np.where(qtm > self.occupancy_threshold)
        for idx in range(len(indices1)):
            self.quadtree.insert(Point(indices2[idx], indices1[idx]))

    def drawQuadTreeMapByCanvas(self, cv, canvas_height, color="gray", width=2):
        self.quadtree.drawTileByCanvas(cv, canvas_height, color, width)

    @staticmethod
    def from_png(filename, cell_size=1, occupancy_threshold=0.8, tile_capacity=20):
        ogm_data = png_to_ogm(filename, normalized=True)
        ogm_data_arr = np.array(ogm_data)
        qtm = QuadTreeMap(ogm_data_arr, cell_size, occupancy_threshold, tile_capacity)

        return qtm