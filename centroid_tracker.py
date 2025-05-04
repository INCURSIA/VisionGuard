import cv2
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.positions = {}

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.positions[self.next_object_id] = [centroid]
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        # keep the positions log

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = dist.cdist(np.array(object_centroids), np.array(input_centroids))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            assigned = set()
            for row, col in zip(rows, cols):
                if col in assigned:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.positions[object_id].append(input_centroids[col])
                self.disappeared[object_id] = 0
                assigned.add(col)

            unassigned_rows = set(range(D.shape[0])) - set(rows)
            for row in unassigned_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unassigned_cols = set(range(D.shape[1])) - assigned
            for col in unassigned_cols:
                self.register(input_centroids[col])

        return self.objects
