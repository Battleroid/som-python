from random import random
from collections import defaultdict


def ssd(v, o):
    return sum([(i - j) ** 2.0 for i, j in zip(v, o)])


class SOM(object):

    def __init__(self,
                 epochs=100,
                 threshold=3,
                 rate=0.7,
                 rate_lower=25,
                 radius=2,
                 radius_lower=25,
                 grow_threshold=2.0):
        self.epochs = epochs
        self.threshold = threshold
        self.rate = rate
        self.rate_lower = rate_lower
        self.radius = radius
        self.radius_lower = radius_lower
        self.growth_threshold = grow_threshold
        self.outputs = []
        self.canceled = dict()

    def _init_outputs(self, n):
        node = [random() for _ in range(n)]
        self.outputs.append(node)

    def bmu(self, v):
        pos = 0
        lowest = ssd(v, self.outputs[0])
        for i, node in enumerate(self.outputs):
            s = ssd(v, node)
            if s < lowest:
                lowest = s
                pos = i
        return pos

    def should_grow(self, v):
        growth = sum([ssd(v, x) for x in self.outputs])
        return growth > self.growth_threshold

    def _retry(self, canceled_data):
        # begin training
        radius = self.radius
        rate = self.rate
        rate_ticks = 0
        radius_ticks = 0
        epochs = self.epochs

        # retrain until epoch reaches 0
        while epochs > 0:
            print 'Retrain Epoch', epochs
            for v in canceled_data:
                # set data to exclude name
                vd = v[1:]

                # get best match
                x = self.bmu(vd)

                # adjust weight for node and neighbors
                if radius > 0:
                    # increase weight for record and neighborhood
                    hood = self.outputs[x - radius:x + radius]
                    for node in hood:
                        for i, val in enumerate(node):
                            vi = vd[i]
                            val = val + rate * (vi - val)
                else:
                    # increase weight for single record
                    for i, val in enumerate(self.outputs[x]):
                        vi = vd[i]
                        val = val + rate * (vi - val)

            # increase epoch, rate, radius, check if limit reached
            if rate_ticks == self.rate_lower:
                rate = rate * rate
                rate_ticks = 0
            rate_ticks += 1

            if radius_ticks == self.radius_lower:
                if radius > 0:
                    radius -= 1
                radius_ticks = 0
            radius_ticks += 1

            epochs -= 1

    def train(self, data, grow_first_only=True):
        # create initial node with n total features
        n_features = len(data[0][1:])
        self._init_outputs(n_features)

        # begin training
        radius = self.radius
        rate = self.rate
        rate_ticks = 0
        radius_ticks = 0
        epochs = self.epochs

        # until epoch count reaches zero
        while epochs > 0:
            print 'Epoch', epochs
            for v in data:
                # remove name of record
                vd = v[1:]

                # if growth value exceeds given growth threshold add new node
                if grow_first_only and epochs == self.epochs:
                    if self.should_grow(vd):
                        self.outputs.append(vd)

                # get BMU of vector
                x = self.bmu(vd)

                # adjust weight for node and neighbors
                if radius > 0:
                    # increase weight for record and neighborhood
                    hood = self.outputs[x - radius:x + radius]
                    for node in hood:
                        for i, val in enumerate(node):
                            vi = vd[i]
                            val = val + rate * (vi - val)
                else:
                    # increase weight for single record
                    for i, val in enumerate(self.outputs[x]):
                        vi = vd[i]
                        val = val + rate * (vi - val)

            # increase epoch, rate, radius, check if limit reached
            if rate_ticks == self.rate_lower:
                rate = rate * rate
                rate_ticks = 0
            rate_ticks += 1

            if radius_ticks == self.radius_lower:
                if radius > 0:
                    radius -= 1
                radius_ticks = 0
            radius_ticks += 1

            epochs -= 1

        # finished, associate each record with the BMU
        groups = defaultdict(list)
        for _ in range(len(self.outputs)):
            groups[_] = []

        for v in data:
            vd = v[1:]
            x = self.bmu(vd)
            groups[x].append(v[0])

        # record the canceled outputs' records and weights
        for k, v in groups.items():
            if len(v) < self.threshold:
                self.canceled[k] = {'records': v, 'weights': self.outputs[k]}

        # remove the cluster from the outputs
        for i in sorted(self.canceled.keys(), reverse=True):
            del self.outputs[i]

        # redo with records associated with canceled nodes only
        canceled_data = []
        for k, v in self.canceled.items():
            for name in v['records']:
                canceled_data.append(find_record(name, data))

        # retain the SOM with what's leftover
        self._retry(canceled_data)

        # redo association one more time
        del groups
        groups = defaultdict(list)
        for _ in range(len(self.outputs)):
            groups[_] = []

        for v in data:
            vd = v[1:]
            x = self.bmu(vd)
            print v[0], '->', x
            groups[x].append(v[0])

        return groups


def find_record(name, data):
    return [x for x in data if x[0] == name][0]
