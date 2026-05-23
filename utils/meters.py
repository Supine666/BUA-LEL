# utils/meters.py
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += float(val) * int(n)
        self.cnt += int(n)

    @property
    def avg(self):
        return self.sum / max(1, self.cnt)