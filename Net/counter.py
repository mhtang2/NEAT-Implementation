class Counter():
    def __init__(self, start=0):
        self.x = start

    def pre(self):
        self.x += 1
        return self.x

    def post(self):
        temp = self.x
        self.x += 1
        return temp

    def val(self, y):
        self.x = y
