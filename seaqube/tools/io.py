from seaqube.tools.types import Writeable


class DummyWriter(Writeable):
    def __init__(self):
        self.cache = []

    def write(self, data):
        self.cache.append(data)

    def close(self):
        return self.cache
