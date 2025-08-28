from abc import *

class Abs(ABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    @staticmethod
    @abstractmethod
    def do(i, j):
        return i + j


class Der(Abs):
    def __init__(self, x, y, k):
        super().__init__(x, y)
        self.k = k
    
    @staticmethod
    def do(i, j, m):
        return i + j +m


from dataclasses import dataclass

@dataclass
class Out:
    x: int 
    y: int


@dataclass
class Dout(Out):
    z: int


o = Out(1, 2)
d = Dout(3)
