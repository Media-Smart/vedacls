from abc import ABC, abstractmethod


class Base(ABC):

    @abstractmethod
    def add(self, pred, gt):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def result(self):
        pass
