from abc import ABC, abstractmethod


class InfoSender(ABC):
    @abstractmethod
    def send_info(self) -> dict:
        pass

class InfoReceiver(ABC):
    @abstractmethod
    def log_info(self, info: dict):
        pass