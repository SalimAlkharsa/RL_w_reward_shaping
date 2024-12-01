from collections import deque

class LastVisitedElements:
    def __init__(self, max_size=3):
        self.max_size = max_size
        self.elements = deque(maxlen=max_size)

    def add_element(self, element):
        self.elements.append(element)

    def get_elements(self):
        return set(self.elements)
