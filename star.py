"""
The Star class is used to represent a single star detected in an image.
Each Star object stores useful information about the star's position, size, brightness, and the image it was found in.
"""

class Star:
    def __init__(self, x, y, r, b, id_star, pic):
        self.x = x
        self.y = y
        self.r = r
        self.b = b
        self.id = str(id_star)
        self.img = pic


