"""

PGM parsing library for CS674 (Image Processing) at UNR.

"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List


class InvalidPGMFormat(Exception):
    pass


class PGMImage:
    signature: str
    cols: int
    rows: int
    quantization: int
    pixels: List[List[int]]
    comments: List[str]
    name: str

    def __init__(self, pgm_filename):
        """ Read a PGM file given its filename. """
        self.signature = None
        self.cols, self.rows, self.quantization = None, None, None
        self.pixels = []
        self.comments = []

        self.name = Path(pgm_filename).name

        with open(pgm_filename, "rb") as pgm_file:
            self.signature = pgm_file.read(2).decode("ascii")

            if self.signature != "P5":  # Other formats not used in CS674
                raise InvalidPGMFormat(
                    f"Format was {self.signature}, but only raw greyscale bytes"
                    " are acceptable."
                )

            while not (self.cols and self.rows and self.quantization):
                # Read .PGM header. This handles multiple, out-of-order
                # comments (preceded by '#'), and metadata (C, R, Q) in-order
                # but possibly seperated by line breaks.
                line = pgm_file.readline().decode("ascii")

                if line.startswith("#"):
                    self.comments.append(line)
                else:
                    line = line.strip().split()  # Strip trailing newlines

                    for num in line:
                        num = int(num)

                        if not self.cols:
                            self.cols = num
                        elif not self.rows:
                            self.rows = num
                        elif not self.quantization:
                            self.quantization = num

            # Read pixel contents given header parameters
            for i in range(self.rows):
                self.pixels.append(pgm_file.read(self.cols))

            if pgm_file.read() != b"":
                raise InvalidPGMFormat(
                    f"Finished reading {pgm_filename} but content still left."
                )

    def normalize_intensity_values(self):
        hi, lo = float("-inf"), 0
        for i in range(self.rows):
            for b in self.pixels[i]:
                hi = max(b, hi)
                lo = min(b, lo)

        for i in range(self.rows):
            self.pixels[i] = b"".join(
                [
                    bytes([int(((px - lo) / (hi - lo)) * self.quantization)])
                    for px in self.pixels[i]
                ]
            )

    def truncate_intensity_values(self):
        def truncate(_px):
            _px = int(_px)
            _px = min(self.quantization, _px)
            _px = max(0, _px)
            return _px

        for i in range(self.rows):
            self.pixels[i] = b"".join([bytes([truncate(px)]) for px in self.pixels[i]])

    def save(self, pgm_filename, normalize=False):
        """ Write this PGM image to a file. 

        :param pgm_filename Filename to save image to
        :param normalize Whether to scale all pixel values to highest pixel value
        """
        if True:
            print(f"Saving {self.name} to {pgm_filename}.")

        if not normalize:
            # Warn user if they're about to save with out-of-bounds pixel values
            if any(
                any(pxl > self.quantization or pxl < 0 for pxl in row)
                for row in self.pixels
            ):
                raise InvalidPGMFormat(
                    f"Image pixel values > {self.quantization}. Normalize or truncate first."
                )
        else:
            self.normalize_intensity_values()

        def itobs(i: int) -> bytes:
            """ Convert integer to byte string. """
            return str(i).encode("ascii")

        with open(pgm_filename, "wb") as f:
            lines = [
                b"P5\n",
                itobs(self.cols) + b" " + itobs(self.rows) + b"\n",
                itobs(self.quantization) + b"\n",
                *self.pixels,
            ]
            f.writelines(lines)

    @property
    def n_pixels(self) -> int:
        return self.rows * self.cols

    def unrolled_pixels(self) -> List[int]:
        """ Return the image's intensity values as an ordered, row-major, flat array. """
        unrolled_pxls = []

        for row in self.pixels:
            for pxl in row:
                unrolled_pxls.append(int(pxl))

        return unrolled_pxls

    def get_histogram(self, normed: bool = False) -> List[int]:
        """ Return the quantity of pixels as a list indexed by intensity value. """
        histogram = [0] * self.quantization

        for pxl in self.unrolled_pixels():
            histogram[pxl] += 1

        if normed:
            for i in range(self.quantization):
                histogram[i] /= self.n_pixels

        return histogram

    def __add__(self, other):
        """ Sum pixel values in one image with those in another image. """
        if isinstance(other, PGMImage):
            assert (self.rows, self.cols) == (
                other.rows,
                other.cols,
            ), "Images must be of the same sizes to be subtracted."

            res = deepcopy(self)

            for i in range(self.rows):
                my_row = [b for b in self.pixels[i]]
                their_row = [b for b in other.pixels[i]]
                res.pixels[i] = b"".join(
                    [
                        bytes([min(x + y, self.quantization)])
                        for x, y in zip(my_row, their_row)
                    ]
                )

            return res

        super().__add__(other)

    def __sub__(self, other):
        """ Difference of pixel values in one image with those in another image. """
        if isinstance(other, PGMImage):
            assert (self.rows, self.cols) == (
                other.rows,
                other.cols,
            ), "Images must be of the same sizes to be added."

            res = deepcopy(self)

            for i in range(self.rows):
                my_row = [b for b in self.pixels[i]]
                their_row = [b for b in other.pixels[i]]
                res.pixels[i] = b"".join(
                    [bytes([max(x - y, 0)]) for x, y in zip(my_row, their_row)]
                )

            return res

        super().__sub__(other)

    def show_histogram(self, normed: bool = False, title=None):
        """ Display a histogram of the image.

        Note: blocks the main thread.
        """
        import matplotlib.pyplot as plot

        # plot.ion()  # Don't block while showing the histogram

        plot.hist(self.unrolled_pixels(), bins=255, density=normed)

        plot.xlabel("Pixel Value")
        plot.ylabel("Count" if not normed else "Probability")

        if not title:
            title = f"Histogram of {self.name}"
        plot.title(title)

        plot.show()
