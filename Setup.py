import os


def setup_GeoMAN():
    # Fix GeoMAN imports by making them relative
    path = os.sep.join(["Models", "GeoMAN", "GeoMAN.py"])
    with open(path, "r") as f:
        content = f.read()
    content = content.replace("from base_model import BaseModel", "from .base_model import BaseModel")
    content = content.replace("from utils import Linear", "from .utils import Linear")
    with open(path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    setup_GeoMAN()
