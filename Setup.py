import os
from pathlib import Path


def setup_GeoMAN():
    # Fix GeoMAN imports by making them relative
    path = os.sep.join(["Models", "GeoMAN", "GeoMAN.py"])
    with open(path, "r") as f:
        content = f.read()
    content = content.replace("from base_model import BaseModel", "from .base_model import BaseModel")
    content = content.replace("from utils import Linear", "from .utils import Linear")
    with open(path, "w") as f:
        f.write(content)


def setup_MTGNN():
    # Make MTGNN repository a package
    Path(os.sep.join(["Models", "MTGNN", "__init__.py"])).touch()
    # Fix MTGNN imports by making them relative
    path = os.sep.join(["Models", "MTGNN", "net.py"])
    with open(path, "r") as f:
        content = f.read()
    content = content.replace("from layer import *", "from .layer import *")
    with open(path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    setup_GeoMAN()
    setup_MTGNN()
