#!/bin/bash
docker run -it --rm --tty --gpus all --volume=$(pwd)/io:/home/gdfx2/io gdfx2
