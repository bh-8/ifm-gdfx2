#!/bin/bash
docker run --rm --tty --gpus all --volume=$(pwd)/io:/home/gdfx2/io gdfx2 "$@"
exit $?
