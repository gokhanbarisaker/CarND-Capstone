#!/usr/bin/env sh

docker run -p 4567:4567 -v $PWD:/capstone -v $PWD/logs:/root/.ros/ --rm -it capstone
