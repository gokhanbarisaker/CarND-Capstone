#!/usr/bin/env sh

docker ps | grep capstone | awk '{print $1 }' | xargs -I {} docker kill {}
