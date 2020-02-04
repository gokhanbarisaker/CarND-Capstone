#!/usr/bin/env sh

docker ps | grep capstone | awk '{print $1 }' | xargs -o -I {} docker exec -it {} /bin/sh -c "[ -e /bin/bash ] && /bin/bash || /bin/sh"
