#!/bin/sh

if [ "$1" = 'local' ]; then
    echo "start ssh and run infinity"
    service ssh restart && sleep infinity
else
    eval "$@"
fi