#!/bin/sh

timestamp() {
    if [ "$1" = "start" ];then
        echo "Start Time: "
    elif [ "$1" = "end" ];then
        echo "End Time: "
    fi
    date +%F_%T
    echo
}