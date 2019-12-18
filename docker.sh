#!/bin/bash

#docker run --runtime=nvidia -it --rm -p 8888:8888  -v /disk011/usrs/hagio:/hagio nh122112/tensorflow:stylegan_2
docker run --runtime=nvidia -it -v /disk018/usrs/hagio:/hagio nh122112/tensorflow:stylegan_2
