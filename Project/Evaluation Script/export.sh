#!/usr/bin/env bash

./build.sh

docker save detection > detection.tar
