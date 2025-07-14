#!/usr/bin/env bash

# Uses ffmpeg to convert .gif to .m4a
script_dir=$(dirname "$(realpath "$0")")

ffmpeg -y -i $script_dir/../demo.gif -strict -2 -an -b:v 32M $script_dir/../demo.m4a

