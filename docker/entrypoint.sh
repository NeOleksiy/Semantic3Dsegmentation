#!/bin/bash

cd /workspace

CMD="python train.py"

if [ "$1" = "infer" ]; then
    shift
    CMD="python inference.py $@"
fi

exec $CMD