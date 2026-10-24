#!/bin/bash

paths=$(find . -name "checkpoint*.pth")

for path in $paths
do
    rm $path
done
