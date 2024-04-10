#!/bin/bash

papermill \
    -k python3 --log-output \
    "./World-7B-mem-finetune.ipynb" "./World-7B-mem-finetune.output.ipynb" 

papermill \
    -k python3 --log-output \
    "./World-3B-mem-finetune.ipynb" "./World-3B-mem-finetune.output.ipynb" 

papermill \
    -k python3 --log-output \
    "./World-1B5-mem-finetune.ipynb" "./World-1B5-mem-finetune.output.ipynb" 
