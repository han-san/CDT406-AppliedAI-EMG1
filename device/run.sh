#!/bin/bash

mkfifo adc_pipe
mkfifo inference_pipe
sudo systemctl start adc
systemctl --user start model
python3 inference_res.py