'''
see ''connecting an LED'' tutorial
'''

import numpy as np
import RPi.GPIO as gpio
import time

gpio.setmode(gpio.BCM)
gpio.setup(18, gpio.OUT)
runs = 100
dts = np.linspace(.5, .01, runs)
for i in range(runs):
    time.sleep(dts[i])
    gpio.output(18, True)
    time.sleep(dts[i])
    gpio.output(18, False)
