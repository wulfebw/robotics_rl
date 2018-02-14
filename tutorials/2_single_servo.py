
import numpy as np
import time

from Adafruit_PWM_Servo_Driver import PWM

from rlrobo.utils import set_servo_angle

# Initialise the PWM device using the default address
pwm = PWM(0x40)

min_pulse = 150  # Min pulse length out of 4096
max_pulse = 600  # Max pulse length out of 4096
min_angle = 10
max_angle = 170

# Set frequency to 60 Hz
pwm.setPWMFreq(60)

steps = 200
sleep_time = .005
while True:

  for angle in np.linspace(min_angle, max_angle, steps):
    set_servo_angle(pwm, 0, angle, min_angle=min_angle, max_angle=max_angle, min_pulse=min_pulse, max_pulse=max_pulse)
    time.sleep(sleep_time)

  for angle in reversed(np.linspace(min_angle, max_angle, steps)):
    set_servo_angle(pwm, 0, angle, min_angle=min_angle, max_angle=max_angle, min_pulse=min_pulse, max_pulse=max_pulse)
    time.sleep(sleep_time)
