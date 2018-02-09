
import numpy as np
import time

from Adafruit_PWM_Servo_Driver import PWM

def set_servo_angle(
        pwm,
        channel,
        angle,
        min_angle=10,
        max_angle=170,
        min_pulse=150,
        max_pulse=600):
  '''
  Description:
      - set servo angle
      - max_angle corresponds to maximum position clockwise with servo facing upward
  '''
  # clip angle to valid range
  angle = max(min(angle, max_angle), min_angle)

  # linearly interpolate between min_pulse and max_pulse
  # depending on where in the possible range angle falls
  frac = (angle - min_angle) / (max_angle - min_angle)
  pulse = int(min_pulse + frac * (max_pulse - min_pulse))
  pwm.setPWM(channel, 0, pulse)

# Initialise the PWM device using the default address
pwm = PWM(0x40)

servoMin = 150  # Min pulse length out of 4096
servoMax = 600  # Max pulse length out of 4096
min_angle = 10
max_angle = 170

# Set frequency to 60 Hz
pwm.setPWMFreq(60)

steps = 200
sleep_time = .005
while True:

  for angle in np.linspace(min_angle, max_angle, steps):
    set_servo_angle(pwm, 0, angle, min_angle=min_angle, max_angle=max_angle)
    time.sleep(sleep_time)

  for angle in reversed(np.linspace(min_angle, max_angle, steps)):
    set_servo_angle(pwm, 0, angle, min_angle=min_angle, max_angle=max_angle)
    time.sleep(sleep_time)
