
import time

import rlrobo.manipulator

if __name__ == '__main__':
    manipulator = rlrobo.manipulator.build_RR_manipulator(l1=1,l2=2)
    while True:
        pos = manipulator.random_position()
        manipulator.set_end_effector_position(pos)
        time.sleep(1)
