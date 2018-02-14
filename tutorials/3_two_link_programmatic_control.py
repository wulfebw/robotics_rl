import rlrobo.manipulator

if __name__ == '__main__':
    manipulator = rlrobo.manipulator.build_RR_manipulator()
    while True:
        pos = manipulator.random_position()
        manipulator.set_end_effector_position(pos)