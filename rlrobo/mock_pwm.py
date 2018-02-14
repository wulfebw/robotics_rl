'''
A Mock PWM class for development purposes
'''

class PWM(object):

    def __init__(self, address=0x40, debug=True):
        self.address = address
        self.debug = debug

    def setPWMFreq(self, freq):
        if self.debug:
            print('PWM frequency set to {}'.format(freq))

    def setPWM(self, channel, on, off):
        if self.debug:
            print('PWM set channel: {}\ton: {}\toff: {}'.format(channel, on, off))