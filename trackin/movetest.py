from maestro import Controller
import time

MOTORS = 0
ROLL_SPEED = 6000

tango = Controller()

for i in range(4):
    ROLL_SPEED = 5000
    tango.setTarget(MOTORS,ROLL_SPEED)
    time.sleep(1)
    ROLL_SPEED = 6000
    tango.setTarget(MOTORS,ROLL_SPEED)
    time.sleep(3)
    ROLL_SPEED = 7000
    tango.setTarget(MOTORS,ROLL_SPEED)
    time.sleep(1)
    ROLL_SPEED = 6000
    tango.setTarget(MOTORS,ROLL_SPEED)
    time.sleep(3)
