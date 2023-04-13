import keyboard
from maestro import Controller
tango = Controller()

def forward():
    tango.setTarget(1,6000)
    tango.setTarget(0,5000)

def back():
    tango.setTarget(1,6000)
    tango.setTarget(0,7000)

def left():
    tango.setTarget(0,6000)
    tango.setTarget(1,5000)

def right():
    tango.setTarget(0,6000)
    tango.setTarget(1,7000)

def stop():
    tango.setTarget(0,6000)
    tango.setTarget(1,6000)

keyboard.on_press_key("w",forward)
keyboard.on_release_key("w",stop)
keyboard.on_press_key("s",back)
keyboard.on_release_key("s",stop)
