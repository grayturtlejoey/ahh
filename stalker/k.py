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

while True:
    # Wait for the next event.
    event = keyboard.read_event()
    if event.event_type == keyboard.KEY_DOWN and event.name == 'w':
        forward()
    if event.event_type == keyboard.KEY_UP:
        stop()
    if event.event_type == keyboard.KEY_DOWN and event.name == 's':
        back()
    if event.event_type == keyboard.KEY_DOWN and event.name == 'a':
        left()
    if event.event_type == keyboard.KEY_DOWN and event.name == 'd':
        right()
    if event.event_type == keyboard.KEY_DOWN and event.name == 'escape':
        stop()
        break



