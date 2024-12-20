import pyglet
import time 

sound=pyglet.media.load("sound.wav",streaming=False)
left_sound=pyglet.media.load("left.wav",streaming=False)
right_sound=pyglet.media.load("right.wav",streaming=False)
sound.play()
time.sleep(1)
left_sound.play()
left_sound.play()
time.sleep(1)
right_sound.play()
time.sleep(1)