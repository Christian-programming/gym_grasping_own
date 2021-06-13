from gym_grasping.envs.robot_sim_env import RobotSimEnv
import gym
import cv2 
from robot_io.input_devices.keyboard_input import KeyboardInput
import numpy as np
import matplotlib.pyplot as plt

cv2.imshow("win", np.zeros((300, 300)))
cv2.waitKey(1)

def main():
    print("start keyboad")
    print("end keyboad")
    env_name = "kuka_block_grasping-v0"
    env  = gym.make(env_name, renderer='egl')

    print("reset")
    env.reset()

    print("start")
    for i in range(1000):
        #action = keyboard.handle_keyboard_events()
        action = env.action_space.sample()
        print("action ", action)
        ob, reward, done, info = env.step(action)
        #img = cv2.resize(ob['img'][:, :, ::-1], (300, 300))
        #img = cv2.resize(ob[:, :, ::-1], (300, 300))
        img = (cv2.resize(ob, (300, 300)) / 255).astype(np.float32)
        print(img.dtype)
        cv2.imshow("win", img)
        cv2.waitKey(0)


#if __name__ ==  "__main__":
#    main()
print("start keyboad")
print("end keyboad")
env_name = "kuka_block_grasping-v0"
env  = gym.make(env_name, renderer='egl')
keyboard = KeyboardInput()

print("reset")
env.reset()
print("start")
for i in range(1000):
    action = keyboard.handle_keyboard_events()
    #action = env.action_space.sample()
    print("action ", action)
    ob, reward, done, info = env.step(action)
    #img = cv2.resize(ob['img'][:, :, ::-1], (300, 300))
    img = cv2.resize(ob[:, :, ::-1], (300, 300)) / 255
    #img = (cv2.resize(ob, (300, 300)) / 255).astype(np.float32)
    print(img)
    cv2.imshow("win", img)
    cv2.waitKey(0)
