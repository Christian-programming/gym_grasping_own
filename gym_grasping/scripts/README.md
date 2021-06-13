# Current Developments

# Development Notes

I'm currently working on suction gripper code. This should do the following
1. On contact with object create a fixed constraint
2. Releasable trought mouse control.

For this I have to move additional information from the collision tuple to the
robot_contact_callback function about the relative geometry.

After this I want to create a demo video of the whole box being inserted (side/side).



# Getting Started
Use the following script `grasping_env_test.py` to start simulations in an interactive way. Change between the different modes avaliable in this script to debug the enviroment.


# Making Videos

Using the `Viewer` class, as in the  `enjoy` mode of `grasping_env_test.py` create a bunch of frames, then convert these to a video using:

ffmpeg -framerate 8  -pix_fmt yuv420p -i %03d.png output.mp4
ffmpeg -start_number 150 -framerate 20  -pix_fmt yuv420p -i %03d.png sim_episodes.mp4 
ffmpeg -start_number 150 -framerate 20  -pix_fmt yuv420p  -i %03d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 sim_episodes.mp4
ffmpeg -framerate 20 -i %03d.png -vcodec libvpx-vp9  video.mp4


# Deubg Viewer
F1: to record frames

Conver Physics Server 00.png -> 00.png
ls |awk -F'ver' '{print "mv \"" $0"\"", $2}'|bash

