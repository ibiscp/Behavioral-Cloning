##Resources For Completing the Project
There are a few files needed to complete the Behavioral Cloning project.

Besides the simulator, we have also included a python script, drive.py, which is used to drive the car autonomously around the track once your deep neural network model is trained.

The simulator contains two tracks. We have included sample driving data for the first track, which can optionally be used to help train your network; you will probably need to collect more data in order to get the vehicle to stay on the road.

The lectures in the Behavioral Cloning lesson will explain how to use these files and will give more information about how to complete the project. Here are links to the resources that you will need:

* [drive.py](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/589244ed_drive/drive.py)
* [sample data for track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
* [project rubric](https://review.udacity.com/#!/rubrics/432/view)

##Simulator Download
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
* [Windows 32-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
* [Windows 64-bit](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)

##Beta Simulators
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894ee55_beta-simulator-linux/beta-simulator-linux.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894ecbd_beta-simulator-mac/beta-simulator-mac.zip)
* [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894ea69_beta-simulator-windows/beta-simulator-windows.zip)

Here are the main differences between the stable simulator and the beta simulator:

1. Steering is controlled via position mouse instead of keyboard. This creates better angles for training. Note the angle is based on the mouse distance. To steer hold the left mouse button and move left or right. To reset the angle to 0 simply lift your finger off the left mouse button.
2. You can toggle record by pressing R, previously you had to click the record button (you can still do that).
3. Recording saves the images to disk all at once recording is toggled off. You can see a save status and play back of the captured data.
4. You can takeover in autonomous mode. While W or S are held down you can control the car the same way you would in training mode. This can be helpful for debugging. As soon as W or S are let go autonomous takes over again.
5. Pressing the spacebar in training mode toggles on and off cruise control (effectively presses W for you).
6. Only the center camera is recorded (open to change).
7. Added a Control screen
8. Track 2 was replaced from a mountain theme to Jungle with free assets , Note the track is challenging
9. You can use brake input in drive.py by issuing negative throttle values