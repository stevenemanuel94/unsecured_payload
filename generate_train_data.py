import pybullet as p
import pandas as pd
import datetime
import os
import math
import random as r
import bullet_func as bf
import numpy as np

#set number of trials and samples per trial (# of vertices for safe control region)
num_samples = 18
num_trials = 10000

#set up camera
cam = bf.set_cam()

#set up storage
ts = datetime.datetime.now().strftime('%a%d%b%Y_%H_%M_%S.%f')[:-3]
num_folder = (num_trials - 1) // 100 + 1
os.mkdir(f"{ts}")

for i in range(num_folder):
    os.mkdir(f"./{ts}/{i}")

data = []

#set 1 for GUI, 0 for no UI during tests
gui = 0

for trial in range(num_trials):
    planeId = bf.start_server(gui)

    robotId = bf.base_box(.5,.5,.1,.226,1,[0,0,.1])
    box_list = bf.random_box(r.randint(2, 3), r.randint(3, 4), .1, .1, .1, 1, 0.8, .5, [-.15, .15, 0.1], 0, 0)

    #generate random initial velocity
    theta = np.pi * 2 * r.random()
    w = r.random() * 1
    speed = r.random() * 3
    x_dot = speed * np.cos(theta)
    y_dot = speed * np.sin(theta)

    #allow boxes to settle
    for i in range(1000):
        p.stepSimulation()

    #set initial veolocities
    for i in range(10):
        p.stepSimulation()
        p.resetBaseVelocity(robotId, linearVelocity = [x_dot, y_dot, 0], angularVelocity = [0, 0, w])
        for box in box_list:
            boxPos, boxOrn = p.getBasePositionAndOrientation(box)
            robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
            boxV = [x_dot, y_dot, 0] + np.cross([0, 0, w], (np.array(boxPos) - np.array(robotPos)))
            p.resetBaseVelocity(box, linearVelocity=boxV)

    #remove dropped boxes
    robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
    init_rel_pos = np.array([])
    for box in box_list:
        if len(p.getContactPoints(box,planeId))>0:
            p.removeBody(box)
            box_list.remove(box)
    for box in box_list:
        boxPos, boxOrn = p.getBasePositionAndOrientation(box)
        relShift = np.linalg.norm(np.array(robotPos) - np.array(boxPos))
        init_rel_pos = np.insert(init_rel_pos,len(init_rel_pos),relShift)

    #get heading
    phi = np.arctan2(y_dot, x_dot)

    #set sim length
    t_len = 0.1
    num_steps = math.ceil(t_len * 240)
    t = np.linspace(0, t_len, num_steps)
    t_count = t_len * 240

    #save current state
    stateId = p.saveState()

    #set amount of allowable box shifting (meters)
    tol = 0.005


    #find max force
    for sample in range(num_samples):
        # force time and direction
        f_mag = 100  # r.random()*10
        f_low = 0
        f_high = 200
        err = 0

        while (abs(f_high-f_low) > 0.25):
            #init state
            p.restoreState(stateId)
            init = bf.getInit(robotId, cam)

            #apply force
            for i in range(len(t)):
                f = f_mag*np.array([np.cos(phi), np.sin(phi), 0]) #np.array([np.cos(phi), np.sin(phi), 0]) * r.random() * 25
                robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)

                if i < t_count:
                    p.applyExternalForce(robotId, -1, f, robotPos, p.WORLD_FRAME)
                p.stepSimulation()

            #find if there was shift in relative position
            robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
            fin_rel_pos = np.array([])
            for box in box_list:
                boxPos, boxOrn = p.getBasePositionAndOrientation(box)
                relShift = np.linalg.norm(np.array(robotPos)-np.array(boxPos))
                fin_rel_pos = np.insert(fin_rel_pos, len(fin_rel_pos), relShift)
            err = np.max(np.abs(fin_rel_pos - init_rel_pos))
            if err > tol:
                f_next = (f_mag + f_low) / 2
                f_high = f_mag
            else:
                f_next = (f_high + f_mag) / 2
                f_low = f_mag
            f_mag = f_next


        #increment phi
        phi += 2 * np.pi / num_samples  # r.random()*np.pi*2

        # get image and box state
        save_im = 0
        if sample == 0:
            save_im=1
        bf.save_trial(trial, ts, robotId, init, f_mag, phi, data, cam, t_count, sample, save_im)
        df = pd.DataFrame(data)
        df.to_csv(f"./{ts}/meta.csv", index=False)

    p.disconnect()
