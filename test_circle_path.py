import pybullet as p
import bullet_func as bf
import numpy as np
import mpc_test as mpc
import datetime
import os
import random as r
import configparser


#neural network path
PATH = '0.001_Tue02Jul2024_12_42_nn.pth'

#make directory
ts = datetime.datetime.now().strftime('%a%d%b%Y_%H_%M_%S.%f')[:-3]
os.mkdir(f"./{ts}/")
dir = f"{ts}"

cam = bf.set_cam()

#sample num
num_samples = 100
w_vect = [0.75, 1, 1.25, 1.5]
constrained_vect = [0, 1]

#set 1 for GUI, 0 for no UI during tests
gui = 0

for sample in range(num_samples):
    #start physics server
    planeId = bf.start_server(gui)

    #set mass of base
    m_base = .226

    #generate tray and random boxes
    robotId = bf.base_box(.5,.5,.1,m_base,1,[1,0,.1])
    boxInitList = bf.random_box(r.randint(1,5), r.randint(1,3), .1, .1, .1, 1, 0.8, .5, [-.15, .15, 0.1], 1, 0)


    #generate initial velocity
    theta = 0 #heading
    ang = 0 #initial rotational velocity
    speed = 0 #initial speed
    x_dot = speed * np.cos(theta)
    y_dot = speed * np.sin(theta)

    #allow boxes to settle
    for i in range(1000):
        p.stepSimulation()

    #set initial velocities
    for i in range(10):
        p.stepSimulation()
        p.resetBaseVelocity(robotId, linearVelocity = [x_dot, y_dot, 0], angularVelocity = [0, 0, ang])
        for box in boxInitList:
            boxPos, boxOrn = p.getBasePositionAndOrientation(box)
            robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
            boxV = [x_dot, y_dot, 0] + np.cross([0, 0, ang], (np.array(boxPos) - np.array(robotPos)))
            p.resetBaseVelocity(box, linearVelocity=boxV)

    #remove dropped boxes
    boxInitList = bf.remove_dropped_box(robotId, boxInitList, planeId)

    #set sys dynamics
    m_box, B = bf.calc_mass(boxInitList, m_base)
    A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
    C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    D = 0

    #save tray and box state
    stateId = p.saveState()

    for w in w_vect:
        # set simulation rate
        hz = 120
        dt = 1 / hz
        t_final = int(np.ceil(2 * np.pi / w))

        #prediction horizon (in seconds)
        horizon = .2

        #time vector
        t = np.linspace(0, t_final, hz * t_final + 1)
        N = int(horizon * hz)
        num_steps = len(t) * 2

        #create desired traj
        x_traj = np.array([np.cos(w * t)])
        y_traj = np.array([np.sin(w * t)])
        x_desired = np.concatenate((x_traj.T, y_traj.T), 1)

        #Cost matrices
        Q = 1 / 10 * np.eye(N)
        Q[0][0] = 10
        Q[1][1] = 10

        #r in the mpc implementation
        cost_u = 0.000005

        #run for the constrained and unconstrained cases
        for constrained in constrained_vect:
            #restore initial state
            p.restoreState(stateId)

            #initialize counters and arrays
            i = 0
            u_hist = np.zeros((2,len(t) - N))
            x_hist = np.zeros((4,len(t) - N))
            v_hist = np.zeros((18, len(t) - N))
            ds = np.zeros((1, len(t) - N))
            vertices = np.zeros((18,2))

            #load neural net
            net, device = bf.load_net(PATH)

            #get init image
            init_d, init_rgb, f_bounds = bf.get_f_bounds(robotId, cam, net, device)
            box_init = bf.record_box_data(boxInitList)
            boxFinalList = boxInitList.copy()

            #control loop
            for i in range(num_steps-2*N):
                #move any dropped boxes before getting new image
                boxFinalList = bf.move_dropped_box(robotId, boxFinalList, planeId)

                # call mpc every other time step
                if i % 2 == 0:
                    #update camera and get force limits
                    payload_d, payload_rgb, f_bounds = bf.get_f_bounds(robotId, cam, net, device)

                    #get states
                    robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
                    robotVel, robotW = p.getBaseVelocity(robotId)
                    phi = np.arctan2(robotVel[1], robotVel[0])

                    #assign x-y components to each safe control vertex
                    for n in range(18):
                        vertices[n, :] = [f_bounds[0,n] * np.cos(phi + n*2*np.pi/18), f_bounds[0,n] * np.sin(phi + n*2*np.pi/18)]

                    #construct current state vector
                    x = [robotPos[0], robotPos[1], robotVel[0],  robotVel[1]]

                    #get control input
                    u_out = mpc.compute_MPC(A, B, C, D, dt, N, x_desired[(i//2):(N + i//2),:], x, Q, cost_u, vertices.T, constrained)

                #apply 1st value in control trajectory
                p.applyExternalForce(robotId, -1, [u_out[0], u_out[1], 0], robotPos, p.WORLD_FRAME)

                #record at 120Hz
                if i % 2 == 0:
                    for n in range(4): x_hist[n, i // 2] = x[n]
                    v_hist[:, i // 2] = f_bounds
                    u_hist[:, i // 2] = u_out
                    ds[:, i // 2] = np.linalg.norm(payload_d - init_d, 'fro')

                #step sim
                p.stepSimulation()

            #get box data
            box_final = bf.record_box_data(boxFinalList)

            #save test data
            bf.save_test(x_hist, u_hist, v_hist, ds, init_d, payload_d, init_rgb, payload_rgb, box_init, box_final, w, sample, constrained, dir)
    p.disconnect()

