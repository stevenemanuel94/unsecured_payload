import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
import random as r
import numpy as np
import os, glob
import torch
from torchvision.transforms import Compose, CenterCrop, Resize

def start_server(gui):
    #start server
    if gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Loads the plane urdf file

    plane = p.loadURDF("plane.urdf")
    planeId = p.getBodyUniqueId(plane)
    p.changeDynamics(planeId,-1, lateralFriction=0, spinningFriction=0)

    #set gravity
    p.setGravity(0, 0, -10)

    p.setRealTimeSimulation(0)

    return planeId

def set_cam():
    # camera setup params
    offset = [0, 0, 0]
    yaw = 30
    pitch = -90
    dist = .4

    # set up camera
    width = 640  # 640  # 128
    height = 480  # 480  # 128

    aspect = width / height
    near = 0.01
    far = 100

    projection_matrix = (
    0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
    -0.02000020071864128, 0.0)

    cam = [width, height, projection_matrix, far, near, dist, pitch]

    return cam
def calc_mass(box_list, m_base):
    #calculate mass of tray-box system
    m_box = 0
    for box in box_list:
        m_box += p.getDynamicsInfo(box, -1)[0]

    #create B matrix for system
    B = np.array([[0, 0], [0, 0], [1 / (m_box + m_base), 0], [0, 1 / (m_box + m_base)]])
    return m_box, B

def random_box(num_box, num_layers, l_max, w_max, h_max, m_max, fric_max, fric_min, pos_bounds, x_off, y_off):
    #path to box textures
    texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)

    # set minimum dimension parameters
    m_min = 0.25
    l_min = 0.05
    h_min = 0.05
    w_min = 0.05

    #initialize box vector
    box_id = []

    for i in range(num_box):
        for j in range(num_layers):
            #randomize centroid position
            pos = [r.uniform(pos_bounds[0], pos_bounds[1]) + x_off, r.uniform(pos_bounds[0], pos_bounds[1]) + y_off, pos_bounds[2] + h_max*(j)]

            # randomize mass
            mass_update = r.random() * (m_max - m_min) + m_min

            # randomize length
            l = (l_max - l_min) * r.random() + l_min
            h = (h_max - h_min) * r.random() + h_min
            w = (w_max - w_min) * r.random() + w_min

            # randomize friction
            fric = (fric_max - fric_min) * r.random() + fric_min

            # calculate inertia diagonal
            ixx = 1/12 * (l**2 + h**2)
            iyy = 1/12 * (w**2 + h**2)
            izz = 1/12 * (l**2 + w**2)

            #randomize texture
            random_texture_path = texture_paths[r.randint(0, len(texture_paths) - 1)]
            textureId = p.loadTexture(random_texture_path)

            #create object in pyBullet
            visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                halfExtents=[l / 2, w / 2, h / 2])
            collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                      halfExtents=[l / 2, w / 2, h / 2])
            new_id = p.createMultiBody(baseMass=mass_update,
                                        basePosition=pos,
                                        baseCollisionShapeIndex=collisionShapeId,
                                        baseVisualShapeIndex=visualShapeId)
            p.changeVisualShape(new_id,-1, textureUniqueId=textureId)
            p.changeDynamics(new_id, -1,
                             localInertiaDiagonal=[ixx, iyy, izz],
                             lateralFriction=fric)
            box_id.append(new_id)
    return box_id

def base_box(l, w, h, m, fric, pos):
    #get path to random textures
    texture_paths = glob.glob(os.path.join('images', '**', '*.jpg'), recursive=True)

    # calculate inertia diagonal
    ixx = 1/12 * (l**2 + h**2)
    iyy = 1/12 * (w**2 + h**2)
    izz = 1/12 * (l**2 + w**2)

    # randomize texture
    random_texture_path = texture_paths[r.randint(0, len(texture_paths) - 1)]
    textureId = p.loadTexture(random_texture_path)

    visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                        halfExtents=[l / 2, w / 2, h / 2])
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                              halfExtents=[l / 2, w / 2, h / 2])
    robot_id = p.createMultiBody(baseMass=m,
                                       basePosition=pos,
                                       baseCollisionShapeIndex=collisionShapeId,
                                       baseVisualShapeIndex=visualShapeId)
    p.changeVisualShape(robot_id, -1, textureUniqueId=textureId)
    p.changeDynamics(robot_id, -1,
                     localInertiaDiagonal=[ixx, iyy, izz],
                     lateralFriction=fric)
    return robot_id

def get_payload_depth_img(robotId, cam):
    #get robot state
    robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
    robotOrn = p.getEulerFromQuaternion(robotOrn)

    #set camera parameters
    width = cam[0]
    height = cam[1]
    projection_matrix = cam[2]
    far = cam[3]
    near = cam[4]
    dist = cam[5]
    pitch = cam[6]

    #set view matrix
    view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=robotPos, distance=dist,
                                                      yaw=robotOrn[2] * 180 / np.pi, pitch=pitch,
                                                      roll=0, upAxisIndex=2)

    #
    images = p.getCameraImage(width,
                              height,
                              view_matrix,
                              projection_matrix,
                              shadow=True,
                              renderer=p.ER_BULLET_HARDWARE_OPENGL)
    depth_buffer_opengl = np.reshape(images[3], [height, width])
    depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    return depth_opengl, images[2]

def save_trial(trial_id, ts, robotId, init, f,phi, data, cam, t_count, sample_id, save_im):
    #save a trial during training data generation

    #get images
    payload_d, payload_rgb = get_payload_depth_img(robotId, cam)

    #get robot state
    robotOrn = init[1]
    robotVel = init[2]
    robotW = init[3]
    init_d = init[4]
    init_rgb = init[5]
    folder = trial_id // 100

    #find difference between initial and final payload depth images
    box_state = np.linalg.norm(init_d - payload_d, 'fro')

    #append to data list
    data.append({"init_path": f"{folder}/{trial_id}_id.npy",
                 "final_path": f"{folder}/{trial_id}_{sample_id}_fd.npy",
                 "robot_orn": robotOrn[2],
                 "robotV_x": robotVel[0],
                 "robotV_y": robotVel[1],
                 "robotW": robotW[2],
                 "f": f,
                 "phi": phi,
                 "ds":box_state,
                 "t": t_count})

    #save final depth array, final rgb image, and final depth image
    np.save(f"{ts}/{folder}/{trial_id}_{sample_id}_fd", payload_d)
    plt.imsave(f"{ts}/{folder}/{trial_id}_{sample_id}_frgb.png", payload_rgb)

    #if first sample, save initial depth array, initial rgb image, and initial depth image
    if save_im:
        np.save(f"{ts}/{folder}/{trial_id}_id", init_d)
        plt.imsave(f"{ts}/{folder}/{trial_id}_irgb.png", init_rgb)
        plt.imsave(f"{ts}/{folder}/{trial_id}_idepth.png", init_d)

def getInit(robotId, cam):
    #get initial state and images
    initPos, robotOrn = p.getBasePositionAndOrientation(robotId)
    initOrn = p.getEulerFromQuaternion(robotOrn)
    initVel, initW = p.getBaseVelocity(robotId)
    init_d, init_rgb = get_payload_depth_img(robotId, cam)
    init = [initPos, initOrn, initVel, initW, init_d, init_rgb]

    return init
def load_net(PATH):
    #loads a model for inference

    # select Cuda device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    #model path
    net = torch.load(PATH, map_location=device)
    net.eval()

    return net, device

def get_f_bounds(robotId, cam, net, device):
    #get the vertices of the safe control region from the neural network

    #get image
    payload_d, payload_rgb = get_payload_depth_img(robotId, cam)

    #get robot state
    robotVel, robotW = p.getBaseVelocity(robotId)
    state = np.zeros((1,3))
    state[0,0] = robotVel[0]
    state[0,1] = robotVel[1]
    state[0,2] = robotW[2]
    state = torch.from_numpy(state)

    #process image
    transform = Compose([CenterCrop((320, 320)), Resize((160, 160))])
    image = torch.from_numpy(payload_d)
    image = torch.unsqueeze(image, 0)
    image = torch.unsqueeze(image, 0)
    image = transform(image)

    #get vertices from eural network
    f_bounds = net(image.to(device), state.to(device))
    return payload_d, payload_rgb, f_bounds.cpu().detach().numpy()

def remove_dropped_box(robotId, box_list, planeId):
    #remove dropped boxes from simulations - CRASHES SIMULATION
    for box in box_list:
        if len(p.getContactPoints(box, planeId)) > 0:
            p.removeBody(box)
            box_list.remove(box)
    return box_list

def move_dropped_box(robotId, box_list, planeId):
    #move dropped box to far away location during controller testing

    robotPos, robotOrn = p.getBasePositionAndOrientation(robotId)
    for box in box_list:
        if len(p.getContactPoints(box, planeId)) > 0:
            p.resetBasePositionAndOrientation(box, [100, 100, 0], robotOrn)
            box_list.remove(box)
    return box_list

def record_box_data(box_list):
    #get box positions and mass/friction values

    box_data = np.zeros((len(box_list),4))
    n = 0
    for box in box_list:
        boxPos, boxOrn = p.getBasePositionAndOrientation(box)
        box_data[n, 0] = boxPos[0] #x
        box_data[n, 1] = boxPos[1] #y
        box_data[n, 2] = p.getDynamicsInfo(box, -1)[0] #mass
        box_data[n, 3] = p.getDynamicsInfo(box, -1)[1]  #fric
        n+=1
    return box_data

def save_test(x_hist, u_hist, v_hist, ds, d_init, d_final, rgb_init, rgb_final, box_init, box_final, w, n, constrained, dir):
    #save controller test data
    filename = dir + f"/trial{n}_w{w}_c{constrained}"

    #save state trajectory, control trajectory, neural net outout (v_hist), shift in image, initial and final box states
    np.savez_compressed(filename, x_hist=x_hist, u_hist=u_hist, v_hist=v_hist, ds=ds, box_init=box_init, box_final=box_final)

    #save initial rgbd and depth images
    plt.imsave(filename+"_irgb.png", rgb_init)
    plt.imsave(filename+"_idepth.png", d_init)

    #save final RGBD and depth images
    plt.imsave(filename + "_frgb.png", rgb_final)
    plt.imsave(filename + "_fdepth.png", d_final)
def rgbd_camera_setup():
    # Mimic RealSense D415 RGB-D camera parameters. Not used in code
    image_size = (480, 640)
    intrinsics = (450., 0, 320., 0, 450., 240., 0, 0, 1)

def base_tray(m):
    #uses the tray in Pybullet - not used in test or data generation scripts at all
    # get path to random textures
    texture_paths = glob.glob(os.path.join('images', '**', '*.jpg'), recursive=True)

    # randomize texture
    random_texture_path = texture_paths[r.randint(0, len(texture_paths) - 1)]
    textureId = p.loadTexture(random_texture_path)

    robot_id = p.loadURDF("tray/traybox.urdf")

    p.changeVisualShape(robot_id, -1, textureUniqueId=textureId)
    p.changeDynamics(robot_id, -1, mass=m)
    return robot_id