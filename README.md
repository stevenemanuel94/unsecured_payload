The 0.001_Tue02Jul2024_12_42_nn.pth file is the trained neural network for inference. 

The images and dtd folders provide textures for the tray and boxes in the simulator. 

There are 3 main scripts:

generate_train_data.py saves the maximum force in a given direction, robot state, and initial and final depth images and numpy arrays
for each trial. This data is intended for training the neural network. The force data is output into a CSV file. 

nn_train.py is the script for training the neural network. The training data is too large to include here, but can be transferred over from PACE if desired. The hyperparameters in the scrpt were the final parameters used for model training. 

test_circle_path.py implements the MPC controller with the trained network for testing. 

A few modules provide necessary functions:

bullet_func.py contains a number of functions that help set up the simulation environment

nn_safe_reg.py contains the pytorch model definition for the network

mpc_test.py contains the mpc implementation

