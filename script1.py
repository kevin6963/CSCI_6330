from mpi4py import MPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def load_data():
    # Load the data in the master process (rank 0)
    if rank == 0:
        return pd.read_csv("tndata.csv")
    else:
        return None

def broadcast_data(data):
    # Broadcast data to all processes
    return comm.bcast(data, root=0)

def preprocess_data(data, scaler, cs="yes"):
    # Reverse index
    tdata = data.reindex(index=data.index[::-1])

    # Get the infected and recovered data
    I = tdata['TOTAL_CONFIRMED']
    R = tdata['TOTAL_INACTIVE_RECOVERED']

    # Get the length of the data
    nn = len(I)

    # Show whether scaling is required
    if cs == "yes":
        tt = np.linspace(0, nn, nn)
        y1 = np.array(I[:nn]).reshape((-1, 1))
        y2 = np.array(R[:nn]).reshape((-1, 1))

        # Scaling
        II = scaler.fit_transform(y1)
        RR = scaler.fit_transform(y2)

        # Plot scaled data in the master process
        if rank == 0:
            plot_data(tt, II, RR)
    else:
        tt = np.linspace(0, nn, nn)
        II = np.array(I[:nn]).reshape((-1, 1))
        RR = np.array(R[:nn]).reshape((-1, 1))

        # Plot unscaled data in the master process
        if rank == 0:
            plot_data(tt, II, RR)

def plot_data(tt, II, RR):
    # Plotting function
    plt.plot(tt, II, '--r')
    plt.plot(tt, RR, '--b')
    plt.legend(['Infected', 'Recovered'])
    plt.title('Scaled Data' if cs == "yes" else 'Unscaled Data')
    plt.show()

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
scaler = MinMaxScaler()

# Load and broadcast data
data = load_data()
data = broadcast_data(data)

# Define the preprocess function
cs = "no"  # Set cs as needed
preprocess_data(data, scaler, cs)

if rank == 0:
    print("Rank 0 finished")
