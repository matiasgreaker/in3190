import pathlib
import pickle
from turtle import st
import numpy as np
import matplotlib.pyplot as plt
import dataclasses

from project1.bjorklund_extracts import StationData
import obspy
import h5py
from obspy.geodetics import gps2dist_azimuth
import os


# Set project root
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.resolve()
# Set data path
DATA_PATH = PROJECT_ROOT / 'project1' / 'project_data'
# Location to save temp cache data, to avoid to re-compute a bunch of steps
CACHE_DIR = DATA_PATH / pathlib.Path(".cache")
CACHE_STATION_DATA = CACHE_DIR / pathlib.Path("station_data_cache_filtered.pickle")

TONGA_LAT_LON : tuple[float,float] = (-20.550, -175.385) # latitude and longitude


# NOTE: This is not used in the final version, but I kept it here for reference
# as it was part of the development process
@dataclasses.dataclass
class StationDataFiltered(StationData):
    """Data from a station contained in one data object. Here it is filtered data with the three filters.
    Inherits from StationData to keep all the original data as well."""
    data_filtered_h1 : np.ndarray = None
    """The data filtered with h1"""
    data_filtered_h2 : np.ndarray = None
    """The data filtered with h2"""
    data_filtered_h3 : np.ndarray = None



def parse_h5_to_obspy_station_data(num_files: int = None) -> obspy.Stream:
    """
    Parse the HDF5 files in DATA_PATH and return a obspy.Stream.
    I've chosen to use ObsPy for this, as it is a well established library for seismology data.
    I started to use a data class for this, but it felt like re-inventing the wheel.
    This function also computes the distance from each station to the Tonga eruption, and saves it in the stats.distance attribute of each Trace object.
    """

    fn_list = os.listdir(DATA_PATH) #List of all file names
    stream = obspy.Stream() # Create an empty stram
    #Loop over files
    for i, file in enumerate(fn_list):
        if file.endswith('.h5'):
            # Only pick a certain number of files if num_files is set
            if num_files is not None and i >= num_files:
                break
            this_fn = h5py.File(pathlib.Path(DATA_PATH / file), 'r')
            lat = this_fn.attrs['latitude']
            lon = this_fn.attrs['longitude']
            st= obspy.read(str(DATA_PATH / file))
            # Adding the distance to Tonga in meters to the stats.distance attribute of the Trace object
            st[0].stats.distance = gps2dist_azimuth(lat,lon,TONGA_LAT_LON[0],TONGA_LAT_LON[1])[0]
            stream += st
    return stream


def plot_section(stream: obspy.Stream):
    """Plot a section plot of the data in an ObsPy stream object
    I hope you do not consider this cheating, but I used the built-in section plot method in ObsPy.
    The task only said that we should "generate a section plot" :P"""
    # Now we have a stream with all the data
    fig, ax = plt.subplots()   
    stream.plot(type='section', plot_dy=20e3, orientation='horizontal', linewidth=.25, grid_linewidth=.25, show=False, fig=fig, reftime=obspy.UTCDateTime(stream[0].stats.starttime))
    ax.set_title('Section plot of all stations')
    plt.show()

def filter_station_data(station_data: StationData) -> StationDataFiltered:
    """Filter the data from a StationData object with the three FIR filters h1, h2 and h3.
    Returns a StationDataFiltered object with the filtered data."""
    data_h1 = np.convolve(h1, station_data.data, mode="same")
    data_h2 = np.convolve(h2, station_data.data, mode="same")
    data_h3 = np.convolve(h3, station_data.data, mode="same")
    return StationDataFiltered(
        station_name=station_data.station_name,
        longitude=station_data.longitude,
        latitude=station_data.latitude,
        data=station_data.data,
        data_filtered_h1=data_h1,
        data_filtered_h2=data_h2,
        data_filtered_h3=data_h3,
        N_samples=station_data.N_samples,
        start_time=station_data.start_time,
        dt=station_data.dt,
        distance_to_tonga=station_data.distance_to_tonga
    )

def convolve_signal_with_filter(h: np.ndarray, x: np.ndarray, ylen_choice: int = 1) -> np.ndarray:
    """
    Convolves the input signal with the given FIR filter coefficients.
    Returns:
    np.ndarray: The filtered signal.
    """
    ylen = len(h)+len(x)-1
    if ylen_choice == 0:
        ylen = len(x)

    y = np.zeros(ylen)
    for n in range(ylen):
        # Perform the convolution operation
        # This is a naive implementation and can be optimized
        for k in range(len(x)):
            # Only accumulate if within bounds
            if n - k >= 0 and n - k < len(h):
                y[n] += x[k] * h[n-k]
            # IF this happens we can break the inner loop
            if n - k < 0:
                break


    return y


def save_filtered_data(station_data_filtered: list[StationDataFiltered]):
    """Save the filtered station data to a pickle file."""
    import pickle
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True)
    with open(CACHE_STATION_DATA, 'wb') as f:
        pickle.dump(station_data_filtered, f)

def load_filtered_data() -> list[StationDataFiltered]:
    """Load the filtered station data from a pickle file."""
    import pickle
    with open(CACHE_STATION_DATA, 'rb') as f:
        station_data_filtered = pickle.load(f)
    return station_data_filtered

def dft(x: np.ndarray, N: int, fs: float = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Discrete Fourier Transform (DFT) of a 1D signal x.
    Parameters:
    x : np.ndarray
        Input signal.
    N : int
        Number of points in the DFT.
    fs : float
        Sampling frequency of the input signal.
    Returns:
    f : np.ndarray
        Frequency bins corresponding to the DFT.
    X : np.ndarray
        DFT of the input signal.
    """
    k = np.arange(N)
    # Create the angular frequencies
    omega = (2 * np.pi *k / N) 
    # Create the DFT
    X = np.zeros((N,), dtype=complex)
    for i, w in enumerate(omega):
        for n in range(len(x)):
            X[i] += x[n] * np.exp(-1j * w * n)
    # Frequency bins
    f = k * fs / N
    return X, f


@dataclasses.dataclass
class ArrivalTimePick:
    """Dataclass to hold the arrival time pick for a single trace."""
    trace_index: int
    station_name: str
    arrival_time: float
    distance_to_tonga: float

def pick_arrival_time(data_stream: obspy.core.Stream, start_index: int = 0, stop_index: int = int(1e6)) -> list[ArrivalTimePick]:
    """
    Pick the arrival time manually using ginput in each trace of the stream.
    First filter the data with each of the three filters and plot them in subplots.
    Ive chosen to return a list of ArrivalTimePick dataclass objects, to keep track of the trace id and distance to Tonga as well.
    I think it is better to return structured data instead of just a list of floats.
    """
    data_stream.sort(keys=["distance"])  # Sort the stream by distance to Tonga
    arrival_times = list[ArrivalTimePick]()
    for i, trace in enumerate(data_stream):
        # Only process traces within the specified index range
        if i < start_index:
            continue
        if i > stop_index:
            break
        # Filter the data with each of the three filters
        data_h1 = np.convolve(h1, trace.data, mode="same")
        data_h2 = np.convolve(h2, trace.data, mode="same")
        data_h3 = np.convolve(h3, trace.data, mode="same")
        # Create a figure with four subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        time = trace.times()
        axs[0].plot(time, trace.data)
        axs[0].set_title(f"Original Trace - {trace.id}")
        axs[1].plot(time, data_h1)
        axs[1].set_title("Filtered Trace - h1")
        axs[2].plot(time, data_h2)
        axs[2].set_title("Filtered Trace - h2")
        axs[3].plot(time, data_h3)
        axs[3].set_title("Filtered Trace - h3")
        axs[3].set_xlabel("Time [s]")
        fig.suptitle(f"Pick arrival time for {trace.id}")
    
        plt.tight_layout()
        arrival_time = plt.ginput(1, timeout=0)[0][0]

        arrival_times.append(ArrivalTimePick(
            trace_index=i,
            station_name=trace.stats.station,
            arrival_time=arrival_time,
            distance_to_tonga=trace.stats.distance
        ))

        # Plot the trace and get user input for the arrival time
        plt.close()

    # Save the arrival times in a pickle file
    with open(CACHE_DIR / pathlib.Path(f"arrival_times_start_{start_index}_stop_{i}.pkl"), "wb") as f:
        pickle.dump(arrival_times, f)

    return arrival_times


def load_arrival_times(fileName: pathlib.Path) -> list[ArrivalTimePick]:
    """Load the arrival times from a pickle file."""
    import pickle
    with open(CACHE_DIR / fileName, "rb") as f:
        arrival_times = pickle.load(f)
        print(f"Loaded {len(arrival_times)} arrival time picks from {fileName}")
    return arrival_times


# Filter coefficients for three different FIR filters
h1 = [9.3102e-04,
  -1.2991e-18,
  -1.1771e-03,
  -8.9350e-04,
   1.1279e-03,
   2.3259e-03,
  -3.0497e-18,
  -3.7419e-03,
  -2.8954e-03,
   3.5886e-03,
   7.1273e-03,
  -6.7002e-18,
  -1.0473e-02,
  -7.7679e-03,
   9.2793e-03,
   1.7882e-02,
  -1.0958e-17,
  -2.5342e-02,
  -1.8731e-02,
   2.2575e-02,
   4.4596e-02,
  -1.4316e-17,
  -7.1659e-02,
  -6.0472e-02,
   9.2253e-02,
   3.0157e-01,
   3.9980e-01,
   3.0157e-01,
   9.2253e-02,
  -6.0472e-02,
  -7.1659e-02,
  -1.4316e-17,
   4.4596e-02,
   2.2575e-02,
  -1.8731e-02,
  -2.5342e-02,
  -1.0958e-17,
   1.7882e-02,
   9.2793e-03,
  -7.7679e-03,
  -1.0473e-02,
  -6.7002e-18,
   7.1273e-03,
   3.5886e-03,
  -2.8954e-03,
  -3.7419e-03,
  -3.0497e-18,
   2.3259e-03,
   1.1279e-03,
  -8.9350e-04,
  -1.1771e-03,
  -1.2991e-18,
   9.3102e-04]

h2 = [6.8867e-04,
  -1.0409e-18,
  -8.7071e-04,
  -1.6144e-04,
   2.4454e-03,
   4.3979e-03,
   2.9653e-03,
   1.8510e-04,
   1.9464e-03,
   9.1274e-03,
   1.2922e-02,
   5.3683e-03,
  -6.4293e-03,
  -6.1213e-03,
   7.3124e-03,
   1.0978e-02,
  -1.3170e-02,
  -4.5946e-02,
  -4.7642e-02,
  -1.5176e-02,
  -2.2060e-03,
  -5.5677e-02,
  -1.3549e-01,
  -1.3111e-01,
   1.6668e-02,
   2.2307e-01,
   3.2035e-01,
   2.2307e-01,
   1.6668e-02,
  -1.3111e-01,
  -1.3549e-01,
  -5.5677e-02,
  -2.2060e-03,
  -1.5176e-02,
  -4.7642e-02,
  -4.5946e-02,
  -1.3170e-02,
   1.0978e-02,
   7.3124e-03,
  -6.1213e-03,
  -6.4293e-03,
   5.3683e-03,
   1.2922e-02,
   9.1274e-03,
   1.9464e-03,
   1.8510e-04,
   2.9653e-03,
   4.3979e-03,
   2.4454e-03,
  -1.6144e-04,
  -8.7071e-04,
  -1.0409e-18,
   6.8867e-04]

h3 = [-2.4366e-04,
  -2.6135e-19,
   3.0807e-04,
   7.3294e-04,
   1.3147e-03,
   2.0667e-03,
   2.9630e-03,
   3.9300e-03,
   4.8428e-03,
   5.5289e-03,
   5.7792e-03,
   5.3643e-03,
   4.0571e-03,
   1.6578e-03,
  -1.9803e-03,
  -6.9275e-03,
  -1.3160e-02,
  -2.0549e-02,
  -2.8859e-02,
  -3.7759e-02,
  -4.6838e-02,
  -5.5636e-02,
  -6.3672e-02,
  -7.0487e-02,
  -7.5676e-02,
  -7.8923e-02,
   9.2033e-01,
  -7.8923e-02,
  -7.5676e-02,
  -7.0487e-02,
  -6.3672e-02,
  -5.5636e-02,
  -4.6838e-02,
  -3.7759e-02,
  -2.8859e-02,
  -2.0549e-02,
  -1.3160e-02,
  -6.9275e-03,
  -1.9803e-03,
   1.6578e-03,
   4.0571e-03,
   5.3643e-03,
   5.7792e-03,
   5.5289e-03,
   4.8428e-03,
   3.9300e-03,
   2.9630e-03,
   2.0667e-03,
   1.3147e-03,
   7.3294e-04,
   3.0807e-04,
  -2.6135e-19,
  -2.4366e-04]

# Ive been using this main block to test different parts of the code during development
if __name__ == "__main__":
    # Some test code to visualize the filters and their frequency responses
    #H1_dft, freq_H1 = dft(h1, N=2048)
    #H2_dft, freq_H2 = dft(h2, N=2048)
    #H3_dft, freq_H3 = dft(h3, N=2048)
    #
    #fig, ax = plt.subplots()
    #ax.plot(freq_H1, np.abs(H1_dft), label='h1 DFT')
    #ax.plot(freq_H2, np.abs(H2_dft), label='h2 DFT')
    #ax.plot(freq_H3, np.abs(H3_dft), label='h3 DFT')
    #ax.set_xlabel('Normalized frequency [Fs]')
    #ax.set_ylabel('Magnitude')
    #ax.set_title('FIR filter frequency responses using DFT')
    #ax.legend()
    #plt.show()

    # Task 3a was done here, but since Jupyter is non-interactive Ive moved the code here
    stream = parse_h5_to_obspy_station_data(num_files=1)
    arrival_time_test = pick_arrival_time(stream)
    exit()
    

    exit()


    station_data = load_filtered_data()
    fig, ax = plt.subplots(figsize=(10, 6))

    for station in station_data:
        # Plotting data in a section plot
        time = station.get_time_vector()
        pos = station.distance_to_tonga / 1000  # Convert to km
        # Offset each signal by its position
        ax.plot(time, station.data_filtered_h3 + pos, label=station.station_name)
        # Mark the P-wave arrival time
    
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Distance to Tonga [km]')
    ax.set_title('Filtered signals from all stations using h2 filter')
    plt.show()