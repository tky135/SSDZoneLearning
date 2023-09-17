# def read_trace(file_name):
#     trace_dict = {}
#     for att_name in ["major_num", "minor_num", "cpu_id", "record_id", "timestamp", "proc_id", "trace_action", "operation_type", "sector_num", "io_size", "proc_name"]:
#         trace_dict[att_name] = []
#     f = open(file_name)
#     for line in f.readlines():
#         line = line.split()
#         major_num, minor_num = line[0].split(',')
#         major_num, minor_num = int(major_num), int(minor_num)

#         cpu_id = int(line[1])
#         record_id = int(line[2])

#         print(line)
#         raise Exception("break")
        
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import bisect
import pickle


def read_trace(file_name):

    # Define column names based on the README
    column_names = [
        "Device Major Number", "Device Minor Number", "CPU Core ID", "Record ID", 
        "Timestamp (in nanoseconds)", "ProcessID", "Trace Action", "OperationType", 
        "SectorNumber + I/O Size", "ProcessName"
    ]

    processed_data =  []
    with open(file_name, "r") as file:
        file = file.readlines()
        for line in tqdm(file, desc="Reading file"):
            line = line.split()
            if len(line) == 11:
                major_minor = line[0].split(',')
                sector_io = line[7] + ' ' + line[8] + ' ' + line[9]
                processed_data.append([
                    major_minor[0], major_minor[1], *line[1:7], sector_io, line[10]
                ])
            elif len(line) == 9:
                major_minor = line[0].split(',')
                processed_data.append([
                    major_minor[0], major_minor[1], *line[1:7], None, line[7]
                ])

    # Converting the processed data into a DataFrame
    df = pd.DataFrame(processed_data, columns=column_names)

    # Splitting the "SectorNumber + I/O Size" column
    df[['SectorNumber', 'Operation', 'I/O Size']] = df['SectorNumber + I/O Size'].str.split(' ', expand=True)
    df.drop(columns=['SectorNumber + I/O Size', 'Operation'], inplace=True)

    # Filling None values and converting columns to appropriate data types
    int_columns = ["Device Major Number", "Device Minor Number", "CPU Core ID", "Record ID", "ProcessID", "SectorNumber", "I/O Size"]
    float_columns = ["Timestamp (in nanoseconds)"]

    for col in int_columns:
        df[col].fillna(-1, inplace=True)
        df[col] = df[col].astype(int)

    for col in float_columns:
        df[col] = df[col].astype(float)

    # only keep Trace Action == 'Q'
    df = df.loc[df['Trace Action'] == 'Q']

    # (temp) select 
    df = df[['Timestamp (in nanoseconds)', 'I/O Size']]
    return df.reset_index(drop=True)
def bw_entry_aligned(df, window_size=0.0005, plot_bw=True):
    """
    A window-based method to generate a bandwidth column based on nearby I/O activities 
    """
    # Initialize an empty list to store bandwidth values
    bandwidth_values = []

    # For each timestamp, compute the bandwidth
    for index, row in tqdm(df.iterrows()):
        start_time = row['Timestamp (in nanoseconds)'] - window_size / 2
        end_time = row['Timestamp (in nanoseconds)'] + window_size / 2
        
        # Filter data within the window
        window_data = df[(df['Timestamp (in nanoseconds)'] >= start_time) & 
                        (df['Timestamp (in nanoseconds)'] <= end_time)]
        
        # Calculate bandwidth
        total_io_size = window_data['I/O Size'].sum()
        bandwidth = total_io_size / window_size
        bandwidth_values.append(bandwidth)

    # Add the computed bandwidth values to the dataframe
    df['Bandwidth'] = bandwidth_values

    # plot the bandwidth
    if plot_bw:
        plt.figure(figsize=(12,7))
        plt.plot(df['Timestamp (in nanoseconds)'], df['Bandwidth'])
        plt.xlabel('Timestamp (in nanoseconds)')
        plt.ylabel('Bandwidth')
        plt.title('Bandwidth with respect to Timestamp')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig("bandwidth.png")
    return df
def chunck_bw_entry_aligned(df_chunk, df, window_size):
    """
    This function calculates bandwidth for a chunk of data.
    """
    bandwidth_values = []
    for _, row in tqdm(df_chunk.iterrows(), total=len(df_chunk)):
        start_time = row['Timestamp (in nanoseconds)'] - window_size / 2
        end_time = row['Timestamp (in nanoseconds)'] + window_size / 2
        
        # Filter data within the window
        # window_data = df[(df['Timestamp (in nanoseconds)'] >= start_time) & 
        #                 (df['Timestamp (in nanoseconds)'] <= end_time)]

        timestamps = df['Timestamp (in nanoseconds)'].values
    
        # Find the start and end indices using binary search
        start_idx = bisect.bisect_left(timestamps, start_time)
        end_idx = bisect.bisect_right(timestamps, end_time)

        window_data = df.iloc[start_idx:end_idx]
        
        # Calculate bandwidth
        total_io_size = window_data['I/O Size'].sum()
        bandwidth = total_io_size / window_size
        bandwidth_values.append(bandwidth)
    
    return bandwidth_values

def bw_entry_aligned_para(df, window_size=0.0005, plot_bw=True):
    """
    A parallelized version of generate_bandwidth.
    """
    # Number of processes to run in parallel
    num_processes = cpu_count()
    
    # Split dataframe into chunks for each process
    chunk_size = len(df) // num_processes
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    # Use a Pool to process each chunk in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(chunck_bw_entry_aligned, [(chunk, df, window_size) for chunk in chunks])
    
    # Flatten results and assign bandwidth values to the dataframe
    bandwidth_values = [val for sublist in results for val in sublist]
    df['Bandwidth'] = bandwidth_values
    
    # Plot the bandwidth if required
    if plot_bw:
        plt.figure(figsize=(12,7))
        plt.plot(df['Timestamp (in nanoseconds)'], df['Bandwidth'])
        plt.xlabel('Timestamp (in nanoseconds)')
        plt.ylabel('Bandwidth')
        plt.title('Bandwidth with respect to Timestamp')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig("bandwidth_parallel%f.png" % window_size)
    
    return df

def bw_time_aligned(data, window_size, step_size):
    """Generate bandwidth based on a window-based approach."""
    bandwidths = []
    window_start = data['Timestamp (in nanoseconds)'].iloc[0]
    window_end = window_start + window_size
    while window_end <= data['Timestamp (in nanoseconds)'].iloc[-1]:
        window_data = data[(data['Timestamp (in nanoseconds)'] >= window_start) & 
                           (data['Timestamp (in nanoseconds)'] < window_end)]
        bandwidth = window_data['I/O Size'].sum() / window_size
        bandwidths.append(bandwidth)
        window_start += step_size
        window_end = window_start + window_size
    return bandwidths

def chunk_bw_time_aligned(args):
    """Adjusted function to fix the issue with window alignment."""
    import bisect
    data, window_size, step_size, chunk_start, chunk_end = args
    bandwidths = []
    aligned_start = data['Timestamp (in nanoseconds)'].iloc[0]
    window_start = aligned_start + np.ceil((chunk_start - aligned_start) / step_size) * step_size

    window_end = window_start + window_size
    timestamps = data['Timestamp (in nanoseconds)'].values
    while window_start < chunk_end and window_end <= data['Timestamp (in nanoseconds)'].iloc[-1]:
        start_index = bisect.bisect_left(timestamps, window_start)
        end_index = bisect.bisect_left(timestamps, window_end)
        window_data = data.iloc[start_index:end_index]
        bandwidth = window_data['I/O Size'].sum() / window_size
        bandwidths.append(bandwidth)
        window_start += step_size
        window_end = window_start + window_size
    # print(len(bandwidths))
    return bandwidths

def bw_time_aligned_para(data, window_size, step_size):
    """Adjusted parallel function using the fixed chunk processing function."""
    num_cpus = cpu_count()
    total_time = data['Timestamp (in nanoseconds)'].iloc[-1] - data['Timestamp (in nanoseconds)'].iloc[0]
    chunk_size = total_time / num_cpus
    chunks = [(data['Timestamp (in nanoseconds)'].iloc[0] + i*chunk_size, 
               data['Timestamp (in nanoseconds)'].iloc[0] + (i+1)*chunk_size) for i in range(num_cpus)]
    pool = Pool(num_cpus)
    results = pool.map(chunk_bw_time_aligned, [(data, window_size, step_size, start, end) for start, end in chunks])
    pool.close()
    bandwidths = [item for sublist in results for item in sublist]
    return bandwidths

def verify_bw_time_aligned_para():
    """
    helper function to verify the correctness of the parallel bandwidth computation
    """
    data = pd.read_csv('preprocessed.csv')
    window_size = 0.1
    step_size = 0.005
    sequential_bandwidths = bw_time_aligned(data, window_size, step_size)
    parallel_bandwidths = bw_time_aligned_para(data, window_size, step_size)
    plt.figure(figsize=(15, 6))
    plt.plot(sequential_bandwidths, label='Sequential Bandwidth', alpha=0.7)
    plt.plot(parallel_bandwidths, label='Parallel Bandwidth', alpha=0.7, linestyle='--')
    plt.title('Comparison of Sequential and Parallel Bandwidth Computations')
    plt.xlabel('Window Index')
    plt.ylabel('Bandwidth')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_bw_list():
    concat_data = []
    for i in range(27):
        file = 'bw_l_%02d.pkl' % i
        if os.path.exists(file):
            with open(file, 'rb') as f:
                data = pickle.load(f)
                concat_data += data
    plt.figure(figsize=(15, 6))
    plt.plot(concat_data, markersize=4, linewidth=2, color='purple')
    plt.title("Bandwidth Data")
    plt.xlabel("Data Points")
    plt.ylabel("Bandwidth")
    plt.grid(True)
    plt.savefig("Bandwith.png")

if __name__ == "__main__":
    PATH_PREFIX = '/mnt/nvme1n1/kt19'
    WINDOW_SIZE = 100.   # in seconds
    STEP_SIZE = 25.
    NUM = 11
    # df_sample = read_trace(os.path.join(PATH_PREFIX, "ssdtrace-00"))
    # df_sample.to_csv("preprocessed.csv")
    # def process_trace(i):
    #     print("processing %d" % i)
    #     df = read_trace(os.path.join(PATH_PREFIX, "ssdtrace-%02d" % i))
    #     df.to_csv("preprocessed-%02d.csv" % i)
    # with Pool(5) as pool:
    #     pool.map(process_trace, range(5))

    for i in range(20,27):
        print("processing %d" % i)
        df = read_trace(os.path.join(PATH_PREFIX, "ssdtrace-%02d" % i))
        df.to_csv("preprocessed-%02d.csv" % i)
    raise Exception("done")

    # read df_sample from csv
    df = pd.read_csv("preprocessed-%02d.csv" % NUM)
    # bw_df = bw_entry_aligned(df, window_size=WINDOW_SIZE)
    bw_l = bw_time_aligned_para(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
    with open('bw_l_%02d.pkl' % NUM, 'wb') as f:
        pickle.dump(bw_l, f)
    
    # write bw_l to pickle file

    # plot the bandwidth
    plt.figure(figsize=(12,7))
    plt.plot(bw_l)
    plt.xlabel('Chunk Index')
    plt.ylabel('Bandwidth')
    plt.title('Bandwidth with respect to Chunk Index')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("bandwidth_time_aligned.png")
