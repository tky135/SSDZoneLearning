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



def read_trace(file_name):

    # Define column names based on the README
    column_names = [
        "Device Major Number", "Device Minor Number", "CPU Core ID", "Record ID", 
        "Timestamp (in nanoseconds)", "ProcessID", "Trace Action", "OperationType", 
        "SectorNumber + I/O Size", "ProcessName"
    ]

    processed_data =  []
    with open(file_name, "r") as file:
        for line in file:
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

    return df
def clean_df(df):
    # only keep Trace Action == 'Q'
    df = df.loc[df['Trace Action'] == 'Q']

    # (temp) select 
    df = df[['Timestamp (in nanoseconds)', 'I/O Size']]
    return df.reset_index(drop=True)

def generate_bandwidth(df, window_size=0.0005, plot_bw=True):
    """
    A window-based method to generate a bandwidth column based on nearby I/O activities 
    """
    # Initialize an empty list to store bandwidth values
    bandwidth_values = []

    # For each timestamp, compute the bandwidth
    for index, row in df.iterrows():
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

if __name__ == "__main__":
    df_sample = read_trace("ssdtrace-00")
    df_sample = clean_df(df_sample)

    df_sample.to_csv("preprocessed.csv")
    df_sample = generate_bandwidth(df_sample, window_size=0.1)
    print(df_sample.head())
