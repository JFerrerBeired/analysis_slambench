"""THE PURPOSE OF THIS FILE IS TO CHECK THE INFORMATION AMONG THE VARIOS SOURCES (SLAMBENCH, 
SLAMTIMER, PERF_SCRIPT, ETC) AND CHECK THAT IT IS COHERENT."""

import matplotlib.pyplot as plt
from slamread import *
from process_perf_script import *

def time_by_frame(data_slambench, data_frames_perf):
    assert len(data_slambench["Runs"]) == 1 #ONLY ONE EXPERIMENT
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    data_slambench = data_slambench["Runs"][0]
    
    time_frames_slambench = np.array(data_slambench[slambench_name_dict]["Duration_Frame"])*1000
    
    ax.plot(data_slambench[slambench_name_dict]["Frame Number"],
             time_frames_slambench,
             lw=1, label="Slambench")
    
    time_frames_slamtimer = []
    
    for frame in data_slambench[slamtimer_name_dict]:
        time_frames_slamtimer.append(frame["process"][0])
    
    ax.plot(data_slambench[slambench_name_dict]["Frame Number"],
             time_frames_slamtimer,
             lw=1, label="Slamtimer")
    
    time_frames_frame_stamps = (data_frames_perf[1:-1] - data_frames_perf[:-2])/1e6
    
    ax.plot(data_slambench[slambench_name_dict]["Frame Number"],
             time_frames_frame_stamps,
             lw=1, label="FRAME_STAMPS")
    
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time elapsed per frame (ms)")
    
    fig.legend()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    mean_time = np.mean([time_frames_slambench, time_frames_slamtimer, time_frames_frame_stamps], axis=0)
    
    ref_time = time_frames_slambench
    
    ax.plot(data_slambench[slambench_name_dict]["Frame Number"],
             time_frames_slambench - ref_time,
             lw=1, label="Slambench")
    
    ax.plot(data_slambench[slambench_name_dict]["Frame Number"],
             time_frames_slamtimer - ref_time,
             lw=1, label="Slamtimer")
    
    ax.plot(data_slambench[slambench_name_dict]["Frame Number"],
             time_frames_frame_stamps - ref_time,
             lw=1, label="FRAME_STAMPS")
    
    ax.set_xlabel("Frame")
    ax.set_ylabel("Deviation from the mean of time elapsed (ms)")
    
    fig.legend()
    
    #plt.show()
    

def time_by_function(data_slambench, data_perf_script, function_names_slamtimer, function_names_perf_script):
    assert len(data_slambench["Runs"]) == 1 #ONLY ONE EXPERIMENT
    assert len(function_names_slamtimer) == len(function_names_perf_script) #Same number of functions
    
    data_slamtimer = data_slambench["Runs"][0][slamtimer_name_dict]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    total_time_slamtimer = 0
    for frame in data_slamtimer:
        total_time_slamtimer += frame["process"][0]
    
    n_events_perf_script = count_events(data_perf_script, "cycles")
    
    x_axis = np.arange(len(function_names_slamtimer))
    times_perf_script = []
    times_slamtimer = []
    
    for function_name_slamtimer, function_name_perf_script in zip(function_names_slamtimer, function_names_perf_script):
        time_slamtimer = 0
        for frame in data_slamtimer:
            if function_name_slamtimer in frame:
                time_slamtimer += np.sum(frame[function_name_slamtimer])
        
        times_slamtimer.append(time_slamtimer/total_time_slamtimer)        
        times_perf_script.append(data_perf_script["cycles"][function_name_perf_script]["total"]/n_events_perf_script)
    
    
    
    ax.bar(x_axis-0.2, times_slamtimer, 0.3, label = 'Slamtimer')
    ax.bar(x_axis+0.2, times_perf_script, 0.3, label = 'Perf script')
    
    x_axis_ticks = np.concatenate([x_axis-0.2, x_axis+0.2])
    x_axis_labels = function_names_slamtimer + function_names_perf_script
    plt.xticks(x_axis_ticks, x_axis_labels)
    plt.xlabel("Functions")
    plt.ylabel("Relative frequency")
    plt.xticks(rotation=30, horizontalalignment='right')
    plt.legend()