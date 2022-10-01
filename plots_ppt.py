from slamread import *
from plotutils import *

import os

savedir = "PPT_images"
init_mono_id = 2
work_mono_id = 3
work_rgbd_id = 2

data_rgbd_multi = load_data_from_directory("C:/dev/analysis_slambench/results/new/results/rgbd_multi/living_room_traj0_loop", "Multi RGBD")
data_rgbd_single = load_data_from_directory("C:/dev/analysis_slambench/results/new/results/rgbd_single/living_room_traj0_loop", "Single RGBD")
data_mono_multi = load_data_from_directory("C:/dev/analysis_slambench/results/new/results/mono_multi/living_room_traj0_loop", "Multi Mono")
data_mono_single = load_data_from_directory("C:/dev/analysis_slambench/results/new/results/mono_single/living_room_traj0_loop", "Single Mono")

all_data = [data_rgbd_multi, data_rgbd_single, data_mono_multi, data_mono_single]

plot_fps(all_data, savedir=savedir)

for data in all_data:
    print_callgraphs_info(data, os.path.join(savedir, data["Name"] + ".txt"))
    plot_individual_fps(data, savedir=savedir)
    
    for depth in [3, 4, 5]:
        for stacked in [True, False]:
            #Plot the init and work phase taking by hand the id of the relevant callgraph in the first run
            if "Mono" in data["Name"]: 
                plot_function_time(data, init_mono_id, 0, depth, serial = stacked, savedir=savedir)
                plot_function_time(data, work_mono_id, 0, depth, serial = stacked, savedir=savedir)
            else: #RGBD
                plot_function_time(data, work_rgbd_id, 0, depth, serial = stacked, savedir=savedir)
            plot_function_time2(data, depth, min_frames=50, serial = stacked, savedir=savedir)
            
        plot_histogram_time(data, depth, 500, savedir=savedir)
    plot_histogram_time(data, 0, savedir=savedir)
