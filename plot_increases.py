from slamread import *
from plotutils import *

data_rgbd_multi = load_data_from_directory("C:/dev/analysis_slambench/results/new/results/rgbd_multi/living_room_traj0_loop", "Multi RGBD")
#data_rgbd_single = load_data_from_directory("C:/dev/analysis_slambench/results/new/results/rgbd_single/living_room_traj0_loop", "Single RGBD")
data_mono_multi = load_data_from_directory("C:/dev/analysis_slambench/results/new/results/mono_multi/living_room_traj0_loop", "Multi Mono")
#data_mono_single = load_data_from_directory("C:/dev/analysis_slambench/results/new/results/mono_single/living_room_traj0_loop", "Single Mono")


data = data_mono_multi

run_id = 0
call_id = 3
depth = 5
percentile = 95
savedir = "Function_increase_images"
default_size = plt.rcParams['xtick.labelsize']

for data, call_id in zip([data_rgbd_multi, data_mono_multi], [2, 3]):
    for depth in range(3, 7):
        for relative in [False, True]:
            plt.rcParams['xtick.labelsize'] = default_size
            plot_function_relative(data, call_id, run_id, depth, percentile, relative=relative, savedir=savedir)

            plt.rcParams['xtick.labelsize'] = 6
            plot_function_increase(data, call_id, run_id, depth, percentile, relative=relative, savedir=savedir)

plt.show()