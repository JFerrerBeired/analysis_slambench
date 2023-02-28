from math import inf
import os
import string
import time
import numpy as np
from collections import defaultdict

from utils import *
from callgraph import *

slambench_name_dict = "slambench"
slamtimer_name_dict = "slamtimer"
callgraph_name_dict = "callgraphs"
local_map_name_dict = "local_map_idle"

rolling_widths = [5, 10]
rolling_metrics = ["Duration_Frame", ]


def load_data_from_directory(dirname, custom_name = None, showtime=True):
    """
    Load benchmark data from a directory that contains several runs of the same dataset
    
    Assume all the dictionary keys match
    """
    start_time = time.time()
    
    filelist = []
    res = {"Runs":[]}
    
    try :
        filelist += [os.path.join(dirname, f) for f in os.listdir(dirname) if f[-4:] == ".log" and os.path.isfile(os.path.join(dirname, f))]
    except OSError :
        printerr("Working directory %s not found.\n" % dirname )
        return None
    
    for filename in filelist:
        data = load_data_from_file(filename, False)
        
        res["Runs"] += [data]
        res["Name"] = custom_name if custom_name is not None else "No name"
        #TODO: Cambiar y definir en el propio fichero
        res["slambench_time_units"] = "s"
        res["slamtimer_time_units"] = "ms"
    
    #Compute derived metrics
    res["mean"] = {slambench_name_dict:{}}
    res["std"] = {slambench_name_dict:{}}
    for rolling_width in rolling_widths:
        res["mean%d" % rolling_width] = {slambench_name_dict:{}}
        res["std%d" % rolling_width] = {slambench_name_dict:{}}
        
    
    for metric in data[slambench_name_dict].keys(): #Take metrics from the last file computed
        values = []
        for run in res["Runs"]:
            values += [run[slambench_name_dict][metric]]
        
        res["mean"][slambench_name_dict][metric] = np.mean(values,0)
        res["std"][slambench_name_dict][metric] = np.std(values,0)
        
        #Rolling averages
        if metric in rolling_metrics:
            for rolling_width in rolling_widths:
                rolling_values = []
                for value in values:
                    rolling_values += [rolling_average(value, rolling_width)]
                res["mean%d" % rolling_width][slambench_name_dict][metric] = np.mean(rolling_values,0)
                res["std%d" % rolling_width][slambench_name_dict][metric] = np.std(rolling_values,0)
                    
    
    if showtime:
        printinfo("Files inside %s checked in %.3f ms.\n" % (dirname, (time.time()-start_time)*1000))
    
    return res


def load_data_from_file(filename, showtime=True):
    """
    Load benchmark data from a file that contains several runs of the same dataset
    """

    start_time = time.time()
    
    with open(filename) as f:
        lines = f.read().splitlines()
        f.close()
    
    data = {"filename":filename, slambench_name_dict:{}, slamtimer_name_dict:[{}], callgraph_name_dict:[], local_map_name_dict:[{}]}

    frame = 1
    headers = None
    stats_section = False
    
    #For dynamic callgraph building
    callgraph = CallGraph()
    last_depth = inf
    callgraph_dict = defaultdict(list)
        
    for i in range(len(lines)):
        line = lines[i]
        
        if line == "" or line.find('=') == 0: #Empty line
            continue
        
        if stats_section:
            if headers:
                if line[0].isdigit(): #slambench data (NEW FRAME!)
                    fields = line.split("\t")
                    
                    for h, header in enumerate(headers):
                        try :
                            current_value = float(fields[h])
                        except ValueError:
                            current_value = float("NaN")
                        data[slambench_name_dict][header] += [current_value]
                        
                    data[slamtimer_name_dict] += [{}]
                    data[local_map_name_dict] += [{}]
                    
                    callgraph_found = False
                    for reference_callgraph, frames in data[callgraph_name_dict]:
                        if reference_callgraph == callgraph:
                            frames.append(frame)
                            callgraph_found = True
                            break
                    if not callgraph_found:
                        #printinfo("New callgraph found at frame %d\n" % frame)
                        data[callgraph_name_dict].append([callgraph, [frame]])
                    
                    frame += 1
                    callgraph = CallGraph()
                    last_depth = inf
                    callgraph_dict.clear()
                        
                elif line[0].isspace(): #slamtimer data
                    depth = line.count('\t')
                    info = line[depth:].split(" ")
                    function_name = info[0]
                    
                    #DATA LOGGING
                    try :
                        function_time = float(info[1])
                    except ValueError:
                        function_time = float("NaN")
                    
                    if not function_name in data[slamtimer_name_dict][-1].keys():
                        data[slamtimer_name_dict][-1][function_name] = [function_time]
                        data[local_map_name_dict][-1][function_name] = [info[2]]
                    else:
                        data[slamtimer_name_dict][-1][function_name].append(function_time)
                        data[local_map_name_dict][-1][function_name].append(info[2])
                                        
                    #CALLGRAPH BUILDING
                    callgraph_dict[depth].append(function_name)
                    if last_depth > depth: #This is a parent
                        for orphan in callgraph_dict[last_depth]: #Set this function as a parent to all the orphan children
                            callgraph.add_parent(orphan, function_name)
                        callgraph_dict[last_depth].clear()
                    
                    last_depth = depth
                    
                    callgraph.add_function(function_name)
                elif "created with" == line[8:20] or line == "End of program.":
                    pass
                elif any(w in line for w in ["START", "FRAME", "END", "REAL_END"]): #Identifiers for tracking frames for perf_script analysis
                    pass
                else:
                    printwarning("%s: Stat line %d not expected: %s\n" % (filename, i, line))
            else: #Define headers and create arrays
                headers = line.split("\t")

                for header in headers:
                    data[slambench_name_dict][header] = []
                
        else: #Look for stats header
            if line.find("Statistics") == 0: #Found it on the first character
                stats_section = True
    
    #Remove last (empty) element of slamtimer
    data[slamtimer_name_dict].pop()
    data[local_map_name_dict].pop()
    
    #Convert lists to numpy arrays
    for key, value in data[slambench_name_dict].items():
        data[slambench_name_dict][key] = np.array(value)
    
    if showtime:
        printinfo("File \"%s\" processed in %.3f ms.\n" % (filename, (time.time()-start_time)*1000))
    return data


def rolling_average(x, w):
    """ Returns the rolling average of data x with width w. """
    return np.convolve(x, np.ones(w), 'same') / w


def get_slamtimer_functions(data, showtime=True):
    """
    The first frame functions are taken as a reference. Then check if all the other frames agree with this.
    If the number of functions are the same and all of them are included in the original, it is a correct
    match.
    
    TODO: also check callgraph?
    """
    start_time = time.time()
    
    error = False
    data_frame = data[slamtimer_name_dict][0]
    functions = list(data_frame.keys())
    
    for i, data_frame in enumerate(data[slamtimer_name_dict][1:]):
        frame_functions = data_frame.keys()
        if len(functions) != len(frame_functions):
            error = True
            break
        
        for frame_function in frame_functions:
            if not frame_function in functions:
                error = True
                break
        
        if error: 
            break
        
    if error:
        printerr("Different functions at frame %d\n\tOriginal functions: %s\n\tFrame functions: %s\n" 
                    % (i+1, functions, list(frame_functions)))
        return None
    
    if showtime:
        printinfo("Functions of %s checked in %.3f ms.\n" % (data["name"], (time.time()-start_time)*1000))
    
    return functions
            
        
def get_time_spent_per_frame(data, function_name):
    """
    Get the time spent per frame of the fiven function
    """
    res = [] #Could preallocate numpy array, but..... zzzz
    for data_frame in data[slamtimer_name_dict]:
        try:
            res += [data_frame[function_name]]
        except:
            printerr("Function name %s not valid.\n" % function_name)
            return None
    
    return res


def get_total_time_spent(data, functions):
    """
    Get the total time spent per function for full execution
    """
    res = {}
    for function_name in functions:
        res[function_name] = 0
    
    for data_frame in data[slamtimer_name_dict]:
        for function_name in functions:
            try:
                for function_call_time in data_frame[function_name]: #Function can be called several times in a frame
                    res[function_name] += function_call_time
            except:
                printerr("Function name %s not valid.\n" % function_name)
                return None
    
    return res


def frames_to_interval_string(frames):
    """ Given a secuence of frames, produces a string with interval format. """
    def interval_to_string(ini, end):
        if ini == end:
            return "%d, " % ini
        else:
            return "%d-%d, " % (ini, end)
    
    str = ""
    if len(frames):
        ini = frames[0] #Set to first frame
        last_frame = frames[0]
        end = frames[0]
        
        if len(frames) == 1:
            str += interval_to_string(ini, end)
        else:
            for frame in frames[1:]:
                if frame == last_frame + 1: #Extend interval
                    end = frame
                else: #End interval
                    str += interval_to_string(ini, end)
                    ini = frame
                    end = frame
                
                last_frame = frame
            
            str += interval_to_string(ini, end)
    
    return str[:-2]
 

def print_callgraphs_info(datas, out_dir=None):
    """ Print info (callgraph tree and frames interval) of the data provided. """
    
    if out_dir is not None:
        f = open(out_dir, 'w')
    print_dest = print if out_dir is None else lambda text: f.write(text+'\n')
    
    if not isinstance(datas, list):
        datas = [datas]
    
    for data in datas:
        name = data["Name"]
        print_dest("-"*10 + name + "-"*10)
        callgraphs = []
        callgraphs_frames = []
        n_runs = len(data["Runs"])
        for id_run, run in enumerate(data["Runs"]):
            for (callgraph, frames) in run[callgraph_name_dict]:
                found = False
                for idx, callgraph_compare in enumerate(callgraphs):
                    if callgraph_compare == callgraph:
                        found = True
                        callgraphs_frames[idx][id_run] = frames
                        break
                if not found:
                    callgraphs.append(callgraph)
                    callgraphs_frames.append([[] for i in range(n_runs)]) #One array of frames for each run for each callgraph
                    callgraphs_frames[-1][id_run] = frames
        
        print_dest("%d different callgraph found: " % len(callgraphs))
        
        for i, frames in enumerate(callgraphs_frames):
            print_dest("*"*5 + "Callgraph %d:" % i)
            print_dest(str(callgraphs[i]))
            print_dest("Runs:")
            for run_frames in frames:
                print_dest("\tFound in %d frames:\t%s" % (len(run_frames), frames_to_interval_string(run_frames)))
        
    if out_dir is not None:
        f.close() 


def get_times_callgraph(data, id_callgraph=None, id_run=None, callgraph=None):
    if callgraph == None: #If not defined, search by id
        callgraph_compare = data["Runs"][id_run][callgraph_name_dict][id_callgraph][0]
    else:
        callgraph_compare = callgraph
        
    res = defaultdict(list)
        
    for run in data["Runs"]:
        #Find frames in which the callgraph appears in the current frame
        callgraph_frames = []
        for (callgraph, frames) in run[callgraph_name_dict]:
            if callgraph_compare == callgraph:
                callgraph_frames = frames
                break
        
        for function in callgraph_compare.get_functions():
            function_res = []
            for frame, frame_times in enumerate(run[slamtimer_name_dict]):
                    function_res.append(np.sum(frame_times[function]) if frame+1 in callgraph_frames else np.nan)

            res[function].append(function_res)
    
    return res, callgraph_compare        


def get_frames_local_map_active(data, function_reference):
    """ Return a list with the frames that the local_map was active on the when the function provided started. """
    frames_active = []
    
    for frame, local_map_idle in zip(data[slambench_name_dict]["Frame Number"], data[local_map_name_dict]):
        #TODO: Change the reading so it reads booleans instead of strings
        if '0' in local_map_idle[function_reference]: #There is some instance of the function that was not idle
            frames_active.append(int(frame))
    
    return frames_active


def get_frames_peak_time(data, percentile):
    """ Return a list with the frames where the root time was over the percentile given. """
    #TODO: Very similar (again to the percentiles of plotutils.py) -> REFACTOR Candidate. Although this ises slambench instead of callgraphs
    
    base_times = data[slambench_name_dict]["Duration_Frame"]
    mask = base_times > np.percentile(base_times, percentile)
    
    return np.array(data[slambench_name_dict]["Frame Number"][mask], dtype=int)



if __name__ == "__main__":
    """data_rgbd_multi = load_data_from_directory("C:/dev/analysis_slambench/results/rbgd_multi/living_room_traj0_loop")
    data_rgbd_single = load_data_from_directory("C:/dev/analysis_slambench/results/rbgd_single/living_room_traj0_loop")
    data_mono_multi = load_data_from_directory("C:/dev/analysis_slambench/results/mono_multi/living_room_traj0_loop")
    data_mono_single = load_data_from_directory("C:/dev/analysis_slambench/results/mono_single/living_room_traj0_loop")"""
    data_local_map_idle = load_data_from_directory("C:/dev/analysis_slambench/results/test_local_map_idle")
    
    print_callgraphs_info(data_local_map_idle)
    a = get_frames_local_map_active(data_local_map_idle["Runs"][0], "TrackRGBD")
    b = get_frames_peak_time(data_local_map_idle["Runs"][0], 95)
    
    counter = 0
    for peak_frame in b:
        if peak_frame in a:
            counter += 1
    
    print("Local Mapping was active in %d of %d peak frames.\nFrames with Local Mapping active: %d" % (counter, len(b), len(a)))


3;