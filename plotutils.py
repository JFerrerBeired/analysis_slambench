#!/usr/bin/python2


# DO not use DISPLAY within matplotlib
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
# End of BugFix

import os
from slamread import *

class GroupedFigure:
    """
    Class for managing group of pyplot figures and setting the same x/y ranges for all of them.
    """
    def __init__(self, group_x, group_y) -> None:
        self.group_x = group_x
        self.group_y = group_y
        self.axes = []
        self.figures = []
        self.savedirs = []
        self.ylims = []
        self.xlims = []
    
    
    def new_plot(self, function, *args, **kwargs):
        """
        Adds a new figure using the provided function and arguments to pass to it.
        """
        savedir = function(*args, **kwargs, save=False) #Always set save to false so it don't get saved twice by mistake (the latter save from the class would overwrite anyway)
        self.figures.append(plt.gcf())
        self.savedirs.append(savedir)
        ax = plt.gca()
        self.axes.append(ax)
        self.ylims.append(ax.get_ylim())
        self.xlims.append(ax.get_xlim())
    
    
    def regroup_axes(self):
        """
        Call when all figures have been added to compute common min/max and apply it.
        """
        ylims = np.asarray(self.ylims)
        xlims = np.asarray(self.xlims)
        
        if self.group_x:
            min_x = np.min(xlims[:,0])
            max_x = np.max(xlims[:,1])
            
            for ax in self.axes:
                ax.set_xlim((min_x, max_x))

        if self.group_y:
            min_y = np.min(ylims[:,0])
            max_y = np.max(ylims[:,1])
            
            for ax in self.axes:
                ax.set_ylim((min_y, max_y))
    
    
    def save_figures(self):
        """
        Saves the figure to a file (directory given by the plotting functions)
        """
        
        for savedir, fig in zip(self.savedirs, self.figures):
            fig.savefig(savedir)
            plt.close(fig)


def get_reformated_simple_name(data):
    """ Gets the name of the data in nonspace capitalized format. """
    
    if "Name" in data:
        return data["Name"].title().replace(" ", "")
    else:
        raise NotImplementedError("Reformat name when no name is provided.")


def plot_fps(datas, savedir=None, save=False):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    if not isinstance(datas, list):
        datas = [datas]
    
    for data in datas:
        ax.errorbar(data["mean"][slambench_name_dict]["Frame Number"],
             data["mean"][slambench_name_dict]["Duration_Frame"],
             data["std"][slambench_name_dict]["Duration_Frame"],
             lw=1, capsize=2, label=data["Name"])
    
    ax.set_title("Mean comparison")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time elapsed per frame (%s)" % data["slambench_time_units"])
    
    fig.legend()
    
    if savedir is not None:
        if len(datas) > 1:
            file_str = "Comparison_"
            for data in datas:
                file_str += get_reformated_simple_name(data) + "_"
        else:
            file_str = "Mean_" + get_reformated_simple_name(datas[0]) #Breaks if len(datas=0) but what are you doing then... ¯\_(ツ)_/¯
        file_str += ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
        
        return file_str


def plot_individual_fps(data, savedir=None, save=False):
    fig, ax = plt.subplots(figsize=(12, 5))
    n_run=1
    for run in data["Runs"]:
        ax.plot(run[slambench_name_dict]["Frame Number"], 
                run[slambench_name_dict]["Duration_Frame"], c="black", lw=1, label="Run %d"%n_run)
        n_run+=1
    
    ax.errorbar(data["mean"][slambench_name_dict]["Frame Number"],
                data["mean"][slambench_name_dict]["Duration_Frame"],
                data["std"][slambench_name_dict]["Duration_Frame"],
                lw=1, capsize=2, label="Mean")
    
    ax.set_title("Individual runs of %s" % data["Name"])
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time elapsed per frame (%s)" % data["slambench_time_units"])
    
    fig.legend()
    
    if savedir is not None:
        file_str = "Individuals_" + get_reformated_simple_name(data) + ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
        
        return file_str


#TODO: Fuse common features of these two functions

def plot_function_time(data, id_callgraph, id_run, depth, serial = True, savedir=None, save=False):
    """ Plot the time spent per frame on each function for a given callgraph. """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    times, callgraph = get_times_callgraph(data, id_callgraph, id_run)
    functions = callgraph.get_functions(depth)
    
    base_time = np.sum([np.nanmean(times[root], axis=0) for root in callgraph.get_roots()], axis=0) #Get total time spent on the roots of the callgraph
    time_in_functions = np.zeros_like(base_time) #Time spent on the measured functions
    
    for function in functions:
        function_time = np.nanmean(times[function], axis=0)
        ax.errorbar(data["mean"][slambench_name_dict]["Frame Number"],
                    function_time if serial else function_time + time_in_functions, 
                    np.nanstd(times[function], axis=0), 
                    capsize=2, label=function)
        
        time_in_functions += function_time
    
    #TODO: Compute std of other through uncertainty propagation
    time_other = base_time - time_in_functions
    ax.plot(data["mean"][slambench_name_dict]["Frame Number"],
            time_other if serial else time_other + time_in_functions,
            label="others")

    #Invert legend to match order of appearence when stack
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1])
    
    ax.set_title("Function breakdown of callgraph %d (run %d) of %s at depth %d (%s)" % (id_callgraph, id_run, 
                                                                                         data["Name"], 
                                                                                         depth,
                                                                                         "stacked times" if not serial else "independent times"))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time elapsed per frame (%s)" % data["slamtimer_time_units"])
    
    if savedir is not None:
        file_str = "Callgraph_%s_d%d_r%d_" % (get_reformated_simple_name(data), depth, id_run) + \
                ("stacked" if not serial else "indep") + ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
        
        return file_str


def plot_function_time2(data, depth, min_frames = 0, serial = True, savedir=None, save=False):
    """ Plot the time spent on each function per frame for all the callgraphs that appear in at least min_frames. """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    #Get all callgraphs
    callgraphs_individual = []
    callgraphs = {}
    id = 0
    for run in data["Runs"]:
        for (callgraph, frames) in run[callgraph_name_dict]:
            if not callgraph in callgraphs_individual:
                callgraphs_individual.append(callgraph)
                if len(frames) > min_frames:
                    callgraphs[id] = callgraph
                id += 1
    
    #Plot data for each callgraph
    for id, callgraph in callgraphs.items():
        times, _ = get_times_callgraph(data, callgraph=callgraph)
        functions = callgraph.get_functions(depth)
        
        base_time = np.sum([np.nanmean(times[root], axis=0) for root in callgraph.get_roots()], axis=0) #Get total time spent on the roots of the callgraph
        time_in_functions = np.zeros_like(base_time) #Time spent on the measured functions
        
        for function in functions:
            function_time = np.nanmean(times[function], axis=0)
            ax.errorbar(data["mean"][slambench_name_dict]["Frame Number"],
                        function_time if serial else function_time + time_in_functions, 
                        np.nanstd(times[function], axis=0), 
                        capsize=2, label=function+str(id))
            time_in_functions += function_time
    
        #TODO: Compute std of other through uncertainty propagation
        time_other = base_time - time_in_functions
        ax.plot(data["mean"][slambench_name_dict]["Frame Number"],
                time_other if serial else time_other + time_in_functions,
                label="others"+str(id))
        
    #Invert legend to match order of appearence when stack
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1])
    
    ax.set_title("Function breakdown of callgraphs (n > %d) of %s at depth %d (%s)" % (min_frames, data["Name"], 
                                                                                       depth,
                                                                                       "stacked times" if not serial else "independent times"))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Time elapsed per frame (%s)" % data["slamtimer_time_units"])
    
    if savedir is not None:
        file_str = "Callgraphs_%s_d%d_n%d_" % (get_reformated_simple_name(data), depth, min_frames) + \
                ("stacked" if not serial else "indep") + ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
        
        return file_str


def plot_histogram_time(data, depth, min_frames = 0, threshold = 1, functions=None, savedir=None, save=False):
    """ Plot the histogram of the time spent on each function for all the callgraphs that appear in at least min_frames. 
        If functions is given, depth is ignored and only those functions are plotted. 
        Ignores measures with a time under the threshold (only relevant functions). """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    functions_arg = functions
    
    #Get all callgraphs
    callgraphs_individual = []
    callgraphs = {}
    id = 0
    for run in data["Runs"]:
        for (callgraph, frames) in run[callgraph_name_dict]:
            if not callgraph in callgraphs_individual:
                callgraphs_individual.append(callgraph)
                if len(frames) > min_frames:
                    callgraphs[id] = callgraph
                id += 1
    
    #NOTE: Does it make sense to use the mean instad to compute the histogram? I will use ALL the measures from all the runs so I don't average the peaks
    
    file_functions_str = []
    for id, callgraph in callgraphs.items():
        times, _ = get_times_callgraph(data, callgraph=callgraph)
        if functions_arg is None:
            functions = callgraph.get_functions(depth)
        else:
            functions = functions_arg
        
        for function in functions:
            function_times = np.array(times[function]).ravel() #TODO: Convert to np.array on the building phase already
            function_times = function_times[function_times > threshold]
            ax.hist(function_times, bins='auto', label=function+str(id), histtype='step')
            
            file_functions_str.append(function+str(id))
    
    fig.legend()
    
    #Unnecesarily complicated title formating depending on custom functions or full callgraph depth 
    title_str = "Custom functions" if functions_arg is not None else "Function"
    title_str += " breakdown (n > %d)" % min_frames
    title_str += " of all callgraphs " if functions_arg is None else ""
    title_str += " of %s " % data["Name"]
    title_str += "at depth %d" % depth if functions_arg is None else ""
    
    ax.set_title(title_str)
    ax.set_xlabel("Time elapsed per frame (%s)" % data["slamtimer_time_units"])
    ax.set_ylabel("Frequency (thr = %.2f %s)" % (threshold, data["slamtimer_time_units"]))
    
    if savedir is not None:
        file_str = "Histogram_%s_t%d" % (get_reformated_simple_name(data), threshold)
        if functions_arg is None:
            file_str += "_d%d_n%d" % (depth, min_frames)
        else:
            for func_str in file_functions_str:
                file_str += "_" + func_str
        file_str += ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
        
        return file_str


def plot_function_relative(data, id_callgraph, id_run, depth, percentile, relative=False, savedir=None, save=False):
    """ Plot the time spent on each function for a given callgraph in stacked bars form. """
    def compute_mask(mask, name):
        print("\n" + "*"*20 + name + "*"*20)
        frame_time = base_times[mask]
        print("TOTAL", end=": ")
        
        print("%.2f ± %.2f" % (np.mean(frame_time), np.std(frame_time)))
        time_in_functions = np.zeros_like(frame_time) #Time spent on the measured functions
    
        for i, function in enumerate(functions):
            function_time = times[function][mask]
            function_time_relative = function_time / frame_time
            print(function, end=": ")
            print("%.2f ± %.2f (%.2f %%)" % (np.mean(function_time), np.std(function_time),
                                             100*np.std(function_time)/np.mean(function_time)), end="\t//\t")
            print("%.2f ± %.2f (%.2f %%)" % (np.mean(function_time_relative), np.std(function_time_relative),
                                             100*np.std(function_time_relative)/np.mean(function_time_relative)))
            
            function_times[i].append(np.mean(function_time) if not relative else np.mean(function_time_relative))
            function_std[i].append(np.std(function_time) if not relative else np.std(function_time_relative))
            
            time_in_functions += function_time
    
        time_other = frame_time - time_in_functions
        time_other_relative = time_other / frame_time
        print("Others", end=": ")
        print("%.2f ± %.2f (%.2f %%)" % (np.mean(time_other), np.std(time_other), 
                                         100*np.std(time_other)/np.mean(time_other)), end="\t//\t")
        print("%.2f ± %.2f (%.2f %%)" % (np.mean(time_other_relative), np.std(time_other_relative), 
                                         100*np.std(time_other_relative)/np.mean(time_other_relative)))
        function_times[-1].append(np.mean(time_other) if not relative else np.mean(time_other_relative))
        function_std[-1].append(np.std(time_other) if not relative else np.std(time_other_relative))
        
        computations_names.append(name)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    times, callgraph = get_times_callgraph(data, id_callgraph, id_run)
    functions = callgraph.get_functions(depth)
    
    #Get total time spent on the roots of the callgraph for each run (Haven't checked if it works for more than one root)
    base_times = np.concatenate(np.sum([times[root] for root in callgraph.get_roots()], axis=0))
    
    #Flatten all runs (I don't care about individual runs)
    #Also reserve arrays for each function + other for plotting afterwards
    function_times = [[]]
    function_std = [[]]
    for function in functions:
        times[function] = np.concatenate(times[function])
        function_times.append([])
        function_std.append([])
    
    computations_names = list()
    
    mask = base_times < np.nanpercentile(base_times, percentile)
    compute_mask(mask, "BOTTOM %d %%" % percentile)
    median_indx = np.argsort(base_times)[len(base_times)//2] #Ignores nans
    mask = np.zeros_like(base_times, dtype=bool)
    mask[median_indx] = True #Selects only the median element
    compute_mask(mask, "MEDIAN")
    mask = np.logical_not(np.isnan(base_times))
    compute_mask(mask, "MEAN")
    mask = base_times > np.nanpercentile(base_times, percentile)
    compute_mask(mask, "TOP %d %%" % (100 - percentile))
    
    legend_array = []
    cum = np.zeros_like(function_times[0])
    for i in range(len(functions)+1):
        legend_array.append(ax.barh(computations_names, function_times[i], left=cum, xerr=function_std[i], capsize=2))
        cum += function_times[i]
    
    fig.legend(legend_array, functions+["Others"])
    
    ax.set_title("%s time spent per function of callgraph %d (run %d) of %s at depth %d" % ("Relative" if relative else "Absolute", 
                                                                                            id_callgraph, id_run, 
                                                                                            data["Name"], 
                                                                                            depth))
    if relative:
        ax.set_xlabel("Fraction of time elapsed")
    else:
        ax.set_xlabel("Time elapsed (%s)" % data["slamtimer_time_units"])
    
    if savedir is not None:
        file_str = "Time_function_callgraph_%s_d%d_r%d_" % (get_reformated_simple_name(data), depth, id_run) + \
                ("relative" if relative else "absolute") + ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
        
        return file_str


def plot_function_increase(data, id_callgraph, id_run, depth, percentile, relative=False, savedir=None, save=False):
    """Plot increase in function time of the percentile samples against the median."""
    def compute_mask(mask):
        frame_time = base_times[mask]
        function_times = []
        time_in_functions = np.zeros_like(frame_time) #Time spent on the measured functions
    
        for function in functions:
            function_time = times[function][mask]
            function_times.append(np.mean(function_time))
            time_in_functions += function_time
    
        time_other = frame_time - time_in_functions
        function_times.append(np.mean(time_other))

        return np.array(function_times)

    fig, ax = plt.subplots(figsize=(12, 5))
    
    times, callgraph = get_times_callgraph(data, id_callgraph, id_run)
    functions = callgraph.get_functions(depth)
    
    #Get total time spent on the roots of the callgraph for each run (Haven't checked if it works for more than one root)
    base_times = np.concatenate(np.sum([times[root] for root in callgraph.get_roots()], axis=0))
    base_times = base_times[~np.isnan(base_times)]
    
    #Flatten all runs (I don't care about individual runs)
    for function in functions:
        times_function = np.concatenate(times[function])
        times[function] = times_function[~np.isnan(times_function)]
    
    median_indx = np.argsort(base_times)[len(base_times)//2]
    mask = np.zeros_like(base_times, dtype=bool)
    mask[median_indx] = True #Selects only the median element
    median_functions = compute_mask(mask)
    
    mask = base_times > np.nanpercentile(base_times, percentile)
    top_functions = compute_mask(mask)
    
    if relative:
        ax.bar(functions+["Others"], 100*(top_functions - median_functions)/median_functions)
        ax.set_ylabel("Increase of time spent (%)")
    else:
        ax.bar(functions+["Others"], top_functions - median_functions)
        ax.set_ylabel("Increase of time spent (%s)" % data["slamtimer_time_units"])
        
    ax.set_title("%s increase of time spent (percentile %d against median) of callgraph %d (run %d) of %s at depth %d" % ("Relative" if relative else "Absolute", 
                                                                                                                        percentile, 
                                                                                                                        id_callgraph, id_run, 
                                                                                                                        data["Name"], 
                                                                                                                        depth))
    
    if savedir is not None:
        file_str = "Time_increase_callgraph_%s_d%d_r%d_" % (get_reformated_simple_name(data), depth, id_run) + \
                ("relative" if relative else "absolute") + ".png"
        file_str = os.path.join(savedir, file_str)
        
        if save:
            fig.savefig(file_str)
            plt.close()
    
        return file_str


def plot_function_time_whiskers(datas, id_callgraphs, id_runs, depth, percentile, savedir=None, save=False):
    """ Plot the time spent on each function for a given callgraph in whiskers form. 
        Datas can be a list to plot several experiments together.""" 
    if not isinstance(datas, list):
        datas = [datas]
        id_runs = [id_runs]
        id_callgraphs = [id_callgraphs]
    
    #Check that the callgraphs are the same for all datas
    #TODO: Update to check functions instead of callgraphs (this can match ergo good plot, even if the callgraphs are different as a whole)
    try:
        callgraph_compare = datas[0]['Runs'][id_runs[0]][callgraph_name_dict][id_callgraphs[0]]
        for i, data in enumerate(datas[1:]):
            if callgraph_compare != data['Runs'][id_runs[i+1]][callgraph_name_dict][id_callgraphs[i+1]]:
                raise
    except:
        printerr("Callgraphs between datas do not match.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for i, (data, id_callgraph, id_run) in enumerate(zip(datas, id_callgraphs, id_runs)):
        times, callgraph = get_times_callgraph(data, id_callgraph, id_run)
        functions = callgraph.get_functions(depth)
        times_functions = []
        
        #Get total time spent on the roots of the callgraph for each run (Haven't checked if it works for more than one root)
        base_times = np.concatenate(np.sum([times[root] for root in callgraph.get_roots()], axis=0))
        base_times = base_times[~np.isnan(base_times)]
        
        #Flatten all runs (I don't care about individual runs)
        for function in functions:
            times_function = np.concatenate(times[function])
            times_functions.append(times_function[~np.isnan(times_function)])
        times_functions.append(base_times - np.sum(times_functions, axis=0))
        functions.append("Others")

        whiskers_positions = np.sort((100-percentile, percentile)) #Sorting so it is still min, max when percentile < 50
        
        box = ax.boxplot(times_functions, whis=whiskers_positions, patch_artist=True)
        
        for patch in box['boxes']:
            patch.set_alpha(0.6)
        ax.set_xticklabels(functions)
        ax.set_ylabel("Time spent (%s)" % data["slamtimer_time_units"])
        ax.set_title("Time spent with whiskers at percentiles %d, %d of callgraph %d (run %d) of %s at depth %d" % (whiskers_positions[0], whiskers_positions[1],
                                                                                                                    id_callgraph, id_run, 
                                                                                                                    data["Name"], 
                                                                                                                    depth))
    

def plot_fraction_frames_local_map_active(data, percentile_parameters, function_reference):
    fig, ax = plt.subplots(figsize=(12, 5))
    min_percentile, max_percentile, step_percentile = percentile_parameters
    
    percentiles = range(min_percentile, max_percentile, step_percentile)
    fractions_mean = []
    fractions_std = []

    
    for percentile in percentiles:
        fraction_percentile = []
        
        for data_run in data["Runs"]:
            a = get_frames_local_map_active(data_run, function_reference)
            b = get_frames_peak_time(data_run, percentile)
            
            counter = 0
            for peak_frame in b:
                if peak_frame in a:
                    counter += 1
            
            fraction_percentile.append(100*counter/len(b))
            
        fractions_mean.append(np.mean(fraction_percentile))
        fractions_std.append(np.std(fraction_percentile))
    
    ax.errorbar(percentiles, fractions_mean, fractions_std, capsize=2)
    ax.set_ylabel("Fraction of frames with local map active (%)")
    ax.set_xlabel("Percentile of time spent per frame")
    ax.set_title("Frequency of parallel local mapping")


def plot_extra_time_percentile(data, percentiles):
    fig, ax = plt.subplots(figsize=(12, 5))

    mean_times_normal = []
    mean_times_peak = []
    std_times_normal = []
    std_times_peak = []
    
    for percentile in percentiles:
        times_peak_run = []
        times_normal_run = []
        for data_run in data["Runs"]:
            b = get_frames_peak_time(data_run, percentile)
            
            time_peak = 0
            time_normal = 0
            
            for frame, time in zip(data_run[slambench_name_dict]['Frame Number'], data_run[slambench_name_dict]['Duration_Frame']):
                if frame in b:
                    time_peak += time
                else:
                    time_normal += time
            
            times_peak_run.append(time_peak)
            times_normal_run.append(time_normal)
    
        mean_times_normal.append(np.mean(times_normal_run))
        std_times_normal.append(np.std(times_normal_run))
        mean_times_peak.append(np.mean(times_peak_run))
        std_times_peak.append(np.std(times_peak_run))

    percentiles_str = [str(x) for x in percentiles]
        
    ax.barh(percentiles_str, mean_times_normal, xerr=std_times_normal, capsize=2)
    ax.barh(percentiles_str, mean_times_peak, left=mean_times_normal, xerr=std_times_peak, capsize=2)
        
    ax.set_ylabel("Peak percentile threshold")
    ax.set_xlabel("Total Execution Time (s)")
    ax.set_title("Weight of peak time frames")
    
    fig.legend(["Time normal", "Time peak"])


def plot_thread_activity(data, tids, event='cycles'):
    """Plots thread activity in a grid visualization. Extracts the data from frame_counts
    of perf_script processed data."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    grid = np.zeros((len(tids), len(data[event]['tid'])))
    
    for i, frame in enumerate(data['cycles']['tid']):
        for j, tid in enumerate(tids):
            grid[j][i] = frame[tid]

            
    ax.imshow(np.log(grid[:,:-2]), aspect='auto', cmap='Reds')
    ax.yaxis.set_ticks(np.arange(1, len(tids))-0.5)
    ax.set_yticklabels([])
    ax.set_yticks(np.arange(len(tids)), minor=True)
    ax.set_yticklabels(tids, minor=True)
    ax.tick_params(axis='y', which="both",length=0)
    ax.set_ylabel("Thread ID")
    #ax.set_xlabel("Frame")
    ax.set_xlabel("Time (100ms blocks)")
    
    ax.grid(which='major', color='black', linestyle='-', linewidth=1, axis='y')
    
    


if __name__ == "__main__":
    #data_rgbd_multi = load_data_from_directory("C:/dev/analysis_slambench/results/rbgd_multi/living_room_traj0_loop", "Multi RGBD")
    #data_rgbd_single = load_data_from_directory("C:/dev/analysis_slambench/results/rbgd_single/living_room_traj0_loop", "Single RGBD")
    #data_mono_multi = load_data_from_directory("C:/dev/analysis_slambench/results/mono_multi/living_room_traj0_loop", "Multi Mono")
    #data_mono_single = load_data_from_directory("C:/dev/analysis_slambench/results/mono_single/living_room_traj0_loop", "Single Mono")

    """data = load_data_from_directory("C:/dev/analysis_slambench/results/mono_multi/living_room_traj0_loop", "test")
    
    plot_function_time_whiskers(data, 2, 0, 3, 95)
    plot_function_time_whiskers([data, data], [3,2], [0,0], 3, 95)
    plt.show()"""
    
   
    data_local_map_idle = load_data_from_directory("C:/dev/analysis_slambench/results/test_local_map_idle")  
    plot_histogram_time(data_local_map_idle, 0)
    plot_fraction_frames_local_map_active(data_local_map_idle, (5, 99, 1), "TrackRGBD")
    plt.show()
    
    """plot_fps(data, savedir="test_out")
    plot_individual_fps(data, savedir="test_out")"""

    #print_callgraphs_info([data_rgbd_single], ["as"])

    """plot_function_time(data, 2, 0, 5, serial = False, savedir="test_out")
    plot_function_time2(data, 4, min_frames=50, serial = True, savedir="test_out")
    plot_histogram_time(data, 4, 500, functions=["ORBextractor"], savedir="test_out")
    plot_histogram_time(data, 4, 500, savedir="test_out")
    plt.show()"""

    """plot_fps([data_rgbd_multi, data_rgbd_single, data_mono_multi, data_mono_single], 
            ["rgbd_multi", "rgbd_single", "mono_multi", "mono_single"])"""

    """data = data_rgbd_multi
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(data["mean"][slambench_name_dict]["Frame Number"],
                data["mean"][slambench_name_dict]["Duration_Frame"],
                data["std"][slambench_name_dict]["Duration_Frame"],
                lw=1, capsize=2, label="mean")

    ax.errorbar(data["mean"][slambench_name_dict]["Frame Number"],
                data["mean5"][slambench_name_dict]["Duration_Frame"],
                data["std5"][slambench_name_dict]["Duration_Frame"],
                lw=1, capsize=2, label="mean5")

    ax.errorbar(data["mean"][slambench_name_dict]["Frame Number"],
                data["mean10"][slambench_name_dict]["Duration_Frame"],
                data["std10"][slambench_name_dict]["Duration_Frame"],
                lw=1, capsize=2, label="mean10")
        
    fig.legend()
    plt.show()"""
         
