import numpy as np
from collections import defaultdict
import re
import time
from multiprocessing import Process, Manager, cpu_count
import json

FIELD_PID = 1
FIELD_TIME = 2
FIELD_EVENT_NAME = 4
TOT_FIELDS = 7 #All the other fields are wrong because of spaces in the symbols

def read_file(file_dir):
    t0 = time.time()
    # Open the input file
    with open(file_dir, "r") as f:
        lines = f.readlines()
    
    t = time.time()
    #print(f"TIME TO READ FILE: {(t-t0):.3f} s")
    return lines


def process_script_file(lines, frame_times, data_return=None, **kwargs):
    """
        Procesa las líneas del fichero leídas por read_file.
        Los filtros se pasan como argumentos y son leídos por kwargs.
        
        Si se da un filtro include, se ignora el exclude.
        
        El filtro de threads acepta "main" para los que pid y tid coincidan.
        
        Si se pasa un diccionario por data_return, se devuelve
        por referencia en lugar de por el return.
    """
    t0 = time.time()
    
    def parse_line_event(i):
        """Process the line with the event info, returns wether the loop should be continued"""
        nonlocal event_name
        nonlocal pid
        nonlocal tid
        nonlocal timestamp
        
        fields = re.findall(r'(\S+)', lines[i])
        if not fields: #Empty line
            return True

        event_name = fields[FIELD_EVENT_NAME][:-1] #Ignore ':' at the end
        
        pid_string = fields[FIELD_PID]
        pid_string = pid_string.split('/')
        pid = pid_string[0]
        tid = pid_string[1]
        
        timestamp_string = fields[FIELD_TIME][:-1] #Ignore ':' at the end
        timestamp_string = timestamp_string.split('.') 
        timestamp = int(timestamp_string[0])*1e9 + int(timestamp_string[1])*1e3
        
        return False
        
    
    def parse_line_stack(i):
        """Process a line from the stack trace and return the relevant fields"""
        fields = lines[i].strip().split()
            
        function_name = "".join(fields[1:-1]).split('+')[0]

        text = fields[-1]
        start = text.find("(")  # Find the position of the opening bracket
        end = text.find(")")    # Find the position of the closing bracket
        if start != -1 and end != -1:  # Check if both brackets were found
            shared_library = text[start + 1:end]  # Extract the text between the brackets
        else:
            shared_library = ""
        
        return function_name, shared_library
    
    
    def look_for_parents(i):
        """Starts at line i and parses the file looking for all the parents."""
        parents = []
        while i < len(lines) and lines[i].startswith("\t"):
            function_name, shared_library = parse_line_stack(i)
            parents.append(function_name)
            i += 1
        
        return parents
            
    event_name = ""
    pid = 0
    tid = 0
    timestamp = 0
    
    filters_library_include = set()
    filters_parent_include = set()
    filters_thread_include = set()
    filters_library_exclude = set()
    filters_parent_exclude = set()
    filters_thread_exclude = set()
    
    sort_frame_tid = False
    
    #PARSE KWARGS
    filter_thread_check = [False, False]
    for key, value in kwargs.items():
        filter_set = None
        match key:
            case "filter_library_include":
                filter_set = filters_library_include
            case "filter_parent_include":
                filter_set = filters_parent_include
            case "filter_thread_include":
                filter_set = filters_thread_include
                filter_thread_check[0] = True
            case "filter_library_exclude":
                filter_set = filters_library_exclude
            case "filter_parent_exclude":
                filter_set = filters_parent_exclude
            case "filter_thread_exclude":
                filter_set = filters_thread_exclude
                filter_thread_check[1] = True
            case _:
                print(f"Warning: {key} is not an accepted argument. Ignored.")
                continue
        
        if isinstance(value, (list, tuple)):
            filter_set.update(value)
        else:
            filter_set.add(value)
    
    parse_line_event(0)
    if filter_thread_check[0]:
        filters_thread_include.remove('main')
        filters_thread_include.add(pid)
    elif filter_thread_check[1]:
        filters_thread_exclude.remove('main')
        filters_thread_exclude.add(pid)
    
    # Initialize a dictionary to store the counts
    function_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    function_libraries = defaultdict(lambda: defaultdict(lambda: 0))
    pid_counts = defaultdict(lambda: defaultdict(lambda: 0))
    tid_counts = defaultdict(lambda: defaultdict(lambda: 0))
    frame_counts = defaultdict(lambda: defaultdict(lambda: [defaultdict(lambda: 0) for _ in range(len(frame_times))]))
    
    # Process the lines
    first_timestamp = timestamp
    frame = 1
    i = 0
    while i < len(lines):
        # Get the first line of the event
        if parse_line_event(i):
            i += 1
            continue
        
        # Process frame counting
        relative_timestamp = timestamp - first_timestamp
        
        while (relative_timestamp > frame_times[frame] - frame_times[0]): #next frame
            frame += 1

        # Process the call-stack lines
        i += 1
        depth = -1 #So it is 0 on the first iteration
        caller = True
        while i < len(lines) and lines[i].startswith("\t"):
            function_name, shared_library = parse_line_stack(i)
            depth += 1
            #TODO: Explore using depth instead of caller
            #TODO: Explore puting i++ here to save it on every continue condition
            
            #LIBRARY FILTER
            if filters_library_include:
                if not shared_library in filters_library_include:
                    i += 1
                    caller = False
                    continue
            else:
                if shared_library in filters_library_exclude:
                    i += 1
                    caller = False
                    continue
            
            #PARENT FILTER
            if caller: #Optimization: Only look for parents on the first iteration
                parents = look_for_parents(i+1)
            
            if filters_parent_include:
                if not filters_parent_include.intersection(set(parents[depth:])):
                    i += 1
                    caller = False
                    continue
            else:
                if filters_parent_exclude.intersection(set(parents[depth:])):
                    i += 1
                    caller = False
                    continue
            
            #THREAD FILTER
            if filters_thread_include:
                if not tid in filters_thread_include:
                    i += 1
                    caller = False
                    continue
            else:
                if tid in filters_thread_exclude:
                    i += 1
                    caller = False
                    continue
            
            function_libraries[function_name][shared_library] += 1
            
            if caller:
                function_counts[event_name][function_name]["self"] += 1
                pid_counts[event_name][pid] += 1
                tid_counts[event_name][tid] += 1

                #Frame-1 because starts in frame=1 (interval 0-1)
                frame_counts[event_name]["function_name"][frame-1][function_name] += 1
                frame_counts[event_name]["tid"][frame-1][tid] += 1
                frame_counts[event_name]["total"][frame-1]["foo"] += 1 #TODO: Fix this foo
            else:
                function_counts[event_name][function_name]["child"] += 1
            function_counts[event_name][function_name]["total"] += 1
            
            i += 1
            caller = False
    
    
    
    return_dict = {"function_counts":function_counts, "function_libraries":function_libraries,
                   "pid_counts":pid_counts, "tid_counts":tid_counts, "frame_counts":frame_counts}
    
    if data_return is not None:
        return_dict = json.loads(json.dumps(return_dict)) #Seems to be around 10ms. Is it worth it to change the default dicts so they can be serialized by pickle?
        data_return.update(return_dict)
        return
    
    t = time.time()
    print(f"TIME TO PROCESS FILE: {(t-t0):.3f} s")
    return return_dict
    #return function_counts, function_libraries, pid_counts, tid_counts, frame_counts


def process_script_file_paralell(lines, frame_times, n_threads = None, **kwargs):
    t0 = time.time()
    if not isinstance(n_threads, int):
        n_threads = cpu_count()
    
    chunks = np.linspace(0, len(lines), n_threads+1, dtype=int)
    
    threads = []
    return_dicts = []
    
    #Align equal-size chunks so all of them start in the next event
    for c in range(1, n_threads+1):
        i = chunks[c]
        while i < len(lines) and lines[i].startswith(("\t", "\n")):
            i += 1
        chunks[c] = i
    
    manager = Manager()
    for n in range(n_threads):
        curr_dict = manager.dict()
        return_dicts.append(curr_dict)
        threads.append(Process(target=process_script_file, 
                               args=(lines[chunks[n]:chunks[n+1]], frame_times, curr_dict), kwargs=kwargs))
        threads[n].start()
    
    [thread.join() for thread in threads]
    
    tx = time.time()
    function_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    function_libraries = defaultdict(lambda: defaultdict(lambda: 0))
    pid_counts = defaultdict(lambda: defaultdict(lambda: 0))
    tid_counts = defaultdict(lambda: defaultdict(lambda: 0))
    frame_counts = defaultdict(lambda: defaultdict(lambda: [defaultdict(lambda: 0) for _ in range(len(frame_times))]))

    for rd in return_dicts:
        for dict_name, d in rd.items():
            curr_dict = None
            match dict_name:
                case 'function_counts':
                    curr_dict = function_counts
                case 'function_libraries':
                    curr_dict = function_libraries
                case 'pid_counts':
                    curr_dict = pid_counts
                case 'tid_counts':
                    curr_dict = tid_counts
                case 'frame_counts':
                    curr_dict = None
                    continue #TODO: Fuse this dict
                case _:
                    raise(NotImplementedError)
            
            #Recursively traverse the dictionary and add the last-depth element to the global one
            stack = [(list(d.items()), curr_dict)] # create a list of key-value pairs and a its destination dictionary
            while stack: # while the stack is not empty
                kv_pairs, dest_dict = stack.pop() # pop the last pair of key-value pairs and parent dictionary
                for key, value in kv_pairs: # iterate over them
                    if isinstance(value, dict): # if the value is a dictionary
                        stack.append((list(value.items()), dest_dict[key])) # push its key-value pairs and the updated list of keys to the stack
                    else: # otherwise
                        dest_dict[key] += value


    t = time.time()
    print(f"TIME TO PROCESS FILE: {(t-t0):.3f} s")
    print(f"SUBTIME: {(t-tx):.3f} s")
    
    return function_counts, function_libraries, pid_counts, tid_counts, frame_counts
    
    

def process_slambench_output_file(lines):
    t0 = int(lines[0].split('\t')[1])
    times = []
    i = 0
    while i<len(lines):
        fields = lines[i].split('\t')
        if fields[0] in ['FRAME', 'END']: #TODO: Remove end (useless)
            times.append(int(fields[1]) - t0)
        elif fields[0] == 'REAL_END':
            times.append(int(fields[1])*100 - t0) #Ridiculous time so all events aside the last frame, fit here
            return np.array(times)
        
        i += 1    
    raise NotImplementedError


def get_counts_by_event_and_sort_keyA(data, event_name, sort_key):
    counts = [[], [], [], []] #Function name, 3 fields of sort_key (self, child, total)
    if event_name in data:
        functions = data[event_name]
        for function_name, count_types in functions.items():
            counts[0].append(function_name)
            counts[1].append(count_types["self"])
            counts[2].append(count_types["child"])
            counts[3].append(count_types["total"])

        #counts = sorted(counts, key=lambda x: sort_key(x[0]), reverse=True)
    return counts


def get_counts_by_event_and_sort_keyB(data, event_name, sort_key):
    assert event_name in data
    
    functions = data[event_name]
    
    mem = [0] * len(functions)
    counts = [mem.copy() for i in range(4)] #Function name, 3 fields of sort_key (self, child, total)
    for i, (function_name, count_types) in enumerate(functions.items()):
        counts[0][i] = function_name
        counts[1][i] = count_types["self"]
        counts[2][i] = count_types["child"]
        counts[3][i] = count_types["total"]

    #counts = sorted(counts, key=lambda x: sort_key(x[0]), reverse=True)
    return counts


def get_counts_by_event_and_sort_keyC(data, event_name, sort_key):
    assert event_name in data
    
    functions = data[event_name]
    
    counts = [[0] * len(functions) for i in range(4)] #Function name, 3 fields of sort_key (self, child, total)
    for i, (function_name, count_types) in enumerate(functions.items()):
        counts[0][i] = function_name
        counts[1][i] = count_types["self"]
        counts[2][i] = count_types["child"]
        counts[3][i] = count_types["total"]

    #counts = sorted(counts, key=lambda x: sort_key(x[0]), reverse=True)
    return counts


def get_counts_by_event_and_sort_key(data, event_name, sort_key):
    assert event_name in data
    sort_key = ["function_name", "self", "child", "total"].index(sort_key)
    
    functions = data[event_name]
    
    counts = [[0]*4 for _ in range(len(functions))]
    for arr, function_name, count_types in zip(counts, functions.keys(), functions.values()):      
        arr[0] = function_name
        arr[1] = count_types["self"]
        arr[2] = count_types["child"]
        arr[3] = count_types["total"]

    counts = sorted(counts, key=lambda x: x[sort_key], reverse=True)
    return list(zip(*counts))


def count_events(data, event_name):
    assert event_name in data
    n_events = 0
    
    for count_type in data[event_name].values():
        n_events += count_type["self"]
    
    return n_events


if __name__ == "__main__":
    slambench_file = r"C:\dev\analysis_slambench\results\perf_scripts\sanity\output.log"
    perf_script_file = r"C:\dev\analysis_slambench\results\perf_scripts\sanity\slambench_script"

    times = process_slambench_output_file(read_file(slambench_file))
    lines = read_file(perf_script_file)
    
    measures = []
    sub_measures = []
    for k in range(5):
        t0 = time.perf_counter()
        function_counts, function_libraries, pid_counts, tid_counts, frame_counts = \
            process_script_file(lines, times)
        sub_measures.append(time.perf_counter()-t0)
    measures.append(sub_measures)
    
    for n in range(2, 13):
        sub_measures = []
        for k in range(5):
            t0 = time.perf_counter()
            function_counts, function_libraries, pid_counts, tid_counts, frame_counts = \
                process_script_file_paralell(lines, times, n)
            sub_measures.append(time.perf_counter()-t0)
        measures.append(sub_measures)
    
    
    for i, sm in enumerate(measures):
        print(f"{i+1}: {np.mean(sm):.3f} ± {np.std(sm):.3f} s")


    """function_counts, function_libraries, pid_counts, tid_counts, frame_counts = \
        process_script_file(lines, times)
                    #filter_library_exclude='[kernel.kallsyms]', 
                    #filter_parent_include='main')#,
                    #filter_thread_include='main')"""

