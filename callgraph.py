from ast import Call
from collections import defaultdict

class CallGraph:
    """ Graph data structure, undirected by default. """

    def __init__(self):
        self._graph = defaultdict(list)
    
    
    def add_functions(self, functions):
        """ Add functions to the graph with the given format of ((function, parent), (func., par.), ...) """
        for function, parent in functions:
            self.add_function(function, parent)
    

    def add_function(self, function, parent=None):
        """ Add function with optional parent """

        if parent != None:
            self._graph[function].append(parent)
            if not parent in self._graph:
                self._graph[parent] = []
        else:
            self._graph[function] = []
    
    
    def add_parent(self, function, parent):
        """ Add parent to function """
        
        self._graph[function].append(parent)


    def find_path(self, node1, node2, path=[]):
        """ Find any path between node1 and node2 (may not be shortest) """

        path = path + [node1]
        if node1 == node2:
            return path
        if node1 not in self._graph:
            return None
        for node in self._graph[node1]:
            if node not in path:
                new_path = self.find_path(node, node2, path)
                if new_path:
                    return new_path
        return None
    
    
    def get_roots(self):
        """ Return all functions with no parent """
        
        res = []
        
        for k, v in self._graph.items():
            if not len(v):
                res += [k]

        return res
    
    
    def find_children(self, parent):
        """ Return all children of a given parent """
        
        res = []
        
        for k, v in self._graph.items():
            if parent in v:
                res += [k]

        return res
    
    
    def get_functions(self, depth=None, inclusive=False, roots=None):
        """ Return functions of a given depth. If no depth is given, returns all functions.
            If inclusive is set to False (default) includes all functions that are the end of its
            tree even if the depth is smaller. Otherwise, only functions with the specified depth are
            returned. """
        if depth == None: #return all functions
            return self._graph.keys()
        
        if roots == None: #Original call, get roots
            roots = self.get_roots()
        
        if not depth: #End of the tree
            return roots
        
        functions = []
        for root in roots:
            children = self.find_children(root)
            if len(children):
                functions += self.get_functions(depth-1, roots=children)
            elif not inclusive:
                functions += [root] #If there are no more children, return itself
        
        return functions
    
    
    def is_empty(self):
        """ Returns wether the callgraph is empty """
        
        return not bool(len(self._graph))
    
    
    def _build_recursive_string(self, root, str, depth):
        """ Recursively builds a string in a tree shape from a given root. """
            
        children = self.find_children(root)
        
        for child in children:
            str += "-" * depth + child + "\n"
            str = self._build_recursive_string(child, str, depth+1)
        
        return str
    

    def __str__(self):
        str = ""
        for root in self.get_roots():
            str += root + "\n"
            str = self._build_recursive_string(root, str, 1)
        return str
            
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
    
    def __eq__(self, other):
        if not isinstance(other, CallGraph): #Other object
            return False
        
        functions = self._graph.keys()
        other_functions = other._graph.keys()
        
        if len(functions) != len(other_functions):
            return False # Different number of functions
        
        for other_function in other_functions:
            if not other_function in self._graph: 
                return False # Function does not match
            
            for other_parent in other._graph[other_function]: 
                if not other_parent in self._graph[other_function]:
                    return False # Parent does not match
        
        return True
    
if __name__ == "__main__":
    callgraph = CallGraph()
    callgraph.add_functions((("A1", None), 
                            ("B1", "A1"), 
                            ("B2", "A1"),
                            ("C1", "B1"), 
                            ("C2", "B1"),
                            ("C3", "B2"), 
                            ("C4", "B2"),
                            ("D1", "C2"),
                            ))

    print(callgraph)
    print(callgraph.get_functions(3))
