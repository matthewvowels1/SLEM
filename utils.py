
import networkx as nx
import numpy as np

def reorder_dag(dag):
    '''Takes a networkx digraph object and returns a topologically sorted graph.'''

    assert nx.is_directed_acyclic_graph(dag), 'Graph needs to be acyclic.'

    old_ordering = list(dag.nodes())  # get old ordering of nodes
    adj_mat = nx.to_numpy_array(dag)  # get adjacency matrix of old graph

    index_old = {v: i for i, v in enumerate(old_ordering)}
    topological_ordering = list(nx.topological_sort(dag))  # get ideal topological ordering of nodes

    permutation_vector = [index_old[v] for v in topological_ordering]  # get required permutation of old ordering

    reordered_adj = adj_mat[np.ix_(permutation_vector, permutation_vector)]  # reorder old adj. mat

    dag = nx.from_numpy_array(reordered_adj, create_using=nx.DiGraph)  # overwrite original dag

    mapping = dict(zip(dag, topological_ordering))  # assign node names
    dag = nx.relabel_nodes(dag, mapping)

    return dag



def get_full_ordering(DAG):
    ''' Note that the input networkx DiGraph DAG MUST be topologically sorted <before> using this function'''
    ordering_info = {}
    current_level = 0
    var_names = list(DAG.nodes)

    for i, var_name in enumerate(var_names):

        if i == 0:  # if first in list
            ordering_info[var_name] = 0

        else:
            # check if any parents
            parent_list = list(DAG.predecessors(var_name))

            # if no parents ()
            if len(parent_list) == 0:
                ordering_info[var_name] = current_level

            elif len(parent_list) >= 1:  # if some parents, find most downstream parent and add 1 to ordering
                for parent_var in parent_list:
                    parent_var_order = ordering_info[parent_var]
                    ordering_info[var_name] = parent_var_order + 1

    return ordering_info



def find_element_in_list(input_list: list, target_string: str):
    matching_indices = []
    for index, element in enumerate(input_list):
        # Check if the element is equal to the target string
        if element == target_string:
            # If it matches, add the index to the matching_indices list
            matching_indices.append(index)
    return matching_indices
