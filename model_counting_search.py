# use python2
import numpy as np
import networkx as nx
import nlopt

################################################################################
# function to compute the complement of a binary string
def get_neg(in_str):
    neg_str = ''
    for s in in_str:
        if s == '0':
            neg_str += '1'
        if s == '1':
            neg_str += '0'
    return neg_str

################################################################################
# function to generate a string, which can be loaded into the crossbar with greater ease
# format: {T, F, A0, !A0, B0, !B0, A1, !A1, B1, !B1, ..., An, !An, Bn, !Bn}
def weave_str(a, b, neg_a, neg_b):
    rev_a = a[::-1]
    rev_neg_a = neg_a[::-1]
    rev_b = b[::-1]
    rev_neg_b = neg_b[::-1]
    tt_str = ''

    for i in range(len(rev_a)):
        tt_str += rev_a[i]+rev_neg_a[i]+rev_b[i]+rev_neg_b[i]

    return '01'+tt_str

################################################################################
# function to generate truth table for n-bit addition
def get_tt_adder(nbits):
    max_val = 2**nbits
    tt_list = []
    out_list = []
    for i in range(max_val):
        for j in range(max_val):
            a = format(i, '0'+str(nbits)+'b')
            b = format(j, '0'+str(nbits)+'b')
            a_neg = get_neg(a)
            b_neg = get_neg(b)
            c_out = format(i+j, '0'+str(nbits+1)+'b')[0]
            tt_str = weave_str(a, b, a_neg, b_neg)
            tt_list.append(tt_str)
            out_list.append(c_out)
    return tt_list, out_list

################################################################################
# function to generate truth table for n-bit binary comparison
def get_tt_comparator(nbits):
    max_val = 2**nbits
    tt_list = []
    out_list = []
    for i in range(max_val):
        for j in range(max_val):
            a = format(i, '0'+str(nbits)+'b')
            b = format(j, '0'+str(nbits)+'b')
            a_neg = get_neg(a)
            b_neg = get_neg(b)
            comp_out = int(i > j)
            tt_str = weave_str(a, b, a_neg, b_neg)
            tt_list.append(tt_str)
            out_list.append(comp_out)
    return tt_list, out_list

################################################################################
# objective function to evaluate a given crossbar design
# in this function, we want to find a crossbar that computes the MSB of n-bit binary addition
# change names of truth table and output variables for other functions
def xbar_eval_adder(xbar_in, grad):
    xbar_arr = np.digitize(xbar_in, bins)-1 # generate crossbar design as integers
    xbar_design = np.reshape(xbar_arr, (nrows, ncols)) # reshape xbar array to xbar matrix of size nrows x ncols

    xbar_out = []
    for input_sample in adder_truthtable:
        xbar_instance = np.zeros((nrows, ncols), dtype=int) # xbar design loaded with inputs
        for i in range(nrows):
            for j in range(ncols):
                xbar_instance[i][j] = int(input_sample[xbar_design[i][j]])

        # build graph from the instantiated xbar design
        # each nanowire is a node
        # each memristor is an edge
        # an edge exists between two nodes if
        # the memristor between the two nanowires is ON for a given set of inputs
        xbar_graph = nx.Graph()
        xbar_graph.add_nodes_from(range(nrows+ncols))
        for i in range(nrows):
            for j in range(ncols):
                if xbar_instance[i][j] == 1:
                    xbar_graph.add_edge(i, nrows+j)

        tmp_out = nx.has_path(xbar_graph, 0, nrows-1) # check if there exists a path between the top row and bottom row
        xbar_out.append(int(tmp_out))

    # count the number of unsatisfiable instances
    # we want to maximize the number of satisfiable instances
    # and minimize the number of unsatisfiable instances
    obj_score = 0.0
    for i in range(len(adder_msb)):
        if int(adder_msb[i]) != xbar_out[i]:
            obj_score += 1.0

    return obj_score

################################################################################

nbits = 4 # number of bits per input variable
n_memr = 2*2*nbits+2 # number of possibilities for each memristor
nrows = 8 # number of rows in the xbar
ncols = 5 # number of columns in the xbar
tmp_xbar = np.random.random(nrows*ncols)
bins = np.linspace(0, 1, n_memr+1) # binarizing the design space
adder_truthtable, adder_msb = get_tt_adder(nbits)

# comparator_truthtable, comparator_out = get_tt_comparator(nbits) # truth table and output for comparator
# print(n_memr)
# xbar_eval_adder(tmp_xbar)

scaling_param = 0.5*(bins[-1]+bins[-2]) # scaling parameter to constrain nlopt solutions
print scaling_param
maxtime_val = 86400 # maximum time to run search for
opt = nlopt.opt(nlopt.GN_ESCH, nrows*ncols) # choose nlopt algorithm
opt.set_maxtime(maxtime_val)
opt.set_lower_bounds(np.zeros(nrows*ncols)) # lower bounds
opt.set_upper_bounds(scaling_param*np.ones(nrows*ncols)) # upper bounds
opt.set_min_objective(xbar_eval_adder) # set objective function
# opt.set_xtol_rel(0) # tolerance value
x = opt.optimize(scaling_param*np.random.random(nrows*ncols)) # start optimization with random solution
minf = opt.last_optimum_value()
print ("minimum value = ", minf)
print ("result code = ", opt.last_optimize_result())
xbar_final = np.reshape(np.digitize(x, bins)-1, (nrows, ncols)) # format final solution to print
print(xbar_final)
