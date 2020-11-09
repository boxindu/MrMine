# from __future__ import division
import networkx as nx
import os,sys,json,logging
import numpy as np
from time import time
from networkx.readwrite import json_graph
from joblib import Parallel,delayed
from utils import get_graph_files
from utils import get_files
from fastdtw import fastdtw
from numpy import genfromtxt
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
from tempfile import gettempdir
import utils
from itertools import chain



def plot_with_labels(low_dim_embs, labels, filename):
  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  color_val_1 = cm.rainbow(np.linspace(0, 1, len(labels)/2))
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    if len(label) != 0:
        # graph_id = int(label.split('_')[0])
        # node_id = int(label.split('_')[1])
        # if graph_id == 0:
        #     color = color_val_1[node_id-1]
        # if graph_id == 1:
        #     if node_id == 1:
        #         color = color_val_1[-1]
        #     else:
        #         color = color_val_1[node_id-2]
        if int(label) < 34:
            graph_id = 0
            node_id = int(label)
            color = color_val_1[node_id]
        if int(label) >=34 and int(label) < 68:
            graph_id = 1
            node_id = int(label) - 34
            if node_id == 0:
                color = color_val_1[0]
            else:
                color = color_val_1[node_id ]
        if int(label) >= 68 and int(label) < 80:
            color = 'black'

        if int(label) >= 80:
            color = 'gray'

        plt.scatter(x, y, c=color, s= 900)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(12,-15),
            fontsize= 24,
            textcoords='offset points',
            ha='right',
            va='bottom')

  plt.savefig(filename)



def wlk_relabel(g,h):
    '''
    Perform node relabeling (coloring) according 1-d WL relabeling process (refer Shervashidze et al (2009) paper)
    :param g: networkx graph
    :param h: height of WL kernel
    :return: relabeled graph
    '''
    for n in list(g.nodes()):
        g.node[n]['relabel'] = {}
        g.node[n]['h_degree'] = {}
    count_map = {}
    for i in xrange(0,h+1): #xrange returns [min,max)
        for n in list(g.nodes()):
            # degree_prefix = 'D' + str(i)
            degree_prefix = ''
            if i == 0:
                g.node[n]['relabel'][0] = degree_prefix + str(g.node[n]['label']).strip() + degree_prefix
                if nx.is_directed(g):
                    g.node[n]['h_degree'][0] = [[g.out_degree(n)]]
                else:
                    g.node[n]['h_degree'][0] = [[g.degree(n)]]
                if g.node[n]['relabel'][0] in count_map.keys():
                    count_map[g.node[n]['relabel'][0]] += 1
                else:
                    count_map[g.node[n]['relabel'][0]] = 1
            else:
                nei_labels = [g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g,n)]
                nei_labels.sort()
                sorted_nei_labels = (','*i).join(nei_labels)

                current_in_relabel = g.node[n]['relabel'][i-1] +'#'*i+ sorted_nei_labels
                g.node[n]['relabel'][i] = degree_prefix + current_in_relabel.strip() + degree_prefix

                if g.node[n]['relabel'][i] in count_map.keys():
                    count_map[g.node[n]['relabel'][i]] += 1
                else:
                    count_map[g.node[n]['relabel'][i]] = 1

                i_nei = []
                for j in range(i + 1):
                    j_nei = [node for node, le in nx.single_source_shortest_path_length(g, n, cutoff=j).items() if le == j] # be careful for the difference!
                    i_nei.append(j_nei)
                # i_nei = [node for node, le in nx.single_source_shortest_path_length(g, n, cutoff=i).items() if le == i]
                # i_nei = [node for node, le in nx.single_source_shortest_path_length(g, n, cutoff=i).items()]
                deg_list = []
                for nei in i_nei:
                    if nx.is_directed(g):
                        out_degree = [deg[1] for deg in g.out_degree(nei)]
                        deg_list.append(sorted(out_degree))
                    else:
                        degree = [deg[1] for deg in g.degree(nei)]
                        deg_list.append(sorted(degree))
                # deg_list = sorted(list(g.degree(i_nei)))
                if len(deg_list) == 0:
                    g.node[n]['h_degree'][i] = [0]
                else:
                    g.node[n]['h_degree'][i] = deg_list
                # print n, g.node[n]['h_degree'][1]

    return g, count_map #relabled graph


def generate_subgraph_edgelist(graph_list, G, window, thresholds, version):
    '''
    generate subgraph edgelist for subgraph level
    :param G: multi resolution graph
    :param h: WL tree max height
    :param window: window size
    :param thresholds: list of thresholds for each subgraph
    :return: edge list
    '''
    edge_list = []
    U = nx.disjoint_union_all(graph_list)
    num_all_nodes = nx.number_of_nodes(G)
    num_only_nodes = nx.number_of_nodes(U)
    num_only_graphs = len(graph_list)
    sub_node_ids = range(num_only_nodes, num_all_nodes - num_only_graphs)
    if version == 3:
        sub_node_ids = range(num_only_nodes, num_all_nodes)
    second_order_deg = []

    for i in sub_node_ids:
        deg = G.node[i]['deg_list'][0]
        second_order_deg.append((i, deg))

    degrees = [(node, val) for (node, val) in U.degree()]
    sorted_nodes_tuple = sorted(second_order_deg, key=lambda x: x[1], reverse=True)
    # print sorted_nodes_tuple
    sorted_nodes_list = [node for (node, deg) in sorted_nodes_tuple]
    # print sorted_nodes_list
    for i, n in enumerate(sub_node_ids):
        if i < window/2:
            window_nodes = sorted_nodes_list[:i] + sorted_nodes_list[i + 1:window]
        elif i >= len(sorted_nodes_list) - window/2:
            window_nodes = sorted_nodes_list[-window - 1:i] + sorted_nodes_list[i + 1:]
        else:
            window_nodes = sorted_nodes_list[i - window / 2:i] + sorted_nodes_list[i + 1:i + window / 2 + 1]
        for j in window_nodes:
            if G.node[n]['height'] != G.node[j]['height']:
                window_nodes.remove(j)

        score_dict = {}

        for selected_node in window_nodes:
            height = G.node[selected_node]['height']
            if height != G.node[n]['height']:
                continue

            score = get_structural_score(n, selected_node, G)

            score_dict[selected_node] = score

            if score < thresholds[height]:
                edge_list.append((selected_node, n))
                # print 'two nodes: {}, {}; degree list:{}, {};height: {}, {} threshold: {}; score: {}'.format(n, selected_node, G.node[n]['deg_list'], G.node[selected_node]['deg_list'],G.node[n]['height'],G.node[selected_node]['height'], thresholds[height], score)
    return edge_list


def get_structural_score (u, v, g):
    '''

    :param u: the first node
    :param g1: the graph in which the first node lies
    :param v: the second node
    :param g2: the graph in which the second node lies
    :param h: WL kernel height
    :return: the structural score between the subtrees rooted at u and v
    '''
    score = 0
    a = np.array(g.node[u]['deg_list'])
    b = np.array(g.node[v]['deg_list'])
    for i in range(len(g.node[u]['deg_list'])):
        if a[i] == []:
            a[i] = [0]
        if b[i] == []:
            b[i] = [0]
        score1, path = fastdtw(a[i], b[i], dist=distance)
        score = score + score1
    # la1 = len(g.node[u]['deg_list'])
    # la2 = len(g.node[v]['deg_list'])
    # score2 = float(max(la1, la2))/float(min(la1, la2)) - 1

    return score



def distance(a, b):
    '''
    self-defined distance function between a and b
    :param a: the first element
    :param b: the second element
    :return: the distance between a and b
    '''
    if max(a, b) == 0:
        return 0
    if min(a,b) == 0:
        return float(max(a, b))/0.01 - 1
    return float(max(a, b))/float(min(a, b)) - 1


def generate_MR_graph_v1(g_list, h_max, OPT1, OPT2, thresholds, window):
    '''
    generate a new multi-resolution graph
    :param g_list: networkx graph list of input graphs
    :param OPT1: option 1 for efficiency improvement
    :param OPT2: option 2 for efficiency improvement
    :return: multi-resolution graph in networkx format
    '''
    G = nx.Graph()
    num_all_nodes = 0
    for g in g_list:
        num_all_nodes = nx.number_of_nodes(g) + num_all_nodes # new node ids of all graphs should be from 0 to num_nodes - 1
    print 'num of only nodes (all graph): {}'.format(num_all_nodes)
    sub_to_id_dict = {}
    G_id = range(-1*len(g_list), 0)
    G_id.sort(reverse=True)
    count = 0
    edge_list = []
    sub_node_attr_deg = {}
    sub_node_attr_hei = {}
    num_g_nodes = 0
    for i, g in enumerate(g_list):
        if i > 0:
            num_g_nodes = nx.number_of_nodes(g_list[i-1]) + num_g_nodes
        for n in list(g.nodes()):
            for h in range(1, h_max +1):
                if g.node[n]['relabel'][h] not in sub_to_id_dict:
                    sub_to_id_dict[g.node[n]['relabel'][h]] = int(num_all_nodes + count)
                    count += 1
                if i == 0:
                    edge_list.append((int(n), sub_to_id_dict[g.node[n]['relabel'][h]]))
                    edge_list.append((sub_to_id_dict[g.node[n]['relabel'][h]], G_id[i]))
                else:
                    edge_list.append((int(num_g_nodes + n), sub_to_id_dict[g.node[n]['relabel'][h]]))
                    edge_list.append((sub_to_id_dict[g.node[n]['relabel'][h]], G_id[i]))

    G.add_edges_from(edge_list)
    sub_length = len(sub_to_id_dict)  # subgraph ids should be from num_nodes to num_nodes + sub_length - 1
    mapping = {j: -1 * j + sub_length + num_all_nodes - 1 for j in
               G_id}  # graph ids should be from num_nodes + sub_length to num_node + sub_length + len(g_list) - 1
    G = nx.relabel_nodes(G, mapping)

    print 'unique nodes from edge list:{}'.format(len(set(list(chain(*G.edges())))))
    # print 'unique nodes from G edge list:{}'.format(len(G.edges()))
    print 'num of graphs: {}'.format(len(g_list))
    print 'num of nodes in mr graph: {}'.format(nx.number_of_nodes(G))
    print 'num of subgraphs: {}'.format(len(sub_to_id_dict))
    print 'num of only nodes: {}'.format(num_all_nodes)


    if OPT1 == True and OPT2 == True:
        return G, sub_to_id_dict

    if OPT1 == True and OPT2 == False:
        for i, g in enumerate(g_list):
            for n in list(g.nodes()):
                for h in range(1, h_max +1):
                    sub_node_attr_deg[sub_to_id_dict[g.node[n]['relabel'][h]]] = g.node[n]['h_degree'][h - 1]
                    sub_node_attr_hei[sub_to_id_dict[g.node[n]['relabel'][h]]] = h
                    # if h ==0:
                        # print g.node[n]['h_degree'][h]
                        # print g.node[n]['h_degree'][h+1]
        # print sub_node_attr_deg
        # print sub_node_attr_deg
        # print sub_node_attr_hei
        nx.set_node_attributes(G, sub_node_attr_deg, 'deg_list')
        nx.set_node_attributes(G, sub_node_attr_hei, 'height')
        threshold_dict = {}
        for l, k in enumerate(range(0, h_max + 1)):
            threshold_dict[k] = thresholds[l]

        # generate a set of edge lists that only among nodes in subgraph resolution with different similarity threshold:
        edge_list2 = generate_subgraph_edgelist(g_list, G, window, threshold_dict, 1)
        print 'length of edge list 2: {}'.format(len(edge_list2))
        G.add_edges_from(edge_list2)
        return G, sub_to_id_dict


    if OPT1 == False and OPT2 == False:

        return


def generate_MR_graph_v2(g_list, h_max):
    G = nx.Graph()
    num_graphs = len(g_list) # graph node ids are from 0 to num_graphs - 1
    count = 0
    sub_to_id_dict = {}
    edge_list = []
    L1_node_list = []
    for i in range(h_max, 0, -1):
        for j, g in enumerate(g_list):
            for k, n in enumerate(g.nodes()):
                if g.node[n]['relabel'][i] not in sub_to_id_dict.keys():
                    sub_to_id_dict[g.node[n]['relabel'][i]] = num_graphs + count
                    count += 1
                if i == h_max:
                    edge_list.append((j, sub_to_id_dict[g.node[n]['relabel'][i]]))
                else:
                    edge_list.append((sub_to_id_dict[g.node[n]['relabel'][i + 1]], sub_to_id_dict[g.node[n]['relabel'][i]]))
                if i == 1:
                    L1_node_list.append(sub_to_id_dict[g.node[n]['relabel'][i]])
    for l in L1_node_list:
        edge_list.append((l, num_graphs + count))
    G.add_edges_from(edge_list)
    sub_num = count
    return G


def generate_MR_graph_v3(g_list, h_max, thresholds, window):
    G = nx.Graph()
    num_graphs = len(g_list) # graph node ids are from 0 to num_graphs - 1
    num_nodes = 0
    for g in g_list:
        num_nodes = nx.number_of_nodes(g) + num_nodes
        print 'node id and degree:'
        print g.nodes()
        print g.degree()

    count = 0
    sub_to_id_dict = {}
    edge_list = []
    L1_node_list = []

    for i in range(h_max, 0, -1):
        num_g_nodes = 0
        for j, g in enumerate(g_list):
            if j != 0:
                num_g_nodes = nx.number_of_nodes(g_list[j - 1]) + num_g_nodes
            for k, n in enumerate(g.nodes()):
                if g.node[n]['relabel'][i] not in sub_to_id_dict.keys():
                    sub_to_id_dict[g.node[n]['relabel'][i]] = num_nodes + count
                    count += 1
                if i == h_max:
                    if j == 0:
                        edge_list.append((n, sub_to_id_dict[g.node[n]['relabel'][i]]))
                    else:
                        edge_list.append((n + num_g_nodes, sub_to_id_dict[g.node[n]['relabel'][i]]))
                else:
                    edge_list.append((sub_to_id_dict[g.node[n]['relabel'][i + 1]], sub_to_id_dict[g.node[n]['relabel'][i]]))
                if i == 1:
                    L1_node_list.append(sub_to_id_dict[g.node[n]['relabel'][i]])
    for l in L1_node_list:
        edge_list.append((l, num_graphs + count))
    G.add_edges_from(edge_list)
    sub_num = count

    sub_node_attr_deg = {}
    sub_node_attr_hei = {}
    for i, g in enumerate(g_list):
        for n in list(g.nodes()):
            for h in range(1, h_max + 1):
                sub_node_attr_deg[sub_to_id_dict[g.node[n]['relabel'][h]]] = g.node[n]['h_degree'][h - 1]
                sub_node_attr_hei[sub_to_id_dict[g.node[n]['relabel'][h]]] = h
    print sub_node_attr_deg
    # print sub_node_attr_deg
    # print sub_node_attr_hei
    nx.set_node_attributes(G, 'deg_list', sub_node_attr_deg)
    nx.set_node_attributes(G, 'height', sub_node_attr_hei)
    threshold_dict = {}
    for l, k in enumerate(range(1, h_max + 1)):
        threshold_dict[k] = thresholds[l]

    # generate a set of edge lists that only among nodes in subgraph resolution with different similarity threshold:
    edge_list2 = generate_subgraph_edgelist(g_list, G, window, threshold_dict, 3)
    print 'length of edge list 2: {}'.format(len(edge_list2))
    G.add_edges_from(edge_list2)
    print G.nodes()
    print num_nodes
    print num_g_nodes

    return G, num_nodes



def generate_random_walks(G, num_walks, walk_length):


    return


def dump_mrmine_sentences(dir, h, opt1, opt2, thresholds, window, extn, version):
    '''
    Get WL features and make the MRML sentence of the format "<target> <context1> <context2> ..." and dump the
    same into a text file.
    :param dir: directory of graph files
    :param extn: the suffix of graph file name (.edgelist, .csv, .json, .gexf, etc.)
    :param h: max height of WL trees
    :return: None
    '''
    graph_files = get_files(dirname=dir, max_files=0, extn=extn)
    graph_id = [i.rstrip('.' + extn).split('/')[-1] for i in graph_files]
    print graph_id
    g_list = []
    num_only_nodes = 0
    prob_map = {}
    for gid, f in enumerate(graph_files):
        extn = str(f.split('.')[-1])
        if extn == 'json' or extn == 'gexf':
            g = utils.read_from_json_gexf(fname=f,label_field_name='Label')
        elif extn == 'edgelist':
            g = utils.read_from_edgelist(f, conv_undir=True)
        elif extn == 'csv':
            g = utils.read_from_csv(f)
        else:
            logging.info("unknown graph file type.")
        if not g:
            return
        g, count_map = wlk_relabel(g, h)
        g_list.append(g)
        prob_map[gid] = count_map
        num_only_nodes = num_only_nodes + nx.number_of_nodes(g)
    logging.info('Loaded {} graph files from {}'.format(len(graph_files), dir))

    # dirname = gettempdir()/
    # plot_networks(g_list, dirname)

    if version == 1:
        T0 = time()

        G, sub_to_id = generate_MR_graph_v1(g_list, h, opt1, opt2, thresholds, window)
        # plt.figure()
        # nx.draw_networkx(G)
        # plt.show()

        logging.info('generating MR graph time {}'.format(round(time()-T0)))

        utils.save_graph_as_edgelist(G, '../mr_graph/')

        T0 = time()

        # generate_random_walks(G, num_walks, walk_length)

        logging.info('generating randkom walk time {}'.format(round(time()-T0)))

        print 'num of nodes in mr graph: {}'.format(nx.number_of_nodes(G))
        print 'num of subgraphs: {}'.format(len(sub_to_id))
        print 'num of only nodes: {}'.format(num_only_nodes)

        # return sub_to_id, num_only_nodes, prob_map, graph_files

    if version == 2:
        G = generate_MR_graph_v2(g_list, h_max= h)
        # plt.figure()
        # nx.draw_networkx(G)
        # plt.show()
        utils.save_graph_as_edgelist(G, '../mr_v2_graph/')
        num_graphs = len(g_list)

        # return num_graphs


    if version == 3:
        G, num_nodes = generate_MR_graph_v3(g_list, h_max= h, thresholds=thresholds, window=window)
        # plt.figure()
        # nx.draw_networkx(G)
        # plt.show()
        utils.save_graph_as_edgelist(G, '../mr_v3_graph/')

        # return num_nodes

    return nx.number_of_nodes(G), len(sub_to_id), num_only_nodes



if __name__ == '__main__':
    dir = '../data/alignment/flickr-lastfm/'
    # dir = '../data/kdd_datasets/mutag/'
    h = 2
    graph_files = get_files(dirname=dir, max_files=0, extn='edgelist')
    for f in graph_files:
        print f
    extn = str(f.split('.')[-1])
    print extn
    _ = dump_mrmine_sentences(dir, h, True, False, [0, 0.1, 0.04], 16, 'edgelist', 1)



