import os, json, logging
import networkx as nx
from networkx.readwrite import json_graph
from numpy import genfromtxt


def get_files(dirname, max_files=0, extn=None):
    all_files = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
    # for root, dirs, files in os.walk(dirname):
    #     for f in files:
    #         all_files.append(os.path.join(root, f))

    all_files = list(set(all_files))
    all_files.sort()
    if max_files:
        return all_files[:max_files]
    else:
        return all_files


def get_graph_files(dirname, max_files=0):
    all_files = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    # for root, dirs, files in os.walk(dirname):
    #     for f in files:
    #         all_files.append(os.path.join(root, f))

    all_files = list(set(all_files))
    all_files.sort()
    if max_files:
        return all_files[:max_files]
    else:
        return all_files


def save_embeddings(corpus, final_embeddings, embedding_size, opfname0, opfname1):
    lines_to_write1 = []
    lines_to_write1.append(str(corpus._vocabsize) + ' ' + str(embedding_size))
    lines_to_write1.extend([corpus._id_to_word_map[i] + ' ' +
                           ' '.join(final_embeddings[i].astype('str').tolist()) for i in xrange(corpus._vocabsize) if corpus._id_to_word_map[i].startswith('0')])
    with open(opfname0, 'w') as fh:
        for l in lines_to_write1:
            print >>fh, l
    fh.close()

    lines_to_write2 = []
    lines_to_write2.append(str(corpus._vocabsize) + ' ' + str(embedding_size))
    lines_to_write2.extend([corpus._id_to_word_map[i] + ' ' +
                            ' '.join(final_embeddings[i].astype('str').tolist()) for i in xrange(corpus._vocabsize) if
                            corpus._id_to_word_map[i].startswith('1')])
    with open(opfname1, 'w') as fh:
        for l in lines_to_write2:
            print >> fh, l
    fh.close()


def save_graph_as_edgelist(G, dir):
    # filename1 = dir + 'mr_graph.edgelist'
    filename2 = dir + 'mr_graph.adjlist'
    # nx.write_edgelist(G, filename1, data=False)
    nx.write_adjlist(G, filename2)

    return


def get_class_labels(graph_files, class_labels_fname):
    graph_to_class_label_map = {l.split()[0].split('.')[0]: int(l.split()[1].strip()) for l in open (class_labels_fname)}
    # print graph_files
    # raw_input()
    labels = [graph_to_class_label_map[os.path.basename(g).split('.')[0]] for g in graph_files]
    return labels


def read_from_json_gexf(fname=None,label_field_name='APIs',conv_undir = False):
    '''
    Load the graph files (.gexf or .json only supported)
    :param fname: graph file name
    :param label_field_name: filed denoting the node label
    :param conv_undir: convert to undirected graph or not
    :return: graph in networkx format
    '''
    if not fname:
        logging.error('no valid path or file name')
        return None
    else:
        try:
            try:
                with open(fname, 'rb') as File:
                    org_dep_g = json_graph.node_link_graph(json.load(File))
                    # num_nodes = nx.number_of_nodes(org_dep_g)
                    mapping = {}
                    for i in org_dep_g.nodes():
                        mapping[i] = int(i)
                    org_dep_g1 = nx.relabel_nodes(org_dep_g, mapping)
            except:
                org_dep_g = nx.read_gexf (path=fname)
                mapping = {}
                for i in org_dep_g.nodes():
                    mapping[i] = int(i)
                org_dep_g1 = nx.relabel_nodes(org_dep_g, mapping)

            g = nx.DiGraph()
            for n, d in list(org_dep_g1.nodes(data=True)):
                if 'Label' in d:
                    # g.add_node(n, attr_dict={'label': '-'.join(d[label_field_name].split('\n'))}) # original node attr is not used
                    # g.add_node(n, attr_dict={'label': '1'})
                    g.add_node(n, label='1')
                else:
                    # g.add_node(n, attr_dict={'label': u'-10000'})
                    g.add_node(n, label=u'-10000')
            g.add_edges_from(org_dep_g1.edges_iter())
            print 'num of nodes: {}'.format(nx.number_of_nodes(g))
        except:
            logging.error("unable to load graph from file: {}".format(fname))
        # return 0
    logging.debug('loaded {} a graph with {} nodes and {} egdes'.format(fname, g.number_of_nodes(),g.number_of_edges()))
    if conv_undir:
        g = nx.Graph (g)
        logging.debug('converted {} as undirected graph'.format (g))
    return g


def read_from_edgelist(fname= None, conv_undir = False):
    '''
    Load graph files (.edgelist)
    :param fname: graph file name
    :param conv_undir: convert to undirected graph or not
    :return: graph in networkx format
    '''
    if not fname:
        logging.error('no valid path or file name')
        return None
    else:
        try:
            g = nx.DiGraph()
            e_list = []
            single_node_list = []
            with open(fname) as f:
                for l in f:
                    if(len(l.strip().split()[:2]) > 1):
                        x, y = l.strip().split()[:2]
                        x = int(x)
                        y = int(y)
                        e_list.append((x, y))
                    else:
                        x = l.strip().split()[:2]
                        x = int(x[0])
                        single_node_list.append(x)
                g.add_edges_from(e_list)
                if len(single_node_list) > 0:
                    g.add_nodes_from(single_node_list)
            for n in list(g.nodes()):
                g.node[n]['label'] = '1'
        except:
            logging.info("unable to load graph from file: {}".format(fname))
        logging.debug(
            'loaded {} a graph with {} nodes and {} egdes'.format(fname, g.number_of_nodes(), g.number_of_edges()))
        if conv_undir:
            g = nx.Graph(g)
            logging.debug('converted {} as undirected graph'.format(g))
    return g

def read_from_csv(fname= None, conv_undir = False):
    if not fname:
        logging.error('no valid path or file name')
        return None
    else:
        if fname.endswith('csv'):
            edge_list = genfromtxt(fname, delimiter=',')
            g = nx.DiGraph()
            g.add_edges_from(edge_list)
            for n in list(g.nodes()):
                g.node[n]['label'] = '1'
            if conv_undir:
                g = nx.Graph(g)
    return g

if __name__ == '__main__':
    print 'nothing to do'
