import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
from scipy.sparse import lil_matrix

import graph
import walks as serialized_walks
from gensim.models import Word2Vec
from skipgram import Skipgram

from six import text_type as unicode
from six import iteritems
from six.moves import range

import psutil
from multiprocessing import cpu_count


# train deepwalk model and save embeddings
def process(format, undirected, matfile_variable_name, number_walks, walk_length, input, output, max_memory_data_size=1000000000,
            representation_size=32, window_size=5, workers=1, vertex_freq_degree=False, seed=0):

    if format == "adjlist":
        G = graph.load_adjacencylist(input, undirected=undirected)
    elif format == "edgelist":
        G = graph.load_edgelist(input, undirected=undirected)
    elif format == "mat":
        G = graph.load_matfile(input, variable_name=matfile_variable_name, undirected=undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % format)

    print("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * number_walks

    print("Number of walks: {}".format(num_walks))

    data_size = num_walks * walk_length

    print("Data size (walks*length): {}".format(data_size))

    if data_size < max_memory_data_size:
        print("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                        path_length=walk_length, alpha=0, rand=random.Random(seed))
        print("Training...")
        model = Word2Vec(walks, size=representation_size, window=window_size, min_count=0, sg=1, hs=1, workers=workers)
    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size, max_memory_data_size))
        print("Walking...")

        walks_filebase = output + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                         path_length=walk_length, alpha=0, rand=random.Random(seed),
                                         num_workers=workers)

        print("Counting vertex frequency...")
        if not vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(walk_files, workers)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,size=representation_size,
                         window=window_size, min_count=0, trim_rule=None, workers=workers)

    model.wv.save_word2vec_format(output)

# split the mr_graph embedding files to node, subgraph, graph embedding files
def post_process(embedding_file, num_nodes, num_sub, num_all_nodes, emb_size):
    with open(embedding_file, 'r') as embed:
        try:
            vectors1_str = embed.readlines()
        except:
            print 'cannot read embedding file 1...'
        embed.close()
    num_all_nodes = len(vectors1_str) - 1
    print 'num of mr graph nodes: {}'.format(num_all_nodes)
    embed_mat = lil_matrix((num_all_nodes, emb_size))
    label = []
    subgraph_embedding_dict = {}
    node_embedding_dict = {}
    graph_embedding_dict = {}
    for index, i in enumerate(vectors1_str):
        if index == 0:
            continue
        node_id_str = i.split(' ')[0]
        node_id = int(
            node_id_str)  # because this node id starts with "1", to avoid "-1", let the node id start with "0" in the original csv file
        # node_id = int(node_id_str)
        label.append(node_id_str)
        vec = [float(i.split(' ')[j]) for j in xrange(1, len(i.split(' ')))]
        embed_mat[index - 1, :] = vec
        if node_id >= num_nodes and node_id < num_nodes + num_sub:
            subgraph_embedding_dict[str(node_id)] = ' '.join(i.split(' ')[1:])
        if node_id < num_nodes:
            node_embedding_dict[str(node_id)] = ' '.join(i.split(' ')[1:])
        if node_id >= num_nodes + num_sub:
            graph_embedding_dict[str(node_id)] = ' '.join(i.split(' ')[1:])

    subgraph_embedding_file = '../embeddings/mr_only_subgraph.embeddings'
    with open(subgraph_embedding_file, 'w') as n_embed_f:
        for i, d in enumerate(subgraph_embedding_dict):
            line_to_write = ' '.join((d, subgraph_embedding_dict[d]))
            n_embed_f.write(line_to_write)
        n_embed_f.close()

    graph_embedding_file = '../embeddings/mr_only_graph.embeddings'
    with open(graph_embedding_file, 'w') as n_embed_f:
        for i,d in enumerate(graph_embedding_dict):
            line_to_write = ' '.join((d, graph_embedding_dict[d]))
            n_embed_f.write(line_to_write)
        n_embed_f.close()

    node_embedding_file = '../embeddings/mr_only_node.embeddings'
    with open(node_embedding_file, 'w') as n_embed_f:
        for i, d in enumerate(node_embedding_dict):
            line_to_write = ' '.join((d, node_embedding_dict[d]))
            n_embed_f.write(line_to_write)
        n_embed_f.close()
