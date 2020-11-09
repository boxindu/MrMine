import argparse,os,logging,psutil,time
from joblib import Parallel,delayed

from utils import get_files
from make_mrmine_corpus import dump_mrmine_sentences
from train import process
from train import post_process

logger = logging.getLogger()
logger.setLevel("INFO")


def main(args):
    '''
    :param args: arguments for
    1. training the skigram model for learning subgraph representations
    2. construct the deep WL kernel using the learnt subgraph representations
    3. performing graph classification using  the WL and deep WL kernel
    :return: None
    '''
    corpus_dir = args.corpus
    align_data_dir = args.dataset
    wlk_length = args.walk_length
    task = args.task
    version = args.version
    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs
    embedding_size = args.embedding_size
    num_negsample = args.num_negsample
    learning_rate = args.learning_rate
    valid_size = args.valid_size
    n_cpus = args.n_cpus
    wlk_h = args.wlk_h
    label_file_name = args.label_filed_name
    class_labels_fname = args.class_labels_file_name
    gnd_truth_fname = args.ground_truth_file_name
    wl_extn = 'new_WL'+str(wlk_h)
    data_extn = args.extension
    # wl_extn = 'WL'+str(wlk_h)
    thresholds = args.thresholds
    window = args.window
    mr_graph_extn = 'adjlist'
    mr_embedding_dir = '../embeddings/mr_graph.embeddings'
    if version == 1:
        mr_graph_file = '../mr_graph/mr_graph.adjlist'
    if version == 2:
        mr_graph_file = '../mr_v2_graph/mr_graph.adjlist'
    if version == 3:
        mr_graph_file = '../mr_v3_graph/mr_graph.adjlist'

    assert os.path.exists(corpus_dir), "File {} does not exist".format(corpus_dir)
    assert os.path.exists(output_dir), "Dir {} does not exist".format(output_dir)

    graph_files = get_files(dirname=align_data_dir, extn= data_extn, max_files=0)
    logging.info('Loaded {} graph file names form {}'.format(len(graph_files),corpus_dir))

    t0 = time.time()
    # Dump MrMine CRCN relation network. The two optimization option is set as True and False by default.
    num_all_vertices, num_sub, num_nodes = dump_mrmine_sentences(align_data_dir, wlk_h, True, False, thresholds, window, data_extn, version)
    logging.info('Generated MrMine MR network for all {} graphs in {} in {} sec'.format(len(graph_files),
                                                                                          corpus_dir, round(time.time()-t0)))

    t0 = time.time()
    process(mr_graph_extn, True, 'network', number_walks=10, walk_length=wlk_length, input=mr_graph_file,
            output=mr_embedding_dir)

    post_process(mr_embedding_dir, num_nodes, num_sub, num_all_vertices, embedding_size)
    # embedding_fname1, embedding_fname2 = train_skipgram(corpus_dir, wl_extn, learning_rate, embedding_size, num_negsample, epochs, batch_size, output_dir,valid_size)

    logging.info('Trained the skipgram model in {} sec.'.format(round(time.time()-t0, 2)))
    print('Embedding gneneration done. Check embeddings folder for node, subgraph and graph embeddings.')
    # if task == 'classification':
    #     perform_classification(corpus_dir, wl_extn, embedding_fname, class_labels_fname)
    # if task == 'alignment':
    #     perform_alignment(embedding_fname1, embedding_fname2, gnd_truth_fname)



def parse_args():
    '''
    Usual pythonic way of parsing command line arguments
    :return: all command line arguments read
    '''
    args = argparse.ArgumentParser("MrMine")
    # args.add_argument("--corpus", default = "wlfile/DrebinADGs_5k_malware/",
    args.add_argument("-c","--corpus", default = "../data/kdd_datasets/ptc",
                      help="Path to directory containing graph files to be used for graph classification or clustering")

    args.add_argument("-f", "--dataset", default="../data/alignment/karate/",
                      help="Path to directory containing graph files to be used for alignment")

    args.add_argument("-w", "--walk_length", default=10,
                      help="length of random walks in deepwalk model")

    args.add_argument('-win','--window',default=10, type=int,
                      help='window size when building cross network links in CRCN network by sorted degree list')

    args.add_argument("-t", "--task", default="alignment",
                      help="specify tasks (input: alignment/classification) for network alignment or graph "
                           "classification, default task is alignment")

    args.add_argument('-th','--thresholds', default=[0.2,0.3,0.3], type=list,
                      help='threshold list for the similarity of each layer of WL subtree; the length should be equal to'
                           'the height of WL subtree')

    args.add_argument('-l','--class_labels_file_name', default='../data/kdd_datasets/ptc.Labels',
                      help='File name containg the name of the sample and the class labels')

    # args.add_argument("--output_dir", default = "embeddings/DrebinADGs_5k_malware/",
    args.add_argument('-o', "--output_dir", default = "../embeddings",
                      help="Path to directory for storing output embeddings")

    args.add_argument('-b',"--batch_size", default=128, type=int,
                      help="Number of samples per training batch")

    args.add_argument('-e',"--epochs", default=3, type=int,
                      help="Number of iterations the whole dataset of graphs is traversed")

    args.add_argument('-extn', '--extension', default='csv',help='extension of data files; default is edgelist'
                                                                      '; also might be gexf or something else')

    args.add_argument('-d',"--embedding_size", default=32, type=int,
                      help="Intended subgraph embedding size to be learnt")

    args.add_argument('-neg', "--num_negsample", default=10, type=int,
                      help="Number of negative samples to be used for training")

    args.add_argument('-lr', "--learning_rate", default= 1.0, type=float,
                      help="Learning rate to optimize the loss function")

    args.add_argument("--n_cpus", default=psutil.cpu_count(), type=int,
                      help="Maximum no. of cpu cores to be used for WL kernel feature extraction from graphs")

    args.add_argument("--wlk_h", default=2, type=int, help="Height of WL subtree (i.e., degree of rooted subgraph features to be considered for representation learning)")

    args.add_argument('-lf', '--label_filed_name', default='Label', help='Label field to be used for coloring nodes in graphs '
                                                                  'using WL kenrel')

    args.add_argument('-v',"--valid_size", default=10, type=int,
                      help="Number of samples to validate training process from time to time")

    args.add_argument('-g', "--ground_truth_file_name", default='../data/alignment/DBLP_gnd.csv',
                      help='File name containing the name of the groundtruth for alignment')

    args.add_argument('-version','--version', default=1, type=int, help='Version of CRCN (Cross-resolution Cross-Network) networks. '
                                                        'v=1: CRCN; v=2: H-CRCN with graphs;v=3: H-CRCN with nodes')

    return args.parse_args()



if __name__=="__main__":
    args = parse_args()
    main(args)
