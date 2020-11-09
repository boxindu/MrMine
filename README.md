# MrMine
This is a python implementation of MrMine: Multi-Resolution Multi-Network Embedding.

# Please cite the following paper:

Du, Boxin, and Hanghang Tong. "MrMine: Multi-resolution Multi-network Embedding."
Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.

# Dependencies:
The code is written in Python 2.7. Tested on Windows, Ubuntu 14.04, 16.04. The following packages are used:

gensim (version == 3.6.0)
networkx (version == 2.2)
joblib (version == 0.14.1)
scikit-learn (version == 0.20.4) (+scipy==1.1.0, +numpy==1.15.4)
six (version == 1.11.0)

# Remarks:
1. Only a small dataset is included in the data folder for a quick demo. More dataset can be added by users or
contact the author. .csv file, .edgelist file, and .adjlist file are supported. For an example, just run:

python main.py

(inside src folder)

2. The output embeddings will be in the embeddings folder, in three files (mr_only_graph.embeddings, mr_only_node.embeddings,
and mr_only_subgraph.embeddings) containing embeddings of three resolutions. Embeddings of different networks for nodes
and graphs are concatenated by the order of the input graph files.

3. The perform_alignment.py and perform_classification.py are temporarily removed (commented out), and will be added back
 in the next version.

4. All arguments (including hyper-parameters) can be found by using (inside src folder):

python main.py --help

