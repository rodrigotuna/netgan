import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import networkx as nx
import pickle
import sys

from netgan import utils
from netgan.netgan import NetGAN




def main():
    ##Check the different datasets

    dataset = sys.argv[1]
    _A_obs, _X_obs, _z_obs = utils.load_npz(f'data/{dataset}.npz')
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    #lcc = utils.largest_connected_components(_A_obs)
    #_A_obs = _A_obs[lcc,:][:,lcc]
    _N = _A_obs.shape[0]

    val_share = 0.0
    test_share = 0.00
    seed = 481516234

    train_ones, val_ones, val_zeros, test_ones, test_zeros = utils.train_val_test_split_adjacency(_A_obs, val_share, test_share, seed, undirected=True, set_ops=False, every_node=False, connected=False, asserts=True)
    

    train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
    assert (train_graph.toarray() == train_graph.toarray().T).all()

    rw_len = 16
    batch_size = 128

    walker = utils.RandomWalker(train_graph, rw_len, p=1, q=1, batch_size=batch_size)

    netgan = NetGAN(_N, rw_len, walk_generator= walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
                W_down_discriminator_size=128, W_down_generator_size=128,
                l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                generator_layers=[40], discriminator_layers=[30], temp_start=5, learning_rate=0.0003)
    
    stopping = 0.5
    eval_every = 2000

    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    log_dict = netgan.train(A_orig=_A_obs, val_ones=val_ones, val_zeros=val_zeros, stopping=stopping,
                        eval_every=eval_every, max_patience=20, max_iters=200000)


    sample_many = netgan.generate_discrete(10000, reuse=True)
    samples = []

    for _ in range(6000):
        if (_+1) % 500 == 0:
            print(_)
        samples.append(sample_many.eval({netgan.tau: 0.5}))

    rws = np.array(samples).reshape([-1, rw_len])
    scores_matrix = utils.score_matrix_from_random_walks(rws, _N).tocsr()
    A_select = sp.csr_matrix((np.ones(len(train_ones)), (train_ones[:,0], train_ones[:,1])))
    sampled_graph = utils.graph_from_scores(scores_matrix, A_select.sum())
    
    #Save sampled_graph as a nx graph
    G = nx.from_numpy_array(sampled_graph)
    pickle.dump(G, open(f'{dataset}_netgan.pickle', 'wb'))




if __name__ == '__main__':
    main()