import os
from collections import OrderedDict

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import numpy as np
np.random.seed(0)


from sosa_sim.graph import convert_keras_to_graph

import argparse

import logging
logging.basicConfig(filename='logger.out', filemode='w', level=logging.INFO)

import json 

def split_mat(x_size, w_size, array_size, partition_size=None):
    no_row, no_col = array_size

    if partition_size is None:
        partition_size = (no_row, no_row, no_col)

    dim1 = int(x_size[0])
    dim2 = int(w_size[0])
    dim3 = int(w_size[1])

    no_row_tile = int( np.ceil (dim2 / partition_size[1]) )
    no_col_tile = int( np.ceil (dim3 / partition_size[2]) )

    w_tile_dim = {i:{} for i in range(no_row_tile)}
    for j in range(no_col_tile):
        for i in range(no_row_tile):
            tile_dim1 = min((i+1)*partition_size[1], dim2) - i*partition_size[1]
            tile_dim2 = min((j+1)*partition_size[2], dim3) - j*partition_size[2]
            w_tile_dim[i][j] = [tile_dim1,tile_dim2]

    
    no_batch_tile = int( np.ceil (dim1 / partition_size[0]) )
    x_tile_dim = {i:{} for i in range(no_batch_tile)}

    for j in range(no_row_tile):
        for i in range(no_batch_tile):
            tile_dim1 = min((i+1)*partition_size[0], dim1) - i*partition_size[0]
            tile_dim2 = min((j+1)*partition_size[1], dim2) - j*partition_size[1]
            x_tile_dim[i][j] = [tile_dim1, tile_dim2]

    return x_tile_dim, w_tile_dim, no_batch_tile, no_row_tile, no_col_tile


def partition_layer(layer_node, array_size, partition_size):
    if layer_node.layer_type == 'Conv2D' or layer_node.layer_type == 'Dense' or layer_node.layer_type == 'DepthwiseConv2D':
        gemm_info = {}  

        input_size = layer_node.layer_attr["input_size"]
        weight_size = layer_node.layer_attr["weight_size"]

        if layer_node.layer_type == 'Conv2D':
            gemm_info["kernel_size"] = layer_node.layer_attr["kernel_size"]

        gemm_info["input_size"] = input_size
        gemm_info["weight_size"] = weight_size

        x_tile_dim, w_tile_dim, no_batch_tile, no_row_tile, no_col_tile = split_mat(input_size, weight_size, array_size, partition_size)
        no_tiles = (no_batch_tile, no_row_tile, no_col_tile)

        gemm_info["x_tile_dim"] = x_tile_dim
        gemm_info["w_tile_dim"] = w_tile_dim
        gemm_info["no_tiles"] = no_tiles

        return gemm_info
    else:
        logging.debug("No GEMM operation found at layer {} of type {}.".format(layer_node.layer_name, layer_node.layer_type))
        return None

def all_dependencies_scheduled(layer_name, graph, layer_schedules):
    node = graph.get_node(layer_name)

    for s in node.src:
        if s.layer_name not in layer_schedules:
            return False

    return True

def precompile_model(model, array_size=[512,512], partition_size=None):
    graph = convert_keras_to_graph(model)

    raw_input = 1

    layers = OrderedDict()
    for layer_name in graph.get_layer_names():
        layer_node = graph.get_node(layer_name)
        gemm_op = partition_layer(layer_node, array_size, partition_size)
        dependencies = [s.layer_name for s in layer_node.src]

        if gemm_op is not None: 
            raw_input = 0 #toggle raw_input after the first layer with gemm ops

        layers[layer_name] = {"gemm_op": gemm_op, "deps": dependencies, "raw_input":raw_input, "layer_type": layer_node.layer_type}
    return layers

def calc_no_ops(layers):
    no_ops = 0
    for layer_name in layers:
        gemm_op = layers[layer_name]['gemm_op']
        if gemm_op is not None:
            X = gemm_op['input_size']
            W = gemm_op['weight_size']
            no_ops += (2 * X[0] * X[1] * W[1])
    return no_ops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='inception')
    parser.add_argument('--batch_size', type=int, required=False, default=1, help='Batch size')
    parser.add_argument('--sentence_len', type=int, required=False, default=100, help='Sentence length for transformer model')
    parser.add_argument('--imsize', type=int, required=False, default=299)  
    parser.add_argument('--array_size', type=int, nargs='+', required=False, default=[32,32], help='Array size')
    parser.add_argument('--out_dir', type=str, required=False, default="experiments/tmp")
    parser.add_argument('--partition_size', type=int, nargs='+', required=False, default=None)

    args = parser.parse_args()

    model_name = args.model
    batch_size = args.batch_size
    sentence_len = args.sentence_len
    
    array_size = args.array_size

    imsize = args.imsize
    partition_size = args.partition_size
    out_dir = args.out_dir

    bm = get_benchmarks(model_name, batch_size, imsize, sentence_len)
    if bm.model_type == "BERT":
        model = bm.get_keras_model(no_layers=1)
        no_repeat = bm.no_layers
    else:
        model = bm.get_keras_model() 
        no_repeat = 1

    layers = precompile_model(model, array_size=array_size, partition_size=partition_size)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir+"/precompiled_model.json", "w") as outfile:  
        json.dump({"args":args.__dict__, "order":list(layers.keys()), "layers":layers, "no_repeat":no_repeat}, outfile)

    print("precompiled model is saved at: {}".format(out_dir))

if __name__ == "__main__":
    main()