
import argparse

from mobile_cv.model_zoo.models.fbnet_v2 import fbnet

from convert_keras import build_fbnet
from sosa_sim.precompile import precompile_model, calc_no_ops
from sosa_sim.sim_binder import run_csim
from sosa_sim.hw import HardwareModel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



parser = argparse.ArgumentParser()
args = parser.parse_args()
args.clk_freq = 1e9

hw_model = HardwareModel(hw_type='systolic', batch_size=64, memory_bw=80, array_size=[128,128], device='cpu', args=args)

model_name = "FBNetV2_F4"

torch_model = fbnet(model_name, pretrained=True)

keras_model = build_fbnet(torch_model)

layers = precompile_model(keras_model, array_size=[128,128], partition_size=None)
no_ops = calc_no_ops(layers)

json_out = {"args":args.__dict__, "model1":{"order":list(layers.keys()), "layers":layers, "no_repeat":1, "no_ops":no_ops}}

sim_res = run_csim(json_out)

csim_runtime = sim_res['no_cycles'] * 1e-9 * 1e3
throughput = sim_res['no_ops'] / (csim_runtime/1e3)
csim_util = throughput / (2*hw_model.peak_throughput())

no_params = count_parameters(torch_model)

print("runtime: {} ms\t util: {}\t no_params:{}\t no_ops: {}".format(csim_runtime, csim_util, no_params, sim_res['no_ops']))

