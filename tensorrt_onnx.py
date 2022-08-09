import os
import cv2
import time
import argparse
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
                 
def GiB(val):
    return val * 1 << 30

def exec_inference(context, bindings, inputs, outputs, stream):
    # transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # synchronize the stream
    stream.synchronize()
    # return only the host outputs.
    return [out.host for out in outputs]

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
    
class TrtEngineOnnx():
    def __init__(self, trt_params):
        self.trt_params = trt_params
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.DTYPE = trt.float32
    
    def build_engine_onnx(self):
        self.builder = trt.Builder(self.TRT_LOGGER)
        self.network = self.builder.create_network(self.EXPLICIT_BATCH)
        self.config = self.builder.create_builder_config()
        if self.trt_params["fp_16"]:
            self.config.flags = 1 << int(trt.BuilderFlag.FP16)
        self.config.max_workspace_size = GiB(1)
        self.parser = trt.OnnxParser(self.network, self.TRT_LOGGER)
        
        if os.path.exists(self.trt_params["trt_engine_path"]):
            with open(self.trt_params["trt_engine_path"], "rb") as f, \
                trt.Runtime(self.TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            # load the ONNX model and parse it in order to populate the TensorRT network.
            with open(self.trt_params["onnx_model_path"], 'rb') as model:
                if not self.parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(self.parser.num_errors):
                        print (self.parser.get_error(error))
                    return None
            self.engine = self.builder.build_engine(self.network, self.config)
            with open(self.trt_params["trt_engine_path"], "wb") as f:
                f.write(self.engine.serialize())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

    def benchmark_perf(self):
        dummy_rgbd = np.zeros((1, 
                               4, 
                               self.trt_params["res_h"], 
                               self.trt_params["res_w"]), 
                              dtype=trt.nptype(self.DTYPE)).ravel()
        
        np.copyto(self.inputs[0].host, dummy_rgbd)
        
        for i in range(100):
            t1 = time.time()
            trt_outputs = exec_inference(self.context, 
                                     bindings=self.bindings, 
                                     inputs=self.inputs, 
                                     outputs=self.outputs,
                                     stream=self.stream)
            print("inference time: %f FPS" % (1 / (time.time() - t1)))
        
    
    def evaluate(self, eval_rgb_path, eval_depth_path, eval_output_path):
        rgb   = np.transpose(cv2.resize(cv2.imread(eval_rgb_path), 
                                        (trt_params["res_w"], trt_params["res_h"]), 
                                        interpolation=cv2.INTER_CUBIC), 
                             (2, 0, 1))[:3,:,:] / 255.0
        depth = cv2.resize(cv2.imread(eval_depth_path), 
                           (trt_params["res_w"], trt_params["res_h"]), 
                           interpolation=cv2.INTER_CUBIC) / 255.0
        if len(depth.shape) == 3:   
            depth = depth[:,:,0]
        depth = depth[None,:,:]
        rgbd_val = np.concatenate((rgb, depth), axis=0)
        rgbd_val = rgbd_val[None,:,:,:].astype(trt.nptype(self.DTYPE)).ravel()
        
        np.copyto(self.inputs[0].host, rgbd_val)
        trt_outputs = exec_inference(self.context, 
                                     bindings=self.bindings, 
                                     inputs=self.inputs, 
                                     outputs=self.outputs, 
                                     stream=self.stream)
        
        phs = np.transpose(trt_outputs[0].reshape((3, trt_params["res_h"], trt_params["res_w"])), [1,2,0])
        amp = np.transpose(trt_outputs[1].reshape((3, trt_params["res_h"], trt_params["res_w"])), [1,2,0])
        cv2.imwrite(os.path.join(eval_output_path, "holo_amp.png"), amp * 255.0)
        cv2.imwrite(os.path.join(eval_output_path, "holo_phs.png"), phs * 255.0)
    
if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default="full_loss", type=str, help='Model name')
    parser.add_argument('--pitch', default=0.008, type=float, help='pixel pitch in mm')
    parser.add_argument('--num-layers', default=30, type=int, help='Number layers')
    parser.add_argument('--num-filters-per-layer', default=24, type=int, help='Number of filters per layer')
    parser.add_argument('--trt-res-h', default=1080, type=int, help='Input image height')
    parser.add_argument('--trt-res-w', default=1920, type=int, help='Input image width')
    parser.add_argument('--fp-16', action='store_true', help='Input image width in export (tensorrt) mode')
    
    parser.add_argument('--benchmark-mode', action='store_true', help='Benchmark performance')
    
    parser.add_argument('--v2', action='store_true', help='Use v2 model')
    parser.add_argument('--active-max-ldi-layer', default=0, type=int, help='Active max LDI layer')
    parser.add_argument('--activate-ddpm', action='store_true', help='Load ddpm network together with hologram rendering network; depth shift specified by --train-depth-shift')
    parser.add_argument('--bypass-ddpm-network', action='store_true', help='Train/evaluate ddpm without using ddpm network (typical for 0 mm offset)')
    
    parser.add_argument('--eval-mode', action='store_true', help='Benchmark performance')
    parser.add_argument('--eval-rgb-path', default=os.path.join(cur_dir, "data", "example_input", "couch_rgb.png"), help='Input rgb image path in evaluation mode')
    parser.add_argument('--eval-depth-path', default=os.path.join(cur_dir, "data", "example_input", "couch_depth.png"), help='Input depth image path in evaluation mode')
    parser.add_argument('--eval-output-path', default=os.path.join(cur_dir, "data", "example_input"), help='Output directory for results')
    
    opt = parser.parse_args()
    
    trt_params = {
        "name": opt.model_name,                                    # model name
        "pitch": opt.pitch,                                        # pixel pitch in mm
        "num_layers": opt.num_layers,                              # number of layers
        "num_filters_per_layer": opt.num_filters_per_layer,        # number of filters per layer
        "res_h": opt.trt_res_h,                                    # input image height
        "res_w": opt.trt_res_w,                                    # input image width
        "fp_16": opt.fp_16,                                        # half-precision float mode for the use of Tensor Cores
    }
    
    postfix_model = "" if not opt.v2 else "_ldi_%d%s%s" % (
        opt.active_max_ldi_layer,
        "_ddpm_0" if opt.activate_ddpm else "",
        "_bypass" if opt.bypass_ddpm_network else ""
    )
    
    trt_params["model_base_path"] = os.path.join(cur_dir, 'model', "ckpt_%s_pitch_%d_layers_%d_filters_%d" % 
        (trt_params["name"], 
         trt_params["pitch"]*1000, 
         trt_params["num_layers"], 
         trt_params["num_filters_per_layer"]) + postfix_model)

    postfix_onnx = "_v2" if opt.v2 else ""
    trt_params["onnx_model_path"] = os.path.join(trt_params["model_base_path"], "inference_graph%s.onnx" % postfix_onnx)
    trt_params["trt_engine_path"] = os.path.join(trt_params["model_base_path"], "inference_graph%s%s.engine" % 
                                                 (postfix_onnx, "_fp16" if opt.fp_16 else "_fp32"))
    
    # build TensorRT engine
    trt_engine = TrtEngineOnnx(trt_params)
    trt_engine.build_engine_onnx()
    
    # run in benchmark mode
    if opt.benchmark_mode:
        trt_engine.benchmark_perf()
    
    # run in evaluation mode
    if opt.eval_mode:
        trt_engine.evaluate(opt.eval_rgb_path, opt.eval_depth_path, opt.eval_output_path)