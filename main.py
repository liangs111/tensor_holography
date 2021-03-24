import os
import cv2
import optics
import argparse
import tfrecord
import numpy as np
import tensorflow as tf
from util import *


class TensorHolographyModel():
    def __init__(self,
                 hologram_params,
                 training_params,
                 model_params,
                 loss_params,
                 path_params,
                 train_dataset_params,
                 test_dataset_params,
                 validate_dataset_params):

        self.hologram_params = hologram_params
        self.training_params = training_params
        self.model_params = model_params
        self.loss_params = loss_params
        self.path_params = path_params
        self.train_dataset_params = train_dataset_params
        self.test_dataset_params = test_dataset_params
        self.validate_dataset_params = validate_dataset_params
        self.model_vars = None

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        # build tfrecord if path_params["gen_record"] is true
        if path_params["gen_record"]:
            generator = tfrecord.TFRecordGeneratorforTH(self.path_params["train_output_path"],
                                            self.path_params["labels"],
                                            self.path_params["train_source_paths"])
            generator.generate_record()
            generator.update_record_paths(self.path_params["test_output_path"],
                                            self.path_params["labels"],
                                            self.path_params["test_source_paths"])
            generator.generate_record()
            generator.update_record_paths(self.path_params["validate_output_path"],
                                            self.path_params["labels"],
                                            self.path_params["validate_source_paths"])
            generator.generate_record()
            print("tfrecord generation done!") 
            

    def _build_model_vars(self):
        fw   = np.full((self.model_params["num_layers"]), self.model_params["filter_width"], dtype=int)
        fnum = np.append(np.full((self.model_params["num_layers"]-1), self.model_params["num_filters_per_layer"], dtype=int), 
                         self.model_params["output_dim"]*(self.model_params["interleave_rate"]**2))
        model_vars = {}

        for i in range(self.model_params["num_layers"]):    
            # first layer
            if i==0:  
                in_dim, out_dim = self.model_params["input_dim"] * (self.model_params["interleave_rate"]**2), fnum[i]
            # last layer
            elif i==self.model_params["num_layers"]-1:
                in_dim, out_dim = fnum[i-1] + self.model_params["input_dim"] * (self.model_params["interleave_rate"]**2),  \
                                  self.model_params["output_dim"] * (self.model_params["interleave_rate"]**2)
            else:
                in_dim, out_dim = fnum[i-1], fnum[i]  

            model_vars[i] = {'weights':tf_init_weights([fw[i], fw[i], in_dim, out_dim], 
                                                        'xavier',
                                                        xavier_params=(in_dim, out_dim),
                                                        r=self.model_params["weight_var_scale"]),
                            'bias':tf.Variable(tf.random.truncated_normal([out_dim],stddev=self.model_params["bias_stddev"]))
                            }

        return model_vars


    def _build_graph(self, x_in, model_vars, data_format='NCHW'):
        layers = {}
        prev_layers = {}

        # build layers
        print("input data:", x_in.shape)
        if self.model_params["renormalize_input"]:
            x_in = x_in - 0.5

        # interleave the input
        if data_format == 'NCHW':
            x_in = tf_interleave_nonnative(self.model_params["interleave_rate"], x_in)
        elif self.model_params["interleave_rate"] != 1:
            # update this for NHWC
            raise Exception('data_format has to be NCHW for interleave')

        # build graph
        for i in range(self.model_params["num_layers"]):
            if i==0:
                prev_layers[i] = x_in   
            elif (i<3) or (i%2==0): 
                prev_layers[i] = layers[i-1]
            else: 
                prev_layers[i] = layers[i-1] + prev_layers[i-2]
                print('(skip connection: %d, %d)'%(i-1, i-3))
            
            if i == self.model_params["num_layers"]-1:
                prev_layers[i] = tf.concat([prev_layers[i], x_in], axis = 1 if data_format == 'NCHW' else 3)

            if not i == self.model_params["num_layers"]-1:
                layers[i] = self.model_params["activation_func"](
                                tf.layers.batch_normalization(
                                    tf.nn.bias_add(
                                        tf.nn.conv2d(prev_layers[i],model_vars[i]['weights'],strides=[1,1,1,1], padding='SAME', data_format=data_format),
                                        model_vars[i]['bias'], 
                                        data_format=data_format
                                    ),
                                    axis=1 if data_format == 'NCHW' else 3
                                )
                            )
                print("layer %d:" % i, layers[i].shape)   
            else:
                # last layer
                field = self.model_params["output_activation_func"](
                            tf.layers.batch_normalization(
                                tf.nn.bias_add(
                                    tf.nn.conv2d(prev_layers[i],model_vars[i]['weights'],strides=[1,1,1,1], padding='SAME', data_format=data_format), 
                                    model_vars[i]['bias'], 
                                    data_format=data_format
                                ),
                                axis=1 if data_format == 'NCHW' else 3
                            ),
                            name='field'
                        )
                if data_format == 'NCHW':
                    field = tf_deinterleave_nonnative(self.model_params["interleave_rate"], field) 
                elif self.model_params["interleave_rate"] != 1:
                    raise Exception('data_format has to be NCHW for interleave')

                # normalize amplitude to [0, sqrt(2)]
                # normalize phase to [0, 1]
                # compute complex field, the phase is renormalized to [-pi, pi]
                if data_format=='NCHW':
                    amp = tf.add(field[:,:self.model_params["output_dim"]//2,:,:]*np.sqrt(0.5), np.sqrt(0.5), name="output_field_amp")
                    phs = tf.add(field[:,self.model_params["output_dim"]//2:,:,:]*0.5, 0.5, name="output_field_phs")
                else:
                    amp = tf.strided_slice(field, [0,0,0,0], [1, field.shape[1], field.shape[2], 3], [1,1,1,1], shrink_axis_mask=1)
                    phs = tf.strided_slice(field, [0,0,0,3], [1, field.shape[1], field.shape[2], 6], [1,1,1,1], shrink_axis_mask=1)
                    amp = tf.add(amp*np.sqrt(0.5), np.sqrt(0.5), name="output_field_amp")
                    phs = tf.add(phs*0.5, 0.5, name="output_field_phs")
                out_field = optics.tf_compl_val(amp, (phs-0.5)*2.0*np.pi, name="out_field")
                
                print("output tensor:", out_field.shape)

        return out_field, amp, phs


    def _get_dataset_iterators(self):
        # train iterator
        extractor = tfrecord.TFRecordExtractorforTH(self.path_params["train_output_path"],
                                                    self.train_dataset_params,
                                                    self.path_params["labels"])
        train_iterator = extractor.build_dataset()
        extractor.update_record_path(self.path_params["test_output_path"],
                                     self.test_dataset_params,
                                     self.path_params["labels"])
        test_iterator = extractor.build_dataset()
        extractor.update_record_path(self.path_params["validate_output_path"],
                                     self.validate_dataset_params,
                                     self.path_params["labels"])
        validate_iterator = extractor.build_dataset()
        return train_iterator, test_iterator, validate_iterator


    def _preprocess_input(self, example):
        # load RGB-D input
        rgbd = tf.concat([example["img"], example["depth"]], 1)

        # create complex hologram
        holo = optics.tf_compl_val(example["amp"], (example["phs"]-0.5) * 2 * np.pi)
        return rgbd, holo, example["amp"], example["phs"]


    def _setup_train(self):
        # get train and test dataset handle
        train_iterator, test_iterator, validate_iterator = self._get_dataset_iterators()
        train_handle = self.sess.run(train_iterator.string_handle())
        test_handle  = self.sess.run(test_iterator.string_handle()) 
        validate_handle  = self.sess.run(validate_iterator.string_handle()) 

        # create feedable handle
        handle   = tf.compat.v1.placeholder(tf.string, shape=[])
        iterator = tf.compat.v1.data.Iterator.from_string_handle(handle, output_types=tf.compat.v1.data.get_output_types(train_iterator))
        example  = iterator.get_next()

        # get data placeholder
        rgbd, holo_in, amp_in, phs_in = self._preprocess_input(example)

        # build model
        self.model_vars = self._build_model_vars()
        holo_out, amp_out, phs_out = self._build_graph(rgbd, self.model_vars)

        return train_handle, test_handle, validate_handle, handle, rgbd, holo_in, amp_in, phs_in, holo_out, amp_out, phs_out


    def validate(self):
        _, _, validate_handle, handle, rgbd, holo_in, amp_in, phs_in, holo_out, amp_out, phs_out = self._setup_train()

        # compute psnr and ssim on the amplitude map 
        ssim_amp_loss = tf.reduce_mean(tf.image.ssim(tf.transpose(amp_out, [0,2,3,1]), tf.transpose(amp_in, [0,2,3,1]), 1.0))
        psnr_amp_loss = tf.reduce_mean(tf.image.psnr(tf.transpose(amp_out, [0,2,3,1]), tf.transpose(amp_in, [0,2,3,1]), 1.0))

        # create model saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=5)

        # initialize variables
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)

        # reload pre-trained model
        if self.training_params["restore_trained_model"]:
            ckpt = tf.train.get_checkpoint_state(self.path_params["ckpt_parent_path"])
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path_params["ckpt_parent_path"]))
                print("model restored from %s" % self.path_params["ckpt_parent_path"])
            else:
                print("ERROR: NO RESTORED MODEL...")

        # test on validation dataset
        validate_steps = self.validate_dataset_params["sample_count"] // self.validate_dataset_params["batch"]
        avg_ssim_amp_loss = []
        avg_psnr_amp_loss = []
        for v_step in range(validate_steps):
            print("validate step %d/%d\n" % (v_step, validate_steps))
            ssim_amp_loss_val, psnr_amp_loss_val \
                 = self.sess.run([ssim_amp_loss, psnr_amp_loss], feed_dict={handle: validate_handle})
            avg_ssim_amp_loss.append(ssim_amp_loss_val)
            avg_psnr_amp_loss.append(psnr_amp_loss_val)

        print("validation results: ssim amp loss = %f(%f; %f/%f), pnsr amp loss = %f(%f; %f/%f)\n" % \
            (np.mean(avg_ssim_amp_loss), np.std(avg_ssim_amp_loss), np.amax(avg_ssim_amp_loss), np.amin(avg_ssim_amp_loss),
             np.mean(avg_psnr_amp_loss), np.std(avg_psnr_amp_loss), np.amax(avg_psnr_amp_loss), np.amin(avg_psnr_amp_loss)))


    def evaluate(self, eval_params):
        # define placeholder for input rgbd images
        rgbd = tf.compat.v1.placeholder(tf.float32, [1, self.model_params["input_dim"], eval_params['res_h'], eval_params['res_w']])

        # define wave propagator
        propagator = optics.tf_propagator(
            (eval_params['res_h'], eval_params['res_w']),
            self.hologram_params["pitch"],
            self.hologram_params["wavelengths"],
            method="as",
            double_pad=eval_params['double_pad']
        )

        # build network
        self.model_vars = self._build_model_vars()
        _, amp_out, phs_out = self._build_graph(rgbd, self.model_vars)
        
        # center phase for each color channel
        phs_out = phs_out - tf.reduce_mean(phs_out, [2,3], keepdims=True) + 0.5

        # compute phase-only hologram
        cpx_out = optics.tf_compl_val(amp_out, (phs_out-0.5) * 2.0 * np.pi)
        if eval_params["use_maimone_dpm"]:
            phs_only, amp_max = optics.tf_dpm_maimone(cpx_out, 
                                                      propagator, 
                                                      depth_shift=eval_params["depth_shift"],
                                                      adaptive_phs_shift=eval_params['adaptive_phs_shift'],
                                                      batch=1, 
                                                      num_channels=3, 
                                                      res_h=eval_params['res_h'], 
                                                      res_w=eval_params['res_w'],
                                                      axis=3,
                                                      phs_max=eval_params['phs_max'],
                                                      amp_max=None, 
                                                      clamp=True,
                                                      normalize=True,
                                                      wavelength=self.hologram_params["wavelengths"])
        else:
            phs_only, amp_max = optics.tf_aadpm(cpx_out, 
                                                propagator, 
                                                depth_shift=eval_params["depth_shift"],
                                                adaptive_phs_shift=eval_params['adaptive_phs_shift'],
                                                batch=1, 
                                                num_channels=3, 
                                                res_h=eval_params['res_h'], 
                                                res_w=eval_params['res_w'],
                                                sigma=eval_params['gaussian_sigma'], 
                                                kernel_width=eval_params['gaussian_width'], 
                                                phs_max=eval_params['phs_max'],
                                                amp_max=None, 
                                                clamp=True,
                                                normalize=True,
                                                wavelength=self.hologram_params["wavelengths"])

        # restore pre-trained model
        self.saver = tf.compat.v1.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(self.path_params["ckpt_parent_path"])
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path_params["ckpt_parent_path"]))
            print("model restored from %s" % self.path_params["ckpt_parent_path"])
        else:
            print("ERROR: NO RESTORED MODEL...")      

        # load input
        rgb   = np.transpose(cv2.resize(cv2.imread(eval_params['rgb_path']), 
            dsize=(eval_params['res_w'], eval_params['res_h']), interpolation=cv2.INTER_CUBIC), [2,0,1])[:3,:,:] / 255.0
        depth = cv2.resize(cv2.imread(eval_params['depth_path']), 
            dsize=(eval_params['res_w'], eval_params['res_h']), interpolation=cv2.INTER_CUBIC) / 255.0
        if len(depth.shape) == 3:
            depth = depth[:,:,0]
        depth = depth[None,:,:]
        rgbd_val = np.concatenate((rgb, depth), axis=0)
        rgbd_val = rgbd_val[None,:,:,:].astype(np.float32)

        # evaluate output
        amp_out_val, phs_out_val, phs_only_val= self.sess.run([amp_out, 
                                                               phs_out, 
                                                               phs_only], 
                                                               feed_dict={rgbd: rgbd_val})
        amp_out_val = np.transpose(amp_out_val[0,:,:,:], [1,2,0])
        phs_out_val = np.transpose(phs_out_val[0,:,:,:], [1,2,0])
        phs_only_val = np.transpose(phs_only_val[0,:,:,:], [1,2,0])


        # save output
        cv2.imwrite(os.path.join(eval_params["output_path"], "phs.png"), phs_out_val * 255.0)
        cv2.imwrite(os.path.join(eval_params["output_path"], "amp.png"), amp_out_val * 255.0)
        cv2.imwrite(os.path.join(eval_params["output_path"], "blue.png"), phs_only_val[:,:,0] * 255.0)
        cv2.imwrite(os.path.join(eval_params["output_path"], "green.png"), phs_only_val[:,:,1] * 255.0)
        cv2.imwrite(os.path.join(eval_params["output_path"], "red.png"), phs_only_val[:,:,2] * 255.0)

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    # evaluation parameters
    parser.add_argument('--eval-mode', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--eval-res-h', default=1080, type=int, help='Input image height in evaluation mode')
    parser.add_argument('--eval-res-w', default=1920, type=int, help='Input image width in evaluation mode')
    parser.add_argument('--eval-rgb-path', default=os.path.join(cur_dir, "data", "example_input", "couch_rgb.png"), help='Input rgb image path in evaluation mode')
    parser.add_argument('--eval-depth-path', default=os.path.join(cur_dir, "data", "example_input", "couch_depth.png"), help='Input depth image path in evaluation mode')
    parser.add_argument('--eval-output-path', default=os.path.join(cur_dir, "data", "example_input"), help='Output directory for results')
    parser.add_argument('--eval-depth-shift', default=0, type=float, help='Depth shift (in mm) from the predicted midpoint hologram to the target hologram plane')
    parser.add_argument('--gaussian-sigma', default=0.7, type=float, help='Sigma of Gaussian kernel used by AA-DPM')
    parser.add_argument('--gaussian-width', default=3, type=int, help='Width of Gaussian kernel used by AA-DPM')
    parser.add_argument('--phs-max', default=3.0, type=float, help='Maximum phase modulation of SLM in unit of pi')
    parser.add_argument('--use-maimone-dpm', action='store_true', help='Use DPM of Maimone et al. 2017')
    
    # dataset parameters
    parser.add_argument('--dataset-res', default=384, type=int, help='dataset image resolution')
    parser.add_argument('--pitch', default=0.008, type=float, help='pixel pitch in mm')

    # model parameters
    parser.add_argument('--num-filters-per-layer', default=24, type=int, help='Number of filters per layer')
    parser.add_argument('--num-layers', default=30, type=int, help='Number layers')

    # validation parameters
    parser.add_argument('--validate-mode', action='store_true', help='Run in validation mode')

    opt = parser.parse_args()

    # fix random seed
    tf.compat.v1.set_random_seed(0)

    # hologram parameters, units in mm
    hologram_params = {
        "wavelengths" : np.array([0.000450, 0.000520, 0.000638]),  # laser wavelengths in BGR order
        "pitch" : opt.pitch,                                           # hologram pitch
        "res_h" : opt.dataset_res,                                 # dataset image height
        "res_w" : opt.dataset_res,                                 # dataset image width
        "depth_base"  : -3,                                        # input hologram plane (midpoint hologram)
        "depth_scale" : 6,                                         # 3D volume depth
        "double_pad": True                                         # double padding for propagation
    }

    # training parameters
    training_params = {
        "restore_trained_model": True,                             # flag to restore pre-trained model
        "batch" : 2,                                               # training batch 
        "num_epochs": 1000,                                        # training epochs
        "decay_type": None,                                        # learning rate decay
        "decay_params": None,                                      # learning decay parameters
        "learning_rate" : 1e-4,                                    # learning rate 
        "optimizer_type" : "adam",                                 # optimizer type 
        "optimizer_params" : 
            {"beta1":0.9, "beta2":0.99, "epsilon":1e-8},           # optimizer parameters                      
        "num_iter_per_model_save": 1000,                           # number of iterations per model save
        "num_iter_per_test": 1000,                                 # number of iterations per validation
        "num_top_depth_for_img_loss": 15,                          # number of top-k depths for computing focal stack loss
        "num_random_depth_for_img_loss": 5,                        # number of random depths selected from rest of the bins
        "depth_dependent_weight_scale": 0.35,                      # attention weight
        "num_hist_bins": 200                                       # number of focal stack bins
    }

    # model parameters
    model_params = {
        "name": "full_loss",                                       # model name
        "input_dim": 4,                                            # RGBD
        "output_dim": 6,                                           # amplitude+phase
        "num_layers": opt.num_layers,                              # number of convolution layers
        "interleave_rate": 1,                                      # interleaving rate (default to no interleaving)
        "num_filters_per_layer": opt.num_filters_per_layer,        # number of filters per convolution layer
        "filter_width": 3,                                         # filter width
        "bias_stddev": 0.01,                                       # bias standard deviation (for model initialization)
        "weight_var_scale": 0.25,                                  # weight variance (for model initialization)
        "renormalize_input": True,                                 # normalize input to [-0.5, 0.5]
        "activation_func": tf.nn.relu,                             # activation function for intermediate layers
        "output_activation_func": tf.nn.tanh,                      # activation function for the output layer
    }
    
    # loss function parameters
    use_l2_loss = False
    num_imgs_in_fs = training_params["num_top_depth_for_img_loss"] + training_params["num_random_depth_for_img_loss"]
    loss_params = {
        "use_l2_loss":   use_l2_loss,                              # activation function for the output layer
        "loss_op":       tf.losses.mean_squared_error if use_l2_loss else tf.compat.v1.losses.absolute_difference,
        "weight_holo":   1.0,                                      # hologram loss weight
        "weight_fs":     num_imgs_in_fs,                           # focal stack loss weight
        "weight_fs_tv":  num_imgs_in_fs,                           # focal stack tv loss weight
    }
    
    # path parameters
    labels = ["amp", "phs", "img", "depth"]
    train_base_path = os.path.join(cur_dir, "data", "train_%d" % opt.dataset_res)
    test_base_path = os.path.join(cur_dir, "data", "test_%d" % opt.dataset_res)
    validate_base_path = os.path.join(cur_dir, "data", "validate_%d" % opt.dataset_res)
    checkpoint_base_path = os.path.join(cur_dir, 'model', "ckpt_%s_pitch_%d_layers_%d_filters_%d" % 
        (model_params["name"], hologram_params["pitch"]*1000, model_params["num_layers"], model_params["num_filters_per_layer"]))
    path_params = {
        "gen_record" : False,                                                           # generate tf record
        "labels" : labels,                                                              # image labels in the tfrecord
        "train_output_path"    : os.path.join(train_base_path, "train.tfrecord"),       # path to training set tfrecord
        "train_source_paths"   : [os.path.join(train_base_path, x) for x in labels],    # path to training set raw images 
        "test_output_path"     : os.path.join(test_base_path, "test.tfrecord"),         # path to test set tfrecord 
        "test_source_paths"    : [os.path.join(test_base_path, x) for x in labels],     # path to test set raw images
        "validate_output_path" : os.path.join(validate_base_path, "validate.tfrecord"), # path to validate set tfrecord
        "validate_source_paths": [os.path.join(validate_base_path, x) for x in labels], # path to validation set raw images
        "ckpt_path"            : os.path.join(checkpoint_base_path, "ckpt"),            # checkpoint path
        "ckpt_parent_path"     : checkpoint_base_path,                                  # checkpoint parent path 
        "epoch_record_file"    : "epoch_record.txt",                                    # txt file that records how many epochs have been trained
        "log_path"             : "logs",                                                # log file
    }
    
    # training dataset parameters
    train_dataset_params = {
        "repeat": True,
        "sample_count": 3800,
        "batch": training_params["batch"],
        "res_h": hologram_params["res_h"],
        "res_w": hologram_params["res_w"],
        "prefetch_buffer_size": 2,
        "num_parallel_calls": 4,
        "shuffle_buffer_size": 2,
        "num_epochs": training_params["num_epochs"]
    }

    # test dataset parameters
    test_dataset_params = {
        "repeat": True,
        "sample_count": 100,
        "batch": training_params["batch"],
        "res_h": hologram_params["res_h"],
        "res_w": hologram_params["res_w"],
        "num_parallel_calls": 2,
        "prefetch_buffer_size": 4,
        "shuffle_buffer_size": 2,
        "num_epochs": training_params["num_epochs"]
    }

    # validation dataset parameters
    validate_dataset_params = {
        "repeat": True,
        "sample_count": 100,
        "batch": training_params["batch"],
        "res_h": hologram_params["res_h"],
        "res_w": hologram_params["res_w"],
        "num_parallel_calls": 2,
        "prefetch_buffer_size": 4,
        "shuffle_buffer_size": 2,
        "num_epochs": training_params["num_epochs"]
    }

    # build model
    tensor_holo_model = TensorHolographyModel(hologram_params=hologram_params,
                                              training_params=training_params,
                                              model_params=model_params,
                                              loss_params=loss_params,
                                              path_params=path_params,
                                              train_dataset_params=train_dataset_params,
                                              test_dataset_params=test_dataset_params,
                                              validate_dataset_params=validate_dataset_params)
    
    # train model from scratch (will be available in the second release)
    # tensor_holo_model.train()
    
    # validate pre-trained model 
    if opt.validate_mode:
        tensor_holo_model.validate()

    # evaluate pre-trained model
    if opt.eval_mode:
        eval_params = {
            "res_h": opt.eval_res_h,
            "res_w": opt.eval_res_w,
            "rgb_path": opt.eval_rgb_path,
            "depth_path": opt.eval_depth_path,
            "output_path": opt.eval_output_path,
            "double_pad": True,
            "use_maimone_dpm": opt.use_maimone_dpm,
            "adaptive_phs_shift": False,
            "depth_shift": opt.eval_depth_shift,
            "gaussian_sigma": opt.gaussian_sigma,
            "gaussian_width": opt.gaussian_width,
            "phs_max": [opt.phs_max*np.pi] * 3,
            "amp_max": None,
        }
        tensor_holo_model.evaluate(eval_params)
