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
        # Create dictionaries of filter widths and number of filters
        fw = {i: self.model_params["filter_width"] for i in range(self.model_params["num_layers"])}
        fnum = {i: self.model_params["num_filters_per_layer"] if i < self.model_params["num_layers"] - 1 else 
               self.model_params["output_dim"] * (self.model_params["interleave_rate"]**2) for i in range(self.model_params["num_layers"])}
        # Initialize model variables dictionary
        model_vars = {}

        # Iterate through each layer
        for i, (width, num_filters) in enumerate(zip(fw.values(), fnum.values())):    
            # First layer
            if i == 0:  
                in_dim, out_dim = self.model_params["input_dim"] * (self.model_params["interleave_rate"]**2), num_filters
            # Last layer
            elif i == self.model_params["num_layers"]-1:
                in_dim, out_dim = fnum[i-1] + self.model_params["input_dim"] * (self.model_params["interleave_rate"]**2), \
                                  self.model_params["output_dim"] * (self.model_params["interleave_rate"]**2)
            # Intermediate layers
            else:
                in_dim, out_dim = fnum[i-1], num_filters  

            # Initialize weights and biases for the current layer
            model_vars[i] = {
                'weights': tf_init_weights([width, width, in_dim, out_dim], 
                                           'xavier',
                                           xavier_params=(in_dim, out_dim),
                                           r=self.model_params["weight_var_scale"]),
                'bias': tf.Variable(tf.random.truncated_normal([out_dim], stddev=self.model_params["bias_stddev"]))
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


    def _get_loss(self,
                  y_out,
                  y_out_amp,
                  y_out_phs,
                  y_gt,
                  y_gt_amp,
                  y_gt_phs,
                  rgbd,  
                  propagator):
        
        # compute total variation 
        def compute_tv_4d(field):
            dx = field[:, :, :, 1:] - field[:, :, :, :-1]
            dy = field[:, :, 1:, :] - field[:, :, :-1, :]
            return dx, dy

        # compute total variation loss
        def compute_tv_loss(x_in, x_gt):
            x_in_dx, x_in_dy   = compute_tv_4d(x_in)
            x_out_dx, x_out_dy = compute_tv_4d(x_gt)
            tv_loss = 0.5 * tf.reduce_mean(self.loss_params["loss_op"](labels=x_in_dx, predictions=x_out_dx)) + \
                      0.5 * tf.reduce_mean(self.loss_params["loss_op"](labels=x_in_dy, predictions=x_out_dy))
            return tv_loss
        
        # get depth dependent weight for the perceptual image
        def get_depth_dependent_weight(depth, depth_to_focus, depth_diff_max):
            depth_diff = (depth_diff_max - tf.abs(depth - depth_to_focus)) * self.training_params["depth_dependent_weight_scale"]
            depth_weight = tf.exp(depth_diff)
            # normalize weight to have max 1
            depth_weight /= tf.reduce_max(depth_weight)
            return depth_weight

        # compute perceptual image loss at given depth
        def get_img_diff_at_depth(y_out, y_gt, depth, depth_to_focus, depth_diff_max):
            img_gt  = tf.abs(propagator(y_gt, -depth_to_focus))
            img_out = tf.abs(propagator(y_out, -depth_to_focus))
            depth_weight = get_depth_dependent_weight(depth, depth_to_focus, depth_diff_max)
            weighted_img_gt = img_gt*depth_weight
            weighted_img_out = img_out*depth_weight
            img_loss = tf.reduce_mean(self.loss_params["loss_op"](weighted_img_gt, weighted_img_out))
            tv_loss  = compute_tv_loss(weighted_img_gt, weighted_img_out)
            return img_loss, tv_loss, img_gt, img_out

        # split depth
        depth = rgbd[:,3,:,:]

        # compute depth to focus
            # 1. compute histogram, pick top ["num_top_depth_for_img_loss"] depth bins
            # 2. add random depth perturbations (smaller than a single bin width) to the top bins to avoid always optimizing for a particular depth
            # 3. randomly select ["num_random_depth_for_img_loss"] depth in the rest of the bins and add offsets
        for i in range(self.training_params["batch"]):
            hist = tf.histogram_fixed_width(depth[i,:,:], 
                                            value_range=(0, 1), 
                                            nbins=self.training_params["num_hist_bins"])
            idx = (tf.cast(tf.argsort(hist, direction="DESCENDING"), tf.float32) + tf.random.uniform([1], minval = 0, maxval=1.0)) / self.training_params["num_hist_bins"]
            top_depth  = idx[:self.training_params["num_top_depth_for_img_loss"]]
            rand_depth = tf.random.shuffle(idx[self.training_params["num_top_depth_for_img_loss"]:])[:self.training_params["num_random_depth_for_img_loss"]]
            if not i:
                # for the first element in batch, initialize depth_to_focus
                depth_to_focus = tf.concat([top_depth, rand_depth], axis = 0)
                depth_to_focus = depth_to_focus[None,:]
            else:
                # for following elements in batch, concat to depth_to_focus
                tmp = tf.concat([top_depth, rand_depth], axis = 0)
                tmp = tmp[None,:]
                depth_to_focus = tf.concat([depth_to_focus, tmp], axis=0)

        # scale depth, depth_to_focus, and add depth_base
        depth = depth[:,None,:,:] * self.hologram_params["depth_scale"] + self.hologram_params["depth_base"]
        depth_to_focus = depth_to_focus * self.hologram_params["depth_scale"] + self.hologram_params["depth_base"]

        # compute hologram loss
        y_gt_phs_scaled = (y_gt_phs-0.5) * 2.0 * np.pi
        y_out_phs_scaled = (y_out_phs-0.5) * 2.0 * np.pi
        phs_diff = tf.atan2(tf.sin(y_gt_phs_scaled - y_out_phs_scaled), tf.cos(y_gt_phs_scaled - y_out_phs_scaled))
        phs_diff -= tf.reduce_mean(phs_diff, [2,3], keepdims=True) # subtract global phase offset per color channel
        holo_loss = self.loss_params["loss_op"](y_gt_amp * tf.cos(phs_diff), y_out_amp) + \
                    self.loss_params["loss_op"](y_gt_amp * tf.sin(phs_diff), 0.)
        
        # compute focal stack loss
        fs_loss = 0.
        fs_tv_loss = 0.
        ssim_img_loss = 0.
        psnr_img_loss = 0.
        for i in range(self.training_params["batch"]):
            # slice example from batch
            depth_slice = depth[None,i,:,:,:]
            y_out_slice = y_out[None,i,:,:,:]
            y_gt_slice = y_gt[None,i,:,:,:]
            # compute perceptual loss at given depth
            for j in range(depth_to_focus.shape[1]):
                tmp_img_loss, tmp_tv_loss, img_gt, img_out = get_img_diff_at_depth(y_out_slice, y_gt_slice, 
                                                                   depth_slice, depth_to_focus[i,j], self.hologram_params["depth_scale"]
                                                                   )
                fs_loss += tmp_img_loss
                fs_tv_loss += tmp_tv_loss
                ssim_img_loss += tf.reduce_mean(tf.image.ssim(tf.transpose(img_gt, [0,2,3,1]), tf.transpose(img_out, [0,2,3,1]), 1.0))
                psnr_img_loss += tf.reduce_mean(tf.image.psnr(tf.transpose(img_gt, [0,2,3,1]), tf.transpose(img_out, [0,2,3,1]), 1.0))
    
        normalize_scale = tf.cast(tf.size(depth_to_focus), dtype=tf.float32)

        # normalize focal stack loss
        fs_loss /= normalize_scale
        fs_tv_loss /= normalize_scale

        # compose final loss
        loss = holo_loss   * self.loss_params["weight_holo"] + \
               fs_loss     * self.loss_params["weight_fs"] + \
               fs_tv_loss  * self.loss_params["weight_fs_tv"]

        ssim_amp_loss = tf.reduce_mean(tf.image.ssim(tf.transpose(y_gt_amp, [0,2,3,1]), tf.transpose(y_out_amp, [0,2,3,1]), 1.0))
        psnr_amp_loss = tf.reduce_mean(tf.image.psnr(tf.transpose(y_gt_amp, [0,2,3,1]), tf.transpose(y_out_amp, [0,2,3,1]), 1.0))
        ssim_img_loss /= normalize_scale
        psnr_img_loss /= normalize_scale

        # output perceptual image at top histogram bins
        return loss, ssim_amp_loss, ssim_img_loss, psnr_amp_loss, psnr_img_loss


    def _setup_optimizer(self,
                         starter_learning_rate,
                         decay_type,
                         decay_params,
                         opt_type,
                         opt_params,
                         global_step):
        """ Partially adapted from [Sitzmann et al. 2018]
        """
        if decay_type is not None:
            if decay_type == 'polynomial':
                learning_rate = tf.train.polynomial_decay(starter_learning_rate,
                                                          global_step,
                                                          **decay_params)
        else:
            learning_rate = starter_learning_rate

        
        opt_type = opt_type.lower()

        if opt_type == 'adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate,
                                               **opt_params)
        elif opt_type == 'sgd_with_momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                   **opt_params)
        elif opt_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                                   **opt_params)
        elif opt_type == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                                  **opt_params)
        else:
            raise Exception('Unknown opt type')

        return optimizer
    

    def train(self):
        # setup training
        train_handle, test_handle, _, handle, rgbd, holo_in, amp_in, phs_in, holo_out, amp_out, phs_out = self._setup_train()
        
        # create wave propagator
        propagator = optics.tf_propagator(
                (self.hologram_params["res_h"], self.hologram_params["res_w"]),
                self.hologram_params["pitch"],
                self.hologram_params["wavelengths"],
                method="as",
                double_pad=self.hologram_params["double_pad"]
            )

        # get loss
        loss, ssim_amp_loss, ssim_img_loss, psnr_amp_loss, psnr_img_loss = self._get_loss(holo_out,
                                                                                          amp_out,
                                                                                          phs_out,
                                                                                          holo_in,
                                                                                          amp_in,
                                                                                          phs_in,
                                                                                          rgbd,  
                                                                                          propagator)

        # setup optimizer
        global_step = tf.Variable(0, trainable=False)
        optimizer = self._setup_optimizer(starter_learning_rate=self.training_params["learning_rate"],
                                          decay_type=self.training_params["decay_type"],
                                          decay_params=self.training_params["decay_params"],
                                          opt_type=self.training_params["optimizer_type"],
                                          opt_params=self.training_params["optimizer_params"],
                                          global_step=global_step)
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

        # create model saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=5, save_relative_paths=True)

        # initialize variables
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)

        # restore trained model variable
        last_epoch = 0
        if self.training_params["restore_trained_model"]:
            ckpt = tf.train.get_checkpoint_state(self.path_params["ckpt_parent_path"])
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path_params["ckpt_parent_path"]))
                print("model restored from %s" % self.path_params["ckpt_parent_path"])
                last_epoch = self.sess.run(global_step) / (self.train_dataset_params["sample_count"]/self.training_params["batch"])
                last_epoch = int(last_epoch)
                print("last training ends at epoch %d" % (last_epoch))
            else:
                print("INFO: NO RESTORED MODEL, RETRAIN FROM SCRATCH")
        
        # start training
        print("Start the training!\n")
        iter = 0
        train_steps = self.train_dataset_params["sample_count"] // self.train_dataset_params["batch"]
        test_steps = self.test_dataset_params["sample_count"] // self.test_dataset_params["batch"]

        for epoch in range(last_epoch, self.training_params["num_epochs"]):
            print("start epoch %d\n" % (epoch))
            for t_step in range(train_steps):   
                _, loss_val, ssim_amp_val, = self.sess.run([train_op,
                                                            loss,
                                                            ssim_amp_loss],
                                                            feed_dict={handle: train_handle})

                print("Epoch %d, Step %d, total_loss %0.8f, ssim amp loss %0.8f\n" % \
                      (epoch, t_step, loss_val, ssim_amp_val))

                if np.isnan(loss_val) or np.isnan(ssim_amp_val):
                    print('Find nan in loss or prediction\n')
                    raise

                # test on validation dataset (test handle)
                if not iter % self.training_params["num_iter_per_test"] and iter > 0:
                    avg_loss = []
                    avg_ssim_amp_loss = []
                    avg_ssim_img_loss = []
                    avg_psnr_amp_loss = []
                    avg_psnr_img_loss = []
                    for v_step in range(test_steps):
                        print("test step %d/%d\n" % (v_step, test_steps))
                        data_loss_val, ssim_amp_loss_val, ssim_img_loss_val, psnr_amp_loss_val, psnr_img_loss_val = \
                            self.sess.run([loss, ssim_amp_loss, ssim_img_loss, psnr_amp_loss, psnr_img_loss], 
                                          feed_dict={handle: test_handle})
                        avg_loss.append(data_loss_val)
                        avg_ssim_amp_loss.append(ssim_amp_loss_val)
                        avg_ssim_img_loss.append(ssim_img_loss_val)
                        avg_psnr_amp_loss.append(psnr_amp_loss_val)
                        avg_psnr_img_loss.append(psnr_img_loss_val)
                        print('''test results at iter %d: average loss = %f,
                                                ssim amp loss = %f, ssim img loss = %f,
                                                psnr amp loss = %f, psnr img loss = %f,''' % 
                            (iter, np.mean(avg_loss), 
                            np.mean(avg_ssim_amp_loss), np.mean(avg_ssim_img_loss), 
                            np.mean(avg_psnr_amp_loss), np.mean(avg_psnr_img_loss)))
                iter = iter + 1
        
            # save model when one epoch finishes
            print("Saving model at epoch %d ...\n" % (epoch))
            self.saver.save(self.sess, self.path_params["ckpt_path"], global_step=global_step)

        print("Finish the training\n")
        print("Done!\n")

    def export_for_tensorrt(self, trt_res_h, trt_res_w, data_format='NCHW'):
        import onnx
        import tf2onnx

        # define placeholder for input rgbd images
        if data_format == 'NCHW':
            rgbd = tf.placeholder("float", [1, 
                                            self.model_params["input_dim"], 
                                            trt_res_h, 
                                            trt_res_w], 
                                  name="input")
        else:
            rgbd = tf.placeholder("float", [1, 
                                            trt_res_h, 
                                            trt_res_w, 
                                            self.model_params["input_dim"]], 
                                  name="input")

        # build model
        self.model_vars = self._build_model_vars()
        self._build_graph(rgbd, self.model_vars, data_format=data_format)

        # restore trained model variable
        self.saver = tf.train.Saver(max_to_keep=5, save_relative_paths=True)
        ckpt = tf.train.get_checkpoint_state(self.path_params["ckpt_parent_path"])
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path_params["ckpt_parent_path"]))
            print("model restored from %s" % self.path_params["ckpt_parent_path"])
        else:
            raise Exception("ERROR: NO RESTORED MODEL...")
        
        # define input and output nodes
        input_node_names = ["input:0"]     # Input nodes list
        output_node_names = ["output_field_amp:0", "output_field_phs:0"]     # Output nodes list
        output_node_names_no_zero = ["output_field_amp", "output_field_phs"] # Output nodes list without :0
        
         # freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(self.sess, 
                                                                        self.sess.graph_def, 
                                                                        output_node_names_no_zero)

        # output to onnx
        onnx_model, _ = tf2onnx.convert.from_graph_def(frozen_graph_def, 
                                                       input_names=input_node_names, 
                                                       output_names=output_node_names,
                                                       )
        
        # save onnx model
        onnx.save_model(onnx_model, '%s/%s.onnx' % (self.path_params["inference_graph_path"], 
                                                    self.path_params["inference_graph_name"]))

    def validate(self):
        _, _, validate_handle, handle, rgbd, holo_in, amp_in, phs_in, holo_out, amp_out, phs_out = self._setup_train()

        # compute psnr and ssim on the amplitude map 
        ssim_amp_loss = tf.reduce_mean(tf.image.ssim(tf.transpose(amp_out, [0,2,3,1]), tf.transpose(amp_in, [0,2,3,1]), 1.0))
        psnr_amp_loss = tf.reduce_mean(tf.image.psnr(tf.transpose(amp_out, [0,2,3,1]), tf.transpose(amp_in, [0,2,3,1]), 1.0))

        # create model saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=5, save_relative_paths=True)

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
                raise Exception("ERROR: NO RESTORED MODEL...")

        # test on test dataset (validate handle)
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
        self.saver = tf.compat.v1.train.Saver(max_to_keep=5, save_relative_paths=True)
        ckpt = tf.train.get_checkpoint_state(self.path_params["ckpt_parent_path"])
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.path_params["ckpt_parent_path"]))
            print("model restored from %s" % self.path_params["ckpt_parent_path"])
        else:
            raise Exception("ERROR: NO RESTORED MODEL...") 

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
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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
    
    # train parameters
    parser.add_argument('--train-mode', action='store_true', help='Run in train mode')
    parser.add_argument('--train-iters', default=1200, type=int, help='Training iterations')

    # export parameters 
    parser.add_argument('--export-mode', action='store_true', help='Export model for tensorrt optimization')
    parser.add_argument('--trt-res-h', default=1080, type=int, help='Input image height in export (tensorrt) mode')
    parser.add_argument('--trt-res-w', default=1920, type=int, help='Input image width in export (tensorrt) mode')
    
    # model name for running different modes
    parser.add_argument('--model-name', default="full_loss", type=str, help='Model name')
    
    
    # ** users can add their own input arguments, and replace the corresponding ones in the dicts**
    
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
        "num_epochs": opt.train_iters,                             # training epochs
        "decay_type": None,                                        # learning rate decay
        "decay_params": None,                                      # learning decay parameters
        "learning_rate" : 1e-4,                                    # learning rate 
        "optimizer_type" : "adam",                                 # optimizer type 
        "optimizer_params" : 
            {"beta1":0.9, "beta2":0.99, "epsilon":1e-8},           # optimizer parameters                      
        "num_iter_per_test": 1000,                                 # number of iterations per validation
        "num_top_depth_for_img_loss": 15,                          # number of top-k depths for computing focal stack loss
        "num_random_depth_for_img_loss": 5,                        # number of random depths selected from rest of the bins
        "depth_dependent_weight_scale": 0.35,                      # attention weight
        "num_hist_bins": 200                                       # number of focal stack bins
    }

    # model parameters
    model_params = {
        "name": opt.model_name,                                       # model name
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
        "inference_graph_path" : checkpoint_base_path,                                  # inference graph path
        "inference_graph_name" : "inference_graph",                                     # inference graph name
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
    
    # train model
    if opt.train_mode:
        tensor_holo_model.train()
    
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

    if opt.export_mode:
        tensor_holo_model.export_for_tensorrt(opt.trt_res_h, opt.trt_res_w)
