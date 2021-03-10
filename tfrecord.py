import os
import abc
import cv2
import numpy as np
import tensorflow as tf

class TFRecordGenerator(abc.ABC):
    def __init__(self, tfrecord_path, labels, dir_paths=None, file_paths=None):
        # tfrecord_path : record tfrecord_path
        # dir_paths     : dir paths of different image sources
        # labels        : label for each dir path
        # file_paths    : files that each contains list of images
        self.tfrecord_path = None
        self.file_paths = None
        self.labels = None
        self.file_count = None
        self.update_record_paths(tfrecord_path, labels, dir_paths, file_paths)

    def update_record_paths(self, tfrecord_path, labels, dir_paths=None, file_paths=None):
        if file_paths is None and dir_paths is None:
            raise ValueError("Both dir_paths and file_paths are none")
        elif file_paths is None:
            if len(dir_paths) != len(labels):
                raise ValueError("Length of file_paths and labels are not equal")
            file_paths = self._convert_dir_to_file_path(dir_paths)
        
        files_count = np.array([len(files) for files in file_paths])
        if not np.all(files_count == files_count[0]):
            raise ValueError("File paths have different number of files")

        self.tfrecord_path = tfrecord_path
        self.file_paths = zip(*file_paths)
        self.labels = labels
        self.file_count = files_count[0]

    def generate_record(self):
        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for count, img_paths in enumerate(self.file_paths):
                example = self._convert_one_example(img_paths)
                writer.write(example.SerializeToString())
                print("complete {:0>4d}/{:0>4d} example".format(count+1, self.file_count))

    def _convert_dir_to_file_path(self, dir_paths):
        file_paths = []
        for dir_path in dir_paths:
            file_path = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            # sort file path, make sure files in different folders are matched
            file_path.sort()
            file_paths.append(file_path)
        return file_paths

    @abc.abstractmethod
    def _convert_one_example(self, img_paths):
        """ define how each example should be processed
        """


class TFRecordGeneratorforTH(TFRecordGenerator):
    def __init__(self, tfrecord_path, labels, dir_paths=None, file_paths=None):
        super(TFRecordGeneratorforTH, self).__init__(tfrecord_path, labels, dir_paths, file_paths)

    def _convert_one_example(self, img_paths):
        features = dict()
        # all images have same shape
        for count, img_path in enumerate(img_paths):
            # save exr image as float32 1d-array in **NCHW** format
            # for best GPU inference performance
            tmp = np.transpose(cv2.imread(img_path, -1), [2,0,1])
            if self.labels[count].startswith("depth"):
                # keep depth image as single channel to reduce memory cost
                tmp = tmp[0,:,:]
            tmp = tmp.flatten()
            features[self.labels[count]] = tf.train.Feature(float_list = tf.train.FloatList(value=tmp))

        example = tf.train.Example(features = tf.train.Features(feature = features))
        return example


class TFRecordExtractor(abc.ABC):
    def __init__(self, tfrecord_path, dataset_params, labels):
        # tfrecord_path  : record tfrecord_path
        # dataset_params : parameters for constructing dataset pipeline
        # labels        : label for each image
        self.tfrecord_path = None
        self.dataset_params = None
        self.labels = None
        self.iterator = None
        self.update_record_path(tfrecord_path, dataset_params, labels)

    def update_record_path(self, tfrecord_path, dataset_params, labels):
        self.tfrecord_path = os.path.abspath(tfrecord_path)
        self.dataset_params = dataset_params
        self.labels = labels

    def _extract_fn(self, tfrecord):
        """ define how each example should be parsed
        """

    def build_dataset(self):
        # Pipeline of dataset and iterator 
        dataset = tf.data.TFRecordDataset([self.tfrecord_path])
        dataset = dataset.shuffle(buffer_size=self.dataset_params["shuffle_buffer_size"])
        if self.dataset_params["repeat"]:
            dataset = dataset.repeat()
        dataset = dataset.map(self._extract_fn, num_parallel_calls=self.dataset_params["num_parallel_calls"])
        dataset = dataset.batch(self.dataset_params["batch"])
        dataset = dataset.prefetch(buffer_size=self.dataset_params["prefetch_buffer_size"])
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        self.iterator = iterator
        return iterator 


class TFRecordExtractorforTH(TFRecordExtractor):
    def __init__(self, tfrecord_path, dataset_params, labels):
        super(TFRecordExtractorforTH, self).__init__(tfrecord_path, dataset_params, labels)

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        features = dict()
        for element in self.labels:
            # restore image in to 3d with provided shape
            if element.startswith("depth"):
                # load as single channel image
                features[element] = tf.io.FixedLenFeature((1, self.dataset_params["res_h"], self.dataset_params["res_w"]), tf.float32) 
            else:
                features[element] = tf.io.FixedLenFeature((3, self.dataset_params["res_h"], self.dataset_params["res_w"]), tf.float32) 

        # Extract the data record
        imgs = tf.io.parse_single_example(tfrecord, features)        
        return imgs