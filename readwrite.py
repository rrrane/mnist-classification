import tensorflow as tf

class DataReader(object):

    def __init__(self,
                 data_path,
                 input_feature_key,
                 target_feature_key,
                 batch_size,
                 filename_queue_epochs = None,
                 one_hot_labels=False,
                 one_hot_depth=None,
                 capacity=30,
                 num_threads=1,
                 min_after_dequeue=10):

        self._data_path = data_path
        self._input_feature_key = input_feature_key
        self._target_feature_key = target_feature_key
        self._batch_size = batch_size
        self._filename_queue_epochs = filename_queue_epochs
        self._one_hot_labels = one_hot_labels
        self._one_hot_depth = one_hot_depth
        self._capacity = capacity
        self._num_threads = num_threads
        self._min_after_dequeue = min_after_dequeue
        
        self._feature =\
        {self._input_feature_key: tf.FixedLenFeature([], tf.string),
         self._target_feature_key: tf.FixedLenFeature([], tf.int64)}
        
        self._filename_queue = tf.train.string_input_producer(
            [self._data_path], num_epochs=filename_queue_epochs)

        self._reader = tf.TFRecordReader()

        self._read_op_name, self._serialized_example = self._reader.read(\
            self._filename_queue)

        self._read_features = tf.parse_single_example(self._serialized_example,
                                                      features=self._feature)

        self._data = tf.decode_raw(self._read_features[self._input_feature_key],
                                   tf.float32)

        self._label = tf.cast(self._read_features[self._target_feature_key],
                              tf.int32)
        
        
        

    @property
    def data_path(self):
        return self._data_path

    @property
    def input_feature_key(self):
        return self._input_feature_key

    @property
    def target_feature_key(self):
        return self._target_feature_key

    @property
    def filename_queue_epochs(self):
        return self._filename_queue_epochs

    @property
    def one_hot_labels(self):
        return self._one_hot_labels

    @property
    def one_hot_depth(self):
        return self._one_hot_depth

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_threads(self):
        return self._num_threads

    @property
    def min_after_dequeue(self):
        return self._min_after_dequeue

    @property
    def feature(self):
        return self._feature

    @property
    def filename_queue(self):
        return self._filename_queue

    @property
    def reader(self):
        return self._reader

    def read(self):
        data, labels = tf.train.shuffle_batch(\
            [self._data, self._label],
            batch_size=self._batch_size,
            capacity=self._capacity,
            num_threads=self._num_threads,
            min_after_dequeue=self._min_after_dequeue)
        
        
        if self._one_hot_labels:
            if isinstance(self._one_hot_depth, int):
                labels = tf.one_hot(labels, depth=self._one_hot_depth)
            else:
                raise Exception("one_hot_depth should be of type int, found " + \
                                str(type(self._one_hot_depth)))
        return data, labels

class ImageDataReader(DataReader):

    def __init__(self,
                 data_path,
                 input_feature_key,
                 target_feature_key,
                 batch_size,
                 shape,
                 filename_queue_epochs = None,
                 one_hot_labels=False,
                 one_hot_depth=None,
                 capacity=30,
                 num_threads=1,
                 min_after_dequeue=10):

        super(ImageDataReader, self).__init__(data_path,
                                              input_feature_key,
                                              target_feature_key,
                                              batch_size,
                                              filename_queue_epochs,
                                              one_hot_labels,
                                              one_hot_depth,
                                              capacity,
                                              num_threads,
                                              min_after_dequeue)

        self._shape = shape
        self._data = tf.reshape(self._data, shape)

    @property
    def shape(self):
        return self._shape
                 
                 
