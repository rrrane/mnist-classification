import tensorflow as tf
import math

class Cell(object):

    def __init__(self, descriptor):
        self._descriptor = descriptor
        self._built = False

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def built(self):
        return self._built

class SensorCell(Cell):

    def __init__(self, descriptor):
        super(SensorCell, self).__init__(descriptor)

class NetworkCell(Cell):

    def __init__(self, descriptor):
        super(NetworkCell, self).__init__(descriptor)

    @property
    def variable_collection(self):
        return tf.get_collection(self._descriptor.variable_collections[0])

    def _add_variables_to_collections(self):
        raise NotImplementedError
        

class GlimpseSensorCell(SensorCell):
    """Sensor mimicing retina-like structure to capture
    glimpse of an image"""
    
    def __init__(self, descriptor):
        """Initializes the sensor
        
        Args:
            glimpse_descriptor: RetinaGlimpseDescriptor object
            image_descriptor: ImageDescriptor object
        """
        super(GlimpseSensorCell, self).__init__(descriptor)
    
    @property
    def shapes(self):
        return self._glimpse_shapes_list
    
    
    def __call__(self, image, location):
        """Glimpse sensor
        
        Args:
            images: [batch_size x height x width x channels] tensor
                    of input images
            locations: [batch_size x 2] tensor representing location
                    of sensor scaled to ([-1, 1], [-1, 1])
                    
        Returns:
            Encoded glimpse
        """
        if not self._built:
            self._build(image, location)
            self._built = True
        
        return tf.concat([self._create_glimpse(image, size, location) for size in self._glimpse_shapes_list], 
                        axis=1)
    
    def _build(self, image, location):
        self._glimpse_shapes_list = list()
        
        for i in range(self._descriptor.number_of_scales):
            self._glimpse_shapes_list.append(tf.constant([int(math.pow(2, i) * self._descriptor.scan_height), 
                                                         int(math.pow(2, i) * self._descriptor.scan_width)]))
    
    def _create_glimpse(self, image, size, location):
        return tf.contrib.layers.flatten(
            tf.image.resize_images(tf.image.extract_glimpse(image, size, location), 
                                   tf.constant([int(self._descriptor.scan_height), 
                                                int(self._descriptor.scan_width)])))


class GlimpseNetworkCell(NetworkCell):
    
    def __init__(self, descriptor):
        super(GlimpseNetworkCell, self).__init__(descriptor)
    
    
    @property
    def kernel_in_hg(self):
        return self._kernel_in_hg
    
    @property
    def bias_hg(self):
        return self._bias_hg
    
    @property
    def kernel_loc_hl(self):
        return self._kernel_loc_hl
    
    @property
    def bias_hl(self):
        return self._bias_hl
    
    @property
    def kernel_hg_out(self):
        return self._kernel_hg_out
    
    @property
    def kernel_hl_out(self):
        return self._kernel_hl_out
    
    @property
    def bias_out(self):
        return self._bias_out

    
    def __call__(self, glimpse, location):

        if not self._built:
            self._build(glimpse, location)
            self._built = True
            
        h_g = self._descriptor.activation_hg(
            tf.add(tf.matmul(glimpse, self._kernel_in_hg), 
                   self._bias_hg))
        
        h_l = self._descriptor.activation_hl(
            tf.add(tf.matmul(location, self._kernel_loc_hl), 
                   self._bias_hl))
        
        z_g = tf.add(tf.add(tf.matmul(h_g, self._kernel_hg_out),
                            tf.matmul(h_l, self._kernel_hl_out)),
                     self._bias_out)
        
        g = self._descriptor.output_activation(z_g)
        return g
    
    def _build(self, glimpse, location):
        self._kernel_in_hg = tf.get_variable(
            "kernel_in_hg",
            shape=[glimpse.shape[1],
                   self._descriptor.hg_vector_length],
            initializer=self._descriptor.kernel_in_hg_initializer,
            trainable=self._descriptor.backprop_trainable)
        
        self._bias_hg = tf.get_variable(
            "bias_hg",
            shape=[1,
                   self._descriptor.hg_vector_length],
            initializer=self._descriptor.bias_hg_initializer,
            trainable=self._descriptor.backprop_trainable)
        
        self._kernel_loc_hl = tf.get_variable(
            "kernel_loc_hl",
            shape=[location.shape[1],
                   self._descriptor.hl_vector_length],
            initializer=self._descriptor.kernel_loc_hl_initializer,
            trainable=self._descriptor.backprop_trainable)
        
        self._bias_hl = tf.get_variable(
            "bias_hl",
            shape=[1,
                   self._descriptor.hl_vector_length],
            initializer=self._descriptor.bias_hl_initializer,
            trainable=self._descriptor.backprop_trainable)
        
        self._kernel_hg_out = tf.get_variable(
            "kernel_hg_out",
            shape=[self._descriptor.hg_vector_length,
                   self._descriptor.output_dimensions],
            initializer=self._descriptor.kernel_hg_out_initializer,
            trainable=self._descriptor.backprop_trainable)
        
        self._kernel_hl_out = tf.get_variable(
            "kernel_hl_out",
            shape=[self._descriptor.hl_vector_length,
                   self._descriptor.output_dimensions],
            initializer=self._descriptor.kernel_hl_out_initializer,
            trainable=self._descriptor.backprop_trainable)
        
        self._bias_out = tf.get_variable(
            "bias_out",
            shape=[1,
                   self._descriptor.output_dimensions],
            initializer=self._descriptor.bias_out_initializer,
            trainable=self._descriptor.backprop_trainable)

        self._add_variables_to_collections()

    def _add_variables_to_collections(self):
        for collection in self._descriptor.variable_collections:
            tf.add_to_collection(collection, self._kernel_in_hg)
            tf.add_to_collection(collection, self._bias_hg)
            tf.add_to_collection(collection, self._kernel_loc_hl)
            tf.add_to_collection(collection, self._bias_hl)
            tf.add_to_collection(collection, self._kernel_hg_out)
            tf.add_to_collection(collection, self._kernel_hl_out)
            tf.add_to_collection(collection, self._self._bias_out)
        

class CoreNetworkCell(NetworkCell):

    def __init__(self,
                 descriptor):
        super(CoreNetworkCell, self).__init__(descriptor)

        self._initial_state = \
        tf.nn.rnn_cell.LSTMStateTuple(\
            tf.Variable(\
                tf.zeros(\
                    [self._descriptor.batch_size,
                     self._descriptor.output_dimensions])),
            tf.Variable(\
                tf.zeros(\
                    [self._descriptor.batch_size,
                     self._descriptor.output_dimensions])))
        

    @property
    def lstm_cell(self):
        return self._lstm_cell

    @property
    def initial_state(self):
        return self._initial_state

    def __call__(self, inputs, state):
        if not self._built:
            self._build(inputs, state)
            self._built = True

        h, state = self._lstm_cell(inputs, state)
        
        return self._descriptor.output_activation(h), state

    def _build(self, inputs, state):
        self._lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._descriptor.output_dimensions)
        self._add_variables_to_collections()

    def _add_variables_to_collections(self):
        for collection in self._descriptor.variable_collections:
            tf.add_to_collection(collection, self._initial_state)

class ActionNetworkCell(NetworkCell):

    def __init__(self,
                 descriptor):
        super(ActionNetworkCell, self).__init__(descriptor)
        

    @property
    def kernel_in_fa(self):
        return self._kernel_in_fa

    @property
    def bias_fa(self):
        return self._bias_fa
    

    def __call__(self, inputs):
        if not self._built:
            self._build(inputs)
            self._built = True

        return tf.nn.softmax(self._descriptor.output_activation(\
            tf.add(tf.matmul(inputs, self._kernel_in_fa), self._bias_fa)))

    def _build(self, inputs):
        self._kernel_in_fa = tf.get_variable("kernel_in_fa",
                                             shape=[inputs.shape[1],
                                                    self._descriptor.output_dimensions],
                                             initializer=self._descriptor.kernel_in_fa_initializer,
                                             trainable=self._descriptor.backprop_trainable)
        self._bias_fa = tf.get_variable("bias_fa",
                                        shape=[1,
                                               self._descriptor.output_dimensions],
                                        initializer=self._descriptor.bias_fa_initializer,
                                        trainable=self._descriptor.backprop_trainable)

        self._add_variables_to_collections()

    def _add_variables_to_collections(self):
        for collection in self._descriptor.variable_collections:
            tf.add_to_collection(collection, self._kernel_in_fa)
            tf.add_to_collection(collection, self._bias_fa)


class LocationNetworkCell(NetworkCell):

    def __init__(self,
                 descriptor):
        super(LocationNetworkCell, self).__init__(descriptor)

        
    @property
    def kernel_in_fl(self):
        return self._kernel_in_fl

    @property
    def bias_fl(self):
        return self._bias_fl

    def __call__(self, inputs):
        if not self._built:
            self._build(inputs)
            self._built = True
        
        return self._descriptor.output_activation(\
            tf.matmul(tf.concat([tf.ones([inputs.shape[0], 1]), inputs], 1), self._kernel_in_fl))

    def _build(self, inputs):
        
        self._kernel_in_fl = tf.get_variable("kernel_in_fl",
                                             shape=[inputs.shape[1] + tf.Dimension(1),
                                                    self._descriptor.output_dimensions],
                                             initializer=self._descriptor.kernel_in_fl_initializer,
                                             trainable=self._descriptor.backprop_trainable)

        self._add_variables_to_collections()

    def _add_variables_to_collections(self):
        for collection in self._descriptor.variable_collections:
            tf.add_to_collection(collection, self._kernel_in_fl)


class BaselineNetworkCell(NetworkCell):
    def __init__(self,
                 descriptor):
        super(BaselineNetworkCell, self).__init__(descriptor)

    @property
    def kernel_in_fb(self):
        return self._kernel_in_fb

    @property
    def bias_fb(self):
        return self._bias_fb

    def __call__(self, inputs):
        
        if not self._built:
            self._build(inputs)
            self._built = True

        return self._descriptor.output_activation(\
            tf.add(tf.matmul(inputs, self._kernel_in_fb), self._bias_fb))

    def _build(self, inputs):
        
        self._kernel_in_fb = tf.get_variable("kernel_in_fb",
                                             shape=[inputs.shape[1],
                                                    self._descriptor.output_dimensions],
                                             initializer=self._descriptor.kernel_in_fb_initializer,
                                             trainable=self._descriptor.backprop_trainable)
        
        self._bias_fb = tf.get_variable("bias_fb",
                                        shape=[1,
                                               self._descriptor.output_dimensions],
                                        initializer=self._descriptor.bias_fb_initializer,
                                        trainable=self._descriptor.backprop_trainable)
        
        self._add_variables_to_collections()

    def _add_variables_to_collections(self):
        for collection in self._descriptor.variable_collections:
            tf.add_to_collection(collection, self._kernel_in_fb)
            tf.add_to_collection(collection, self._bias_fb)

class ClippedRandomNormalSamplerCell:

    def __init__(self,
                 descriptor):
        self._descriptor = descriptor
        self._built = False

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def built(self):
        return self._built

    def __call__(self, inputs):
        if not self._built:
            self._build(inputs)
            self._built = True

        return tf.clip_by_value(\
            tf.contrib.distributions.MultivariateNormalDiag(\
                inputs, self._scales).sample(),
            self._descriptor.min_val,
            self._descriptor.max_val)

    def _build(self, inputs):
        self._scales = tf.ones([self._descriptor.batch_size, inputs.shape[1]])
