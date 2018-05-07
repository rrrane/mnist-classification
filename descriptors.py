import tensorflow as tf

class Descriptor(object):
    def __init__(self):
        pass

class SensorDescriptor(Descriptor):
    def __init__(self, scan_length):
        super(SensorDescriptor, self).__init__()
        self._scan_length = scan_length

    @property
    def scan_length(self):
        return self._scan_length

class NetworkDescriptor(Descriptor):
    def __init__(self,
                 output_dimensions,
                 output_activation,
                 apply_softmax,
                 include_bias_in_kernel,
                 variable_collections,
                 train_with_reinforce=False):
        super(NetworkDescriptor, self).__init__()
        self._output_dimensions = output_dimensions
        self._output_activation = output_activation
        self._apply_softmax = apply_softmax
        self._include_bias_in_kernel = include_bias_in_kernel
        self._variable_collections = variable_collections
        self._train_with_reinforce = train_with_reinforce
        self._backprop_trainable = not self._train_with_reinforce

    @property
    def output_dimensions(self):
        return self._output_dimensions

    @property
    def output_activation(self):
        return self._output_activation

    @property
    def apply_softmax(self):
        return self._apply_softmax

    @property
    def include_bias_in_kernel(self):
        return self._include_bias_in_kernel

    @property
    def variable_collections(self):
        return self._variable_collections

    @property
    def train_with_reinforce(self):
        return self._train_with_reinforce

    @property
    def backprop_trainable(self):
        return self._backprop_trainable


class ImageSensorDescriptor(SensorDescriptor):
    def __init__(self,
                 scan_height,
                 scan_width):
        super(ImageSensorDescriptor, self).__init__(scan_height * scan_width)
        self._scan_height = scan_height
        self._scan_width = scan_width

    @property
    def scan_height(self):
        return self._scan_height

    @property
    def scan_width(self):
        return self._scan_width


class GlimpseDescriptor(ImageSensorDescriptor):
    def __init__(self, 
                 scan_height, 
                 scan_width,
                 num_scales):

        super(GlimpseDescriptor, self).__init__(scan_height, scan_width)
        self._num_scales = num_scales

    @property
    def number_of_scales(self):
        return self._num_scales



class GlimpseNetworkDescriptor(NetworkDescriptor):
    def __init__(self,
                 output_dimensions,
                 output_activation,
                 len_hg, 
                 len_hl,  
                 activation_hg, 
                 activation_hl, 
                 kernel_in_hg_initializer, 
                 bias_hg_initializer, 
                 kernel_loc_hl_initializer, 
                 bias_hl_initializer, 
                 kernel_hg_out_initializer, 
                 kernel_hl_out_initializer, 
                 bias_out_initializer,
                 apply_softmax=False,
                 include_bias_in_kernel=False,
                 variable_collections=list(),
                 train_with_reinforce = False):
        
        super(GlimpseNetworkDescriptor, self).__init__(output_dimensions,
                                                       output_activation,
                                                       apply_softmax,
                                                       include_bias_in_kernel,
                                                       variable_collections,
                                                       train_with_reinforce)
        
        self._len_hg = len_hg
        self._len_hl = len_hl
        
        self._activation_hg = activation_hg
        self._activation_hl = activation_hl
        
        self._kernel_in_hg_initializer = kernel_in_hg_initializer
        self._bias_hg_initializer = bias_hg_initializer
        self._kernel_loc_hl_initializer = kernel_loc_hl_initializer
        self._bias_hl_initializer = bias_hl_initializer
        self._kernel_hg_out_initializer = kernel_hg_out_initializer
        self._kernel_hl_out_initializer = kernel_hl_out_initializer
        self._bias_out_initializer = bias_out_initializer
        
    
    @property
    def hg_vector_length(self):
        return self._len_hg
    
    @property
    def hl_vector_length(self):
        return self._len_hl
    
    @property
    def activation_hg(self):
        return self._activation_hg
    
    @property
    def activation_hl(self):
        return self._activation_hl
    
    @property
    def kernel_in_hg_initializer(self):
        return self._kernel_in_hg_initializer
    
    @property
    def bias_hg_initializer(self):
        return self._bias_hg_initializer
    
    @property
    def kernel_loc_hl_initializer(self):
        return self._kernel_loc_hl_initializer
    
    @property
    def bias_hl_initializer(self):
        return self._bias_hl_initializer
    
    @property
    def kernel_hg_out_initializer(self):
        return self._kernel_hg_out_initializer
    
    @property
    def kernel_hl_out_initializer(self):
        return self._kernel_hl_out_initializer
    
    @property
    def bias_out_initializer(self):
        return self._bias_out_initializer


class CoreNetworkDescriptor(NetworkDescriptor):

    def __init__(self,
                 output_dimensions,
                 output_activation,
                 batch_size,
                 apply_softmax=False,
                 include_bias_in_kernel=False,
                 variable_collections=list(),
                 train_with_reinforce = False):
        super(CoreNetworkDescriptor, self).__init__(output_dimensions,
                                                    output_activation,
                                                    apply_softmax,
                                                    include_bias_in_kernel,
                                                    variable_collections,
                                                    train_with_reinforce)
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size


class ActionNetworkDescriptor(NetworkDescriptor):

    def __init__(self,
                 output_dimensions,
                 output_activation,
                 kernel_in_fa_initializer,
                 bias_fa_initializer,
                 apply_softmax=True,
                 include_bias_in_kernel=False,
                 variable_collections=list(),
                 train_with_reinforce = False):
        super(ActionNetworkDescriptor, self).__init__(output_dimensions,
                                                      output_activation,
                                                      apply_softmax,
                                                      include_bias_in_kernel,
                                                      variable_collections,
                                                      train_with_reinforce)

        self._kernel_in_fa_initializer = kernel_in_fa_initializer
        self._bias_fa_initializer = bias_fa_initializer

    @property
    def kernel_in_fa_initializer(self):
        return self._kernel_in_fa_initializer

    @property
    def bias_fa_initializer(self):
        return self._bias_fa_initializer

class LocationNetworkDescriptor(NetworkDescriptor):

    def __init__(self,
                 output_dimensions,
                 output_activation,
                 kernel_in_fl_initializer,
                 bias_fl_initializer,
                 apply_softmax=False,
                 include_bias_in_kernel=True,
                 variable_collections=list(),
                 train_with_reinforce=True):

        super(LocationNetworkDescriptor, self).__init__(output_dimensions,
                                                        output_activation,
                                                        apply_softmax,
                                                        include_bias_in_kernel,
                                                        variable_collections,
                                                        train_with_reinforce)
        
        self._kernel_in_fl_initializer = kernel_in_fl_initializer
        self._bias_fl_initializer = bias_fl_initializer

    @property
    def kernel_in_fl_initializer(self):
        return self._kernel_in_fl_initializer

    @property
    def bias_fl_initializer(self):
        return self._bias_fl_initializer

class BaselineNetworkDescriptor(NetworkDescriptor):
    def __init__(self,
                 output_dimensions,
                 output_activation,
                 kernel_in_fb_initializer,
                 bias_fb_initializer,
                 apply_softmax=False,
                 include_bias_in_kernel=False,
                 variable_collections=list(),
                 train_with_reinforce=False):
        super(BaselineNetworkDescriptor, self).__init__(output_dimensions,
                                                        output_activation,
                                                        apply_softmax,
                                                        include_bias_in_kernel,
                                                        variable_collections,
                                                        train_with_reinforce)
        
        self._kernel_in_fb_initializer = kernel_in_fb_initializer
        self._bias_fb_initializer = bias_fb_initializer

    @property
    def kernel_in_fb_initializer(self):
        return self._kernel_in_fb_initializer

    @property
    def bias_fb_initializer(self):
        return self._bias_fb_initializer

class ClippedRandomNormalSamplerDescriptor:

    def __init__(self,
                 batch_size,
                 min_val=-1.0,
                 max_val=1.0):
        self._batch_size = batch_size
        self._min_val = min_val
        self._max_val = max_val

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def min_val(self):
        return self._min_val

    @property
    def max_val(self):
        return self._max_val
