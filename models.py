import tensorflow as tf

from descriptors import GlimpseDescriptor
from descriptors import GlimpseNetworkDescriptor
from descriptors import CoreNetworkDescriptor
from descriptors import ActionNetworkDescriptor
from descriptors import LocationNetworkDescriptor
from descriptors import ClippedRandomNormalSamplerDescriptor
from descriptors import BaselineNetworkDescriptor

from networkcells import GlimpseNetworkCell
from networkcells import GlimpseSensorCell
from networkcells import CoreNetworkCell
from networkcells import ActionNetworkCell
from networkcells import LocationNetworkCell
from networkcells import ClippedRandomNormalSamplerCell
from networkcells import BaselineNetworkCell


class ModelDescriptor(object):

    def __init__(self,
                 sequence_length,
                 number_of_scales,
                 glimpse_width,
                 glimpse_height,
                 num_glimpse_fc,
                 num_loc_fc,
                 glimpse_net_out_dim,
                 core_network_state_units,
                 number_of_actions,
                 location_dimensionality,
                 batch_size):

        self._sequence_length = sequence_length
        self._num_glimpse_scales = number_of_scales
        self._glimpse_width = glimpse_width
        self._glimpse_height = glimpse_height
        self._num_glimpse_fc = num_glimpse_fc
        self._num_loc_fc = num_loc_fc
        self._glimpse_net_out_dim = glimpse_net_out_dim
        self._core_network_state_units = core_network_state_units
        self._number_of_actions = number_of_actions
        self._loc_dim = location_dimensionality
        self._batch_size = batch_size

    @property
    def sequence_length(self):
        return self._sequence_length
    
    @property
    def num_glimpse_scales(self):
        return self._num_glimpse_scales
    
    @property
    def glimpse_width(self):
        return self._glimpse_width

    @property
    def glimpse_height(self):
        return self._glimpse_height

    @property
    def num_glimpse_fc(self):
        return self._num_glimpse_fc
    
    @property
    def num_loc_fc(self):
        return self._num_loc_fc

    @property
    def glimpse_net_out_dim(self):
        return self._glimpse_net_out_dim

    @property
    def core_network_state_units(self):
        return self._core_network_state_units
    
    @property
    def number_of_actions(self):
        return self._number_of_actions
    
    @property
    def loc_dim(self):
        return self._loc_dim
    
    @property
    def batch_size(self):
        return self._batch_size

class Model(object):

    def __init__(self, descriptor):

        self._descriptor = descriptor
        self._built = False
        
        self._construct_descriptors()
        self._initialize_network()

    @property
    def built(self):
        return self._built

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def backprop_outputs(self):
        return self._backprop_outputs
    
    @property
    def glimpse_descriptor(self):
        return self._glimpse_desc

    @property
    def glimpse_network_descriptor(self):
        return self._glimpse_net_desc

    @property
    def core_network_descriptor(self):
        return self._core_net_desc

    @property
    def action_network_descriptor(self):
        return self._act_net_desc

    @property
    def location_network_descriptor(self):
        return self._loc_net_desc
        
    @property
    def baseline_network_descriptor(self):
        return self._baseline_net_desc

    @property
    def random_sampler_descriptor(self):
        return self._rand_sampler_desc

    @property
    def glimpse_sensor(self):
        return self._glimpse_sensor

    @property
    def glimpse_network(self):
        return self._glimpse_network

    @property
    def core_network(self):
        return self._core_network

    @property
    def action_network(self):
        return self._action_network

    @property
    def location_network(self):
        return self._location_network

    @property
    def baseline_network(self):
        return self._baseline_network

    @property
    def sampler(self):
        return self._sampler


    @property
    def current_core_network_state(self):
        return self._current_core_network_state

    @property
    def glimpse_sensor_output(self):
        return self._glimpse_sensor_output

    @property
    def glimpse_network_output(self):
        return self._glimpse_network_output

    @property
    def core_network_output(self):
        return self._core_network_output

    @property
    def action_network_output(self):
        return self._action_network_output

    @property
    def baseline_network_output(self):
        return self._baseline_network_output

    @property
    def location_network_output(self):
        return self._location_network_output

    @property
    def next_location_output(self):
        return self._next_location_output

    

    def __call__(self, inputs):
        if not self._built:
            self._build(inputs)
            self._built = True

        

        for t in range(1, self._descriptor.sequence_length + 1):
            self._glimpse_sensor_output.append(\
                self._glimpse_sensor(inputs,
                                     self._next_location_output[t - 1]))

            self._glimpse_network_output.append(\
                self._glimpse_network(\
                    self._glimpse_sensor_output[t - 1],
                    self._next_location_output[t - 1]))

            h, self._current_core_network_state =\
            self._core_network(self._glimpse_network_output[t - 1],
                               self._current_core_network_state)

            if t < self._descriptor.sequence_length:
                
                self._core_network_output.append(h)
                
                self._location_network_output.append(\
                    self._location_network(h))
                
                self._next_location_output.append(\
                    self._sampler(self._location_network_output[t]))

            self._action_network_output.append(self._action_network(h))

            self._baseline_network_output.append(self._baseline_network(h))

        
        
        return {"ACTIONS": self._action_network_output[-1],
                "BASELINES_REDUCED": tf.transpose(\
                    tf.reduce_mean(\
                        tf.concat(self._baseline_network_output, axis=1),
                        axis=0,
                        keep_dims=True)),
                "BASELINES": tf.transpose(\
                    tf.concat(self._baseline_network_output, axis=1)),
                "LOCATIONS": tf.stack(self._next_location_output),
                "MEANS": tf.stack(self._location_network_output),
                "STATES": tf.concat(\
                    [tf.ones([self._descriptor.sequence_length, self._descriptor.batch_size, 1]),
                     tf.stack(self._core_network_output)], axis=2)}

            
            

    def _build(self, inputs):
        self._current_core_network_state = self._core_network.initial_state
        self._glimpse_sensor_output = list()
        self._glimpse_network_output = list()
        self._core_network_output = [self._current_core_network_state.h]
        self._action_network_output = list()
        self._baseline_network_output = list()
        self._location_network_output =\
        [self._location_network(self._core_network_output[0])]
        self._next_location_output = [self._sampler(self._location_network_output[0])]

    
    def _construct_descriptors(self):

        self._glimpse_desc =\
        GlimpseDescriptor(\
            self._descriptor.glimpse_height,
            self._descriptor.glimpse_width,
            self._descriptor.num_glimpse_scales)

        self._glimpse_net_desc =\
        GlimpseNetworkDescriptor(\
            self._descriptor.glimpse_net_out_dim,
            tf.nn.relu,
            self._descriptor.num_glimpse_fc,
            self._descriptor.num_loc_fc,
            tf.nn.relu,
            tf.nn.relu,
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.xavier_initializer())
        
        self._core_net_desc =\
        CoreNetworkDescriptor(\
            self._descriptor.core_network_state_units,
            tf.identity,
            self._descriptor.batch_size)

        self._act_net_desc =\
        ActionNetworkDescriptor(\
            self._descriptor.number_of_actions,
            tf.nn.relu,
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.xavier_initializer())

        self._loc_net_desc =\
        LocationNetworkDescriptor(\
            self._descriptor.loc_dim,
            tf.identity,
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.xavier_initializer(),
            variable_collections=["location_network_collection"])

        self._baseline_net_desc =\
        BaselineNetworkDescriptor(1,
                                  tf.nn.relu,
                                  tf.contrib.layers.xavier_initializer(),
                                  tf.contrib.layers.xavier_initializer(),
                                  variable_collections=["baseline_network_collection"])

        self._rand_sampler_desc =\
        ClippedRandomNormalSamplerDescriptor(self._descriptor.batch_size)

    def _initialize_network(self):
        self._glimpse_sensor =\
        GlimpseSensorCell(self._glimpse_desc)
        
        self._glimpse_network =\
        GlimpseNetworkCell(self._glimpse_net_desc)
        
        self._core_network =\
        CoreNetworkCell(self._core_net_desc)
        
        self._action_network =\
        ActionNetworkCell(self._act_net_desc)
        
        self._location_network =\
        LocationNetworkCell(self._loc_net_desc)
        
        self._baseline_network =\
        BaselineNetworkCell(self._baseline_net_desc)
        
        self._sampler =\
        ClippedRandomNormalSamplerCell(self._rand_sampler_desc)
