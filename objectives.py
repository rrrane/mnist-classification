import tensorflow as tf


class ObjectiveFunction(object):

    def __init__(self, inputs, targets):
        self._inputs = inputs
        self._targets = targets

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets


class LossFunction(ObjectiveFunction):

    def __init__(self,
                 inputs,
                 targets,
                 loss = tf.losses.softmax_cross_entropy):
        super(LossFunction, self).__init__(inputs, targets)
        self._loss = loss(self._targets, self._inputs)

    @property
    def loss(self):
        return self._loss


class RewardFunction(ObjectiveFunction):

    def __init__(self,
                 inputs,
                 targets,
                 batch_size,
                 sequence_length):

        super(RewardFunction, self).__init__(inputs, targets)
        
        self._rewards = tf.transpose(tf.concat(\
            [tf.zeros([batch_size, sequence_length - 1]),
             tf.expand_dims(tf.cast(tf.equal(tf.argmax(inputs, axis=1),
                              tf.argmax(targets, axis=1)),
                     tf.float32), axis=1)], axis=1))

    @property
    def rewards(self):
        return self._rewards
