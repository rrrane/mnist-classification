import tensorflow as tf


class Trainer(object):

    def __init__(self, variable_collection):
        self._variable_collection = variable_collection

    @property
    def variable_collection(self):
        return self._variable_collection


class BackpropTrainer(Trainer):

    def __init__(self,
                 variable_collection,
                 loss_function,
                 optimizer = tf.train.AdamOptimizer()):
        super(BackpropTrainer, self).__init__(variable_collection)

        self._optimizer = optimizer
        self._loss_function = loss_function

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def loss_function(self):
        return self._loss_function

    @property
    def train_op(self):
        return self._optimizer.minimize(
            self._loss_function, var_list=self._variable_collection)


class ReinforceTrainer(Trainer):

    def __init__(self,
                 variable_collection,
                 variance,
                 learning_rate,
                 batch_size,
                 means,
                 samples,
                 inputs,
                 rewards,
                 baselines):
        super(ReinforceTrainer, self).__init__(variable_collection)
        self._variance = variance
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._means = means
        self._samples = samples
        self._inputs = inputs
        self._rewards = rewards
        self._baselines = baselines

    @property
    def variance(self):
        return self._variance

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def means(self):
        return self._means

    @property
    def samples(self):
        return self._sample

    @property
    def inputs(self):
        return self._inputs

    @property
    def rewards(self):
        return self._rewards

    @property
    def baselines(self):
        return self._baselines

    @property
    def train_op(self):
        return tf.assign_add(\
            self._variable_collection[0],
            (self._learning_rate / self._variance) * self._policy_gradient())
    

    def _policy_gradient(self):
        """
        Calculates policy gradient

        Args:
            mu: [T x M x 2] tensor
            l: [T x M x2] tensor
            s: [T x M x 257] tensor
            R: [T x M] tensor
            b_t: [T x 1] tensor
            batch_size: size of the batch
        """
        step1 = tf.subtract(self._rewards, self._baselines) #[T x M]
        step2 = tf.transpose(self._inputs, [2, 0, 1]) #[257 x T x M]
        step3 = tf.multiply(step1, step2) #[257 x T x M]
        step4 = tf.transpose(step3, [0, 2, 1]) #[257 x M x T]
        step5 = tf.subtract(self._samples, self._means) #[T x M x 2]
        step6 = tf.tensordot(step4, step5, axes=[[2], [0]]) #[257 x M x M x 2]
        step7 = tf.transpose(step6, [0, 3, 1, 2]) #[257 x 2 x M x M]
        step8 = tf.multiply(step7, tf.eye(self._batch_size)) #[257 x 2 x M x M]

        gradient = tf.reduce_mean(step8, axis=[2, 3]) #[257 x 2]

        return gradient
                 
