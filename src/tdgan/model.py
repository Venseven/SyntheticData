from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorpack.tfutils.summary import add_moving_summary
import json
import os
import pickle
import tarfile
from tensorpack import (
    BatchData, BatchNorm, Dropout, ModelDescBase, ModelSaver,
    PredictConfig, QueueInput, SaverRestore, SimpleDatasetPredictor, logger)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.argtools import memoized
from tdgan.data import Preprocessor, RandomZData, TGANDataFlow
from tdgan.trainer import GANTrainer
import base64
from tensorflow.python.framework import ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import math_ops
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()



#GPU SELECTION
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0";




# =============================================================================
# for flattening the batch inputs before feeding into the tf.layers.dense
# =============================================================================
def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.get_shape().as_list()[1:]
    if None not in shape:
        return tf.reshape(x, [-1, int(np.prod(shape))])
    return tf.reshape(x, tf.stack([tf.shape(input=x)[0], -1]))



# TODO - check with K if unsqueeze() the same as tf.newaxis
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq,0), tf.float32)
    # add extra dimensions to add to the paddign to the attention logits
    return seq[:, tf.newaxis, tf.newaxis, :] # batchsize, 1,1 ,seq_length

def scaled_dot_product_attention(q, k, v, mask):
    #import ipdb;ipdb.set_trace();
    matmul_qk = tf.matmul(q, k, transpose_b = True)
    dk = tf.cast(tf.shape(input=k)[-1], tf.float32)
    scaled_attn_logits = matmul_qk /tf.math.sqrt(dk)
    # scale matmul_qkfrom tensorflow.python.framework import ops

    # adding mask to the scaled tensor
    if mask is not None:
        scaled_attn_logits += (mask * -1e9)
        # softmax normalised on the last axis seq_len_k
    attention_weights = tf.nn.softmax(scaled_attn_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

def apply_regularization(regularizer, weights_list=None):
  """Returns the summed penalty by applying `regularizer` to the `weights_list`.
  Adding a regularization penalty over the layer weights and embedding weights
  can help prevent overfitting the training data. Regularization over layer
  biases is less common/useful, but assuming proper data preprocessing/mean
  subtraction, it usually shouldn't hurt much either.
  Args:
    regularizer: A function that takes a single `Tensor` argument and returns
      a scalar `Tensor` output.
    weights_list: List of weights `Tensors` or `Variables` to apply
      `regularizer` over. Defaults to the `GraphKeys.WEIGHTS` collection if
      `None`.
  Returns:
    A scalar representing the overall regularization penalty.
  Raises:
    ValueError: If `regularizer` does not return a scalar output, or if we find
        no weights.
  """
  if not weights_list:
    weights_list = ops.get_collection(ops.GraphKeys.WEIGHTS)
  if not weights_list:
    raise ValueError('No weights to regularize.')
  with ops.name_scope('get_regularization_penalty',
                      values=weights_list) as scope:
    penalties = [regularizer(w) for w in weights_list]
    penalties = [
        p if p is not None else constant_op.constant(0.0) for p in penalties
    ]
    for p in penalties:
      if p.get_shape().ndims != 0:
        raise ValueError('regularizer must return a scalar Tensor instead of a '
                         'Tensor with rank %d.' % p.get_shape().ndims)

    summed_penalty = math_ops.add_n(penalties, name=scope)
    ops.add_to_collection(ops.GraphKeys.REGULARIZATION_LOSSES, summed_penalty)
    return summed_penalty





# =============================================================================
# ModeldescBase is used here for getting input_signature...saves compile time 
# =============================================================================

class GAN(ModelDescBase):
    
	
    def __init__(
            self,
            metadata,
            batch_size=200,
            z_dim=200,
            noise=0.2,
            l2norm=0.00001,
            learning_rate=0.001,
            num_gen_rnn=100,
            num_gen_feature=100,
            num_dis_layers=1,
            num_dis_hidden=100,
            optimizer='AdamOptimizer',
            training=True
                        ):

        self.metadata = metadata
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_feature = num_gen_feature
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.optimizer = optimizer
        self.is_training = training
# =============================================================================
# data processing       
# =============================================================================
    def inputs(self):
        
            inputs = []
            for col_id, col_info in enumerate(self.metadata['details']):
                if col_info['type'] == 'value':
                    cluster_mode = col_info['n']
                    inputs.append(
                        tf.TensorSpec(dtype=tf.float32,shape=(self.batch_size, 1), name='input{}value'.format(col_id)))

                    inputs.append(
                        tf.TensorSpec(
                            dtype=tf.float32,
                            shape=(self.batch_size,cluster_mode),
                            name="input{}cluster".format(col_id)
                        )
                    )

                elif col_info['type'] == 'category':
                    inputs.append(tf.TensorSpec(dtype=tf.int32,shape=(self.batch_size, 1), name='input{}value'.format(col_id)))

                else:
                    raise ValueError(
                        "self.metadata['details'][{}]['type'] must be either `category` or "
                        "`values`. Instead it was {}.".format(col_id, col_info['type'])
                    )

            return inputs
    
    def collect_variables(self, g_scope='gen', d_scope='discrim'):
        """Assign generator and discriminator variables from their scopes.

        Args:
            g_scope(str): Scope for the generator.
            d_scope(str): Scope for the discriminator.

        Raises:
            ValueError: If any of the assignments fails or the collections are empty.

        """
        self.g_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, g_scope)
        self.d_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, d_scope)

        if not (self.g_vars or self.d_vars):
            raise ValueError('There are no variables defined in some of the given scopes')
# =============================================================================
# loss building...according to paper
# =============================================================================
    def losses_(self,logits_real, logits_fake, extra_g=0, l2_norm=0.00001):
            with tf.compat.v1.name_scope("GAN_loss"):
                score_real = tf.sigmoid(logits_real)
                score_fake = tf.sigmoid(logits_fake)
                tf.compat.v1.summary.histogram('score-real', score_real)
                tf.compat.v1.summary.histogram('score-fake', score_fake)
    
                with tf.compat.v1.name_scope("discrim"):
                    d_loss_pos = tf.reduce_mean(
                        input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                            logits=logits_real,
                            labels=tf.ones_like(logits_real)) * 0.7 + tf.random.uniform(
                                tf.shape(input=logits_real),
                                maxval=0.3
                        ),
                        name='loss_real'
                    )
    
                    d_loss_neg = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=logits_fake, labels=tf.zeros_like(logits_fake)), name='loss_fake')
    
                    d_pos_acc = tf.reduce_mean(
                        input_tensor=tf.cast(score_real > 0.5, tf.float32), name='accuracy_real')
    
                    d_neg_acc = tf.reduce_mean(
                        input_tensor=tf.cast(score_fake < 0.5, tf.float32), name='accuracy_fake')
    
                    d_loss = 0.5 * d_loss_pos + 0.5 * d_loss_neg + \
                        apply_regularization(
                            tf.keras.regularizers.l2(0.5 * (l2_norm)),
                            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "discrim"))
    
                    self.d_loss = tf.identity(d_loss, name='loss')
    
                with tf.compat.v1.name_scope("gen"):
                    g_loss = tf.reduce_mean(input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=logits_fake, labels=tf.ones_like(logits_fake))) + \
                        apply_regularization(
                            tf.keras.regularizers.l2(0.5 * (l2_norm)),
                            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, 'gen'))
                    extra_g = tf.identity(extra_g, name='klloss')
                    self.g_loss = tf.identity( g_loss + extra_g, name='final-g-loss')
    
                add_moving_summary(
                    extra_g,self.g_loss,self.d_loss, d_pos_acc, d_neg_acc, decay=0.)

# =============================================================================
# Attention builder
# =============================================================================
    def make_attention(self,states):
        a_cap=tf.math.sigmoid(tf.compat.v1.get_variable("a_cap",shape=(len(states),1,1)))
        #logger.info("states and testing")
        
        states=tf.stack(states,axis=0)*a_cap
       # logger.info(states)
        attention=tf.reduce_sum(input_tensor=states,axis=0)
        
        return attention,attention

# =============================================================================
# Generator with LSTM and Dense Networks
# =============================================================================
    def generator_(self,z):
        with tf.compat.v1.variable_scope('LSTM'):
                    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.num_gen_rnn)
                    state = cell.zero_state(self.batch_size, dtype='float32')
                    attention = tf.zeros(
                        shape=(self.batch_size, self.num_gen_rnn), dtype='float32')
                    input = tf.compat.v1.get_variable(name='go', shape=(1, self.num_gen_feature))  # <GO>
                    input = tf.tile(input, [self.batch_size, 1])
                    input = tf.concat([input, z], axis=1)

                    ptr = 0
                    outputs = []
                    states = []
                    for col_id, col_info in enumerate(self.metadata['details']):
                        if col_info['type'] == 'value':
                            #logger.info(state)
                            output, state = cell(tf.concat([input, attention], axis=1), state)
                            states.append(state[1])
                            logger.info(tf.shape(input=states))
                            cluster_modes = col_info['n']
                            with tf.compat.v1.variable_scope("%02d" % ptr):
                                hidden=tf.compat.v1.layers.dense(inputs=batch_flatten(output),units=100,activation=tf.tanh,name="main_layer")
                                outputs.append(tf.compat.v1.layers.dense(inputs=batch_flatten(hidden),units=1,activation=tf.tanh,name="layer-1"))
                                input = tf.concat([hidden, z], axis=1)
                                #logger.info(len(states))
                                #logger.info(tf.reshape(states,[self.batch_size,len(states),self.num_gen_rnn]))
                                attw = tf.compat.v1.get_variable("attw", shape=(len(states), 1, 1))
                                attw = tf.nn.softmax(attw, axis=0)
                                attention = tf.reduce_sum(input_tensor=tf.stack(states, axis=0) * attw, axis=0)
                                #state=(state[0],stat)

                            ptr += 1

                            output, state = cell(tf.concat([input, attention], axis=1), state)
                            states.append(state[1])
                            with tf.compat.v1.variable_scope("%02d" % ptr):
                                hidden=tf.compat.v1.layers.dense(inputs=batch_flatten(output),units=self.num_gen_feature,activation=tf.tanh,name="main_layer")
                                w = tf.compat.v1.layers.dense(inputs=batch_flatten(hidden),units=cluster_modes, activation=tf.nn.softmax,name="layer-2")
                                outputs.append(w)
                                input = tf.compat.v1.layers.dense(inputs=w,units=self.num_gen_feature,activation=tf.identity,name="layer-3")
                                input = tf.concat([input, z], axis=1)
                                
                                attw = tf.compat.v1.get_variable("attw", shape=(len(states), 1, 1))
                                attw = tf.nn.softmax(attw, axis=0)
                                attention = tf.reduce_sum(input_tensor=tf.stack(states, axis=0) * attw, axis=0)
                                #length=len(states)
                                #states=tf.reshape(states,[self.batch_size,length,self.num_gen_rnn])
                                #state,attention=self.location_based_attention(states,self.num_gen_rnn)
                                #logger.info(state)
                                #state=tf.reshape(state,[self.batch_size,self.num_gen_rnn])
                                #attention=tf.reshape(attention,[self.batch_size,length])
                                #logger.info(state)
                                #logger.info(attention)
                                


                            ptr += 1

                        elif col_info['type'] == 'category':
                            label_mode=col_info["n"]
                            output, state = cell(tf.concat([input, attention], axis=1), state)
                            states.append(state[1])
                            with tf.compat.v1.variable_scope("%02d" % ptr):
                                hidden =tf.compat.v1.layers.dense(inputs=batch_flatten(output),units=self.num_gen_feature, activation=tf.tanh,name="main_layer")
                                w =tf.compat.v1.layers.dense(inputs=batch_flatten(hidden),units=label_mode,activation=tf.nn.softmax,name="layer-2")
                                outputs.append(w)
                                one_hot = tf.one_hot(tf.argmax(input=w, axis=1), label_mode)
                                input = tf.compat.v1.layers.dense(
                                    inputs=w,units=self.num_gen_feature, activation=tf.identity,name="layer-3")
                                input = tf.concat([input, z], axis=1)
                                attw = tf.compat.v1.get_variable("attw", shape=(len(states), 1, 1))
                                attw = tf.nn.softmax(attw, axis=0)
                                attention = tf.reduce_sum(input_tensor=tf.stack(states, axis=0) * attw, axis=0)
                            ptr += 1

                        else:
                            raise ValueError(
                                "self.metadata['details'][{}]['type'] must be either `category` or "
                                "`values`. Instead it was {}.".format(col_id, col_info['type'])
                            )

        return outputs
# =============================================================================
# staticmethod inorder to make it int/float..
# =============================================================================
    @staticmethod
    def BatchDiversity(l, n_kernel=10, kernel_dim=10):
    
        M = tf.compat.v1.layers.dense(name='fc_diversity',inputs=l,units=(n_kernel * kernel_dim),activation=tf.identity)
        M = tf.reshape(M, [-1, n_kernel, kernel_dim])
        M1 = tf.reshape(M, [-1, 1, n_kernel, kernel_dim])
        M2 = tf.reshape(M, [1, -1, n_kernel, kernel_dim])
        diff = tf.exp(-tf.reduce_sum(input_tensor=tf.abs(M1 - M2), axis=3))
        return tf.reduce_sum(input_tensor=diff, axis=0)
# =============================================================================
# DISCRIMINATOR with DENSE and Batchdiversity Architecture
# =============================================================================
    @auto_reuse_variable_scope
    def discriminator_(self,vecs):
        logits = tf.concat(vecs, axis=1)
        for i in range(self.num_dis_layers):
            with tf.compat.v1.variable_scope('dis_fc{}'.format(i)):
                if i == 0:
                    logits = tf.compat.v1.layers.dense(inputs=logits,units=self.num_dis_hidden,activation=tf.identity,name="Main_layer_predictions",kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.1)                    )

                else:
                    logits =tf.compat.v1.layers.dense(inputs=logits,units=self.num_dis_hidden,activation=tf.identity,name="Main_layer_predictions")

                logits = tf.concat([logits, self.BatchDiversity(l=logits)], axis=1)
                logits = BatchNorm('bn', logits, center=True, scale=False)#more powerfull than tf.layers.batchnorm
                logits = Dropout(logits)
                logits = tf.nn.leaky_relu(logits)

        return tf.compat.v1.layers.dense(inputs=logits,units=1,activation=tf.identity,name="discriminator_top")
# =============================================================================
# AdamOptimizer....Other Optimizer gives poor results
# =============================================================================
    @memoized
    def get_optimizer(self):
        
        return tf.compat.v1.train.AdamOptimizer(self.learning_rate,0.5)
# =============================================================================
# Tensorflow Graph building ......Adding 0.2 noise to Discrete Variable
# =============================================================================

    @staticmethod
    def compute_kl(real, pred):
        r"""Compute the Kullback–Leibler divergence, :math:`D_{KL}(\textrm{pred} || \textrm{real})`.
        Args:
            real(tensorflow.Tensor): Real values.
            pred(tensorflow.Tensor): Predicted values.
        Returns:
            float: Computed divergence for the given values.
        """
        return tf.reduce_sum(input_tensor=(tf.math.log(pred + 1e-4) - tf.math.log(real + 1e-4)) * pred)
# =============================================================================
# passing inputs list of tensors as reference ...can also assign inside the function by calling self.inputs()
# =============================================================================
    def build_graph(self, *inputs):
        
        
        z = tf.random.normal(
            [self.batch_size, self.z_dim], name='z_train')

        z = tf.compat.v1.placeholder_with_default(z, [None, self.z_dim], name='z')

        with tf.compat.v1.variable_scope('gen'):
            vecs_gen = self.generator_(z)
            vecs_denorm = []
            ptr = 0
            for col_id, col_info in enumerate(self.metadata['details']):
                if col_info['type'] == 'category':
                    t = tf.argmax(input=vecs_gen[ptr], axis=1)
                    t = tf.cast(tf.reshape(t, [-1, 1]), 'float32')
                    vecs_denorm.append(t)
                    ptr += 1

                elif col_info['type'] == 'value':
                    vecs_denorm.append(vecs_gen[ptr])
                    ptr += 1
                    vecs_denorm.append(vecs_gen[ptr])
                    ptr += 1

                else:
                    raise ValueError(
                        "self.metadata['details'][{}]['type'] must be either `category` or "
                        "`values`. Instead it was {}.".format(col_id, col_info['type'])
                    )

            tf.identity(tf.concat(vecs_denorm, axis=1), name='gen')

        vecs_pos = []
        ptr = 0
        for col_id, col_info in enumerate(self.metadata['details']):
            if col_info['type'] == 'category':
                one_hot = tf.one_hot(tf.reshape(inputs[ptr], [-1]), col_info['n'])
                noise_input = one_hot

                if self.is_training:
                    noise = tf.random.uniform(tf.shape(input=one_hot), minval=0, maxval=self.noise)
                    noise_input = (one_hot + noise) / tf.reduce_sum(
                        input_tensor=one_hot + noise, keepdims=True, axis=1)

                vecs_pos.append(noise_input)
                ptr += 1

            elif col_info['type'] == 'value':
                vecs_pos.append(inputs[ptr])
                ptr += 1
                vecs_pos.append(inputs[ptr])
                ptr += 1

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`values`. Instead it was {}.".format(col_id, col_info['type'])
                )

        KL = 0.
        ptr = 0
        if self.training:
            for col_id, col_info in enumerate(self.metadata['details']):
                if col_info['type'] == 'category':
                    dist = tf.reduce_sum(input_tensor=vecs_gen[ptr], axis=0)
                    dist = dist / tf.reduce_sum(input_tensor=dist)

                    real = tf.reduce_sum(input_tensor=vecs_pos[ptr], axis=0)
                    real = real / tf.reduce_sum(input_tensor=real)
                    KL += self.compute_kl(real, dist)
                    ptr += 1

                elif col_info['type'] == 'value':
                    ptr += 1
                    dist = tf.reduce_sum(input_tensor=vecs_gen[ptr], axis=0)
                    dist = dist / tf.reduce_sum(input_tensor=dist)
                    real = tf.reduce_sum(input_tensor=vecs_pos[ptr], axis=0)
                    real = real / tf.reduce_sum(input_tensor=real)
                    KL += self.compute_kl(real, dist)

                    ptr += 1

                else:
                    raise ValueError(
                        "self.metadata['details'][{}]['type'] must be either `category` or "
                        "`values`. Instead it was {}.".format(col_id, col_info['type'])
                    )

        with tf.compat.v1.variable_scope('discrim'):
            discrim_pos = self.discriminator_(vecs_pos)
            discrim_neg = self.discriminator_(vecs_gen)
        self.losses_(discrim_pos, discrim_neg, extra_g=KL, l2_norm=self.l2norm)
        self.collect_variables()
        
class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray
    with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct
          

            
class TDGANModel:
    """Main model from TGAN.

    Args:
        continuous_columns (list[int]): 0-index list of column indices to be considered continuous.
        output (str, optional): Path to store the model and its artifacts. Defaults to
            :attr:`output`.
        gpu (list[str], optional):Comma separated list of GPU(s) to use. Defaults to :attr:`None`.
        max_epoch (int, optional): Number of epochs to use during training. Defaults to :attr:`5`.
        steps_per_epoch (int, optional): Number of steps to run on each epoch. Defaults to
            :attr:`10000`.
        save_checkpoints(bool, optional): Whether or not to store checkpoints of the model after
            each training epoch. Defaults to :attr:`True`
        restore_session(bool, optional): Whether or not continue training from the last checkpoint.
            Defaults to :attr:`True`.
        batch_size (int, optional): Size of the batch to feed the model at each step. Defaults to
            :attr:`200`.
        z_dim (int, optional): Number of dimensions in the noise input for the generator.
            Defaults to :attr:`100`.
        noise (float, optional): Upper bound to the gaussian noise added to categorical columns.
            Defaults to :attr:`0.2`.
        l2norm (float, optional):
            L2 reguralization coefficient when computing losses. Defaults to :attr:`0.00001`.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to
            :attr:`0.001`.
        num_gen_rnn (int, optional): Defaults to :attr:`400`.
        num_gen_feature (int, optional): Number of features of in the generator. Defaults to
            :attr:`100`
        num_dis_layers (int, optional): Defaults to :attr:`2`.
        num_dis_hidden (int, optional): Defaults to :attr:`200`.
        optimizer (str, optional): Name of the optimizer to use during `fit`,possible values are:
            [`GradientDescentOptimizer`, `AdamOptimizer`, `AdadeltaOptimizer`]. Defaults to
            :attr:`AdamOptimizer`.
    """

    def __init__(
        self, continuous_columns,datecolumns=None, output='output', gpu=None, max_epoch=5, steps_per_epoch=10000,
         batch_size=8, z_dim=200, noise=0.2,
        l2norm=0.00001, learning_rate=0.001, num_gen_rnn=100, num_gen_feature=100,
        num_dis_layers=1, num_dis_hidden=100, optimizer='AdamOptimizer',save_checkpoints=True, restore_session=True
    ):
        """Initialize object."""
        # Output
        self.datecolumns=datecolumns
        self.continuous_columns = continuous_columns
        self.log_dir = os.path.join(output, 'logs')
        self.model_dir = os.path.join(output, 'model')
        self.output = output

        # Training params
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch
        self.save_checkpoints = save_checkpoints
        self.restore_session = restore_session

        # Model params
        self.model = None
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.noise = noise
        self.l2norm = l2norm
        self.learning_rate = learning_rate
        self.num_gen_rnn = num_gen_rnn
        self.num_gen_feature = num_gen_feature
        self.num_dis_layers = num_dis_layers
        self.num_dis_hidden = num_dis_hidden
        self.optimizer = optimizer

        if gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        self.gpu = gpu

    def get_model(self, training=True):
        """Return a new instance of the model."""
        return GAN(
            metadata=self.metadata,
            batch_size=self.batch_size,
            z_dim=self.z_dim,
            noise=self.noise,
            l2norm=self.l2norm,
            learning_rate=self.learning_rate,
            num_gen_rnn=self.num_gen_rnn,
            num_gen_feature=self.num_gen_feature,
            num_dis_layers=self.num_dis_layers,
            num_dis_hidden=self.num_dis_hidden,
            optimizer=self.optimizer,
            training=training
        )
    
    def prepare_sampling(self):
        """Prepare model for generate samples."""
        if self.model is None:
            self.model = self.get_model(training=False)

        else:
            self.model.is_training = False

        predict_config = PredictConfig(
            session_init=SaverRestore(self.restore_path),
            model=self.model,
            input_names=['z'],
            output_names=['gen/gen', 'z'],
        )

        self.simple_dataset_predictor = SimpleDatasetPredictor(
            predict_config,
            RandomZData((self.batch_size, self.z_dim))
        )

    def fit(self, data,config,datapath):
        """Fit the model to the given data.

        Args:
            data(pandas.DataFrame): dataset to fit the model.

        Returns:
            None

        """
        
        
        
        
        if os.path.exists("caches/cluster_data.json") and os.path.getsize("caches/cluster_data.json") > 0:#sometimes if json throws a error it will create a empty json file
            print(True)
            with open("caches/cluster_data.json", 'r') as read_file:
                self.metadata = json.load(read_file,object_hook=json_numpy_obj_hook)
            self.preprocessor = Preprocessor(continuous_columns=self.continuous_columns,metadata=self.metadata,datecolumns=self.datecolumns)
                #self.preprocessor = Preprocessor(continuous_columns=self.continuous_columns,metadata=self.metadata)
            data=self.metadata["transformed_data"]
        else:
            print(False)
            self.preprocessor = Preprocessor(continuous_columns=self.continuous_columns,datecolumns=self.datecolumns)
            data = self.preprocessor.fit_transform(data)
            self.metadata = self.preprocessor.metadata
            config["Dataname"]=datapath
            with open("Config.json",'w') as f:
                f.write(json.dumps(config))
            #print(self.metadata)
            self.metadata["transformed_data"]=data
            #print(self.metadata["transformed_data"])
            with open("caches/cluster_data.json",'w') as f:
                f.write(json.dumps(self.metadata,cls=NumpyEncoder))
        dataflow = TGANDataFlow(data, self.metadata)
        batch_data = BatchData(dataflow, self.batch_size)
        input_queue_= QueueInput(batch_data)

        self.model = self.get_model(training=True)

        trainer = GANTrainer(
            model=self.model,
            input_queue=input_queue_,
        )

        self.restore_path = os.path.join(self.model_dir, 'checkpoint')

        if os.path.isfile(self.restore_path) and self.restore_session:
            session_init = SaverRestore(self.restore_path)
            with open(os.path.join(self.log_dir, 'stats.json')) as f:
                starting_epoch = json.load(f)[-1]['epoch_num'] + 1

        else:
            session_init = None
            starting_epoch = 1

        action = 'k' if self.restore_session else None
        logger.set_logger_dir(self.log_dir, action=action)

        callbacks = []
        if self.save_checkpoints:
            callbacks.append(ModelSaver(checkpoint_dir=self.model_dir))

        trainer.train_with_defaults(
            callbacks=callbacks,
            steps_per_epoch=self.steps_per_epoch,
            max_epoch=self.max_epoch,
            session_init=session_init,
            starting_epoch=starting_epoch
        )

        self.prepare_sampling()

    def sample(self, num_samples):
        """Generate samples from model.

        Args:
            num_samples(int)

        Returns:
            None

        Raises:
            ValueError

        """
        max_iters = (num_samples // self.batch_size)

        results = []
        for idx, o in enumerate(self.simple_dataset_predictor.get_result()):
            results.append(o[0])
            if idx + 1 == max_iters:
                break

        results = np.concatenate(results, axis=0)

        ptr = 0
        features = {}
        
        for col_id, col_info in enumerate(self.metadata['details']):
            if col_info['type'] == 'category':
                features['f%02d' % col_id] = results[:, ptr:ptr + 1]
                ptr += 1

            elif col_info['type'] == 'value':
                gaussian_components = col_info['n']
                val = results[:, ptr:ptr + 1]
                ptr += 1
                pro = results[:, ptr:ptr + gaussian_components]
                ptr += gaussian_components
                features['f%02d' % col_id] = np.concatenate([val, pro], axis=1)

            else:
                raise ValueError(
                    "self.metadata['details'][{}]['type'] must be either `category` or "
                    "`values`. Instead it was {}.".format(col_id, col_info['type'])
                )

        return self.preprocessor.reverse_transform(features)[:num_samples].copy()

    def tar_folder(self, tar_name):
        """Generate a tar of :self.output:."""
        with tarfile.open(tar_name, 'w:gz') as tar_handle:
            for root, dirs, files in os.walk(self.output):
                for file_ in files:
                    tar_handle.add(os.path.join(root, file_))

            tar_handle.close()

    @classmethod
    def load(cls, path):
        """Load a pretrained model from a given path."""
        with tarfile.open(path, 'r:gz') as tar_handle:
            destination_dir = os.path.dirname(tar_handle.getmembers()[0].name)
            tar_handle.extractall()

        with open('{}/TGANModel'.format(destination_dir), 'rb') as f:
            instance = pickle.load(f)

        instance.prepare_sampling()
        return instance

    def save(self, path, force=False):
        """Save the fitted model in the given path."""
        if os.path.exists(path) and not force:
            logger.info('The indicated path already exists. Use `force=True` to overwrite.')
            return

        base_path = os.path.dirname(path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model = self.model
        dataset_predictor = self.simple_dataset_predictor

        self.model = None
        self.simple_dataset_predictor = None

        with open('{}/TGANModel'.format(self.output), 'wb') as f:
            pickle.dump(self, f)

        self.model = model
        self.simple_dataset_predictor = dataset_predictor

        self.tar_folder(path)

        logger.info('Model saved successfully.')

