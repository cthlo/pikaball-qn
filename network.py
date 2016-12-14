import config as cfg
import tensorflow as tf
import math

class _NN(object):
    def __init__(self, active_nn=None):
        with tf.name_scope('Input'):
            self.state = state = tf.placeholder(tf.float32, [None, cfg.state_size], name='State')

        with tf.name_scope('Hidden'):
            d = 1/math.sqrt(cfg.state_size+1) # input nodes count
            W1 = tf.Variable(tf.random_normal([cfg.state_size, cfg.hidden_size], -d, d), name='Input_weights')
            b1 = tf.Variable(tf.random_normal([1, cfg.hidden_size], -d, d), name='Input_bias')
            hidden = tf.tanh(tf.matmul(state, W1) + b1, name='Hidden')

        with tf.name_scope('Q-values'):
            d = 1/math.sqrt(cfg.hidden_size+1) # hidden nodes count
            W2 = tf.Variable(tf.random_normal([cfg.hidden_size, cfg.num_actions], -d, d), name='Hidden_weights')
            b2 = tf.Variable(tf.random_normal([1, cfg.num_actions], -d, d), name='Hidden_bias')
            self.qvals = qvals = tf.add(tf.matmul(hidden, W2), b2, name='Q-values')

        with tf.name_scope('Best_action'):
            self.max_q = tf.reduce_max(qvals, 1, name='Maximum_Q-value')
            self.best_action = tf.gather(tf.argmax(qvals, 1), 0, name='Best_action')

        model = [W1, b1, W2, b2]
        if active_nn:
            with tf.name_scope('Copy_from_active_network'):
                self.copy_from_active_nn = [vt.assign(vf) for vt, vf in zip(model, active_nn.model)]
        else:
            self.model = model
            self.__build_optim()

    def __build_optim(self):
        with tf.name_scope('Mean_squared_error'):
            self.targ_qvals = tf.placeholder(tf.float32, [None], name='Target_Q-values')
            self.actions = tf.placeholder(tf.int32, [None], name='Actions')
            est_qvals = tf.reduce_sum(self.qvals*tf.one_hot(self.actions, cfg.num_actions), 1, name='Estimated_Q-values')
            loss = tf.reduce_mean(tf.square(self.targ_qvals-est_qvals), name='Loss')

        with tf.name_scope('Counters'):
            self.glob_step = glob_step = tf.Variable(0, name='Global_step', trainable=False)
            #f_glob_step = tf.cast(gs, tf.float32, name='Global step (float)')

        with tf.name_scope('Hyper'):
            learn_rate = tf.maximum(
                cfg.min_alpha,
                tf.train.exponential_decay(
                    cfg.init_alpha,
                    glob_step,
                    cfg.alpha_decay_steps,
                    cfg.alpha_decay_rate,
                    staircase=True
                ), name='Learning_rate'
            )

        with tf.name_scope('Optimizer'):
            optim = tf.train.RMSPropOptimizer(learning_rate=learn_rate, momentum=cfg.momentum)
            self.train = optim.minimize(loss, global_step=glob_step)

        with tf.name_scope('Summary'):
            tf.scalar_summary('Learning_rate', learn_rate)
            tf.scalar_summary('Loss', loss)
            self.summary = tf.merge_all_summaries()

class Network(object):
    def __init__(self):
        with tf.name_scope('Active_net'):
            self.active_net = _NN()

        self.freeze_net = cfg.targ_net_update_freq > 0
        if self.freeze_net:
            with tf.name_scope('Target_net'):
                self.target_net = _NN(active_nn=self.active_net)
        else:
            self.target_net = self.active_net

        self.sess = sess = tf.Session()
        self.writer = tf.train.SummaryWriter(cfg.summary_folder, sess.graph)
        self.extra_summaries = {}

        sess.run(tf.initialize_all_variables())

    def ffwd(self, state):
        an = self.active_net
        return self.sess.run([an.qvals, an.best_action, an.glob_step], {an.state: state})

    def train(self, pre_states, actions, rewards, pos_states):
        an, tn = self.active_net, self.target_net
        maxqs = self.sess.run(tn.max_q, {tn.state: pos_states})
        target_qs = cfg.gamma*maxqs + rewards
        _, summary, gs = self.sess.run([an.train, an.summary, an.glob_step], {
            an.state: pre_states,
            an.actions: actions,
            an.targ_qvals: target_qs
        })
        self.writer.add_summary(summary, gs)

        if self.freeze_net and gs % cfg.targ_net_update_freq == 0:
            print('Update target network at global step %d' % gs)
            self.sess.run(tn.copy_from_active_nn)

    def record_scalar(self, name, value):
        try:
            op, ph = self.extra_summaries[name]
        except KeyError:
            ph = tf.placeholder(tf.float32, name=name)
            op = tf.scalar_summary(name, ph)
            self.extra_summaries[name] = (op, ph)
        summary, gs = self.sess.run([op, self.active_net.glob_step], {ph: value})
        self.writer.add_summary(summary, gs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.sess.close()
