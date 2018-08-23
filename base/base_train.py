import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, config, logger, data_loader=None):
        '''
        Contructing the trainer
        :param sess: TF.Session() instance
        :param model: The model instance
        :param config: config namespace which will contain all configurations you have specified in the jason
        :param logger: logger class which will summarize and write the values to the tensorboard
        :param data_loader: the data loader if specified.
        '''
        # Assign all class attributes
        self.sess = sess
        self.model = model
        self.config = config
        self.logger = logger
        if data_loader is not None:
            self.data_loader = data_loader

        # initialize all variables of the graph
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        '''
        this is the main loop of training
        Looping on the epochs
        :return:
        '''
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, epoch=None):
        '''
        implement the logic of epoch:
        - loop over the number of iterations in the config and call the train step
        - add any summaries you want using the summary
        :param epoch: take the number of epoch if you are interested
        :return:
        '''
        raise NotImplementedError

    def train_step(self):
        '''
        implement the logic of the train step
        - run the tensorflow session
        :return:
        '''
        raise NotImplementedError