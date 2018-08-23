from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader):
        super(ExampleTrainer, self).__init__(sess, model, config, logger, data_loader)

    def train_epoch(self, epoch=None):
        loop = tqdm(range(self.config.num_epochs))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        loss_ = np.mean(losses)
        acc_ = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)

        print("epoch {}: loss->{}, acc->{}.".format(cur_epoch, loss_, acc_))

        summaries_dict = {
            'loss': loss_,
            'acc': acc_
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data_loader.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy], feed_dict=feed_dict)
        return loss, acc
