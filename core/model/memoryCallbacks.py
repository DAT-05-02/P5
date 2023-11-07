import tensorflow as tf
import psutil


# a simple callback for logging gpu memory usage
class GpuMemoryCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir: str, freq: int):
        super().__init__()
        self.total_step = 0
        self.freq = freq

        # setting up file writer
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_train_batch_end(self, *_):
        self.log_memory_use()

    def on_test_batch_end(self, *_):
        self.log_memory_use()

    def on_predict_batch_end(self, *_):
        self.log_memory_use()

    def log_memory_use(self):
        self.total_step += 1

        # only logging memory use at the given frequency
        if self.total_step % self.freq == 0:
            with self.writer.as_default():
                total_usage = tf.config.experimental.get_memory_info('GPU:0')['current']
                tf.summary.scalar("gpu_memory_use [mB]", total_usage / 1e6, step=self.total_step)
                self.writer.flush()


# a simple callback for logging cpu memory usage
class CpuMemoryCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir: str, freq: int):
        super().__init__()
        self.totalStep = 0
        self.freq = freq

        # setting up file writer
        self.writer = tf.summary.create_file_writer(log_dir)

        # getting virtual_memory object from psutil
        self.svmem = psutil.virtual_memory()

    def on_train_batch_end(self, *_):
        self.log_memory_use()

    def on_test_batch_end(self, *_):
        self.log_memory_use()

    def on_predict_batch_end(self, *_):
        self.log_memory_use()

    def log_memory_use(self):
        self.totalStep += 1

        # only logging memory use at the given frequency
        if self.totalStep % self.freq == 0:
            with self.writer.as_default():
                total_usage = self.svmem.used
                tf.summary.scalar("total_cpu_memory_use [mB]", total_usage / 1e6, step=self.totalStep)
                self.writer.flush()
