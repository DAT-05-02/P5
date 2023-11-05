
import tensorflow as tf
import psutil

# a simple callback for logging gpu memory usage
class GpuMemoryCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir:str, freq: int):
        self.totalStep = 0
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
        self.totalStep += 1
        
        # only logging memory use at the given frequency
        if self.totalStep % self.freq == 0 :
            with self.writer.as_default():
                totalUsage = tf.config.experimental.get_memory_usage('GPU:0')
                tf.summary.scalar("gpu_memory_use [mB]", totalUsage / 1e6, step = self.totalStep)
                self.writer.flush()

# a simple callback for logging cpu memory usage
class CpuMemoryCallback(tf.keras.callbacks.Callback):

    def __init__(self, log_dir:str, freq: int):
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
        if self.totalStep % self.freq == 0 :
            with self.writer.as_default():
                totalUsage = self.svmem.used
                tf.summary.scalar("total_cpu_memory_use [mB]", totalUsage / 1e6, step = self.totalStep)
                self.writer.flush()