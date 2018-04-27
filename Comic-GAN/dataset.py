import utils

class Dataset:
    def __init__(self, data_files, image_width=96, image_height=96, scale_func=None):
        """
        Initialize the class
        :param data_files: List of files in the database
        :param image_width:
        :param image_height:
        :param scale_func: Scale function
        """
        image_channels = 3
        
        if scale_func is None:
            self.scaler = utils.scale
        else:
            self.scaler = scale_func
            
        self.image_mode = 'RGB'
        self.data_files = data_files
        self.shape = len(data_files), image_width, image_height, image_channels
        
    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch size
        :return Batches of data
        """
        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode)

            current_index += batch_size
            
            yield self.scaler(data_batch)