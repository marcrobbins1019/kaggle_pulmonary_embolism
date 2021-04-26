from tensorflow.keras.utils import Sequence


class PE3DGenerator(Sequence):

    def __init__(self, train_df, validation_df, batch_size):
        self.x, self.y = train_df, validation_df
        self.batch_size = batch_size

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
            for file_name in batch_x]), np.array(batch_y)
