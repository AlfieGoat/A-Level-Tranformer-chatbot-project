import torch
from torch import nn
from transformer import Transformer
import pre_processing_raw_train_data_database
import torch.utils.data
import pickle
import torch.nn.functional as F
import random

import radam
import ranger
from torch.utils.tensorboard import SummaryWriter


class DataSet(torch.utils.data.Dataset):

    def __init__(self):
        super(DataSet, self).__init__()
        self.db = pre_processing_raw_train_data_database.Database()

    def __getitem__(self, child_length, parent_length):
        # Gets an item from the database
        return self.db.custom_call(f"SELECT * FROM trainData WHERE child_length = {child_length} AND parent_length = "
                                   f"{parent_length}").fetchall()

    def __len__(self):
        # Uses a binary search to find the number of data points
        lower_bound = 0
        upper_bound = 1000000000000
        old_middle = -1
        while True:
            middle = (lower_bound + upper_bound) // 2
            value = self.db.get_train_data_by_id(middle)
            if middle == old_middle:
                return middle
            if value is None:
                upper_bound = middle - 1
            else:
                lower_bound = middle + 1
            old_middle = middle


class DataLoader:

    def __init__(self, batch_size_bytes, open_and_load_index=False):
        self.data_set = DataSet()
        self.batch_size_bytes = batch_size_bytes
        self.num_training_points = 0
        self.num_training_points_per_parent_child = torch.zeros((2000, 2000))
        self.training_point_parent_child_tensor = None

        if open_and_load_index:
            self.get_num_training_data()

    def set_batch_size_bytes(self, size):
        self.batch_size_bytes = size

    def get_num_training_data(self):
        self.training_point_parent_child_tensor = pickle.load(open("num_training_points_per_parent_child.pickle", "rb"))
        self.num_training_points = self.training_point_parent_child_tensor.sum()

    def get_batch(self):  # 4 bytes per 32 bit int
        encoder = []  # stores training data for the encoder
        decoder = []  # stores training data for the decoder
        current_size = 0  # current size of the training data in bytes
        child_length, parent_length = self.generate_number_return_parent_child()
        parent_length_max = parent_length  # stores the parent length max so that we can correctly pad the tensor
        child_length_max = child_length  # stores the child length max so that we can correctly pad the tensor
        query = self.data_set.__getitem__(child_length, parent_length)  # gets rows with certain child and parent length
        # keeps adding data until we exceed the data wanted size of the data
        while current_size + (parent_length_max + child_length_max) * 8 < self.batch_size_bytes:
            for training_point in query:  # iterates over all of the rows within the query
                if current_size + (parent_length_max + child_length_max) * 8 >= self.batch_size_bytes:
                    return torch.stack(encoder), torch.stack(decoder)
                parent = pickle.loads(training_point[2])  # loads the parent comment tensor
                parent = F.pad(parent, [0, parent_length_max-parent.shape[0]], value=14950)  # pads it to the correct length
                encoder.append(parent)  # adds it to the encoder
                child = pickle.loads(training_point[1])  # loads the child comment tensor
                # pads the child comment to the correct length

                child = F.pad(child, [0, child_length_max-child.shape[0]], value=14950)

                # Adds the time steps to the decoder

                decoder.append(child)

                # Changes current size to reflect what has just been added
                current_size += (parent_length_max + child_length_max) * 8
            # print(current_size/self.batch_size_bytes, parent_length, child_length)
            # If we have iterated through the whole query, we will change the parent and child length to load more
            if parent_length >= 3:
                parent_length -= 1

            elif child_length >= 3:
                child_length -= 1
                parent_length = parent_length_max

            else:  # if we have reached the minimum parent and child length, we reset everything and try again
                encoder = []
                decoder = []
                current_size = 0
                child_length, parent_length = self.generate_number_return_parent_child()
                parent_length_max = parent_length
                child_length_max = child_length

            query = self.data_set.__getitem__(child_length, parent_length)

        return torch.stack(encoder), torch.stack(decoder)

    def index_number_of_training_points(self, create=False, save=False):
        if create:
            for child_length in range(2000):
                for parent_length in range(2000):
                    items = self.data_set.__getitem__(child_length, parent_length)
                    self.num_training_points_per_parent_child[parent_length, child_length] = len(items)
                    print(parent_length, child_length, len(items))

            if save:
                pickle.dump(self.num_training_points_per_parent_child,
                            open("num_training_points_per_parent_child.pickle", "wb"))
        else:
            self.num_training_points_per_parent_child = \
                pickle.load(open("num_training_points_per_parent_child.pickle", "rb"))

    def generate_number_return_parent_child(self):
        if self.training_point_parent_child_tensor is None:
            # makes sure the training points indexing has been
            # loaded and the training points have been counted
            self.index_number_of_training_points()
            self.get_num_training_data()

        # Gets a random index used to get a parent and child comment
        index = random.randint(0, self.num_training_points)

        # Goes through the indexed tensor and uses it to find where the index lies
        for child_length in range(2000):
            for parent_length in range(2000):
                index -= self.training_point_parent_child_tensor[parent_length, child_length]
                if index <= 0:
                    return child_length, parent_length


class TrainingSkeleton(nn.Module):

    def __init__(self, epochs=1, batch_size_bytes=195312, save_rate=25):
        super(TrainingSkeleton, self).__init__()
        self.vocab = pickle.load(open("vocab_dict.pickle", "rb"))
        self.DataLoader = DataLoader(batch_size_bytes, True)
        self.transformer = Transformer(6, 8, 8, 64, 512, 2048, 0.1, "cuda:0", "cuda:1", len(self.vocab))
        self.epochs = epochs
        self.optimizer = ranger.Ranger(self.transformer.parameters())
        self.loss = nn.CrossEntropyLoss()
        self.summary_writer = SummaryWriter('runs')
        self.save_rate = save_rate
        # run tensorboard --logdir=runs from scripts folder to get tensorboard server

    @staticmethod
    def make_expected_values(dec_inp):  # Converts the decoder values into values acceptable to the model
        return (dec_inp*0).to("cuda:1").long()  # TODO Add expected values

    def training_loop(self, model_id=None):  # The main training loop
        if model_id is not None:
            checkpoint = self.load_model(model_id)
            self.transformer.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            start_iteration = checkpoint['iteration']

        else:
            start_epoch = 0
            start_iteration = 0

        for epoch in range(start_epoch, self.epochs):
            for iteration in range(start_iteration, int(self.DataLoader.num_training_points)):
                self.optimizer.zero_grad()  # Zeros the gradient
                enc_inp, dec_inp = self.DataLoader.get_batch()  # Gets a batch
                # sends data to the transformer
                print(dec_inp)
                prediction = self.transformer(enc_inp, dec_inp).view(-1, len(self.vocab))
                print(prediction)
                expected = self.make_expected_values(dec_inp).flatten()  # Makes the expected values
                loss = self.loss(prediction, expected)  # Gets the loss of output of the model
                loss.backward()
                self.optimizer.step()  # The backwards propagation step

                # Used for the data visualisation
                self.summary_writer.add_scalar('Training_loss', 5,
                                               epoch*self.DataLoader.num_training_points+iteration)
                self.summary_writer.flush()

                if iteration % self.save_rate == 0:  # Saves the model
                    self.save_model(f"{epoch}_{iteration}", epoch, iteration)

            start_iteration = 0  # Added so after model is loaded the start iteration goes back to 0

    def save_model(self, model_id, epoch, iteration):  # Saves the Model
        torch.save(
            {
                'model_state_dict': self.transformer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'iteration': iteration
            },
            f"./model_saving/model_{model_id}.pt")

    @staticmethod
    def load_model(model_id):  # Loads the model
        checkpoint = torch.load(f"model_saving/model_{model_id}.pt")
        return checkpoint




a = TrainingSkeleton()
a.training_loop()






