import torch
from torch import nn
from transformer import Transformer
import pre_processing_raw_train_data_database
import torch.utils.data
import pickle
import torch.nn.functional as F
import random
import pre_processing_convert_to_bpes
import radam
import ranger
from torch.utils.tensorboard import SummaryWriter
import time
import torch.optim as optim


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


# TODO Think about the implications of having a batch system that gets batch by size, maybe shuffle it?

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
        current_size_enc = 0  # current size of the training data in bytes
        current_size_dec = 0
        child_length, parent_length = self.generate_number_return_parent_child()
        parent_length_max = parent_length  # stores the parent length max so that we can correctly pad the tensor
        child_length_max = child_length  # stores the child length max so that we can correctly pad the tensor
        query = self.data_set.__getitem__(child_length, parent_length)  # gets rows with certain child and parent length
        # keeps adding data until we exceed the data wanted size of the data
        while (current_size_enc + parent_length_max * 8) <= self.batch_size_bytes and (current_size_dec + child_length_max * 8) <= self.batch_size_bytes:
            for training_point in query:  # iterates over all of the rows within the query
                if (current_size_enc + parent_length_max * 8) <= self.batch_size_bytes and (current_size_dec + child_length_max * 8) <= self.batch_size_bytes:

                    parent = pickle.loads(training_point[2])  # loads the parent comment tensor
                    parent = F.pad(parent, [0, parent_length_max-parent.shape[0]], value=14950)  # pads it to the correct length
                    encoder.append(parent)  # adds it to the encoder
                    child = pickle.loads(training_point[1])  # loads the child comment tensor
                    # pads the child comment to the correct length

                    child = F.pad(child, [0, child_length_max-child.shape[0]], value=14950)

                    # Adds the time steps to the decoder

                    decoder.append(child)

                    # Changes current size to reflect what has just been added
                    current_size_enc += parent_length_max * 8
                    current_size_dec += child_length_max * 8
                else:
                    encoder, decoder = torch.stack(encoder), torch.stack(decoder)
                    random_int = random.randint(0, encoder.shape[0])
                    return torch.cat((encoder[random_int:], encoder[:random_int])), torch.cat(
                        (decoder[random_int:], decoder[:random_int]))

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
                current_size_enc = 0
                current_size_dec = 0
                child_length, parent_length = self.generate_number_return_parent_child()
                parent_length_max = parent_length
                child_length_max = child_length

            query = self.data_set.__getitem__(child_length, parent_length)

        encoder, decoder = torch.stack(encoder), torch.stack(decoder)
        random_int = random.randint(0, encoder.shape[0])
        return torch.cat((encoder[random_int:], encoder[:random_int])), torch.cat((decoder[random_int:], decoder[:random_int]))

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
        max_len = 150
        if self.training_point_parent_child_tensor is None:
            # makes sure the training points indexing has been
            # loaded and the training points have been counted
            self.index_number_of_training_points()
            self.get_num_training_data()

        while True:
            # Gets a random index used to get a parent and child comment
            index = random.randint(0, self.num_training_points)
            break_out = False
            # Goes through the indexed tensor and uses it to find where the index lies
            for child_length in range(2000):
                for parent_length in range(2000):
                    index -= self.training_point_parent_child_tensor[parent_length, child_length]
                    if index <= 0:
                        if parent_length <= max_len and child_length <= max_len:
                            return child_length, parent_length
                        else:
                            break_out = True
                    if break_out:
                        break
                if break_out:
                    break


class GetBatch(nn.Module):
    def __init__(self, max_n):
        super(GetBatch, self).__init__()
        self.max_n = max_n
        self.n = 0

    def forward(self):
        enc_inp = None
        while enc_inp is None:
            try:
                enc_inp, dec_inp = pickle.load(open(f"batch_{self.n}", "rb"))
                if enc_inp is None:
                    time.sleep(0.01)
                    self.n += 1
                    if self.n == self.max_n:
                        self.n = 0
            except EOFError:
                pass
        # dec_inp[dec_inp == 1] = 14950
        # print(dec_inp)
        pickle.dump((None, None), open(f"batch_{self.n}", "wb"))
        return enc_inp, dec_inp


class ScheduledOptimiser(nn.Module):
    def __init__(self, optimiser, init_lr, model_dim, warmup_steps):
        super(ScheduledOptimiser, self).__init__()
        self.optimiser = optimiser
        self.init_lr = init_lr
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        self.steps = 0

    def step(self):
        self.update_lr()
        self.optimiser.step()

    def zero_grad(self):
        self.optimiser.zero_grad()

    def update_lr(self):
        self.steps += 1
        lr = self.init_lr * ((self.model_dim ** -0.5) * min(self.steps ** (-0.5), self.steps * self.warmup_steps ** (-1.5)))

        for param in self.optimiser.param_groups:
            param['lr'] = lr


class TrainingSkeleton(nn.Module):

    def __init__(self, epochs=1, save_rate=250):
        super(TrainingSkeleton, self).__init__()
        self.vocab = pickle.load(open("vocab_dict.pickle", "rb"))
        self.DataLoader = DataLoader(3211, True)
        # self.transformer = Transformer(6, 8, 20, 64, 1280, 2048, 0.25, "cuda:0", "cuda:1", "cuda:1", len(self.vocab))
        self.transformer = Transformer(6, 8, 16, 64, 1024, 4096, 0.3, "cuda:0", "cuda:1", "cuda:1", len(self.vocab))

        self.epochs = epochs
        # self.optimizer = ranger.Ranger(self.transformer.parameters())
        self.optimizer = ScheduledOptimiser(optim.Adam(self.transformer.parameters(), betas=(0.9, 0.98), eps=1e-09), 2.0, 1024, 8000)
        self.loss = nn.CrossEntropyLoss(weight=pickle.load(open("representative_vocab.pickle", "rb")).to("cuda:1"))
        self.summary_writer = SummaryWriter('runs')
        self.save_rate = save_rate
        self.batcher = GetBatch(10)

        # run tensorboard --logdir=runs from scripts folder to get tensorboard server

    def make_expected_values(self, dec_inp):  # Converts the decoder values into values acceptable to the model
        expected = dec_inp.transpose(0, 1)[1:].transpose(0, 1).contiguous().view(-1).to("cuda:1").long()
        return expected

    def training_loop(self, model_id=None):  # The main training loop
        if model_id is not None:
            checkpoint = self.load_model(model_id)
            start_epoch = checkpoint['epoch']
            start_iteration = checkpoint['iteration'] + 1
            del checkpoint

        else:
            start_epoch = 0
            start_iteration = 0
        self.transformer.train()
        for epoch in range(start_epoch, self.epochs):
            for iteration in range(start_iteration, int(self.DataLoader.num_training_points)):
                enc_inp, dec_inp = self.batcher()
                expected = self.make_expected_values(dec_inp)
                dec_inp = dec_inp.transpose(0, 1)[:-1].transpose(0, 1).contiguous()

                self.optimizer.zero_grad()  # Zeros the gradient
                print(dec_inp.shape)
                prediction = self.transformer(enc_inp, dec_inp)  # .view(-1, len(self.vocab))
                print(prediction[0].argmax(-1), expected[:dec_inp.shape[1]])

                loss = self.loss(prediction.view(-1, 14951), expected)  # Gets the loss of output of the model
                loss.backward()
                print(f"epoch:{epoch}       iteration: {iteration}      loss: {loss.item()}")
                # self.optimizer.step()  # The backwards propagation step
                self.optimizer.step()

                # Used for the data visualisation
                self.summary_writer.add_scalar('Training_loss', loss.item(),
                                               epoch*self.DataLoader.num_training_points+iteration)
                self.summary_writer.flush()

                if iteration % self.save_rate == 0 and iteration != 0:  # Saves the model
                    self.save_model(f"{epoch}_{iteration}", epoch, iteration)
                    self.test(gen=True)
                    self.transformer.train()

    def save_model(self, model_id, epoch, iteration):  # Saves the Model
        torch.save(
            {
                'model_state_dict': self.transformer.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch,
                'iteration': iteration
            },
            f"./model_saving/model_{model_id}.pt")

    def load_model(self, model_id):  # Loads the model

        checkpoint = torch.load(f"model_saving/model_{model_id}.pt")

        self.transformer.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint

    def test(self, usr="", gen=False):
        if gen:
            enc = self.batcher()[0][0]
            while len(enc) > 40:
                enc = self.batcher()[0][0]

            enc_normalised = ""
            for i in range(len(enc)):
                for k, v in self.vocab.items():
                    if v == enc[i]:
                        enc_normalised += k
                        break
            print(f"encoder input: {enc_normalised}")
        else:
            # Used for converting to BEP tensor
            usr = pre_processing_convert_to_bpes.convert_tokens_to_bpes(usr, self.vocab)
            for count, token in enumerate(usr):
                usr[count] = self.vocab[token]
            usr = [2] + usr + [3]
            enc = torch.zeros((len(usr)))
            for count in range(len(usr)):
                enc[count] = usr[count]

        self.transformer.eval()  # Set transformer to evaluation mode, so dropout + others are disabled
        outputs = torch.zeros(1)  # Sets a tensor to the GEN tag
        dec_normalised = ""  # The string that holds the response
        endgen = False  # The flag for ending generation
        num_tokens = 0  # Holds the number of tokens
        while endgen is False:
            # Gets transformer response
            output = self.transformer(enc.unsqueeze(0), outputs.unsqueeze(0))[-1].argmax(-1)[-1]
            outputs = torch.cat((outputs.cpu().float(), output.cpu().unsqueeze(0).float()))
            # Used to figure out which token was generated
            for k, v in self.vocab.items():
                if output == 1 or num_tokens > 40:
                    dec_normalised += "<|ENDGEN|>"
                    endgen = True
                    break
                if v == output:
                    dec_normalised += k
                    num_tokens += 1
                    break

                elif output == 14947:
                    num_tokens += 1
                    dec_normalised += " "
                    break

        print(f"test: Decoder output: {dec_normalised}")
        return dec_normalised


if __name__ == "__main__":
    a = TrainingSkeleton()
    #a.load_model("0_1750")
    #a.test("Hello, how are you?", True)
    a.training_loop("0_15500")






