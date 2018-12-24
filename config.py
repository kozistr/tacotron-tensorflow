import argparse


args_list = []
parser = argparse.ArgumentParser()


def add_arg_group(name):
    """
    :param name: A str. Argument group.
    :return: An list. Arguments.
    """
    arg = parser.add_argument_group(name)
    args_list.append(arg)
    return arg


def get_config():
    cfg, un_parsed = parser.parse_known_args()
    return cfg, un_parsed


# Network
network_arg = add_arg_group('Network')
network_arg.add_argument('--model', type=str)
network_arg.add_argument('--n_encoder_banks', type=int, default=16)
network_arg.add_argument('--n_decoder_banks', type=int, default=8)
network_arg.add_argument('--n_highway_blocks', type=int, default=4)
network_arg.add_argument('--reduction_factor', type=int, default=5)
network_arg.add_argument('--filter_size', type=int, default=256)
network_arg.add_argument('--fc_unit', type=int, default=1024)
network_arg.add_argument('--dropout', type=float, default=.5)


# DataSet
data_arg = add_arg_group('DataSet')
data_arg.add_argument('--sample_rate', type=int, default=22050,
                      help="sample rate of an audio")
data_arg.add_argument('--n_fft', type=int, default=2048,
                      help="fft points (samples)")
data_arg.add_argument('--frame_shift', type=float, default=0.0125)
data_arg.add_argument('--frame_length', type=float, default=0.05)
data_arg.add_argument('--n_mels', type=int, default=80)
data_arg.add_argument('--power', type=float, default=1.2,
                      help="Exponent for amplifying the predicted magnitude")
data_arg.add_argument('--n_iters', type=int, default=50)
data_arg.add_argument('--preemphasis', type=float, default=.97)
data_arg.add_argument('--max_db', type=float, default=100.)
data_arg.add_argument('--min_db', type=float, default=20.)
data_arg.add_argument('--embed_size', type=int, default=256,
                      help='the size of character/word embedding vector')
data_arg.add_argument('--vocab_size', type=int, default=391587)
data_arg.add_argument('--batch_size', type=int, default=32)


# Train/Test hyper-parameters
train_arg = add_arg_group('Training')
train_arg.add_argument('--is_train', type=bool, default=True)
train_arg.add_argument('--epochs', type=int, default=10)
train_arg.add_argument('--logging_step', type=int, default=500)
train_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adadelta'])
train_arg.add_argument('--l2_reg', type=float, default=5e-4)
train_arg.add_argument('--grad_clip', type=float, default=5.)
train_arg.add_argument('--lr', type=float, default=1e-3)
train_arg.add_argument('--lr_decay', type=float, default=.75)
train_arg.add_argument('--lr_lower_boundary', type=float, default=2e-5)
train_arg.add_argument('--test_size', type=float, default=.2)


# Misc
misc_arg = add_arg_group('Misc')
misc_arg.add_argument('--device', type=str, default='gpu')
misc_arg.add_argument('--n_threads', type=int, default=8,
                      help='the number of workers for speeding up')
misc_arg.add_argument('--seed', type=int, default=1337)
misc_arg.add_argument('--verbose', type=bool, default=True)
