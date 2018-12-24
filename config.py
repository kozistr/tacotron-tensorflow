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
network_arg.add_argument('--mode', type=str, default='non-static', choices=['static', 'non-static'])
network_arg.add_argument('--model', type=str, default='charcnn', choices=['charcnn', 'charrnn'])
network_arg.add_argument('--filter_size', type=int, default=256,
                         help='conv1d filter size')
network_arg.add_argument('--fc_unit', type=int, default=1024)
network_arg.add_argument('--drop_out', type=float, default=.7)


# DataSet
data_arg = add_arg_group('DataSet')
data_arg.add_argument('--embed_size', type=int, default=300,
                      help='the size of character/word embedding vector')
data_arg.add_argument('--vocab_size', type=int, default=391587,
                      help='default is w2v vocab size')
data_arg.add_argument('--character_size', type=int, default=251,
                      help='number of korean chars')
data_arg.add_argument('--sequence_length', type=int, default=400,
                      help='the length of the sentence, default is c2v max words cnt')
data_arg.add_argument('--batch_size', type=int, default=128)


# Train/Test hyper-parameters
train_arg = add_arg_group('Training')
train_arg.add_argument('--is_train', type=bool, default=True)
train_arg.add_argument('--epochs', type=int, default=10)
train_arg.add_argument('--logging_step', type=int, default=500)
train_arg.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adadelta'])
train_arg.add_argument('--l2_reg', type=float, default=5e-4)
train_arg.add_argument('--grad_clip', type=float, default=5.)
train_arg.add_argument('--lr', type=float, default=2e-4)
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
