
import os
import json
# import argparse
import pprint
import datetime
# import torch
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
# from torch.utils import data
from bnaf import *
# from tqdm import tqdm

# from optim.adam import Adam
# from optim.lr_scheduler import ReduceLROnPlateau



from data.gas import GAS
from data.bsds300 import BSDS300
from data.hepmass import HEPMASS
from data.miniboone import MINIBOONE
from data.power import POWER

NAF_PARAMS = {
    'power': (414213, 828258),
    'gas': (401741, 803226),
    'hepmass': (9272743, 18544268),
    'miniboone': (7487321, 14970256),
    'bsds300': (36759591, 73510236)
}


def load_dataset(args):

    #convert datasets
    # data = pd.read_csv(r'C:\Users\just\PycharmProjects\BNAF\data\gas\ethylene_methane.txt', delim_whitespace=True, header='infer')
    # data.to_pickle('data/gas/ethylene_methane.pickle')

    if args.dataset == 'gas':
        # dataset = GAS('data/gas/ethylene_CO.pickle')
        dataset = GAS('data/gas/ethylene_methane.pickle') #actual loading file looked for methane????
    elif args.dataset == 'bsds300':
        dataset = BSDS300('data/BSDS300/BSDS300.hdf5')
    elif args.dataset == 'hepmass':
        dataset = HEPMASS('data/hepmass')
    elif args.dataset == 'miniboone':
        dataset = MINIBOONE('data/miniboone/data.npy')
    elif args.dataset == 'power':
        dataset = POWER('data/power/data.npy')
    else:
        raise RuntimeError()

    # dataset_train = torch.utils.data.TensorDataset(
    #     torch.from_numpy(dataset.trn.x).float().to(args.device))
    # data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_dim, shuffle=True)
    #
    # dataset_valid = torch.utils.data.TensorDataset(
    #     torch.from_numpy(dataset.val.x).float().to(args.device))
    # data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_dim, shuffle=False)
    #
    # dataset_test = torch.utils.data.TensorDataset(
    #     torch.from_numpy(dataset.tst.x).float().to(args.device))
    # data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_dim, shuffle=False)

    dataset_train = tf.data.Dataset.from_tensor_slices(dataset.trn.x)#.float().to(args.device)
    dataset_train.shuffle(buffer_size=len(dataset.trn.x)).repeat().batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    data_loader_train = tf.contrib.eager.Iterator(dataset_train)
    ##data_loader_train.get_next()

    dataset_valid = tf.data.Dataset.from_tensor_slices(dataset.val.x)#.float().to(args.device)
    dataset_valid.shuffle(buffer_size=len(dataset.val.x)).repeat().batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    data_loader_valid = tf.contrib.eager.Iterator(dataset_valid)
    ##data_loader_valid.get_next()

    dataset_test = tf.data.Dataset.from_tensor_slices(dataset.tst.x)#.float().to(args.device)
    dataset_test.shuffle(buffer_size=len(dataset.tst.x)).repeat().batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    data_loader_test = tf.contrib.eager.Iterator(dataset_test)
    ##data_loader_test.get_next()

    args.n_dims = dataset.n_dims
    
    return data_loader_train, data_loader_valid, data_loader_test


def create_model(args, verbose=False):

    flows = []
    for f in range(args.flows):

        #build internal layers for a single flow
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
                                       args.n_dims * args.hidden_dim, dim=args.n_dims))
            layers.append(Tanh())

        ## wrap each flow with layers that ensure consistency in dimensions.  Math to divide out the hidden_dimensions
        # units is performed in the MaskedWeight layer
        flows.append(
            BNAF(*([MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims), Tanh()] + \
                   layers + \
                   [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims)]),\
                 res=args.residual if f < args.flows - 1 else None
            )
        )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, 'flip'))

        model = Sequential(*flows)
        params = sum(sum(p != 0) if len(p.shape) > 1 else p.shape
                     for p in model.trainable_variables())
    
    if verbose:
        print('{}'.format(model))
        print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params, 
            NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims))
                
    if args.save and not args.load:
        with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
            print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params, 
                NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims), file=f)
    
    return model

def save_model(args, root):
    def f():
        if args.save:
            print('Saving model..')
            root.save(os.path.join(args.load or args.path, 'checkpoint.pt'))
    return f
    
    
def load_model(model, optimizer, args, root, load_start_epoch=False):
    def f():
        print('Loading model..')
        # root.restore(tf.train.latest_checkpoint(checkpoint_dir))
        root.restore(os.path.join(args.load or args.path, 'checkpoint.pt'))
        if load_start_epoch:
            args.start_epoch = tf.train.get_global_step().numpy()
    return f


def compute_log_p_x(model, x_mb):

    ## use tf.gradient + tf.convert_to_tensor + tf.GradientTape(persistent=True) to clean up garbage implementation in bnaf.py

    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = tf.reduce_sum(tf.distributions.Normal(tf.zeros_like(y_mb), tf.ones_like(y_mb)).log_prob(y_mb), axis=-1)#.sum(-1)
    return log_p_y_mb + log_diag_j_mb


def train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args):
    
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        t = tqdm(data_loader_train, smoothing=0, ncols=80)
        train_loss = []
        
        for x_mb, in t:
            with tf.GradientTape() as tape:
                loss = - tf.reduce_mean(compute_log_p_x(model, x_mb))

            loss.backward()
            optimizer.com
            tf.clip_grad_norm_(model.parameters(), clip_norm=args.clip_norm)

            optimizer.step()
            optimizer.zero_grad()
            
            t.set_postfix(loss='{:.2f}'.format(loss.item()), refresh=False)
            train_loss.append(loss)

            global_step.assign_add(1)
        
        train_loss = torch.stack(train_loss).mean()
        optimizer.swap()
        validation_loss = - torch.stack([compute_log_p_x(model, x_mb).mean().detach()
                                         for x_mb, in data_loader_valid], -1).mean()
        optimizer.swap()

        print('Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}'.format(
            epoch + 1, args.start_epoch + args.epochs, train_loss.item(), validation_loss.item()))

        stop = scheduler.step(validation_loss,
            callback_best=save_model(args),
            callback_reduce=load_model(model, optimizer, args))
        
        if args.tensorboard:
            with tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar('loss/validation', validation_loss.item(), epoch + 1)
                tf.contrib.summary.scalar('loss/train', train_loss.item(), epoch + 1)
                # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
                # writer.add_scalar('loss/validation', validation_loss.item(), epoch + 1)
                # writer.add_scalar('loss/train', train_loss.item(), epoch + 1)

        if stop:
            break
            
    load_model(model, optimizer, args)()
    optimizer.swap()
    validation_loss = - torch.stack([compute_log_p_x(model, x_mb).mean().detach()
                                     for x_mb, in data_loader_valid], -1).mean()
    test_loss = - torch.stack([compute_log_p_x(model, x_mb).mean().detach()
                           for x_mb, in data_loader_test], -1).mean()

    print('###### Stop training after {} epochs!'.format(epoch + 1))
    print('Validation loss: {:4.3f}'.format(validation_loss.item()))
    print('Test loss:       {:4.3f}'.format(test_loss.item()))
    
    if args.save:
        with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
            print('###### Stop training after {} epochs!'.format(epoch + 1), file=f)
            print('Validation loss: {:4.3f}'.format(validation_loss.item()), file=f)
            print('Test loss:       {:4.3f}'.format(test_loss.item()), file=f)


class parser_:
    pass

def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = True
    tf.enable_eager_execution(config=config)

    args = parser_()
    args.device = '/cpu:0'  # '/gpu:0'
    args.dataset = 'miniboon' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
    args.learning_rate = np.float32(1e-2)
    args.batch_dim = 200
    args.clip_norm = 0.1
    args.epochs = 1000
    args.patience = 20
    args.cooldown = 10
    args.early_stopping = 100
    args.decay = 0.5
    args.min_lr = 5e-4
    args.polyak = 0.998
    args.flows = 5
    args.layers = 1
    args.hidden_dim = 10
    args.residual = 'gated'
    args.expname = ''
    args.load = None
    args.save = False
    args.tensorboard = 'tensorboard'

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--dataset', type=str, default='miniboone',
    #                     choices=['gas', 'bsds300', 'hepmass', 'miniboone', 'power'])
    #
    # parser.add_argument('--learning_rate', type=float, default=1e-2)
    # parser.add_argument('--batch_dim', type=int, default=200)
    # parser.add_argument('--clip_norm', type=float, default=0.1)
    # parser.add_argument('--epochs', type=int, default=1000)
    #
    # parser.add_argument('--patience', type=int, default=20)
    # parser.add_argument('--cooldown', type=int, default=10)
    # parser.add_argument('--early_stopping', type=int, default=100)
    # parser.add_argument('--decay', type=float, default=0.5)
    # parser.add_argument('--min_lr', type=float, default=5e-4)
    # parser.add_argument('--polyak', type=float, default=0.998)
    #
    # parser.add_argument('--flows', type=int, default=5)
    # parser.add_argument('--layers', type=int, default=1)
    # parser.add_argument('--hidden_dim', type=int, default=10)
    # parser.add_argument('--residual', type=str, default='gated',
    #                    choices=[None, 'normal', 'gated'])
    #
    # parser.add_argument('--expname', type=str, default='')
    # parser.add_argument('--load', type=str, default=None)
    # parser.add_argument('--save', action='store_true')
    # parser.add_argument('--tensorboard', type=str, default='tensorboard')
    
    # args = parser.parse_args()

    # print('Arguments:')
    # pprint.pprint(args.__dict__)

    args.path = os.path.join('checkpoint', '{}{}_layers{}_h{}_flows{}{}_{}'.format(
        args.expname + ('_' if args.expname != '' else ''),
        args.dataset, args.layers, args.hidden_dim, args.flows, '_' + args.residual if args.residual else '',
        str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')))

    print('Loading dataset..')
    data_loader_train, data_loader_valid, data_loader_test = load_dataset(args)
    
    if args.save and not args.load:
        print('Creating directory experiment..')
        os.mkdir(args.path)
        with open(os.path.join(args.path, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=4, sort_keys=True)
    
    print('Creating BNAF model..')
    with tf.device('/cpu:0'):
        model = create_model(args, verbose=True)

    print('Creating optimizer..')
    optimizer = tf.train.AdamOptimizer()
    # optimizer = Adam(model.parameters(), lr=args.learning_rate, amsgrad=True, polyak=args.polyak)

    print('Creating scheduler..')
    scheduler = ReduceLROnPlateau(optimizer, factor=args.decay,
                                  patience=args.patience, cooldown=args.cooldown,
                                  min_lr=args.min_lr, verbose=True,
                                  early_stopping=args.early_stopping,
                                  threshold_mode='abs')

    ## tensorboard and saving
    writer = tf.contrib.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
    writer.set_as_default()
    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())

    args.start_epoch = 0
    if args.load:
        load_model(model, optimizer, args, root, load_start_epoch=True)

    with tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()
        # global_step.assign(0)

    print('Training..')
    train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args, root)


if __name__ == '__main__':
    main()
