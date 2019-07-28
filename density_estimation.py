import os
import json
import pprint
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from bnaf import *
from optim.lr_scheduler import *

from scipy.optimize import fmin_l_bfgs_b

import functools

# import argparse
# import torch
# from torch.utils import data
# from data.generate2d import sample2d, energy2d
# from tqdm import tqdm
# from optim.adam import Adam

from data.gas import GAS
from data.bsds300 import BSDS300
from data.hepmass import HEPMASS
from data.miniboone import MINIBOONE
from data.power import POWER
from data.GQ_MS import GQ_MS

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
    elif args.dataset == 'gq_ms_wheat':
        dataset = GQ_MS('data/GQ_MS/wheat_perms.xlsx')
    elif args.dataset == 'gq_ms_soy':
        dataset = GQ_MS('data/GQ_MS/soy_perms.xlsx')
    elif args.dataset == 'gq_ms_corn':
        dataset = GQ_MS('data/GQ_MS/corn_perms.xlsx')
    elif args.dataset == 'gq_ms_canola':
        dataset = GQ_MS('data/GQ_MS/canola_perms.xlsx')
    elif args.dataset == 'gq_ms_barley':
        dataset = GQ_MS('data/GQ_MS/barley_perms.xlsx')
    else:
        raise RuntimeError()


    dataset_train = tf.data.Dataset.from_tensor_slices((dataset.trn.x))#.float().to(args.device)
    # dataset_train = dataset_train.shuffle(buffer_size=len(dataset.trn.x)).repeat().batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    dataset_train = dataset_train.shuffle(buffer_size=len(dataset.trn.x)).batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    # data_loader_train = tf.contrib.eager.Iterator(dataset_train)
    ##data_loader_train.get_next()

    dataset_valid = tf.data.Dataset.from_tensor_slices((dataset.val.x))#.float().to(args.device)
    # dataset_valid = dataset_valid.shuffle(buffer_size=len(dataset.val.x)).repeat().batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    dataset_valid = dataset_valid.shuffle(buffer_size=len(dataset.val.x)).batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    # data_loader_valid = tf.contrib.eager.Iterator(dataset_valid)
    ##data_loader_valid.get_next()

    dataset_test = tf.data.Dataset.from_tensor_slices((dataset.tst.x))#.float().to(args.device)
    # dataset_test = dataset_test.shuffle(buffer_size=len(dataset.tst.x)).repeat().batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    dataset_test = dataset_test.shuffle(buffer_size=len(dataset.tst.x)).batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    # data_loader_test = tf.contrib.eager.Iterator(dataset_test)
    ##data_loader_test.get_next()

    args.n_dims = dataset.n_dims


    # train_size = 3000
    # dataset = np.arcsinh(5*np.random.RandomState(111).normal(0,1,size=[3*train_size,1]).astype(np.float32))
    #
    # dataset_train = tf.data.Dataset.from_tensor_slices((dataset[:train_size]))  # .float().to(args.device)
    # dataset_train = dataset_train.batch(batch_size=args.batch_dim).prefetch( buffer_size=1)
    #
    # dataset_valid = tf.data.Dataset.from_tensor_slices((dataset[train_size:2*train_size]))#.float().to(args.device)
    # dataset_valid = dataset_valid.batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    #
    # dataset_test = tf.data.Dataset.from_tensor_slices((dataset[train_size*2:]))#.float().to(args.device)
    # dataset_test = dataset_test.batch(batch_size=args.batch_dim).prefetch(buffer_size=1)
    #
    # args.n_dims = 1

    return dataset_train, dataset_valid, dataset_test

def smooth_abs_tf(x):
    #asdjksfdajk
    return tf.square(x) / tf.sqrt(tf.square(x) + .01 ** 2)

def eval_loss_and_grads(x, loss_train, var_list, var_shapes, var_locs):
    ## x: updated variables from scipy optimizer
    ## var_list: list of trainable variables
    ## var_sapes: the shape of each variable to use for updating variables with optimizer values
    ## var_locs:  slicing indecies to use on x prior to reshaping

    ## update variables
    grad_list = []
    for i in range(len(var_list)):
        var_list[i].assign(np.reshape(x[var_locs[i]:(var_locs[i+1])], var_shapes[i]))

    # ## make the predictions pass through the origin by setting final layer bias
    # var_list[-1].assign((var_list[-1]-model(np.array([[0]])))[0])

    ## calculate new gradient
    with tf.device("CPU:0"):
        with tf.GradientTape() as tape:
            prediction_loss, regL1_penalty, regL2_penalty = loss_train(x=x)

        for p in tape.gradient(prediction_loss, var_list):
            grad_list.extend(np.array(tf.reshape(p, [-1])))

    grad_list = [v if v is not None else 0 for v in grad_list]

    return np.float64(prediction_loss), np.float64(grad_list), np.float64(regL1_penalty), np.float64(regL2_penalty)


class Evaluator(object):

    # def __init__(self, loss_train_fun, loss_val, global_step, early_stop_limit, var_list, var_shapes, var_locs, model):
    def __init__(self, loss_train_fun, loss_val, global_step, scheduler, var_list, var_shapes, var_locs):
        self.loss_train_fun = loss_train_fun #func_tools partial function with model, features, and labels already loaded
        self.predLoss_val = loss_val #func_tools partial function with model, features, and labels already loaded
        self.global_step = global_step #tf variable for tracking update steps from scipy optimizer step_callback
        self.predLoss_val_prev, _, _, _, _ = np.float64(loss_val()) + 10.0 #state variable of loss for early stopping
        self.predLoss_val_cntr = 0 #counter to watch for early stopping
        # self.early_stop_limit = early_stop_limit #number of cycles of increasing validation loss before stopping
        self.scheduler = scheduler
        self.var_shapes = var_shapes #list of shapes of each tf variable
        self.var_list = var_list #list of trainable tf variables
        self.var_locs = var_locs
        self.loss_value, self.regL1, self.regL2 = self.loss_train_fun(x=np.float64(0))
        self.grads_values = None

    def loss(self, x):
        # assert self.loss_value is None
        self.loss_value, self.grad_values, self.regL1, self.regL2 = eval_loss_and_grads(x, self.loss_train_fun, self.var_list, self.var_shapes, self.var_locs) #eval_loss_and_grads
        return self.loss_value

    def grads(self, x):
        # assert self.loss_value is not None
        # grad_values = np.copy(self.grad_values)
        # self.loss_value = None
        # self.grad_values = None
        return self.grad_values

    def step_callback(self, x):
        ## early stopping tracking
        predLoss_val_temp, _, _ = np.float64(self.predLoss_val())
        if predLoss_val_temp > self.predLoss_val_prev:
            self.predLoss_val_cntr += 1
        else:
            self.predLoss_val_cntr = 0
            self.predLoss_val_prev = predLoss_val_temp

        #write tensorboard variables
        tf.summary.scalar('loss_train', self.loss_value)
        tf.summary.scalar('loss_val', predLoss_val_temp)
        tf.summary.scalar('predLoss_val_cntr', self.predLoss_val_cntr)
        tf.summary.scalar('regL1', self.regL1)
        tf.summary.scalar('regL2', self.regL2)

        # increment the global step counter
        self.global_step = self.global_step + 1

        return self.scheduler.on_epoch_end(epoch=self.global_step, monitor=predLoss_val_temp)

        # #return true to scipy optimizer to stop optimization loop
        # if self.predLoss_val_cntr > self.early_stop_limit:
        #     print("stop")
        #     sys.stdout.flush()
        #     return True
        # else:
        #     return False


def create_model(args, verbose=False):

    manualSeed = 1
    np.random.seed(manualSeed)

    # random.seed(manualSeed)
    # torch.manual_seed(manualSeed)

    flows = []
    for f in range(args.flows):
        #build internal layers for a single flow
        layers = []
        for _ in range(args.layers - 1):
            layers.append(MaskedWeight(args.n_dims * args.hidden_dim,
                                       args.n_dims * args.hidden_dim, dim=args.n_dims))
            layers.append(Tanh())

        flows.append(
            BNAF(layers = [MaskedWeight(args.n_dims, args.n_dims * args.hidden_dim, dim=args.n_dims), Tanh()] + \
               layers + \
               [MaskedWeight(args.n_dims * args.hidden_dim, args.n_dims, dim=args.n_dims)], \
             res=args.residual if f < args.flows - 1 else None
             )
        )

        if f < args.flows - 1:
            flows.append(Permutation(args.n_dims, 'flip'))

        model = Sequential(flows)
        # params = np.sum(np.sum(p.numpy() != 0) if len(p.numpy().shape) > 1 else p.numpy().shape
        #              for p in model.trainable_variables)[0]
    
    # if verbose:
    #     print('{}'.format(model))
    #     print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params,
    #         NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims))

    # if args.save and not args.load:
    #     with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
    #         print('Parameters={}, NAF/BNAF={:.2f}/{:.2f}, n_dims={}'.format(params,
    #             NAF_PARAMS[args.dataset][0] / params, NAF_PARAMS[args.dataset][1] / params, args.n_dims), file=f)
    
    return model

def load_model(args, root, load_start_epoch=False):
    # def f():
    print('Loading model..')
    root.restore(tf.train.latest_checkpoint(args.load or args.path))
    # root.restore(os.path.join(args.load or args.path, 'checkpoint'))
    # if load_start_epoch:
    #     args.start_epoch = tf.train.get_global_step().numpy()
    # return f

@tf.function
def loss_func(model, x_mb, x, regL2 = np.float32(-1.0), regL1 = np.float32(-1.0)):
    loss = - tf.reduce_mean(compute_log_p_x(model, x_mb))
    regL2_penalty = np.float32(0)
    regL1_penalty = np.float32(0)
    if regL2 >= 0:  regL2_penalty = tf.reduce_mean(tf.square(x))
    if regL1 >= 0:  regL1_penalty = tf.reduce_mean(smooth_abs_tf(x))
    return loss + regL2_penalty +regL2_penalty, regL2_penalty, regL1_penalty

@tf.function
def compute_log_p_x(model, x_mb):
    ## use tf.gradient + tf.convert_to_tensor + tf.GradientTape(persistent=True) to clean up garbage implementation in bnaf.py
    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = tf.reduce_sum(tfp.distributions.Normal(tf.zeros_like(y_mb), tf.ones_like(y_mb)).log_prob(y_mb), axis=-1)#.sum(-1)
    return log_p_y_mb + log_diag_j_mb

# def compute_kl(model, args):
#     d_mb = tfp.distributions.Normal(tf.zeros((args.batch_dim, 2)),
#                                       tf.ones((args.batch_dim, 2)))
#     y_mb = d_mb.sample()
#     x_mb, log_diag_j_mb = model(y_mb)
#     log_p_y_mb = tf.reduce_sum(d_mb.log_prob(y_mb), axis=-1)
#     return log_p_y_mb - log_diag_j_mb + energy2d(args.dataset, x_mb) + tf.reduce_sum(tf.nn.relu(x_mb.abs() - 6) ** 2, axis=-1)

# def train_density2d(model, optimizer, scheduler, args):
#     iterator = trange(args.steps, smoothing=0, dynamic_ncols=True)
#     for epoch in iterator:
#         x_mb = torch.from_numpy(sample2d(args.dataset, args.batch_dim)).float().to(args.device)
#
#         loss = - compute_log_p_x(model, x_mb).mean()
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
#
#         optimizer.step()
#         optimizer.zero_grad()
#
#         scheduler.step(loss)
#
#         iterator.set_postfix(loss='{:.2f}'.format(loss.data.cpu().numpy()), refresh=False)
#
#
# def train_energy2d(model, optimizer, scheduler, args):
#     iterator = trange(args.steps, smoothing=0, dynamic_ncols=True)
#     for epoch in iterator:
#         loss = compute_kl(model, args).mean()
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_norm)
#
#         optimizer.step()
#         optimizer.zero_grad()
#
#         scheduler.step(loss)
#
#         iterator.set_postfix(loss='{:.2f}'.format(loss.data.cpu().numpy()), refresh=False)
@tf.function
def train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args):
    
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):

        # t = tqdm(data_loader_train, smoothing=0, ncols=80)
        train_loss = []
        
        for x_mb in data_loader_train:
            with tf.GradientTape() as tape:
                loss = - tf.reduce_mean(compute_log_p_x(model, x_mb)) #negative -> minimize to maximize liklihood

            grads = tape.gradient(loss, model.trainable_variables)
            grads = [None if grad is None else tf.clip_by_norm(grad, clip_norm=args.clip_norm) for grad in grads]
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_loss.append(loss)

            tf.compat.v1.train.get_global_step().assign_add(1)
        train_loss = np.mean(train_loss)
        validation_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)) for x_mb in data_loader_valid])

        # print('Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}'.format(
        #     epoch + 1, args.start_epoch + args.epochs, train_loss, validation_loss))


        stop = scheduler.on_epoch_end(epoch = epoch, monitor=validation_loss)

        if args.tensorboard:
            # with tf.contrib.summary.always_record_summaries():
                tf.summary.scalar('loss/validation', validation_loss,tf.compat.v1.train.get_global_step())
                tf.summary.scalar('loss/train', train_loss, tf.compat.v1.train.get_global_step())
                # writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
                # writer.add_scalar('loss/validation', validation_loss.item(), epoch + 1)
                # writer.add_scalar('loss/train', train_loss.item(), epoch + 1)

        if stop:
            break

    validation_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)) for x_mb in data_loader_valid])
    test_loss = - tf.reduce_mean([tf.reduce_mean(compute_log_p_x(model, x_mb)) for x_mb in data_loader_test])

    print('###### Stop training after {} epochs!'.format(epoch + 1))
    print('Validation loss: {:4.3f}'.format(validation_loss))
    print('Test loss:       {:4.3f}'.format(test_loss))
    
    if args.save:
        with open(os.path.join(args.load or args.path, 'results.txt'), 'a') as f:
            print('###### Stop training after {} epochs!'.format(epoch + 1), file=f)
            print('Validation loss: {:4.3f}'.format(validation_loss), file=f)
            print('Test loss:       {:4.3f}'.format(test_loss), file=f)

class parser_:
    pass

def main():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    # tf.compat.v1.enable_eager_execution(config=config)

    # tf.config.experimental_run_functions_eagerly(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    args = parser_()
    args.device = '/cpu:0'  # '/gpu:0'
    args.dataset = 'gq_ms_wheat' #['gas', 'bsds300', 'hepmass', 'miniboone', 'power']
    args.learning_rate = np.float32(1e-2)
    args.batch_dim = 200
    args.clip_norm = 0.1
    args.epochs = 5000
    args.patience = 10
    args.cooldown = 10
    args.decay = 0.5
    args.min_lr = 5e-4
    args.flows = 1
    args.layers = 1
    args.hidden_dim = 3
    args.residual = 'gated'
    args.expname = ''
    args.load = ''#r'C:\Users\justjo\PycharmProjects\BNAF_tensorflow_eager\checkpoint\gq_ms_wheat_layers1_h3_flows1_gated_2019-07-03-01-58-21'
    args.save = True
    args.tensorboard = 'tensorboard'
    args.early_stopping = 10
    args.maxiter = 500
    args.factr = 1E1
    args.optimizer = 'LBFGS' #or None
    args.regL2 = 0.0001
    args.regL1 = -1

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
            json.dump(str(args.__dict__), f, indent=4, sort_keys=True)
    
    print('Creating BNAF model..')
    with tf.device(args.device):
        model = create_model(args, verbose=True)

    ### debug
    # data_loader_train_ = tf.contrib.eager.Iterator(data_loader_train)
    # x = data_loader_train_.get_next()
    # a = model(x)

    ## tensorboard and saving
    writer = tf.summary.create_file_writer(os.path.join(args.tensorboard, args.load or args.path))
    writer.set_as_default()
    tf.compat.v1.train.get_or_create_global_step()

    root = None
    args.start_epoch = 0
    if not args.optimizer:
        print('Creating optimizer..')
        with tf.device(args.device):
            optimizer = tf.optimizers.Adam()
        root = tf.train.Checkpoint(optimizer=optimizer,
                                   model=model,
                                   optimizer_step=tf.compat.v1.train.get_global_step())

        if args.load:
            load_model(args, root, load_start_epoch=True)

    print('Creating scheduler..')
    # use baseline to avoid saving early on
    scheduler = EarlyStopping(model=model, patience=args.early_stopping, args = args, root = root)

    with tf.device(args.device):
        # print('Training..')
        # if args.experiment == 'density2d':
        #     train_density2d(model, optimizer, scheduler, args)
        # elif args.experiment == 'energy2d':
        #     train_energy2d(model, optimizer, scheduler, args)
        # else:

        ## SGD
        if not args.optimizer:
            train(model, optimizer, scheduler, data_loader_train, data_loader_valid, data_loader_test, args)
        ## LBFGS
        else:
            loss_train = functools.partial(loss_func, x_mb=data_loader_train, model=model,
                                           regL2=args.regL2, regL1=args.regL1)
            loss_val = functools.partial(loss_func, x_mb=data_loader_valid, model=model,
                                         regL2=np.float64(-1.0), regL1=np.float64(-1.0), x=np.float64(0))

            with tf.device("CPU:0"):
                with tf.GradientTape() as tape:
                    prediction_loss, _, _, _, _ = loss_train(x=np.float64(0))
                    var_list = tape.watched_variables()

            x_init = []
            for v in var_list:
                x_init.extend(np.array(tf.reshape(v, [-1])))

            var_shapes = []
            var_locs = [0]
            for v in var_list:
                var_shapes.append(np.array(tf.shape(v)))
                var_locs.append(np.prod(np.array(tf.shape(v))))
            var_locs = np.cumsum(var_locs)

            evaluator = Evaluator(loss_train, loss_val, 0, scheduler, var_list, var_shapes, var_locs)

            x, min_val, info = fmin_l_bfgs_b(func=evaluator.loss, x0=np.float64(x_init),
                                             fprime=evaluator.grads, maxiter=args.maxiter, factr=args.factr,
                                             callback=evaluator.step_callback)

            print(info)

if __name__ == '__main__':
    main()

##"C:\Program Files\Git\bin\sh.exe" --login -i

#### tensorboard --logdir=C:\Users\just\PycharmProjects\BNAF\tensorboard\checkpoint
## http://localhost:6006/