import os
import pickle
import six
import random
from lib.confusion_matrix_pretty_print import *
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.metrics
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.python.keras.losses import Loss
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
from lib import utils, models_tf
from lib.load_data import load_hole_dataset, one_hot_encoder, load_CYC_dataset
from collections import Counter
from tensorflow.python.platform import tf_logging as logging


def calculate_weighted_acc(x, y, model, class_weight):
    y_pred = model.predict(x)
    weighted_acc = utils.calculate_class_weighted_accuracy(y, y_pred, class_weight)
    
    return weighted_acc


class weighted_acc_callback(Callback):
    def __init__(self, train_data, val_data, test_data, model):
        super(weighted_acc_callback, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.model = model
        self.train_weighted_acc = []
        self.val_weighted_acc = []
        self.test_weighted_acc = []
        self.train_cls_weight = calculate_class_weight(train_data[1])
        self.val_cls_weight = calculate_class_weight(val_data[1])
        if test_data is not None:
            self.test_cls_weight = calculate_class_weight(test_data[1])
    
    def on_epoch_end(self, epoch, logs={}):
        self.train_weighted_acc.append(
            calculate_weighted_acc(self.train_data[0], self.train_data[1], self.model, self.train_cls_weight))
        self.val_weighted_acc.append(
            calculate_weighted_acc(self.val_data[0], self.val_data[1], self.model, self.val_cls_weight))
        if self.test_data is not None:
            self.test_weighted_acc.append(
                calculate_weighted_acc(self.test_data[0], self.test_data[1], self.model, self.test_cls_weight))
        else:
            self.test_weighted_acc.append(0.)
        print('Weighted Acc: \n train: {:.2f} val: {:.2f} test: {:.2f}'.format(
            self.train_weighted_acc[-1] * 100., self.val_weighted_acc[-1] * 100.,
            self.test_weighted_acc[-1] * 100. if self.test_data is not None else 0., ))


class customized_earlystopping(EarlyStopping):
    def __init__(self, x_val, y_val, model, monitor='val_acc', min_delta=0, patience=0, verbose=0,
                 mode='max', baseline=None, restore_best_weights=False):
        super(customized_earlystopping, self).__init__(monitor=monitor, min_delta=min_delta, patience=patience,
                                                       verbose=verbose, mode=mode, baseline=baseline,
                                                       restore_best_weights=restore_best_weights)
        self.x_val = x_val
        self.y_val = y_val
        self.model = model
        self.val_cls_weight = calculate_class_weight(y_val)
    
    def get_monitor_value(self, logs):
        return calculate_weighted_acc(self.x_val, self.y_val, self.model, self.val_cls_weight)


class customized_modelcheckpoint(ModelCheckpoint):
    def __init__(self, x_val, y_val, model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch',
                 options=None, ):
        super(customized_modelcheckpoint, self).__init__(filepath=filepath, monitor=monitor, verbose=verbose,
                                                         save_best_only=save_best_only,
                                                         save_weights_only=save_weights_only, mode=mode,
                                                         save_freq=save_freq, options=options, )
        self.x_val = x_val
        self.y_val = y_val
        self.model = model
        self.val_cls_weight = calculate_class_weight(y_val)
    
    def get_monitor_value(self, ):
        return calculate_weighted_acc(self.x_val, self.y_val, self.model, self.val_cls_weight)
    
    def _save_model(self, epoch, logs):
        """Saves the model.
  
        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}
        
        if isinstance(self.save_freq, int) or self.epochs_since_last_save >= self.period:
            # Block only when saving interval is reached.
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)
            
            try:
                if self.save_best_only:
                    current = self.get_monitor_value()
                    if current is None:
                        logging.warning('Can save best model only with %s available, '
                                        'skipping.', self.monitor)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s' % (epoch + 1, self.monitor,
                                                               self.best, current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.model.save_weights(
                                    filepath, overwrite=True, options=self._options)
                            else:
                                self.model.save(filepath, overwrite=True, options=self._options)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(
                            filepath, overwrite=True, options=self._options)
                    else:
                        self.model.save(filepath, overwrite=True, options=self._options)
                
                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
                if 'is a directory' in six.ensure_str(e.args[0]).lower():
                    raise IOError('Please specify a non-directory filepath for '
                                  'ModelCheckpoint. Filepath used is an existing '
                                  'directory: {}'.format(filepath))
                # Re-throw the error for any other causes.
                raise e


def build_model(model_name='EEGNet', num_classes=2, Chans=64, Samples=240, dropoutRate=0.25, kernLength=128):
    if model_name == 'FCN':
        return models_tf.FCN(num_classes=num_classes, Chans=Chans, SamplePoints=Samples, dropoutRate=dropoutRate)
    elif model_name == 'EEGNet':
        print('kernLength: ', kernLength)
        return models_tf.EEGNet(num_classes=num_classes, Chans=Chans, SamplePoints=Samples, dropoutRate=dropoutRate,
                                kernLength=kernLength, )
    elif model_name == 'DeepConvNet':
        return models_tf.DeepConvNet(num_classes=num_classes, Chans=Chans, SamplePoints=Samples,
                                     dropoutRate=dropoutRate)
    elif model_name == 'ShallowConvNet':
        return models_tf.ShallowConvNet(num_classes=num_classes, Chans=Chans, SamplePoints=Samples,
                                        dropoutRate=dropoutRate)
    elif model_name == 'RD3Net':
        return models_tf.RD3Net(num_classes=num_classes, Chans=Chans, SamplePoints=Samples,
                                dropoutRate=dropoutRate)
    else:
        raise Exception('No such model:{}'.format(model_name))


def calculate_class_weight(y):
    # return dictionary mapping class indices (integers) to a weight (float) value
    y = np.array(y, np.int)
    y_num = dict(Counter(y))
    
    for key in list(y_num.keys()):
        y_num[key] = 1. / (y_num[key] / len(y))
    total = np.sum(list(y_num.values()))
    for key in list(y_num.keys()):
        y_num[key] = y_num[key] / total
    
    return y_num


def train_model(train_dataset, val_dataset, test_dataset, model, model_path="./model", batch_size=32,
                epochs=300, patience=100, loss='sparse_categorical_crossentropy'):
    print('-' * 20 + 'model_path' + '-' * 20)
    print(model_path)
    os.makedirs(model_path, exist_ok=True)
    
    x_train, y_train = utils.shuffle_data(train_dataset)
    # train_class_weight = calculate_class_weight(y_train)
    x_val, y_val = val_dataset
    if test_dataset is not None:
        x_test, y_test = test_dataset
    num_train_batch = np.ceil(len(x_train) / batch_size)
    
    # Train Model
    print('-' * 20, 'Start to train model...', '-' * 20)
    lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate=0.005,
                                                    decay_steps=num_train_batch * epochs)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.01, decay_steps=num_train_batch, decay_rate=0.99,
    #                                                              staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc', ])
    ### TensorBoard
    # log_dir = "./log"
    # if os.path.exists(log_dir):
    #     shutil.rmtree(log_dir)
    # Tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_grads=True)
    # best_ckpt = customized_modelcheckpoint(x_val=x_val, y_val=y_val, model=model, filepath=model_path,
    #                                        monitor='val_acc', save_best_only=True, mode='max')
    # early_stop = customized_earlystopping(x_val=x_val, y_val=y_val, model=model, monitor='val_acc', mode='max',
    #                                       patience=patience)
    # test_callback = weighted_acc_callback(train_dataset, val_dataset, test_dataset, model=model)
    early_stopping = EarlyStopping(monitor='val_acc', mode='max', patience=patience, )
    # test_callback = TestCallback(val_dataset, test_dataset, early_stop)
    model_ckpt = ModelCheckpoint(filepath=model_path, monitor='val_acc', mode='max', save_best_only=True)
    
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=tuple(val_dataset), callbacks=[early_stopping, model_ckpt],
                        verbose=2, )  # class_weight=train_class_weight,callbacks=[best_ckpt, early_stop, test_callback],
    
    return {
        'train_acc' : np.array(history.history['acc'], dtype=np.float32),
        'train_loss': np.array(history.history['loss'], dtype=np.float32),
        'val_acc'   : np.array(history.history['val_acc'], dtype=np.float32),
        'val_loss'  : np.array(history.history['val_loss'], dtype=np.float32),
    }
    # return {
    #     'train_acc': np.array(test_callback.train_weighted_acc, dtype=np.float32),
    #     'val_acc'  : np.array(test_callback.val_weighted_acc, dtype=np.float32),
    #     'test_acc' : np.array(test_callback.test_weighted_acc, dtype=np.float32),
    # }


def eva_model(model, dataset):
    '''
    cal_acc
    input:
    model_name,model, model_path,data_name,x_test,y_test

    output:
    acc
    y_pred
    '''
    x_test, y_test = dataset
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=-1)
    y_test = np.squeeze(y_test)
    acc = np.sum(y_pred == y_test) / len(y_pred)
    
    return acc, y_pred, y_pred_prob


def compute_target_gradient(x, model, target):
    """
    Computes the gradient of the input image batch wrt the target output class.

    Note, this gradient is only ever computed from the <Student> model,
    and never from the <Teacher/Attacked> model when using this version.

    Args:
        x: batch of input of shape [B, T, C]
        model: classifier model
        target: integer id corresponding to the target class

    Returns:
        the output of the model and a list of gradients of shape [B, T, C]
    """
    with tf.GradientTape() as tape:
        tape.watch(x)  # need to watch the input tensor for grad wrt input
        out = model(x, training=False)  # in evaluation mode
        target_out = out[:, target]  # extract the target class outputs only
    
    image_grad = tape.gradient(target_out, x)  # compute the gradient
    
    return out, image_grad


class DATASET_CONFIG:
    def __init__(self, ds_dir=None, seg_len='1s', sample_rate=16000, ds_path=None, label_smooth_para=None,
                 label_preprocess=None,
                 ds_preprocess='normalized_denoise_nsnet2', gcc_phat=True, num_class=8, ):
        self.ds_dir = ds_dir
        self.seg_len = seg_len
        self.sample_rate = sample_rate
        self.ds_preprocess = ds_preprocess
        self.gcc_phat = gcc_phat
        self.num_sbj = None
        self.num_class = num_class
        self.num_channel = None
        self.num_samplepoint = None
        self.ds_path = ds_path
        self.label_smooth_para = label_smooth_para
        self.label_preprocess = label_preprocess
        
        with open(ds_path, "rb") as fo:
            ds = pickle.load(fo)
        x = ds['x']
        self.num_sbj = len(x)
        (_, self.num_feature_map, self.num_channel, self.num_samplepoint) = x[0].shape
        del x, ds


class MODEL_CONFIG:
    def __init__(self, md_dir='./model', name='EEGNet', batch_size=32, epochs=1000, dropoutRate=0.25,
                 earlystop_patience=50, num_filter=32, normalization=None, kernLength=None,
                 loss=None):
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropoutRate = dropoutRate
        self.earlystop_patience = earlystop_patience
        self.num_filter = num_filter
        self.normalization = normalization
        self.md_dir = md_dir
        self.ckpt_dir = os.path.join(self.md_dir, 'ckpt')
        self.kernLength = kernLength
        self.loss = loss


# class Gauss_sparse_categorical_crossentropy_Loss(Loss):
#     def __init__(self, num_class=8, mean=0, sigma=1, label_smoothing=0):
#         super(Gauss_sparse_categorical_crossentropy_Loss, self).__init__()
#         self.num_class = num_class
#         weight = np.roll(
#             np.linspace(-self.num_class // 2, self.num_class // 2, endpoint=self.num_class % 2, num=self.num_class),
#             -self.num_class // 2)
#         weight = np.exp(- ((weight - mean) ** 2) / (2 * (sigma ** 2))) / (np.sqrt(2 * np.pi) * sigma)
#         weight = -weight / weight.max()
#         self.weight = weight - weight.min()
#         self.label_smoothing = label_smoothing
#
#     def __call__(self, y_true, y_pred, sample_weight=None):
#         print('Using self_customized loss function')
#
#         weight = np.roll(self.weight, y_true)
#         onehot_labels = one_hot_encoder(y_true)
#         if self.label_smoothing > 0:
#             smooth_positives = 1.0 - self.label_smoothing
#             smooth_negatives = self.label_smoothing / self.num_class
#             onehot_labels = onehot_labels * smooth_positives + smooth_negatives
#
#         cross_entropy = -tf.reduce_mean(weight * onehot_labels * tf.log(tf.clip.by_value(y_pred, 1e-10, 1.0)))
#
#         return cross_entropy

def calculate_acc_and_acc3(x, y, model):
    y_pred = model.predict(x)
    num_cls = y_pred.shape[-1]
    y_pred = np.argmax(y_pred, axis=-1)
    y = np.array(y)
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)
    else:
        y = np.array(y, dtype=int)
    acc = np.sum(y == y_pred) / len(y)
    is_neighbor = []
    for ty, py in list(zip(y, y_pred)):
        neighbor = np.array([ty + num_cls - 1, ty, ty + 1], dtype=int) % num_cls
        is_neighbor.append(py in neighbor)
    acc_3 = np.sum(is_neighbor) / len(y)
    
    return acc, acc_3


if __name__ == '__main__':
    
    # setting keras
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 4)
    tf.config.experimental.set_memory_growth(gpus[0], True)
    K.set_image_data_format('channels_first')
    # setting random seed
    random_seed = 0
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    # ---------------------------------------------------------------------------------------- #
    
    # model_configure & dataset_configure
    model_name = 'EEGNet'
    seg_len = '256ms'
    # ds_name = 'ini_hann_np_gcc_phat_128'
    # ds_name = 'drop_denoised_ini_hann_np_gcc_phat_128'
    # ds_name = 'norm_drop_denoised_ini_hann_np_gcc_phat_128'
    ds_name = 'norm_drop_denoised_norm_ini_hann_np_gcc_phat_128'
    kernLength = 128
    label_preprocess = None  # None  # 'one_hot'  # 'mean_smooth'  # 'gauss_smooth'
    label_smooth_para = ''
    epochs = 100
    md_dir = os.path.join('../model', model_name, seg_len, 'kernLength_' + str(kernLength), ds_name,
                          str(label_preprocess) + '_' + str(label_smooth_para), 'epoch_' + str(epochs))
    md_config = MODEL_CONFIG(name=model_name, md_dir=md_dir, batch_size=64, epochs=epochs, dropoutRate=0.25,
                             earlystop_patience=100, num_filter=32, normalization='sample-wise', kernLength=kernLength,
                             loss='sparse_categorical_crossentropy')
    ds_config = DATASET_CONFIG(ds_dir=None, ds_path=os.path.join(
        "/home/swang/project/SmartWalker/collect_dataset/dataset/4F_CYC/256ms_0.13_400_16000/", ds_name + '.pkl'),
                               seg_len=seg_len, sample_rate=16000, ds_preprocess=ds_name, gcc_phat=True,
                               num_class=8, label_preprocess=label_preprocess,
                               label_smooth_para=label_smooth_para)  # 1s_0.5_800_16000    256ms_0.13_400_16000
    
    folds = 5
    train_accs = []
    val_accs = []
    test_accs = []
    train_acc3s = []
    val_acc3s = []
    test_acc3s = []
    fold_split_idx = utils.gen_cross_val_idx(ds_config.num_sbj, num_fold=5, num_subfold=[3, 1, 1], random_seed=0, )
    for fold in range(folds):
        # set paths
        img_path = os.path.join(md_config.md_dir, str(fold), 'acc_loss.jpg')
        ckpt_dir = os.path.join(md_config.md_dir, str(fold), 'ckpt')
        print("------------------train model------------------")
        K.clear_session()
        # load dataset ---- split dataset for train, val & test
        train_idx, val_idx, test_idx = fold_split_idx[fold]
        print("-----{} for train----{} for val-----{} for test-----".format(train_idx, val_idx, test_idx))
        x_train, y_train = load_CYC_dataset(train_idx, ds_config.ds_path, shuffle=True, split=None,
                                            normalization=md_config.normalization,
                                            label_preprocess=ds_config.label_preprocess,
                                            label_smooth_para=ds_config.label_smooth_para)
        x_val, y_val, = load_CYC_dataset(val_idx, ds_config.ds_path, shuffle=True, split=None,
                                         normalization=md_config.normalization,
                                         label_preprocess=ds_config.label_preprocess,
                                         label_smooth_para=ds_config.label_smooth_para)
        x_test, y_test, = load_CYC_dataset(test_idx, ds_config.ds_path, shuffle=True, split=None,
                                           normalization=md_config.normalization,
                                           label_preprocess=ds_config.label_preprocess,
                                           label_smooth_para=ds_config.label_smooth_para)
        utils.statistic_label_proportion(y_train, y_val, y_test, do_print=True)
        
        # build model
        model = build_model(model_name=md_config.name, num_classes=ds_config.num_class, Chans=ds_config.num_channel,
                            Samples=ds_config.num_samplepoint, dropoutRate=md_config.dropoutRate,
                            kernLength=md_config.kernLength)
        model.summary()
        history = train_model([x_train, y_train], [x_val, y_val], test_dataset=None, model=model,
                              model_path=ckpt_dir, batch_size=md_config.batch_size, epochs=md_config.epochs,
                              patience=md_config.earlystop_patience, loss=md_config.loss)
        train_acc, train_loss = history['train_acc'], history['train_loss']
        val_acc, val_loss = history['val_acc'], history['val_loss']
        print('train_acc: ', train_acc)
        print('train_loss: ', train_loss)
        print('val_acc: ', val_acc)
        print('val_loss: ', val_loss)
        # calculate the final result
        # ckpt_dir='/home/swang/project/SmartWalker/SSL/model/model/EEGNet/ckpt'
        model = keras.models.load_model(ckpt_dir)
        # _, f_train_acc = model.evaluate(x_train, y_train)
        # _, f_val_acc = model.evaluate(x_val, y_val)
        # _, f_test_acc = model.evaluate(x_test, y_test)
        # train_accs.append(f_train_acc)
        # val_accs.append(f_val_acc)
        # test_accs.append(f_test_acc)
        f_train_acc, f_train_acc3 = calculate_acc_and_acc3(x_train, y_train, model)
        f_val_acc, f_val_acc3 = calculate_acc_and_acc3(x_val, y_val, model)
        f_test_acc, f_test_acc3 = calculate_acc_and_acc3(x_test, y_test, model)
        train_accs.append(f_train_acc)
        val_accs.append(f_val_acc)
        test_accs.append(f_test_acc)
        train_acc3s.append(f_train_acc3)
        val_acc3s.append(f_val_acc3)
        test_acc3s.append(f_test_acc3)
        print('-' * 20, 'acc & acc3', '-' * 20, )
        print('-' * 20, f_train_acc, f_val_acc, f_test_acc, '-' * 20, )
        print('-' * 20, f_train_acc3, f_val_acc3, f_test_acc3, '-' * 20, )
        
        # plot the curve of results
        title = 'Acc & Loss of {} Training ({:.1f}%) & Val ({:.1f}%) & Test ({:.1f}%)'.format(
            md_config.name, f_train_acc * 100, f_val_acc * 100, f_test_acc * 100)
        curve_name = ['Training acc', 'Training loss', 'Val acc', 'Val loss', ]
        curve_data = [train_acc, train_loss, val_acc, val_loss, ]
        color = ['r', 'g', 'b', 'cyan']
        img_path = img_path
        utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, y_lim=(0, 1), img_path=img_path)
        
        # train_acc, val_acc, test_acc = history['train_acc'], history['val_acc'], history['test_acc']
        
        # # plot the curve of results
        # title = 'Weighted Acc of {} Training ({:.1f}%) & Val ({:.1f}%) & Test ({:.1f}%)'.format(
        #     md_config.name, train_acc[-md_config.earlystop_patience] * 100,
        #                     val_acc[-md_config.earlystop_patience] * 100, test_acc[-md_config.earlystop_patience] * 100)
        # curve_name = ['Training acc', 'Val acc', 'Test acc', ]
        # curve_data = [train_acc, val_acc, test_acc]
        # color = ['r', 'g', 'b', ]
        # utils.plot_curve(data=list(zip(curve_name, curve_data, color)), title=title, y_lim=(0, 1))
        K.clear_session()
    print('train_accs: ', [round(i, 3) for i in train_accs], '-' * 5, round(np.mean(train_accs), 3))
    print('val_accs: ', [round(i, 3) for i in val_accs], '-' * 5, round(np.mean(val_accs), 3))
    print('test_accs: ', [round(i, 3) for i in test_accs], '-' * 5, round(np.mean(test_accs), 3))
    print('train_accs: ', [round(i, 3) for i in train_acc3s], '-' * 5, round(np.mean(train_acc3s), 3))
    print('val_accs: ', [round(i, 3) for i in val_acc3s], '-' * 5, round(np.mean(val_acc3s), 3))
    print('test_accs: ', [round(i, 3) for i in test_acc3s], '-' * 5, round(np.mean(test_acc3s), 3))
    print(md_config.kernLength, ds_config.ds_preprocess, ds_config.label_preprocess, ds_config.label_smooth_para)
