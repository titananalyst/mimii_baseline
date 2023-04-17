#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1].
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.
"""
########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
########################################################################
import time
import os

from PyQt5.QtCore import QLibraryInfo
# from PySide2.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)


########################################################################
# import additional python-library
########################################################################
import numpy
import librosa  # sound analysing library
import librosa.core  # I/O handling
import librosa.feature  # spectral features
import yaml  # data-serialization language for configuration files
import logging
# from import
from tqdm import tqdm  # progress bar
from sklearn import metrics  # classification metrics used for roc_auc_score
from keras.models import Model  # https://www.activestate.com/resources/quick-reads/what-is-a-keras-model/
from keras.layers import Input, Dense
# Input: instantiate a keras tensor
# Dense: regular densely-connected NN layer
########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.3"
########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save .png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


########################################################################
# file I/O
########################################################################
# pickle I/O
def save_pickle(filename, save_data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    logger.info("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    logger.info("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as lf:
        # rb stands for open in "binary" mode used for audio files
        load_data = pickle.load(lf)
    return load_data


# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    # Load an audio file as floating point time series.
    # sr = None preserves the native sampling rate of the file
    # mono is default set to False, otherwise converts signal to mono
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed mono data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    # "Demux" is short for "demultiplexing," which refers to the process of separating a 
    # combined signal or stream of data into its individual components or channels.
    try:
        # returns audio time series (multi channel is supported)
        # as np.ndarray (multidimensional, homogeneous array of fixed size items)
        # return sr = sampling rate
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            # checks if Number of array dimensions is smaller or equal to 1
            return sr, multi_channel_data
        
        # takes the row of the channel-th column and converted to a numpy array
        # returns array containing the channel 0 along with the sample rate sr
        return sr, numpy.array(multi_channel_data)[channel, :]

    except ValueError as msg:
        logger.warning(f'{msg}')


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    
# In the function file_to_vector_array, frames is an input parameter that 
# determines the number of frames to use for computing the Mel spectrogram.

# Mel spectrograms are typically computed using a short-time Fourier transform (STFT), 
# where the audio signal is divided into small, overlapping time frames, and the Fourier 
# transform is computed for each frame. The resulting spectrogram is a matrix where each 
# column corresponds to a frame and each row corresponds to a frequency bin.

# In this function, the Mel spectrogram is computed using a fixed number of frames, where 
# each frame corresponds to a short-time window of the audio signal. The number of frames is 
# determined by the input parameter frames, which defaults to a value of 5.

# By setting frames to 5, the Mel spectrogram will be computed using a window size of 5 frames, 
# which means that the Mel spectrogram coefficients for each time frame will be computed using 
# information from the current frame and the two preceding and two following frames. This can help 
# capture some of the temporal dynamics of the audio signal and improve the quality of the feature 
# representation.
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr, y = demux_wav(file_name)  # demulitplexing
    mel_spectrogram = librosa.feature.melspectrogram(y=y,  # raw data as input (.wav file)
                                                     sr=sr,  # sampling rate
                                                     n_fft=n_fft,  # length of fft window (fast fourier transformation)
                                                     hop_length=hop_length,  # number of samples between successive frames.
                                                     # number of audio samples between adjacent STFT columns
                                                     # STFT: short-time fourier transfrom
                                                     # represents a signal in the time-frequency domain by computing
                                                     # discrete fourier transforms (DFT) over short overlapping windows.
                                                     n_mels=n_mels,  # number of Mel bands to generate
                                                     # Mel bands are mel-frequency spectre which are computed by bundling subsets
                                                     # of FFT bin magnitudes. (mel derives from melody)
                                                     power=power  # Exponent for the magnitude melspectrogram, 1 for energy
                                                     # 2 for power etc.)
    # returns matrix of mel spectrogram coefficients

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)
    # takes the mel spectrogram coeffitiens and the power for scaling the spectrogram
    # Computes normalized logarithmic spectrogram by adding small constant .epsilon
    # to avoid taking the logarithm of zero and then takes the base-10 log of the result.
    # It scales the amplitude values of the spectrogram logaritmically, which helps to
    # better capture the perceptual properties of the sound.

    # The resulting logarithmic spectrogram is then normalized by dividing by the scaling 
    # factor power, and multiplying by a constant factor of 20. This normalization step ensures 
    # that the resulting spectrogram is on a similar scale as traditional spectrograms and Mel 
    # spectrograms, while still preserving the logarithmic scaling.

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return numpy.empty((0, dims), float)
    # return a new array of specified shape and data type, without initializing its entries.
    # creates a new numpy array with 0 rows and dims cols, where the data type of each entry is float.
    # creates an empty array if vectorrary_size is less than 1. Avoid returning a None value
    # or raising an error where there are no vectors to store in the array.


    # 06 generate feature vectors by concatenating multi_frames
    vectorarray = numpy.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
    # Overall, the code concatenates multiple frames of the log mel-spectrogram into feature vectors by 
    # selecting a slice of the log mel-spectrogram for each frame and assigning it to the corresponding 
    # slice in a new array that stores the feature vectors.
    return vectorarray


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels,
                                            frames=frames,
                                            n_fft=n_fft,
                                            hop_length=hop_length,
                                            power=power)

        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
            # if it is the first file, it initializes dataset with zeros of the shape
            # to hold the extracted feature vectors for all files.
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
        # appends the extracted feature vectors from the current file to dataset. 
        # starting at the index of the corresponding file. this is done by multiplying
        # the current file index by the number of frames in the feature vector and using
        # this to slice the dataset array
    return dataset


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, feature_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnormal = 0/1
    """
    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    # using glob module to search for files with the specified extension in
    # the subdirectories
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                 normal_dir_name=normal_dir_name,
                                                                 ext=ext))))
    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.exception("no_wav_data!!")

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                   abnormal_dir_name=abnormal_dir_name,
                                                                   ext=ext))))
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.exception("no_wav_data!!")

    # 03 separate train & eval
    # train are all after len of abnormal (abnormal are significantly less)
    train_files = normal_files[len(abnormal_files):]  
    train_labels = normal_labels[len(abnormal_files):]
    # eval are all the normal[:len(abnormal)] concatenated with all abnormal files
    eval_files = numpy.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


########################################################################


########################################################################
# keras model
########################################################################
def keras_model(inputDim):
    """
    define the keras model
    the model based on the simple dense auto encoder (64*64*8*64*64)
    """
    # input layer 
    inputLayer = Input(shape=(inputDim,))
    # 5 fully connected hidden layers
    h = Dense(64, activation="relu")(inputLayer)  # 64 units and ReLu activation fct.
    h = Dense(64, activation="relu")(h)  # 64 units and ReLu activation fct.
    h = Dense(8, activation="relu")(h)  # just 8 units and ReLu activation fct.
    h = Dense(64, activation="relu")(h)  # built up to 64 again
    h = Dense(64, activation="relu")(h)
    # ouput layer
    h = Dense(inputDim, activation=None)(h)  # same numbers of layers as input layer

    return Model(inputs=inputLayer, outputs=h)


########################################################################

print("I'm starting...")

########################################################################
# main
########################################################################
if __name__ == "__main__":

    # load parameter yaml
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)

    # make output directory
    os.makedirs(param["pickle_directory"], exist_ok=True)
    os.makedirs(param["model_directory"], exist_ok=True)
    os.makedirs(param["result_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = sorted(glob.glob(os.path.abspath("{base}/*/*/*".format(base=param["base_directory"]))))

    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    results = {}

    # loop of the base directory
    for dir_idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=dir_idx + 1, total=len(dirs)))

        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]

        # setup path
        evaluation_result = {}
        train_pickle = "{pickle}/train_{machine_type}_{machine_id}_{db}.pickle".format(pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id, db=db)
        eval_files_pickle = "{pickle}/eval_files_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        eval_labels_pickle = "{pickle}/eval_labels_{machine_type}_{machine_id}_{db}.pickle".format(
                                                                                       pickle=param["pickle_directory"],
                                                                                       machine_type=machine_type,
                                                                                       machine_id=machine_id,
                                                                                       db=db)
        model_file = "{model}/model_{machine_type}_{machine_id}_{db}.hdf5".format(model=param["model_directory"],
                                                                                  machine_type=machine_type,
                                                                                  machine_id=machine_id,
                                                                                  db=db)
        history_img = "{model}/history_{machine_type}_{machine_id}_{db}.png".format(model=param["model_directory"],
                                                                                    machine_type=machine_type,
                                                                                    machine_id=machine_id,
                                                                                    db=db)
        evaluation_result_key = "{machine_type}_{machine_id}_{db}".format(machine_type=machine_type,
                                                                          machine_id=machine_id,
                                                                          db=db)

        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        if os.path.exists(train_pickle) and os.path.exists(eval_files_pickle) and os.path.exists(eval_labels_pickle):
            train_data = load_pickle(train_pickle)
            eval_files = load_pickle(eval_files_pickle)
            eval_labels = load_pickle(eval_labels_pickle)
        else:
            train_files, train_labels, eval_files, eval_labels = dataset_generator(target_dir)

            train_data = list_to_vector_array(train_files,
                                              msg="generate train_dataset",
                                              n_mels=param["feature"]["n_mels"],
                                              frames=param["feature"]["frames"],
                                              n_fft=param["feature"]["n_fft"],
                                              hop_length=param["feature"]["hop_length"],
                                              power=param["feature"]["power"])

            save_pickle(train_pickle, train_data)
            save_pickle(eval_files_pickle, eval_files)
            save_pickle(eval_labels_pickle, eval_labels)

        # model training
        print("============== MODEL TRAINING ==============")
        model = keras_model(param["feature"]["n_mels"] * param["feature"]["frames"])
        model.summary()
        # show summary of th emodel architecture, including the layers and parameters.

        # training
        if os.path.exists(model_file):
            model.load_weights(model_file)
        else:
            model.compile(**param["fit"]["compile"])
            # compile is used to configure the learning process before training a model
            # specifies loss, optimizer and evaluation metrixs that will be used durning training and testing
            # defiend in baseline.yaml
            history = model.fit(train_data,  # fit used for training with all its parameters
                                train_data,
                                epochs=param["fit"]["epochs"],
                                batch_size=param["fit"]["batch_size"],
                                shuffle=param["fit"]["shuffle"],
                                validation_split=param["fit"]["validation_split"],
                                verbose=param["fit"]["verbose"])
                                # verbose parameter "1" which is used here 
                                # and also is the default, indicates to show
                                # a progress bar (0, 1, 2) look up again for more information

            visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
            visualizer.save_figure(history_img)
            model.save_weights(model_file)

        # evaluation
        print("============== EVALUATION ==============")
        y_pred = [0. for k in eval_labels]
        # initialize a list which has the same length as eval_labels
        # it gets overwritten during the evaluation
        y_true = eval_labels


        # This section of the code performs evaluation on the trained model.
        # It computes the Area Under the Curve (AUC) score using the ROC 
        # curve for binary classification.

        # The evaluation is performed on a separate set of evaluation files,
        # and for each file, it computes the mean squared error between the
        # predicted output and the ground truth. The resulting errors are then used 
        # to compute the AUC score. The AUC score is a commonly used metric for binary 
        # classification problems, and it measures the performance of the model in 
        # distinguishing between positive and negative samples.

        # The AUC score ranges from 0 to 1, with 0.5 indicating random guessing and 1 
        # indicating perfect classification.

        for num, file_name in tqdm(enumerate(eval_files), total=len(eval_files)):
            try:
                data = file_to_vector_array(file_name,
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"])
                error = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
                y_pred[num] = numpy.mean(error)  # <------------------------ y_pred (model)
                # The error is being used as a proxy for the anomaly score. Anomaly detection 
                # algorithms aim to identify unusual data points or patterns that deviate 
                # significantly from the expected normal behavior. In this case, the model is 
                # trained to reconstruct normal data points accurately, so when it is given an 
                # anomalous data point, it may not be able to reconstruct it as well, resulting 
                # in a higher error. 
                
                # Thus, the error is being used as an anomaly score, where a higher error 
                # indicates a higher likelihood of the data point being anomalous.
                # The anomaly scores are then used to compute ROC  curve
                # to evaluate the performance of the anomaly detection algorithm.

            except:
                logger.warning("File broken!!: {}".format(file_name))

        score = metrics.roc_auc_score(y_true, y_pred)
        logger.info("AUC : {}".format(score))
        evaluation_result["AUC"] = float(score)
        results[evaluation_result_key] = evaluation_result
        print("===========================")

    # output results
    print("\n===========================")
    logger.info("all results -> {}".format(result_file))
    with open(result_file, "w") as f:
        f.write(yaml.dump(results, default_flow_style=False))
    print("===========================")
########################################################################
