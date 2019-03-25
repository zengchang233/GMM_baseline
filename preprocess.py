import sidekit
import pandas as pd
import numpy as np
import os
import random
import sys
from multiprocessing import cpu_count

def readDirStruct(audio_path, train = True):
    '''
    param:
        path  : path to wav dir
        train : if True, extract train feature, else, extract Test feature
    return:
        wavlist : wav file list for FeatureServer
    '''
    if not os.path.exists(os.getcwd() + '/log/'):
        os.mkdir(os.getcwd() + '/log/')
    if train:
        wav_dir = audio_path
        speaker_list = os.listdir(wav_dir)
        wavlist = []
        for i in random.sample(speaker_list,100):
            speaker_dir_path = wav_dir + '/' + i
            speech_list = os.listdir(speaker_dir_path) # .wav audio
            for j in speech_list:
                wavlist.append(i + '/' + j.split('.')[0])
        log_file_name = os.getcwd() + '/log/aishell2.log'
    else:
        wav_dir = audio_path
        wavlist = [i.split('.')[0] for i in os.listdir(wav_dir)]
        log_file_name = os.getcwd() + '/log/aishell2_test.log'
    with open(log_file_name, 'w') as fobj:
        for i in wavlist:
            fobj.write(i+'\n')
        return wavlist

def extractFeature(audio_dir, feature_dir, cep = 'mfcc', train = True):
    '''
    param:
        audio_dir   : path to audio dir
        feature_dir : path to feature dir
        cep         : cepstrum type 'mfcc' or 'plp'
        train       : if True, extract feature from train set, else, from test set
    return:
        None
    '''
    # if feature dir not exists, create feature dir
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)

    wavlist = []
    if train:
        wav_list_file = os.getcwd() + '/log/aishell2_train.log'
    else:
        wav_list_file = os.getcwd() + '/log/aishell2_test.log'

    if os.path.exists(wav_list_file):
        with open(wav_list_file,'r') as fobj:
            for i in fobj:
                wavlist.append(i[0:-1])
    else:
        wavlist = readDirStruct(audio_dir, train)
        
    # prepare the necessary variables
    showlist = np.asarray(wavlist)
    channellist = np.zeros_like(showlist, dtype = int) 
    # create feature extractor
    extractor = sidekit.FeaturesExtractor(audio_filename_structure=audio_dir+'/{}.wav',
                                          feature_filename_structure=feature_dir+"/{}.h5",
                                          sampling_frequency=16000,
                                          lower_frequency=133.3333,
                                          higher_frequency=6955.4976,
                                          filter_bank="log",
                                          filter_bank_size=64,
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=20,
                                          vad="snr",
                                          snr=40,
                                          pre_emphasis=0.97,
                                          save_param=["vad", "energy", "cep", "fb"],
                                          feature_type=cep,
                                          keep_all_features=True)                                   
    # save the feature
    print('start extracting feature')
    extractor.save_list(show_list = showlist, 
                        channel_list = channellist, 
                        num_thread = cpu_count() // 2)
    print('extract feature done')

def readEnrollmentPaths(_path):
    data = pd.read_csv(_path)
    spk_indices = {}
    grp_temp_indices = {}
    classes = []
    for label in data['SpeakerID']:
        classes.append(str(label))
    for grpid, label, ind in zip(data['GroupID'], data['SpeakerID'], data['FileID']):
        if not str(label) in spk_indices.keys():
            spk_indices[str(label)] = []
        if not grpid in grp_temp_indices.keys():
            grp_temp_indices[grpid] = []
        spk_indices[str(label)].append(ind)
        grp_temp_indices[grpid].append(str(label))
    grp_indices = {k:set(v) for k, v in grp_temp_indices.items()}
    return spk_indices, grp_indices, classes # 说话人-文件名列表，组编号-说话人列表，所有说话人集合

def readTestPaths(_path):
    data = pd.read_csv(_path)
    indices = {}
    classes = []
    for label in data['GroupID']:
        classes.append(label)
    for grp, idx in zip(classes, data['FileID']):
        if not grp in indices.keys():
            indices[grp] = []
        indices[grp].append(idx) # 组编号到文件名的映射
    return indices, classes # 组编号-文件名列表，所有组集合

def main():
    audio_dir = sys.argv[1]
    feature_dir = os.getcwd() + '/{}_{}_feature'.format(sys.argv[2], sys.argv[3])
    train = True if sys.argv[3] == 'train' else False
    extractFeature(audio_dir, feature_dir, sys.argv[2], train)

if __name__ == '__main__':
    main()
