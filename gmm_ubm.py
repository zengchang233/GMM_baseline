#! /usr/bin/python3
#2018-09-04 12:17:03 

import sidekit
import os
import sys
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

def get_parser():
    parser = argparse.ArgumentParser(description = 'script for GMM-UBM adaptation')
    
    parser.add_argument('name',
                         help = 'model name'
    )
    
    parser.add_argument('--feat_type',
                         type = str,
                         default = 'mfcc',
                         help = 'feature type'
    )
    
    parser.add_argument('--delta',
                         action = 'store_true',
                         help = 'using delta information of feature'
    )
    
    parser.add_argument('--distribNum',
                         type = int,
                         default = 512,
                         help = 'distribution number'
    )
    parser.add_argument('--num_thread',
                         type = int,
                         default = 20,
                         help = 'threads number'
    )
    parser.add_argument('--extract', 
                         action = 'store_true',
                         help = 'extract feature from audio')
    
    parser.add_argument('--train',
                         action = 'store_true',
                         help = 'train the ubm model')

    parser.add_argument('--adaptation',
                         action = 'store_true',
                         help = 'adaptation for speaker model'
    )
    
    parser.add_argument('--score',
                         action = 'store_true',
                         help = 'compute the eer'                    
    )

    args = parser.parse_args()
    
    return args
    
def save_info(args):
    info =  '\nModel name:          {}\n'.format(args.name)
    info += 'Feature type:        {}\n'.format(args.feat_type)
    if args.train or args.adaptation:
        info += 'Feature delta:       {}\n'.format(args.delta)
    info += 'Distribution number: {}\n'.format(args.distribNum)
    info += 'Thread number:       {}\n'.format(args.num_thread)
    if not os.path.exists('log/'):
        os.mkdir('log')
    if not os.path.exists('log/{}_info.txt'):
        with open('log/{}_info.txt'.format(args.name), 'w') as f:
            f.write(info)
    else:
        with open('log/{}_info.txt'.format(args.name), 'a') as f:
            f.write(info)

def preprocess():
    wav_dir = '/home/zeng/zeng/aishell/aishell2/ios/data/wav'
    #dev_dir = '/home/zeng/zeng/aishell/af2019-sr-devset-20190312/data'
    speaker_list = os.listdir(wav_dir)
    wavlist = []
    for i in speaker_list:
        speaker_dir_path = wav_dir + '/' + i
        speech_list = os.listdir(speaker_dir_path)
        for j in speech_list:
            wavlist.append(i + '/' + j.split('.')[0])
    with open(os.getcwd() + '/log/aishell2_wavlist.log', 'w') as fobj:
        for i in wavlist:
            fobj.write(i+'\n')
    return wavlist

def extract_feat(args):
    # wav directory and feature directory
    audio_folder = '/home/zeng/zeng/aishell/aishell2/ios/data/wav'
    features_folder = '/home/zeng/zeng/aishell/aishell2/ios/data/feature'
    wavlist = []
    if os.path.exists(os.getcwd() + '/log/aishell2_wavlist.log'):
        with open(os.getcwd() + '/log/aishell2_wavlist.log','r') as fobj:
            for i in fobj:
                wavlist.append(i[0:-1])
    else:
        wavlist = preprocess()
        
    # prepare the necessary variables
    showlist = np.asarray(wavlist)
    channellist = np.zeros_like(showlist, dtype = int)
        
    # create feature extractor
    extractor = sidekit.FeaturesExtractor(audio_filename_structure=audio_folder+'/{}.wav',
                                          feature_filename_structure=features_folder+"/{}.h5",
                                          sampling_frequency=16000,
                                          lower_frequency=100.0,
                                          higher_frequency=7000.0,
                                          filter_bank="log",
                                          filter_bank_size=64,
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=24,
                                          vad="snr",
                                          snr=40,
                                          pre_emphasis=0.97,
                                          save_param=["vad", "energy", "cep", "fb"],
                                          keep_all_features=True)
    
    # save the feature
    extractor.save_list(show_list = showlist, 
                        channel_list = channellist, 
                        num_thread = args.num_thread)
    
def train_ubm(**args):
    if args['feat_type'] == 'mfcc':
        datasetlist = ["energy", "cep", "vad"]
        mask = "[0-12]"
        features_folder = '/home/zeng/zeng/aishell/aishell2/ios/data/feature'
    if args['feat_type'] == 'fb':
        datasetlist = ["fb", "vad"]
        mask = None
        features_folder = '/home/zeng/zeng/aishell/aishell2/ios/data/feature'
        
    ubmlist = []
    if os.path.exists(os.getcwd() + '/log/aishell2_wavlist.log'):
        with open(os.getcwd() + '/log/aishell2_wavlist.log','r') as fobj:
            for i in fobj:
                ubmlist.append(i[0:-1])
    else:
        ubmlist = preprocess()
        
    # create feature server for loading feature from disk
    server = sidekit.FeaturesServer(features_extractor=None,
                                    feature_filename_structure=features_folder+"/{}.h5",
                                    sources=None,
                                    dataset_list=datasetlist,
                                    mask=mask,
                                    feat_norm="cmvn",
                                    global_cmvn=None,
                                    dct_pca=False,
                                    dct_pca_config=None,
                                    sdc=False,
                                    sdc_config=None,
                                    delta=args['delta'],
                                    double_delta=args['delta'],
                                    delta_filter=None,
                                    context=None,
                                    traps_dct_nb=None,
                                    rasta=True,
                                    keep_all_features=False)
    # create Mixture object for training
    ubm = sidekit.Mixture()
    ubm.EM_split(server, ubmlist, args['distribNum'], iterations = (1,2,2,4,4,4,4,8,8,8,8,8,8), num_thread = args['num_thread'], save_partial = True)
    # write trained ubm to disk
    ubm.write(os.getcwd() + '/model/ubm_512.h5')

def adaptation(args):
    if args.feat_type == 'mfcc':
        datasetlist = ["energy", "cep", "vad"]
        mask = "[0-12]"
        features_folder = '/home/zeng/zeng/aishell/af2019-sr-devset-20190312/feature'
    if args.feat_type == 'fb':
        datasetlist = ["fb", "vad"]
        mask = None
        features_folder = '/home/zeng/zeng/aishell/af2019-sr-devset-20190312/feature'
    
    # create feature server for loading feature from disk    
    feature_server = sidekit.FeaturesServer(features_extractor=None,
                                            feature_filename_structure=features_folder+"/{}.h5",
                                            sources=None,
                                            dataset_list=datasetlist,
                                            mask=mask,
                                            feat_norm="cmvn",
                                            global_cmvn=None,
                                            dct_pca=False,
                                            dct_pca_config=None,
                                            sdc=False,
                                            sdc_config=None,
                                            delta=True if args.delta else False,
                                            double_delta=True if args.delta else False,
                                            delta_filter=None,
                                            context=None,
                                            traps_dct_nb=None,
                                            rasta=True,
                                            keep_all_features=False)

    enroll_idmap = sidekit.IdMap(os.getcwd() + '/task/idmap.h5')
    ndx = sidekit.Ndx(os.getcwd() + '/task/dev_ndx.h5')
    
    ubm = sidekit.Mixture()
    ubm.read(os.getcwd() + '/model/ubm.h5')
    enroll_stat = sidekit.StatServer(enroll_idmap, distrib_nb = ubm.distrib_nb(), feature_size = ubm.dim())
    enroll_stat.accumulate_stat(ubm=ubm, feature_server = feature_server, seg_indices=range(enroll_stat.segset.shape[0]), num_thread=args.num_thread)
    enroll_stat.write(os.getcwd() + '/task/enroll_stat.h5')

    print('MAP adaptation', end = '')
    regulation_factor = 16
    enroll_sv = enroll_stat.adapt_mean_map_multisession(ubm, regulation_factor)
    enroll_sv.write(os.getcwd() + '/task/enroll_sv.h5')
    print('\rMAP adaptation done')

    print('Compute scores', end = '')
    score = sidekit.gmm_scoring(ubm, enroll_sv, ndx, feature_server, num_thread = args.num_thread)
    score.write(os.getcwd() + '/task/dev_score.h5')
    print('\rCompute scores done')
    
def compute_eer(args):
    print('Compute eer', end = '')
    key = sidekit.Key(os.getcwd() + '/task/dev_key.h5')
    score = sidekit.Scores(os.getcwd() + '/task/dev_score.h5')
    print('\rCompute eer done')
    
    dp = sidekit.DetPlot()
    prior = sidekit.logit_effective_prior(0.01, 1, 1)
    dp.set_system_from_scores(score, key)
    minDCF, _, __, ___, eer = sidekit.bosaris.detplot.fast_minDCF(dp.__tar__[0], dp.__non__[0], prior, normalize=True)
    
    with open('task/dev_result.txt', 'w') as f:
        f.write('eer    :    {:7.2%}\n'.format(eer))
        f.write('minDCF :    {:7.2%}\n'.format(minDCF))
    return eer, minDCF
    
def main():
    args = get_parser()
    save_info(args)
    if args.extract:
        extract_feat(args)
    if args.train:
        train_ubm(feat_type = args.feat_type, delta = args.delta, distribNum = args.distribNum, num_thread = args.num_thread)
    if args.adaptation:
        adaptation(args)
    if args.score:
        compute_eer(args)

if __name__ == '__main__':
    main()























