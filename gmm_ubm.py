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
    
def train_ubm(**args):
    if (args['feat_type'] == 'mfcc') or (args['feat_type'] == 'plp'):
        datasetlist = ["energy", "cep", "vad"]
        mask = "[0-19]"
    if args['feat_type'] == 'fb':
        datasetlist = ["fb", "vad"]
        mask = None
    features_folder = os.getcwd() + '/{}_train_feature'.format(args['feat_type'])
        
    ubmlist = []
    try:
        with open(os.getcwd() + '/log/aishell2.log','r') as fobj:
            for i in fobj:
                ubmlist.append(i[0:-1])
    except FileNotFoundError:
        print('please generate ubm wav list as first')
        
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
    if (args.feat_type == 'mfcc') or (args.feat_type == 'plp'):
        datasetlist = ["energy", "cep", "vad"]
        mask = "[0-19]"
    if args.feat_type == 'fb':
        datasetlist = ["fb", "vad"]
        mask = None
    features_folder = os.getcwd() + '/{}_test_feature'.format(args.feat_type)
    
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
    
def main():
    args = get_parser()
    save_info(args)
    if args.train:
        train_ubm(feat_type = args.feat_type, 
                  delta = args.delta, 
                  distribNum = args.distribNum, 
                  num_thread = args.num_thread)
    if args.adaptation:
        adaptation(args)

if __name__ == '__main__':
    main()























