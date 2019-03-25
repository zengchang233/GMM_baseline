#! /bin/bash

stage=0

# generate task definition files. You have to change the path.

if [ $stage -le 0 ]; then
    dev_enrollment='/home/zeng/zeng/aishell/af2019-sr-devset-20190312/enrollment.csv'
    dev_annotation='/home/zeng/zeng/aishell/af2019-sr-devset-20190312/annotation.csv'
    python utils.py --enroll $dev_enrollment --dev $dev_annotation
fi


# feature extraction

if [ $stage -le 1 ]; then
    echo '-----start extract feature-----'
    audio_dir='/home/zeng/zeng/aishell/wav'        # change it to your own audio dir.
    feature_type='mfcc'                            # you can change it to plp.
    for i in train test; do
        python preprocess.py $audio_dir $feature_type $i
    done
    echo '-----feature extraction done-----'
fi


# train model

distribNum=512                # mixture number, 512 is common, but you can set it as you like.
num_thread=4                  # threads number, please be careful, DO NOT more than the cores of
                              # your machine. Half of core amount of your machine is preferred.
if [ $stage -le 2 ]; then
    echo '-----start training-----'
    feature_type='mfcc'                       # you can change it to plp or fb
    python gmm_ubm.py mfcc_512 --feat_type $feature_type \
                               --delta --distribNum $distribNum \
                               --num_thread $num_thread \
                               --train --adaptation --score
    echo '-----training done-----'
fi

# calculate accuracy
if [ $stage -le 3 ]; then
    echo '-----start score-----'
    dev_score_file=task/dev_score.h5
    python utils.py --score $dev_score_file
    echo '-----score done-----'
fi
