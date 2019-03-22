import numpy as np
import pandas as pd
import warnings
import sidekit
import tqdm
import argparse
import os
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser(description = 'some utils for preprocess and score')
    parser.add_argument('--enroll',
                        type = str,
                        default = None,
                        help = 'enrollment csv path')
    parser.add_argument('--dev',
                        type = str,
                        default = None,
                        help = 'annotation csv path')
    parser.add_argument('--test',
                        type = str,
                        default = None,
                        help = 'test csv path')
    parser.add_argument('--score',
                        type = str,
                        default = None,
                        help = 'calculate the score')
    args = parser.parse_args()
    return args                                                   

def remove(path):
    '''
    remove the h5 file which cannot be read
    param:
        path : trainset feature path
    '''
    server = sidekit.FeaturesServer(features_extractor=None,
                                    feature_filename_structure=path+"/{}.h5",
                                    sources=None,
                                    dataset_list=["fb", "vad"],
                                    mask=None,
                                    feat_norm="cmvn",
                                    global_cmvn=None,
                                    dct_pca=False,
                                    dct_pca_config=None,
                                    sdc=False,
                                    sdc_config=None,
                                    delta=False,
                                    double_delta=False,
                                    delta_filter=None,
                                    context=None,
                                    traps_dct_nb=None,
                                    rasta=True,
                                    keep_all_features=False)
    speaker = os.listdir(path)
    for s in speaker:
        speaker_path = os.path.join(path, s)
        speech = os.listdir(speaker_path)
        for sph in speech:
            speech_path = s + '/' + sph.split('.')[0]
            try:
                feature,_ = server.load(speech_path,0)
            except:
                os.remove(os.path.join(speaker_path, sph))
    print('done')

def enroll(path):
    '''
    create the idmap file
    param:
        path : enrollment file path
    '''
    data = pd.read_csv(path)
    models = []
    segments = []
    for j in range(len(data.index)):
        models.extend([data['SpeakerID'][j]])
        segments.extend([data['FileID'][j]])
    idmap = sidekit.IdMap()
    idmap.leftids = np.asarray(models)
    idmap.rightids = np.asarray(segments)
    idmap.start = np.empty(idmap.rightids.shape, '|O')
    idmap.stop = np.empty(idmap.rightids.shape, '|O')
    assert idmap.validate(), 'idmap is not valid'
    idmap.write('task/idmap.h5')

def trial_test(path):
    '''
    create the test ndx file
    param:
        path : the test file path
    '''
    data = pd.read_csv(path)
    models = []
    segments = []
    trials = []
    enroll_idmap = sidekit.IdMap('task/idmap.h5')
    enroll_models = []
    for i in range(0, len(enroll_idmap.leftids), 3):
        enroll_models.append(enroll_idmap.leftids[i])
    print(enroll_models)
    for i in range(len(data.index)):
        for j in range(5):
            ind = data['GroupID'][i] * 5 + j
            models.append(enroll_models[ind])
            segments.append(data['FileID'][i])
            trials.append('nontarget')
    key = sidekit.Key(models=np.array(models),
                      testsegs=np.array(segments), 
                      trials=np.array(trials))
    ndx = key.to_ndx()
    assert ndx.validate(), 'ndx is not valid'
    ndx.write('task/ndx.h5')

def trial_dev(path):
    '''
        create the dev ndx and key file
        param:
            path : the annotation file path
        '''
    data = pd.read_csv(path)
    models = []
    segments = []
    trials = []
    enroll_idmap = sidekit.IdMap('task/idmap.h5')
    enroll_models = []
    for i in range(0, len(enroll_idmap.leftids), 3):
        enroll_models.append(enroll_idmap.leftids[i])
    print(enroll_models)
    for i in range(len(data.index)):
        for j in range(5):
            ind = data['GroupID'][i] * 5 + j
            models.append(enroll_models[ind])
            segments.append(data['FileID'][i])
            if str(data['SpeakerID'][i]) == enroll_models[ind]:
                trials.append('target')
            else:
                trials.append('nontarget')
    key = sidekit.Key(models=np.array(models),
                    testsegs=np.array(segments),
                    trials=np.array(trials))
    assert key.validate(), 'key is not valid'
    key.write('task/dev_key.h5')
    ndx = key.to_ndx()
    assert ndx.validate(), 'ndx is not valid'
    ndx.write('task/dev_ndx.h5')

def score(score):
    '''
    param:
        score      : score object or score file path of model on dev set
    return:
        results    : final score of the model
    '''
    if type(score) == str:
        score = sidekit.Scores(score)
    
    scoremat = score.scoremat
    minimum = scoremat.min()
    maximum = scoremat.max()
    models = score.modelset
    segments = score.segset.tolist()
    dev = pd.read_csv('../af2019-sr-devset-20190312/annotation.csv')
    results = []
    for threshold in tqdm.tqdm(np.linspace(minimum, maximum, 300)):
        correct = 0
        for i, fileid in enumerate(dev['FileID']):
            ind = segments.index(fileid)
            score = scoremat[:, ind]
            score = score[score!=0]
            if (score > threshold).sum() == 0:
                if dev['IsMember'][i] == 'N':
                    correct += 1
            else:
                if dev['IsMember'][i] == 'Y':
                    correct += 1
        result = correct / len(segments)
        results.append(result)
    np.savetxt("result/results.txt", results)
    return results

def show_result(results):
    '''
    show the score which is calculated according to the rule of competition
    param:
        results : score vector or the path of file which store the score vector.
    '''
    if type(results) == str:
        results = np.loadtxt(results)
    
    plt.plot(results)
    plt.show()

def main():
    args = get_args()
    if args.enroll:
        enroll(args.enroll)
    if args.dev:
        trial_dev(args.dev)
    if args.test:
        trial_test(args.test)
    if args.score:
        results = score(args.score)
        show_result(results)

if __name__ == '__main__':
    main()
