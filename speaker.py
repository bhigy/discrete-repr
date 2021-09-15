import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import logging
import pandas as pd
import pickle
from sklearn import linear_model
from collections import Counter
from sklearn.preprocessing import StandardScaler
import plotnine as pn

# setting seeds
SEED = 1234
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

def adjust_for_random(data, columns):
    data_t = data.query("mode=='trained'")
    data_r = data.query("mode=='random'")
    data_d = data_t.merge(
        data_r,
        on=['size', 'level', 'run'],
        suffixes=['_t', '_r'])
    for m in columns:
        data_d[m] = data_d[f'{m}_t'] - data_d[f'{m}_r']
    return data_d

def analyze(versions, experiment_dir, output_dir, layers, runs, model="vg"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Mean pooling; global diagnostic")
    config = dict(versions=versions,
                  directory=experiment_dir,
                  output=output_dir,
                  test_size=1/2,
                  runs = runs,
                  layers=layers,
                  model=model)
    global_diagnostic(config)

def global_diagnostic(config):
    out = config['output'] / '{}_speaker_classification.csv'.format(config['model'])
    del config['output']
    result = meanpooled_diagnostic(directory=config['directory'], layers=config['layers'], runs=config['runs'], versions=config['versions'], model_variant=config['model'])
    df_results = pd.DataFrame.from_records(result)
    df_results.to_csv(out)

def meanpooled_diagnostic(directory='.', layers=[], test_size=1/2, c=1.0, runs=[], versions=[], model_variant="vg"):
    from sklearn.model_selection import train_test_split
    splitseed = random.randint(0, 1024) # this is used to make identical but random splits on mfcc & state data
    result = []

    logging.info("Loading speaker data")
    with open("/roaming/gchrupal/verdigris/platalea.vq/data/datasets/flickr8k/wav2spk.txt") as f:
        speakerdata = f.read().splitlines()
    wav2spk = {}
    for item in speakerdata:
        fields = item.split()
        wav2spk[fields[0]] = int(fields[1])

    for size in versions:
        for run in runs:
            for layer in layers:
                logging.info("Loading data")
                if model_variant=="vg":
                    data = pickle.load(open("{}/vq-{}-q{}-r{}/global_input.pkl".format(directory, size, layer, run), "rb")) # VQ
                elif model_variant=="vn":
                    data = pickle.load(open("{}/vn-{}-r{}/global_input.pkl".format(directory, size, run), "rb")) # Niekerk
                speakers = [wav2spk[wav] for wav in data['audio_id']]
                act = [ np.mean(item[:, :], axis=0) for item in data['audio'] ]
                y, y_val, X, X_val = train_test_split(speakers, act, test_size=test_size, random_state=splitseed)

                logging.info("Training for MFCC features")
                model = linear_model.LogisticRegression(C=c, max_iter = 1000)
                this = train_classifier(model, X, y, X_val, y_val, majority=majority_class, scale=True)
                result.append({**this, 'C': c, 'mode': 'random', 'level': 'mfcc', 'run': run, 'size': size})
                result.append({**this, 'C': c, 'mode': 'trained', 'level': 'mfcc', 'run': run, 'size': size})
                del X, X_val
                logging.info("Maximum accuracy on val: {}".format(result[-1]['acc']))

                for mode in ['random', 'trained']:
                    logging.info("Loading activations for {} layer {} run {}".format(mode, layer, run))
                    if model_variant == 'vn':
                        data = pickle.load(open("{}/vn-{}-r{}/global_{}_codebook.pkl".format(directory, size, run, mode), "rb"))
                    elif model_variant == 'vg':
                        data = pickle.load(open("{}/vq-{}-q{}-r{}/global_{}_codebook.pkl".format(directory, size, layer, run, mode), "rb"))
                    logging.info("Training for {} layer {} run {}".format(mode, layer, run))
                    act = [ np.mean(item[:, :], axis=0) for item in data['codebook'] ]
                    X, X_val = train_test_split(act, test_size=test_size, random_state=splitseed)
                    model = linear_model.LogisticRegression(C=c, max_iter = 1000)
                    this = train_classifier(model, X, y, X_val, y_val, majority=majority_class, scale=True)
                    result.append({**this, 'C': c, 'mode': mode, 'level': layer, 'run': str(run), 'size': str(size)})
                    del X, X_val
                    logging.info("Maximum accuracy on val: {}".format(result[-1]['acc']))
    return result

def collate(items):
    x, y = zip(*items)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)
    return x, y

def majority_binary(y):
    return (y.mean(dim=0) >= 0.5).float()

def majority_class(y):
    return Counter(y).most_common(1)[0][0]

def train_classifier(model, X, y, X_val, y_val, majority=majority_class, scale=False):
    # scale data
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_val = scaler.transform(X_val)

    maj = majority(y)
    baseline = np.mean([(maj == y_i) for y_i in y_val ])
    logging.info("Baseline accuracy: {}".format(baseline))
    # train classifier
    model.fit(X, y)
    accuracy_val = model.score(X_val, y_val)
    scores = dict(acc=accuracy_val, maj_baseline=baseline)
    return scores

def inject(x, e):
    return [ {**xi, **e} for xi in x ]

def get_axislabel(metric):
    labels = {'rsa': "r",
              'vmeasure': 'NMI',
              'abx': 'ABX',
              'diag': 'Accuracy',
              'acc': 'Accuracy'}
    if metric in labels:
        return labels[metric]
    else:
        return metric

def get_title(metric):
    labels = {'rsa': "RSA",
              'vmeasure': 'NMI',
              'abx': 'ABX',
              'diag': 'Diagnostic classifier',
              'acc': 'Speaker identification'}
    if metric in labels:
        return labels[metric]
    else:
        return metric

def plot_metric_vs_size(data, metric):
    # hide legend for color when we have only one level
    guide_color = data.level.unique().shape[0] > 1
    linetypes = {"trained": "solid", "random": "dashed"}
    shapes = {"trained": "o", "random": "^"}
    g = pn.ggplot(data, pn.aes(x='size',
                               color='factor(level)',
                               shape='mode',
                               linetype='mode',
                               y=metric)) + \
    pn.scale_x_continuous(trans="log2", breaks=[32, 64, 128, 256, 512, 1024]) + \
    pn.scale_color_discrete(name="level", guide=guide_color) + \
    pn.scale_linetype_manual(values=linetypes) + \
    pn.scale_shape_manual(values=shapes) + \
    pn.geom_smooth() + \
    pn.geom_point(alpha=0.7, size=3) + \
    pn.labs(x="Codebook size",
            y=get_axislabel(metric),
            title=get_title(metric),
            shape="Mode",
            linetype="Mode") + \
    pn.theme(text=pn.element_text(size=16, family='serif'))
    if guide_color:
        g += pn.guides(color=pn.guide_legend(title='VQ at level'))
    return g

def plot_both():
    speaker_vg = pd.read_csv("data/out/vg_speaker_classification.csv")
    speaker_vn = pd.read_csv("data/out/vn_speaker_classification.csv")
    speaker_vg = speaker_vg[speaker_vg.level.ne("mfcc")] # filtering out mfcc
    speaker_vn = speaker_vn[speaker_vn.level.ne("mfcc")] # filtering out mfcc
    speaker_vg["model"] = "Visually grounded"
    speaker_vn["model"] = "Self-supervised"
    speaker_vn["level"] = "0"
    data = pd.concat([speaker_vg, speaker_vn])
    data['level'] = data['level'].astype(int)
    metric = 'acc'

    linetypes = {"trained": "solid", "random": "dashed"}
    colors = {0: 'grey', 1: '#db5f57', 2:'#57db5f', 3: '#736cdd'}
    labels = ["Self-supervised", "VS - VQ at level 1", "VS - VQ at level 2", "VS - VQ at level 3"]
    shapes = {0: 'o', 1: '^', 2:'s', 3: 'D'}
    fill_values = {'trained': 'black', 'random': 'white'}

    g = pn.ggplot(data, pn.aes(x='size',
                               color='factor(level)',
                               shape='factor(level)',
                               linetype='mode',
                               y=metric)) + \
        pn.scale_x_continuous(trans="log2", breaks=[32, 64, 128, 256, 512, 1024]) + \
        pn.scale_color_manual(colors, labels=labels) + \
        pn.scale_linetype_manual(values=linetypes) + \
        pn.scale_shape_manual(values=shapes, labels=labels) + \
        pn.geom_smooth() + \
        pn.geom_point(inherit_aes=True, mapping=pn.aes(fill='mode'),size=3, alpha=.7) + \
        pn.scale_fill_manual(values=fill_values) + \
        pn.labs(x="Codebook size",
                y=get_axislabel(metric),
                title=get_title(metric),
                shape="Model",
                fill='Mode',
                linetype="Mode") + \
        pn.theme(text=pn.element_text(size=16, family='serif')) + \
	pn.theme(legend_key_width = 35) + \
        pn.guides(color=pn.guide_legend(title='Model'))
    pn.ggsave(g, "fig/speaker_combined.pdf")


def plot_speaker(model):
    # recall vs size
    speaker = pd.read_csv(f"data/out/{model}_speaker_classification.csv")
    mfcc = speaker[speaker.level.eq("mfcc")]
    base = pd.DataFrame.from_records([
        dict(baseline='MFCC', acc=mfcc.acc.mean()),
        dict(baseline='Majority',acc=mfcc.maj_baseline.mean())])
    speaker = speaker[speaker.level.ne("mfcc")] # filtering out mfcc
    if model == 'vg':
        g = plot_metric_vs_size(speaker, 'acc')
        pn.ggsave(g, "fig/speaker_vg.pdf")
    if model == 'vn':
        g = plot_metric_vs_size(speaker, 'acc')
        pn.ggsave(g, "fig/speaker_vn.pdf")
##############################################
#logging.basicConfig(filename='logs/speakerclassification.log',level=logging.DEBUG)
logging.basicConfig(filename='logs/speakerclassification.log',level=logging.DEBUG)
experiments_path = "/roaming/gchrupal/verdigris/vq-analyze-code/experiments"
versions = [32, 64, 128, 256, 512, 1024]
layers = [1, 2, 3]
runs = [0, 1, 2]

##############################################
#analyze(versions, experiments_path, 'data/out/', layers, runs, model="vg")
#analyze(versions, experiments_path, 'data/out/', [1], runs, model="vn")
#plot_speaker("vg")
#plot_speaker("vn")
plot_both()
