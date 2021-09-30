import pickle
import logging
import platalea.dataset as dataset
from platalea.experiments.config import args
import json
import os
import os.path
from pathlib import Path
import torch
import numpy as np
from ABXpy.misc.any2h5features import convert
import shutil
import ABXpy
import ABXpy.task
from zerospeech2020.evaluation.abx import _load_features_2019, _average
import ABXpy.distances.distances
from ABXpy.distance import edit_distance
import ABXpy.score as score
import ABXpy.analyze as analyze
import pandas as pd
import platalea.basicvq as M
import pydub
import random

# Parsing arguments
args.enable_help()
args.parse()


def save_data(nets, directory, batch_size=32,
              alignment_fpath='data/datasets/flickr8k/fa.json',
              use_precomputed_mfcc=True):
    Path(directory).mkdir(parents=True, exist_ok=True)
    save_global_data(nets, directory=directory,
                     alignment_fpath=alignment_fpath,
                     batch_size=batch_size,
                     use_precomputed_mfcc=use_precomputed_mfcc)
    save_local_data(directory=directory,
                    alignment_fpath=alignment_fpath)


def get_audio_and_alignments(alignment_fpath, use_precomputed_mfcc=True):
    logging.info("Loading alignments")
    data = load_alignment(alignment_fpath)
    if use_precomputed_mfcc:
        logging.info("Loading audio features")
        val = dataset.Flickr8KData(root=args.flickr8k_root, feature_fname=args.audio_features_fn,
                                   meta_fname=args.flickr8k_meta, split='val')
        alignments = [data[sent['audio_id']] for sent in val]
        # Only consider cases where alignement does not fail
        alignments = [item for item in alignments if good_alignment(item)]
        sentids = set(item['audio_id'] for item in alignments)
        audio = [sent['audio'] for sent in val if sent['audio_id'] in sentids]
    else:
        from platalea.vq_encode import config, audio_features
        # Only consider cases where alignement does not fail
        alignments = [item for item in data.values() if good_alignment(item)]
        paths = [item['audiopath'] for item in alignments]
        try:
            audio = pickle.load(open(f'{alignment_fpath}_cached_audio.pkl', 'rb'))
        except FileNotFoundError:
            audio = audio_features(paths, config)
            pickle.dump(audio, open(f'{alignment_fpath}_cached_audio.pkl', 'wb'))
    return audio, alignments


def save_global_data(nets, directory, alignment_fpath, batch_size=32,
                     use_precomputed_mfcc=True):
    """Generate data for training a phoneme decoding model."""
    audio, alignments = get_audio_and_alignments(alignment_fpath,
                                                 use_precomputed_mfcc)
    audio_np = [a.numpy() for a in audio]

    # Global data
    global_input = dict(
        audio_id=np.array([datum['audio_id'] for datum in alignments]),
        ipa=np.array([align2ipa(datum) for datum in alignments]),
        text=np.array([datum['transcript'] for datum in alignments]),
        audio=np.array(audio_np))
    global_input_path = Path(directory) / 'global_input.pkl'
    pickle.dump(global_input, open(global_input_path, "wb"), protocol=4)

    for mode, net in nets:
        global_act = collect_activations(net, audio, batch_size=batch_size)
        for layer in global_act:
            path = "{}/global_{}_{}.pkl".format(directory, mode, layer)
            logging.info("Saving global data in {}".format(path))
            pickle.dump({layer: global_act[layer]}, open(path, "wb"),
                        protocol=4)


def good_alignment(item):
    for word in item['words']:
        if word['case'] != 'success' or word['alignedWord'] == '<unk>':
            return False
    return True


def make_indexer(factors, layer):
    def inout(pad, ksize, stride, L):
        return ((L + 2*pad - 1*(ksize-1) - 1) // stride + 1)

    def index(ms):
        t = ms//10
        for l in factors:
            if factors[l] is None:
                pass
            else:
                pad = factors[l]['pad']
                ksize = factors[l]['ksize']
                stride = factors[l]['stride']
                t = inout(pad, ksize, stride, t)
            if l == layer:
                break
        return t

    return index


def label_and_save(act, ali, index, fpath, layer=None, framewise=True,
                   phone_labels=True):
    y, X = label_activations(act, ali, index, framewise, phone_labels)
    local = check_nan(features=X, labels=y)
    if layer is not None:
        local = {layer: local}
    logging.info("Saving local data in {}".format(os.path.basename(fpath)))
    pickle.dump(local, open(fpath, "wb"), protocol=4)


def save_local_data(directory, alignment_fpath):
    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    global_input = pickle.load(open(f'{directory}/global_input.pkl', "rb"))
    adata = load_alignment(alignment_fpath)
    alignments = [adata.get(i, adata.get(i+'.wav')) for i in global_input['audio_id']]
    # Local data
    logging.info("Computing local data for MFCC")
    # - with phoneme labels
    label_and_save(global_input['audio'], alignments, index=lambda ms: ms//10,
                   fpath=f'{directory}/local_input.pkl', framewise=True)
    # - with word labels
    label_and_save(global_input['audio'], alignments, index=lambda ms: ms//10,
                   fpath=f'{directory}/local_input_w.pkl', framewise=True,
                   phone_labels=False)
    try:
        factors = json.load(open(f"{directory}/downsampling_factors.json", "rb"))
    except FileNotFoundError:
        # Try one level up (applicable for /trigrams)
        factors = json.load(open(f"{directory}/../downsampling_factors.json", "rb"))
    for mode in ['trained', 'random']:
        for layer in factors.keys():
            if layer[:4] == "conv":
                pass  # This data is too big
            else:
                global_act = pickle.load(open(f'{directory}/global_{mode}_{layer}.pkl', "rb"))
                index = make_indexer(factors, layer)
                logging.info(f'Computing local data for {mode}, {layer}')
                # with phoneme labels
                label_and_save(global_act[layer], alignments, index=index,
                               fpath=f'{directory}/local_{mode}_{layer}.pkl',
                               layer=layer, framewise=True)
                # with word labels
                label_and_save(global_act[layer], alignments, index=index,
                               fpath=f'{directory}/local_{mode}_{layer}_w.pkl',
                               layer=layer, framewise=True, phone_labels=False)


def save_local_data_word(directory,
                         alignment_fpath='data/datasets/flickr8k/fa.json'):
    Path(directory).mkdir(parents=True, exist_ok=True)
    logging.getLogger().setLevel('INFO')
    logging.info("Loading alignments")
    global_input = pickle.load(open(f'{directory}/global_input.pkl', "rb"))
    adata = load_alignment(alignment_fpath)
    alignments = [adata.get(i, adata.get(i+'.wav')) for i in global_input['audio_id']]
    # Local data
    logging.info("Computing local data for MFCC")
    label_and_save(global_input['audio'], alignments, index=lambda ms: ms//10,
                   fpath=f'{directory}/local_input_w.pkl', framewise=True,
                   phone_labels=False)
    try:
        factors = json.load(open(f'{directory}/downsampling_factors.json', "rb"))
    except FileNotFoundError:
        # Default VGS settings
        factors = default_factors()
    for mode in ['trained', 'random']:
        for layer in factors.keys():
            if layer == "conv1" or layer == "conv2":
                pass  # This data is too big
            else:
                global_act = pickle.load(open(f'{directory}/global_{mode}_{layer}.pkl', "rb"))
                index = make_indexer(factors, layer)
                logging.info(f'Computing local data for {mode}, {layer}')
                # with word labels
                label_and_save(global_act[layer], alignments, index=index,
                               fpath=f'{directory}/local_{mode}_{layer}_w.pkl',
                               framewise=True, phone_labels=False)


def default_factors():
    return dict(conv=dict(pad=0, ksize=6, stride=2),
                rnn_bottom0=None,
                codebook=None,
                rnn_top0=None,
                rnn_top1=None,
                rnn_top2=None)


def make_factors(net):
    try:
        conv = dict(pad=0,
                    ksize=net.SpeechEncoder.Conv.kernel_size[0],
                    stride=net.SpeechEncoder.Conv.stride[0])
    except AttributeError:
        conv = dict(pad=0,
                    ksize=net.SpeechEncoder.Bottom.Conv.kernel_size[0],
                    stride=net.SpeechEncoder.Bottom.Conv.stride[0])
    D = dict(conv=conv)
    try:
        for k in range(net.SpeechEncoder.RNN.num_layers):
            D['rnn{}'.format(k)] = None
    except AttributeError:
        for k in range(net.SpeechEncoder.Bottom.RNN.num_layers):
            D['rnn_bottom{}'.format(k)] = None
        D['codebook'] = None
        for k in range(net.SpeechEncoder.Top.RNN.num_layers):
            D['rnn_top{}'.format(k)] = None
    return D


def load_alignment(path):
    data = {}
    for line in open(path):
        item = json.loads(line)
        item['audio_id'] = os.path.basename(item['audiopath'])
        data[item['audio_id']] = item
    return data


def label_activations(activations, alignments, index=lambda ms: ms//10,
                      framewise=True, phone_labels=True):
    """Return array of phoneme/word labels and array of corresponding
    [mean-pooled] activation states."""
    labels = []
    states = []
    for act, ali in zip(activations, alignments):
        # extract labels and activations for current utterance
        fn = frames if framewise else slices
        fr = list(fn(ali, act, index=index, phone_labels=phone_labels))
        if len(fr) > 0:
            y, X = zip(*fr)
            y = np.array(y)
            X = np.stack(X)
            labels.append(y)
            states.append(X)
    return np.concatenate(labels), np.concatenate(states)


def align2ipa(datum):
    """Extract IPA transcription from alignment information for a sentence."""
    from platalea.ipa import arpa2ipa
    result = []
    for word in datum['words']:
        for phoneme in word['phones']:
            result.append(arpa2ipa(phoneme['phone'].split('_')[0], '_'))
    return ''.join(result)


def slices(utt, rep, index, phone_labels=True,
           aggregate=lambda x: x.mean(axis=0)):
    """Return sequence of slices associated with phoneme labels, given an
       alignment object `utt`, a representation array `rep`, and
       indexing function `index`, and an aggregating function\
       `aggregate`.
    """
    lbl_fn = phones if phone_labels else words
    for label in lbl_fn(utt):
        phone, start, end = label
        assert index(start) < index(end)+1, f'Something funny: {start} {end} {index(start)} {index(end)}'
        yield (label, aggregate(rep[index(start):index(end)+1]))


def frames(utt, rep, index, phone_labels=True):
    """Return pair sequence of (phoneme label, frame), given an
       alignment object `utt`, a representation array `rep`, and
      indexing function `index`.
    """
    lbl_fn = phones if phone_labels else words
    for label, start, end in lbl_fn(utt):
        assert index(start) < index(end)+1, f'Something funny: {start} {end} {index(start)} {index(end)}'
        for j in range(index(start), index(end)+1):
            if j < rep.shape[0]:
                yield (label, rep[j])
            else:
                logging.warning(f'Index out of bounds: {j} {rep.shape}')


def phones(utt):
    """Return sequence of phoneme labels associated with start and end
     time corresponding to the alignment JSON object `utt`.

    """
    for word in utt['words']:
        pos = word['start']
        for phone in word['phones']:
            start = pos
            end = pos + phone['duration']
            pos = end
            label = phone['phone'].split('_')[0]
            if label != 'oov':
                yield (label, int(start*1000), int(end*1000))


def words(utt):
    """
    Return sequence of word labels associated with start and end time
    corresponding to the alignment JSON object `utt`.
    """
    for word in utt['words']:
        label = word['word']
        if label != 'oov':
            yield (label, int(word['start']*1000), int(word['end']*1000))


def check_nan(labels, features):
    # Get rid of NaNs
    ix = np.isnan(features.sum(axis=1))
    logging.info("Found {} NaNs".format(sum(ix)))
    X = features[~ix]
    y = labels[~ix]
    return dict(features=X, labels=y)


def collect_activations(net, audio, batch_size=32):
    data = torch.utils.data.DataLoader(dataset=audio,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=0,
                                       collate_fn=dataset.batch_audio)
    out = {}
    for au, l in data:
        logging.info("Introspecting a batch of size {}x{}x{}".format(au.shape[0], au.shape[1], au.shape[2]))
        act = net.SpeechEncoder.introspect(au.cuda(), l.cuda())
        # print({k:[vi.shape for vi in v] for k,v in act.items()})
        for k in act:
            if k not in out:
                out[k] = []
            out[k] += [item.detach().cpu().numpy() for item in act[k]]

    return {k: np.array(v) for k, v in out.items()}


def spec(d):
    if type(d) != type(dict()):
        return type(d)
    else:
        return {key: spec(val) for key, val in d.items()}


def phonemes(u):
    result = []
    for word in u['words']:
        if word['case'] == 'success':
            current = word['start']
            for phone in word['phones']:
                phone['start'] = current
                phone['end'] = current + phone['duration']
                current = phone['end']
                result.append(phone)
    return result


def shift_times(words):
    start = words[0]['start']
    startOffset = words[0]['startOffset']
    for word in words:
        word['start']       = word['start'] - start
        word['end']         = word['end'] - start
        word['startOffset'] = word['startOffset'] - startOffset
        word['endOffset']   = word['endOffset'] - startOffset
        yield word


def trigrams(xs):
    if len(xs) < 3:
        return []
    else:
        return [xs[0:3]] + trigrams(xs[1:])


def ngrams(xs, n):
    """Split sequence xs into non-overlapping segments of  length n."""
    if len(xs) < n:
        return []
    else:
        return [xs[:n]] + ngrams(xs[n:], n)


def deoov(xs):
    return [x for x in xs if not any(xi['phone'].startswith('oov') or xi['phone'].startswith('sil') for xi in x)]


def prepare_abx(k=1000, overlap=True):
    import csv
    from prepare_flickr8k import load_alignment, good_alignment
    align = {key: value for key, value in load_alignment("data/datasets/flickr8k/fa.json").items() if good_alignment(value)}
    us = random.sample(list(align.values()), k)
    wav = Path(args.flickr8k_root) / args.flickr8k_audio_subdir
    out = Path("data/flickr8k_abx_wav")
    speakers = dict(line.split() for line in open(f"{args.flickr8k_root}/flickr_audio/wav2spk.txt"))
    out.mkdir(parents=True, exist_ok=True)
    with open("data/flickr8k_abx.item", "w") as itemout:
      with open("data/flickr8k_trigrams_fa.json", "w") as tri_fa:
        items = csv.writer(itemout, delimiter=' ', lineterminator='\n')
        items.writerow(["#file", "onset", "offset", "#phone", "speaker", "context", "lang"])
        for u in us:
            filename = os.path.split(u['audiopath'])[-1]
            speaker = speakers[filename]
            bare, _ = os.path.splitext(filename)
            if overlap:
                grams = deoov(trigrams(phonemes(u)))
            else:
                grams = deoov(ngrams(phonemes(u), 3))
            logging.info("Loading audio from {}".format(filename))
            sound = pydub.AudioSegment.from_file(wav / filename)
            for i, gram in enumerate(grams):
                start = int(gram[0]['start']*1000)
                end = int(gram[-1]['end']*1000)
                triple = [phone['phone'].split('_')[0] for phone in gram]
                fragment = sound[start: end]
                target = out / "{}_{}.wav".format(bare, i)
                if end - start < 100:
                    logging.info("SKIPPING short audio {}".format(target))
                else:
                    items.writerow(["{}_{}".format(bare, i), 0, end-start, triple[1], speaker, '_'.join([triple[0], triple[-1]]), "en"])
                    fragment.export(format='wav', out_f=target)
                    # We don't have an orthographic transcript for phoneme trigrams: use ARPA.
                    word = '_'.join(phone['phone'].split('_')[0] for phone in gram)
                    tri_fa.write(json.dumps(dict(audiopath="{}".format(target),
                                                 transcript=word,
                                                 words=[dict(start=0,
                                                             end=sum([phone['duration'] for phone in gram]),
                                                             word=word,
                                                             alignedWord=word,
                                                             case='success',
                                                             phones=gram)])))
                    tri_fa.write("\n")
                    logging.info("Saved {}th trigram in {}".format(i, target))
    generate_triplets('data/flickr8k_abx')
    generate_triplets('data/flickr8k_abx', within_speaker=True)


def prepare_abx_rep(directory, k=1000, overlap=True):
    import pickle
    import csv
    from prepare_flickr8k import make_indexer, load_alignment, good_alignment
    align = {key: value for key, value in load_alignment("data/datasets/flickr8k/fa.json").items() if good_alignment(value)}
    us = random.sample(list(align.values()), k)
    out_trained = Path(directory) / "encoded/trained/flickr8k_val_rep/"
    out_rand = Path(directory) / "encoded/random/flickr8k_val_rep/"
    speakers = dict(line.split() for line in open(f"{args.flickr8k_root}/flickr_audio/wav2spk.txt"))
    factors = json.load(open("{}/downsampling_factors.json".format(directory), "rb"))
    layer = 'codebook'
    global_inp = pickle.load(open("{}/global_input.pkl".format(directory), "rb"))
    global_act_trained = pickle.load(open(f"{directory}/global_trained_{layer}.pkl", "rb"))
    global_act_rand = pickle.load(open(f"{directory}/global_random_{layer}.pkl", "rb"))
    activation_trained = dict(zip(global_inp['audio_id'], global_act_trained[layer]))
    activation_rand = dict(zip(global_inp['audio_id'], global_act_rand[layer]))
    index = make_indexer(factors, layer)
    shutil.rmtree(out_trained, ignore_errors=True)
    out_trained.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_rand, ignore_errors=True)
    out_rand.mkdir(parents=True, exist_ok=True)
    with open(Path(directory) / "flickr8k_abx_rep.item", "w") as itemout:
        items = csv.writer(itemout, delimiter=' ', lineterminator='\n')
        items.writerow(["#file", "onset", "offset", "#phone", "speaker", "context", "lang"])
        for u in us:
            filename = os.path.split(u['audiopath'])[-1]
            speaker = speakers[filename]
            bare, _ = os.path.splitext(filename)
            if overlap:
                grams = deoov(trigrams(phonemes(u)))
            else:
                grams = deoov(ngrams(phonemes(u), 3))
            for i, gram in enumerate(grams):
                start = int(gram[0]['start']*1000)
                end = int(gram[-1]['end']*1000)
                triple = [phone['phone'].split('_')[0] for phone in gram]
                fragment_trained = activation_trained[filename][index(start): index(end)]
                fragment_rand = activation_rand[filename][index(start): index(end)]
                target_trained = out_trained / "{}_{}.txt".format(bare, i)
                target_rand = out_rand / "{}_{}.txt".format(bare, i)
                if end - start < 100 or fragment_trained.shape[0] < 1:
                    logging.info("SKIPPING short audio {}".format(target_trained))
                    logging.info("SKIPPING short audio {}".format(target_rand))
                else:
                    items.writerow(["{}_{}".format(bare, i), 0, end-start, triple[1], speaker, '_'.join([triple[0], triple[-1]]), "en"])
                    np.savetxt(target_trained, fragment_trained.astype(int), fmt='%d')
                    np.savetxt(target_rand, fragment_rand.astype(int), fmt='%d')
                    logging.info("Saved activation of size {} for {}th trigram in {}".format(fragment_trained.shape, i, target_trained))
                    logging.info("Saved activation of size {} for {}th trigram in {}".format(fragment_rand.shape, i, target_rand))
    logging.info("Saved item file in {}".format(Path(directory) / "flickr8k_abx_rep.item"))
    generate_triplets(f'{directory}/flickr8k_abx_rep')
    generate_triplets(f'{directory}/flickr8k_abx_rep', within_speaker=True)


def generate_triplets(f_basepath, within_speaker=False):
    if within_speaker:
        task = ABXpy.task.Task(f"{f_basepath}.item", "phone",
                               by=["speaker", "context", "lang"])
        triplets = f"{f_basepath}_within.triplets"
    else:
        task = ABXpy.task.Task(f"{f_basepath}.item", "phone",
                               by="context", across="speaker")
        triplets = f"{f_basepath}.triplets"
    logging.info("Task statistics: {}".format(task.stats))
    logging.info("Generating triplets")
    if os.path.isfile(triplets):
        os.remove(triplets)
    task.generate_triplets(output=triplets)


def ed(x, y, normalized=None):
    return edit_distance(x, y)


def run_abx(feature_dir, triplet_file, distance=ed):
    logging.info("Running ABX on {} and {}".format(feature_dir, triplet_file))
    logging.info("Converting features {}".format(feature_dir))
    convert(feature_dir, feature_dir / "features", load=_load_features_2019)
    logging.info("Computing distances")
    ABXpy.distances.distances.compute_distances(
            feature_dir / "features",
            'features',
            triplet_file,
            feature_dir / "distance",
            distance,
            normalized=True,
            n_cpu=16)
    logging.info("Computing scores")
    score.score(triplet_file,  feature_dir / "distance", feature_dir / "score")
    analyze.analyze(triplet_file, feature_dir / "score", feature_dir / "analyze")
    data = pd.read_csv(feature_dir / "analyze", delimiter='\t')
    return data


def compute_result(encoded_dir, triplets_fpath, output_dir, within_speaker=False, rep=False, distance=ed):
    base, _ = os.path.splitext(os.path.split(triplets_fpath)[-1])
    result = run_abx(encoded_dir, triplets_fpath, distance)
    result.to_csv("{}/{}_analyze.csv".format(output_dir, base),
                  sep='\t', header=True, index=False)
    if within_speaker:
        avg_error = _average("{}/{}_analyze.csv".format(output_dir, base),
                             "within")
    else:
        avg_error = _average("{}/{}_analyze.csv".format(output_dir, base),
                             "across")
    return avg_error


def abx(k=1000, within_speaker=False):
    from platalea.vq_encode import encode
    from vq_eval import experiments
    shutil.rmtree("data/flickr8k_abx_wav/", ignore_errors=True)
    prepare_abx(k=k, within_speaker=within_speaker)
    result = "abx_within_flickr8k_result.json" if within_speaker else "abx_flickr8k_result.json"
    for modeldir in experiments(result):
        result = [json.loads(line) for line in open(modeldir + "result.json")]
        best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
        oldnet = torch.load("{}/net.{}.pt".format(modeldir, best))
        logging.info("Loading model from {} at epoch {}".format(modeldir, best))
        net = M.SpeechImage(oldnet.config)
        net.load_state_dict(oldnet.state_dict())
        net.cuda()
        encoded_dir = Path(f"{modeldir}/encoded/flickr8k_val/")
        shutil.rmtree(encoded_dir, ignore_errors=True)
        encoded_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Encoding data")
        encode(net, "data/flickr8k_abx_wav/", encoded_dir)
        logging.info("Computing ABX")
        triplets = ("{}/flickr8k_abx_rep_within.triplets" if within_speaker else "{}/flickr8k_abx_rep.triplets").format(modeldir)
        avg_error = compute_result(encoded_dir, triplets, modeldir, within_speaker=within_speaker)
        logging.info("Score: {}".format(avg_error))
