import lyz.methods as L
import pandas as pd
from pathlib import Path
from torch.multiprocessing import Pool
import logging
import random
import torch
import numpy as np
import json
import pickle
from math import log2
import metrics
import os
import prepare_flickr8k as F8
import platalea.basicvq as M
import shutil
import plotnine as pn


def main():
    RS = 123
    random.seed(RS)
    np.random.seed(RS)
    torch.manual_seed(RS)

    logging.getLogger().setLevel(logging.INFO)
    path_outdir = Path('data/out')

    # ana = Analyze(experiment_dir="experiments")

    # Recall
    # data = ana.recall()
    # data.to_csv(Path("data/out") / "recall.csv", header=True, index=False)

    # Prepare
    # ana.prepare()
    # ana.prepare_word()
    # ana.prepare_baseline()

    # RSA
    # data = ana.rsa()
    # data.to_csv(path_outdir / "rsa.csv", header=True, index=False)
    # data = ana.meanpool_rsa()
    # data.to_csv(path_outdir / "meanpool_rsa.csv", header=True, index=False)
    # ABX
    # data = ana.abx()
    # data.to_csv(path_outdir / "abx.csv", header=True, index=False)
    # Other metrics
    # for shortname in ['diag', 'vmeasure', 'meandur']:
    #     data = ana.apply_metric(Metric.get_metric(shortname))
    #     data.to_csv(path_outdir / f'{shortname}.csv', header=True, index=False)

    # plot_vs()
    # vel()
    # plot_meanpool()
    # plot_size_level()

    # Van Niekerk
    # ana_vn = Analyze_van_Niekerk(
    #     experiment_dir="experiments",
    #     srcroot='/home/bjrhigy/dev/bshall-zrsc/submission/flickr8k')

    # Prepare
    # ana_vn.prepare()
    # ana_vn.prepare_word()

    # RSA
    # data = ana_vn.rsa()
    # data.to_csv(path_outdir / "rsa_vn.csv", header=True, index=False)
    # ABX
    # data = ana_vn.abx()
    # data.to_csv(path_outdir / "abx_vn.csv", header=True, index=False)
    # Other metrics
    # for shortname in ['diag', 'vmeasure']:
    #     data = ana_vn.apply_metric(Metric.get_metric(shortname))
    #     data.to_csv(path_outdir / f'{shortname}_vn.csv', header=True, index=False)

    # plot_vs_vn()
    # plot_size_level_vn()

    # plot mean duration for both models in 1 fig
    # plot_meandur()
    # run_rsa_dtw(test_size=1/2)
    # plots for both models in 1 fig
    # plot_meandur()
    # plot_word_correspondence()
    plot_size_level_joined()


class Analyze:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.sizes = [2 ** n for n in range(5, 11)]
        self.levels = [1, 2, 3]
        self.number_runs = 3
        self.splitseed = random.randint(0, 1024)
        self.abx_k = 1000
        self.abx_overlap = False
        self.abx_within_speaker = True
        self.cached_datadir = Path(self.experiment_dir) / "cache"
        (self.cached_datadir / "complete").mkdir(exist_ok=True, parents=True)
        (self.cached_datadir / "trigram").mkdir(exist_ok=True, parents=True)

    def recall(self):
        """Retrieve performance scores from the experiments in
        `self.experiment_dir` and return a data frame with the results."""
        records = []
        for size in self.sizes:
            for level in self.levels:
                for run in range(self.number_runs):
                    scores = sorted(load_results(self._path(size, level, run)),
                                    key=lambda x: x['recall']['10'])[-1]
                    r = dict(size=size, level=level, run=run,
                             epoch=scores['epoch'],
                             recall1=scores['recall']['1'],
                             recall5=scores['recall']['5'],
                             recall10=scores['recall']['10'],
                             mode='trained')
                    records.append(r)
        return pd.DataFrame.from_records(records)

    def prepare(self):
        """Extract data necessary for analyses for experiments in
        `self.experiment_dir`."""
        import platalea.basicvq as VQ
        shutil.rmtree("data/flickr8k_abx_wav/", ignore_errors=True)
        F8.prepare_abx(k=self.abx_k, overlap=self.abx_overlap)
        for size in self.sizes:
            for level in self.levels:
                for run in range(self.number_runs):
                    directory = self._path(size, level, run)
                    torch.cuda.set_device(run)
                    prepare_on(directory, module=VQ)

    def prepare_word(self):
        """
        Run the word preparation step independently (normally part of prepare
        method).
        """

        from prepare_flickr8k import save_local_data_word
        for size in self.sizes:
            for level in self.levels:
                for run in range(self.number_runs):
                    directory = self._path(size, level, run)
                    save_local_data_word(directory)

    def prepare_baseline(self):
        """Extract data necessary for analyzing baseline (non-VQ) models."""
        import platalea.basic as B
        for run in [0, 1, 2]:
            directory = self._path(None, None, run)
            torch.cuda.set_device(run)
            prepare_on(directory, module=B)

    def _runner(self, worker, sizes=None, levels=None, runs=None):
        """Run the specified worker for all settings for the experiments in
        `self.experiment_dir` and return a data frame with the
        results.

        """
        if sizes is None:
            sizes = self.sizes
        if levels is None:
            levels = self.levels
        if runs is None:
            runs = range(self.number_runs)

        settings = []
        for size in sizes:
            for level in levels:
                for run in runs:
                    directory = self._path(size, level, run)
                    logging.info(f"Process started for {directory}")
                    args = (size, level, run, directory, self.cached_datadir,
                            self.splitseed)
                    settings.append(args)
        with Pool() as pool:
            results = pool.map(worker, settings, chunksize=10)
        records = [record for result in results for record in result]
        data = pd.DataFrame.from_records(records)
        return data

    def rsa(self):
        """Compute RSA scores for the experiments in `self.experiment_dir` and
        return a data frame with the results.

        """
        return self._runner(_rsa_worker)

    def meanpool_rsa(self):
        """Compute meanpool RSA scores for the experiments in `self.experiment_dir` and
        return a data frame with the results.

        """
        data_rest = self._runner(_meanpool_rsa_worker,
                                 sizes=self.sizes,
                                 levels=self.levels)
        data_rest.to_csv("data/out/mprsa_rest.csv", index=False)
        data_base = self._runner(_meanpool_rsa_worker,
                                 sizes=[None],
                                 levels=[None])
        data_base.to_csv("data/out/mprsa_base.csv", index=False)
        return pd.concat([data_base, data_rest])

    def abx(self):
        from platalea.vq_encode import encode
        records = []
        for size in self.sizes:
            for level in self.levels:
                for run in range(self.number_runs):
                    modeldir = self._path(size, level, run)
                    result = [json.loads(line) for line in open(modeldir / "result.json")]
                    best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
                    oldnet = torch.load("{}/net.{}.pt".format(modeldir, best), map_location='cuda:0')
                    logging.info(f"Loading model from {modeldir} at epoch {best}")
                    net = M.SpeechImage(oldnet.config)
                    net.load_state_dict(oldnet.state_dict())
                    net.cuda()
                    net_rand = M.SpeechImage(oldnet.config)
                    net_rand.cuda()

                    logging.info("Computing ABX rep")
                    # FIXME this assumes the global_* files are already created
                    F8.prepare_abx_rep(modeldir, k=self.abx_k,
                                       overlap=self.abx_overlap)
                    for mode, net in [('trained', net), ('random', net_rand)]:
                        for input in ['triplets', 'fragments']:
                            suffix = '' if input == 'triplets' else '_rep'
                            # on triplets
                            encoded_dir = Path(f"{modeldir}/encoded/{mode}/flickr8k_val{suffix}/")
                            if input == 'triplets':
                                shutil.rmtree(encoded_dir, ignore_errors=True)
                                encoded_dir.mkdir(parents=True, exist_ok=True)
                                logging.info(f"Encoding data ({mode})")
                                encode(net, "data/flickr8k_abx_wav/", str(encoded_dir))
                                triplets = ("data/flickr8k_abx_within.triplets"
                                            if self.abx_within_speaker else "data/flickr8k_abx.triplets")
                            else:
                                triplets = (f"{modeldir}/flickr8k_abx_rep_within.triplets"
                                            if self.abx_within_speaker else f"{modeldir}/flickr8k_abx_rep.triplets")
                            logging.info(f"Computing ABX ({mode}, {input})")
                            err = F8.compute_result(
                                encoded_dir, triplets, modeldir,
                                within_speaker=self.abx_within_speaker)
                            records.append(dict(
                                size=size, level=level, run=run, mode=mode,
                                layer='codebook', input=input,
                                within=self.abx_within_speaker,
                                abx=1-err/100))
        data = pd.DataFrame.from_records(records)
        return data

    def apply_metric(self, metric):
        """
        Apply a metric (type Metric) to the experiments in
        `self.experiment_dir` and return a data frame with the results.
        """
        return self._runner(metric.worker)

    def _path(self, size, level, run):
        return Path(self.experiment_dir) / f'vq-{size}-q{level}-r{run}'


class Analyze_van_Niekerk(Analyze):
    def __init__(self, experiment_dir, srcroot):
        super(Analyze_van_Niekerk, self).__init__(experiment_dir)
        self.levels = [1]
        self.srcroot = srcroot

    def prepare(self):
        """Extract data necessary for analyses for experiments in
        `self.experiment_dir`."""
        for size in self.sizes:
            for run in range(self.number_runs):
                srcdir = self._srcpath(size, run)
                srcdir_trigrams = self._srcpath(size, run, trigrams=True)
                outdir = self._path(size, 1, run)
                prepare_van_niekerk_on(srcdir, srcdir_trigrams, outdir)

    def prepare_word(self):
        from prepare_flickr8k import save_local_data_word
        for size in self.sizes:
            for run in range(self.number_runs):
                directory = self._path(size, 1, run)
                save_local_data_word(directory)

    def rsa(self):
        """Compute RSA scores for the experiments in `self.experiment_dir` and
        return a data frame with the results.

        """
        return self._runner(_rsa_worker)

    def abx(self):
        records = []
        if self.abx_within_speaker:
            triplets = "data/flickr8k_abx_within.triplets"
        else:
            triplets = "data/flickr8k_abx.triplets"
        for size in self.sizes:
            for run in range(self.number_runs):
                modeldir = self._path(size, 1, run)
                srcroot = self._srcpath(size, run, True)
                logging.info(f'Computing ABX for {srcroot}')
                for mode in ['trained', 'random']:
                    encdir = srcroot / mode / 'encodings'
                    for fname in ['features', 'distance', 'score', 'analyze']:
                        if os.path.isfile(encdir / fname):
                            os.remove(encdir / fname)
                    err = F8.compute_result(
                        encdir, triplets, modeldir,
                        within_speaker=self.abx_within_speaker)
                    records.append(dict(
                        size=size,
                        level=1,
                        run=run,
                        mode=mode,
                        layer='codebook',
                        within=self.abx_within_speaker,
                        abx=1-err/100))
        data = pd.DataFrame.from_records(records)
        return data

    def _path(self, size, level, run):
        return Path(self.experiment_dir) / f'vn-{size}-r{run}'

    def _srcpath(self, size, run, trigrams=False):
        exp = 'english_triplets' if trigrams else 'english'
        return Path(self.srcroot) / f'r{run}' / exp / 'val' / str(size) / \
            'indices'


def load_results(d, fname='result.json'):
    return [json.loads(line) for line in open(Path(d) / fname)]


def load_best_net(modeldir, module):
    result = [json.loads(line) for line in open(Path(modeldir) / "result.json")]
    best = sorted(result, key=lambda x: x['recall']['10'], reverse=True)[0]['epoch']
    oldnet = torch.load(f"{modeldir}/net.{best}.pt")
    logging.info(f"Loading model from {modeldir} at epoch {best}")
    net = module.SpeechImage(oldnet.config)
    net.load_state_dict(oldnet.state_dict())
    return net.cuda()


def prepare_on(modeldir, module):
    from prepare_flickr8k import save_data, make_factors
    net = load_best_net(modeldir, module)
    json.dump(make_factors(net), open(f"{modeldir}/downsampling_factors.json", "w"))
    net_rand = module.SpeechImage(net.config)
    net_rand.cuda()
    logging.info(f"Preparing trigram data on {modeldir}/trigrams")
    save_data([('trained', net), ('random', net_rand)],
              f"{modeldir}/trigrams", batch_size=8,
              alignment_fpath='data/flickr8k_trigrams_fa.json',
              use_precomputed_mfcc=False)
    logging.info(f"Preparing whole utterance data on {modeldir}")
    save_data([('trained', net), ('random', net_rand)], modeldir, batch_size=8)


def prepare_van_niekerk_on(srcdir, srcdir_trigrams, outdir):
    from prepare_van_niekerk import save_data
    logging.info(f'Processing files for {srcdir}')
    os.makedirs(outdir, exist_ok=True)
    shutil.copyfile(Path('data') / 'downsampling_factors_van_niekerk.json',
                    outdir / 'downsampling_factors.json')
    save_data(srcdir, outdir)
    save_data(srcdir_trigrams, outdir / 'trigrams',
              alignment_fpath='data/flickr8k_trigrams_fa.json',
              use_precomputed_mfcc=False)


def _rsa_worker(args):
    size, level, run, directory, cached_datadir, splitseed = args
    torch.cuda.set_device(run)
    logging.info("Process on run {}".format(run))
    results = []
    results += L.ed_rsa(
        directory,
        layers=['codebook'],
        cached_datadir=cached_datadir / "complete",
        info={'size': size, 'level': level, 'run': run, 'input': 'complete'},
        splitseed=splitseed)

    results += L.ed_rsa(
        f"{directory}/trigrams",
        cached_datadir=cached_datadir / "trigram",
        layers=['codebook'],
        info={'size': size, 'level': level, 'run': run, 'input': 'trigram'},
        splitseed=splitseed)
    return results


def _meanpool_rsa_worker(args):
    """Run meanpool RSA on baselines and VQ runs."""
    from lyz.methods import weighted_average_RSA
    size, level, run, directory, cached_datadir, splitseed = args
    if level is None:
        layers = ['rnn0', 'rnn1', 'rnn2', 'rnn3']
    else:
        layers = [f'rnn_bottom{i}' for i in range(0, level)] + \
                 [f'rnn_top{i}' for i in range(0, 4-level)]
    results = weighted_average_RSA(directory,
                                   layers=layers,
                                   attention='mean',
                                   test_size=1/2,
                                   epochs=60,
                                   device='cpu',
                                   cached_datadir=cached_datadir,
                                   splitseed=splitseed)
    # Add standard info
    for result in results:
        result['size'] = size
        result['level'] = level
        result['run'] = run
        result['reference'] = 'phoneme'
        result['input'] = 'complete'
        result['rsa'] = result['cor']
        del result['cor']
        result['pooling'] = 'mean'
    return results


def remove_sil(X, y):
    unsil = y != 'sil'
    return X[unsil], y[unsil]


def diagnostic(X, y, args):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    splitseed = args[5]
    if splitseed is not None:
        splitseed = random.randint(0, 1024)
    X, y = remove_sil(X, y)
    X, X_val, y, y_val = train_test_split(X, y, test_size=1/2,
                                          random_state=splitseed)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)
    C = [10**n for n in range(-3, 2)]
    scores = []
    for Ci in C:
        m = LogisticRegression(C=Ci)
        m.fit(X, y)
        scores.append((m.score(X_val, y_val), Ci))
    result = sorted(scores)[-1]
    return dict(diag=result[0], C=result[1])


def rsa_dtw(directory, test_size=1/2, splitseed=None, info={}):
    import pickle
    from sklearn.model_selection import train_test_split
    import ursa.util as U
    import ursa.similarity as S
    result = []
    for input in ['complete', 'trigrams']:
        logging.info("Loading transcription data")
        if input == 'complete':
            data_in = pickle.load(open(f"{directory}/global_input.pkl", "rb"))
            cached_datadir = "experiments/cache"
        elif input == 'trigrams':
            data_in = pickle.load(open(f"{directory}/trigrams/global_input.pkl", "rb"))
            cached_datadir = "experiments/cache/trigram"
        trans = data_in['ipa']
        aid = data_in['audio_id']

        try:
            trans_sim_full = torch.load(Path(cached_datadir) / "trans_sim.pt")
        except FileNotFoundError:
            logging.info("Computing phoneme edit distances for transcriptions")
            trans_sim_full = torch.tensor(U.pairwise(S.stringsim, trans)).double().to(device)
            torch.save(trans_sim_full, Path(cached_datadir) / "trans_sim.pt")
        if splitseed is None:
            splitseed = random.randint(0, 1024)
        logging.info("Split seed: {}".format(splitseed))
        _, index = train_test_split(range(len(aid)), test_size=test_size, random_state=splitseed)
        aid = aid[index]
        trans_sim = trans_sim_full[index, :][:, index]

        net = load_best_net(directory, module=M)
        net_rand = M.SpeechImage(net.config)
        for mode in ['random', 'trained']:
            net = net if mode == 'trained' else net_rand
            if input == 'complete':
                data = pickle.load(open(f"{directory}/global_{mode}_codebook.pkl", "rb"))['codebook']
            elif input == 'trigrams':
                data = pickle.load(open(f"{directory}/trigrams/global_{mode}_codebook.pkl", "rb"))['codebook']
            codebook = net.SpeechEncoder.Codebook.embedding.cpu().numpy()
            _, data = train_test_split(data, test_size=test_size, random_state=splitseed)
            embedding = [datum @ codebook for datum in data]
            emb_sim = 1-torch.tensor(U.pairwise(dtw, embedding)).double().cpu()
            torch.save(emb_sim, f"{directory}{'/trigrams/' if input=='trigrams' else ''}/emb_sim_{mode}.{test_size}.pt")
            cor = S.pearson_r(S.triu(trans_sim), S.triu(emb_sim))
            result.append({'rsa': cor.item(),
                           'mode': mode,
                           'layer': 'embedding',
                           'reference': 'phoneme',
                           'input': input, **info})
            logging.info(result[-1])
    return result


def dtw(u, v):
    import ABXpy.distance as D
    return D.default_distance(u, v, normalized=True)


def run_rsa_dtw(test_size=1/2):
    splitseed = random.randint(0, 1024)
    records = []
    #for model in ['vq', 'vn']:
    for model in ['vq']:
        for size in [32, 1024]:
            for run in [0, 1, 2]:
                directory = f"experiments/{model}-{size}-{'q1-' if model == 'vq' else ''}r{run}"
                logging.info(f"Doing {directory}")
                info = dict(run=run, model=model, size=size)
                result = rsa_dtw(directory, test_size=test_size, splitseed=splitseed, info=info)
                records += result
    data = pd.DataFrame.from_records(records)
    data.to_csv(f"data/out/rsa_dtw.{test_size}.csv", index=False, header=True)


class Metric:
    @staticmethod
    def get_metric(shortname):
        if shortname == 'vmeasure':
            return Metric(vmeasure, 'V-measure', shortname)
        elif shortname == 'coverage':
            return Metric(coverage, 'coverage', shortname)
        elif shortname == 'condent':
            return Metric(conditional_entropy, 'conditional entropy',
                          shortname)
        elif shortname == 'diag':
            return Metric(diagnostic, 'diagnostic classifier', shortname)
        elif shortname == 'meandur':
            return Metric(mean_duration, 'mean duration', shortname)
        else:
            raise ValueError('Metric {} is unknown'.format(shortname))

    def __init__(self, fn, name, shortname):
        self._name = name
        self._shortname = shortname
        self._fn = fn

    def worker(self, args):
        size, level, run, directory, cached_datadir, splitseed = args
        results = []
        logging.info("Computing {} on {}".format(self._name, directory))
        for layer in ['codebook']:
            for mode in ['trained', 'random']:
                for ref in ['phoneme', 'word']:
                    suffix = '' if ref == 'phoneme' else '_w'
                    fpath = f'{directory}/local_{mode}_{layer}{suffix}.pkl'
                    data = pickle.load(open(fpath, 'rb'))
                    result = self._fn(data[layer]['features'],
                                      data[layer]['labels'],
                                      args)
                    result['size'] = size
                    result['level'] = level
                    result['run'] = run
                    result['layer'] = layer
                    result['mode'] = mode
                    result['reference'] = ref
                    logging.info("Result: {}".format(result))
                    results.append(result)
        return results


def indices(onehot):
    return [oh.nonzero()[0][0] for oh in onehot]


def vmeasure(X, y, args):
    from sklearn.metrics import v_measure_score
    from sklearn.model_selection import train_test_split
    splitseed = args[5]
    if splitseed is not None:
        splitseed = random.randint(0, 1024)
    X, X_val, y, y_val = train_test_split(X, y, test_size=1/2,
                                          random_state=splitseed)
    return dict(vmeasure=v_measure_score(y, indices(X)))


def coverage(codes, labels, args):
    '''
    Computes the coverage of label segments by the most frequent co-occuring
    code.
    '''
    ind = indices(codes)
    return dict(cov_top_code=np.mean(metrics.coverage_top_1(labels, ind)),
                cov_top_label=np.mean(metrics.coverage_top_1(ind, labels)))


def conditional_entropy(codes, labels, args):
    size = args[0]
    ind = indices(codes)
    values, _ = np.unique(ind, return_counts=True)
    num_uniq_ind = len(values)
    cond_entropy_ind = metrics.conditional_entropy(ind, labels)
    cond_entropy_lab = metrics.conditional_entropy(labels, ind)
    return dict(
        cond_ent_code=cond_entropy_ind,
        cond_ent_label=cond_entropy_lab,
        norm_cond_ent_code=cond_entropy_ind / log2(size),
        norm_cond_ent_code2=cond_entropy_ind / log2(num_uniq_ind),
        num_uniq_ind=num_uniq_ind,
        norm_cond_ent_label=cond_entropy_lab / log2(40))


def mean_duration(codes, labels, args):
    return dict(
        mean_duration=np.mean(metrics.count_repetitions(indices(codes))),
        mean_duration_labels=np.mean(metrics.count_repetitions(labels)))


def plot_meandur():
    meandur_vq = load_data(source='vg', metrics=['meandur'])
    meandur_vn = load_data(source='vn', metrics=['meandur'])
    meandur_vq["model"] = "Visually grounded"
    meandur_vn["model"] = "Self-supervised"
    meandur_vn["level"] = 0

    meandur = pd.concat([meandur_vq, meandur_vn])
    base_p = meandur.query('reference=="phoneme"')['mean_duration_labels'].iloc[0]
    base_w = meandur.query('reference=="word"')['mean_duration_labels'].iloc[0]
    base = pd.DataFrame.from_records([
        dict(baseline='Phoneme', mean_duration=base_p),
        dict(baseline='Word', mean_duration=base_w)])
    meandur = meandur.query('reference=="phoneme" & mode=="trained"')

    colors = {0: 'grey', 1: '#db5f57', 2:'#57db5f', 3: '#736cdd'}
    labels = ["Self-supervised", "VS - VQ at level 1", "VS - VQ at level 2", "VS - VQ at level 3"]
    shapes = {0: 'o', 1: '^', 2:'s', 3: 'D'}
    g = pn.ggplot(meandur, pn.aes(x='size', color='factor(level)',
                                  shape='factor(level)', y='mean_duration')) + \
        pn. geom_point() + \
        pn.scale_x_continuous(trans="log2", breaks=[32, 64, 128, 256, 512, 1024]) + \
        pn.scale_color_manual(colors, labels=labels) + \
        pn.scale_shape_manual(values=shapes, labels=labels) + \
        pn.geom_point(alpha=0.7, size=3) + \
        pn.geom_hline(base, pn.aes(yintercept='mean_duration',
                                   linetype='baseline')) + \
        pn.geom_smooth() + \
        pn.labs(x="Codebook size",
                y=get_axislabel('mean_duration'),
                title=get_title('mean_duration'),
                shape='Model',
                linetype="Baseline") + \
        pn.theme(text=pn.element_text(size=16, family='serif')) +\
        pn.guides(color=pn.guide_legend(title='Model'))
    pn.ggsave(g, 'fig/mean_duration_size.pdf')


def plot_vs():
    data = massage_data()

    for x, y in [('diag', 'rsa'), ('abx', 'rsa'), ('vmeasure', 'rsa'),
                 ('vmeasure', 'diag')]:
        g = pn.ggplot(data, pn.aes(x=x, y=y,  shape='factor(level)',
                                   color='factor(size)')) + \
            pn.geom_point() + \
            pn.scale_shape_discrete(name="level") + \
            pn.scale_color_discrete(name="size") + \
            pn.labs(x=get_axislabel(x),
                    y=get_axislabel(y),
                    title=f'{get_title(x)} vs. {get_title(y)}',
                    shape='VQ at level') + \
            pn.guides(color=pn.guide_legend(title='Codebook size')) + \
            pn.theme(text=pn.element_text(size=16, family='serif'))
        pn.ggsave(g, "fig/{}-{}.pdf".format(x, y))

    # phoneme vs word
    rsa = pd.read_csv('data/out/rsa.csv')
    rsa = rsa.query('input=="complete"')
    plot_phoneme_vs_word(rsa, 'rsa')
    plot_phoneme_vs_word(pd.read_csv('data/out/vmeasure.csv'), 'vmeasure')

    # abx vs rsa trigram
    data = pd.read_csv("data/out/rsa.csv").query("reference=='phoneme' & input=='trigram'")
    abx = pd.read_csv("data/out/abx.csv").query("input=='triplets'")
    del abx['input']
    abx['reference'] = 'phoneme'
    data = data.merge(abx)
    data = adjust_for_random(data, ['rsa', 'abx'])
    g = pn.ggplot(data, pn.aes(x='abx', y='rsa', shape='factor(level)',
                               color='factor(size)')) + \
        pn.geom_point() + \
        pn.scale_shape_discrete(name="level") + \
        pn.scale_color_discrete(name="size")
    pn.ggsave(g, "fig/abx-rsa3.pdf")


def plot_phoneme_vs_word(data, colname):
    phon = data.query("reference=='phoneme'")
    word = data.query("reference=='word'")
    ref = phon.merge(word, on=['mode', 'layer', 'size', 'level', 'run'],
                     suffixes=['_phoneme', '_word'])
    ref['reference'] = None
    ref = adjust_for_random(ref, [f'{colname}_phoneme', f'{colname}_word'])
    g = pn.ggplot(ref, pn.aes(x=f'{colname}_phoneme', y=f'{colname}_word',
                              shape='factor(level)', color='factor(size)')) + \
        pn.geom_point() + \
        pn.scale_shape_discrete(name="level") + \
        pn.scale_color_discrete(name="size")

    pn.ggsave(g, f'fig/{colname}_phoneme_word.pdf')


def plot_size_level_joined():
    # phoneme metrics
    data_vg = load_data()
    data_vn = load_data(source="vn")
    data_vg["model"] = "Visually grounded"
    data_vn["model"] = "Self-supervised"
    data_vn["level"] = 0
    data_d = pd.concat([data_vg, data_vn])

    colors = {0: 'grey', 1: '#db5f57', 2: '#57db5f', 3: '#736cdd'}
    labels = ["Self-supervised", "VS - VQ at level 1", "VS - VQ at level 2",
              "VS - VQ at level 3"]
    shapes = {0: 'o', 1: '^', 2:'s', 3: 'D'}
    fill_values = {'trained': 'black', 'random': 'white'}
    for metric in ['diag', 'rsa', 'abx', 'vmeasure']:
        g = pn.ggplot(data_d, pn.aes(x='size',
                                     color='factor(level)',
                                     shape='factor(level)',
                                     linetype='mode',
                                     y=metric)) + \
            pn.scale_x_continuous(trans="log2", breaks=[32, 64, 128, 256, 512, 1024]) + \
            pn.scale_color_manual(colors, labels=labels) + \
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
        pn.ggsave(g, f"fig/joined_{metric}_size.pdf")


def plot_size_level():
    # recall vs size
    recall = pd.read_csv("data/out/recall.csv")
    base = pd.DataFrame.from_records([
        dict(baseline='No VQ layer', recall10=0.416)])
    shapes = {1: '^', 2:'s', 3: 'D'}
    g = pn.ggplot(recall, pn.aes(x='size', color='factor(level)', shape='factor(level)',
                                 y='recall10')) + \
        pn. geom_point() + \
        pn.scale_x_continuous(trans="log2", breaks=[32, 64, 128, 256, 512, 1024]) + \
        pn.scale_color_discrete(name="level") + \
        pn.scale_shape_manual(values=shapes) + \
        pn.geom_point(alpha=0.7, size=3) + \
        pn.geom_hline(base, pn.aes(yintercept='recall10',
                                   linetype='baseline')) + \
        pn.geom_smooth() +\
        pn.labs(x="Codebook size",
                y=get_axislabel('recall10'),
                title=get_title('recall10'),
                linetype="Baseline") + \
        pn.guides(color=pn.guide_legend(title='VQ layer at level')) + \
        pn.guides(shape=pn.guide_legend(title='VQ layer at level')) + \
        pn.theme(text=pn.element_text(size=16, family='serif'))
    pn.ggsave(g, "fig/recall_size.pdf")
    # mean duration vs size
    meandur = pd.read_csv("data/out/meandur.csv")
    base_p = meandur.query('reference=="phoneme"')['mean_duration_labels'].iloc[0]
    base_w = meandur.query('reference=="word"')['mean_duration_labels'].iloc[0]
    base = pd.DataFrame.from_records([
        dict(baseline='Phoneme', mean_duration=base_p),
        dict(baseline='Word', mean_duration=base_w)])
    meandur = meandur.query('reference=="phoneme" & mode=="trained"')
    g = pn.ggplot(meandur, pn.aes(x='size', color='factor(level)',
                                  y='mean_duration')) + \
        pn. geom_point() + \
        pn.scale_x_continuous(trans="log2", breaks=[32, 64, 128, 256, 512, 1024]) + \
        pn.scale_color_discrete(name="level") + \
        pn.geom_point(alpha=0.7, size=3) + \
        pn.geom_hline(base, pn.aes(yintercept='mean_duration',
                                   linetype='baseline')) + \
        pn.geom_smooth() + \
        pn.labs(x="Codebook size",
                y=get_axislabel('mean_duration'),
                title=get_title('mean_duration'),
                linetype="Baseline") + \
        pn.theme(text=pn.element_text(size=16, family='serif')) +\
        pn.guides(color=pn.guide_legend(title='VQ at level'))
    pn.ggsave(g, 'fig/mean_duration_size.pdf')
    # other metrics
    data_d = load_data()
    for metric in ['diag', 'rsa', 'abx', 'vmeasure']:
        g = plot_metric_vs_size(data_d, metric)
        pn.ggsave(g, f"fig/{metric}_size.pdf")
    # word-level
    metrics = ['rsa', 'vmeasure']
    data_d = load_data(ref='word', metrics=metrics)
    for metric in metrics:
        g = plot_metric_vs_size(data_d, metric)
        pn.ggsave(g, f"fig/{metric}_w_size.pdf")


def plot_vs_vn():
    abx = pd.read_csv("data/out/abx_vn.csv")
    abx['reference'] = 'phoneme'
    rsa = pd.read_csv("data/out/rsa_vn.csv")
    data = rsa.query("reference=='phoneme' & input=='complete'").merge(abx)
    data = adjust_for_random(data, ['rsa', 'abx'])
    data3 = rsa.query("reference=='phoneme' & input=='trigram'").merge(abx)
    data3 = adjust_for_random(data3, ['rsa', 'abx'])
    g = pn.ggplot(data, pn.aes(x='abx', y='rsa', shape='factor(level)',
                               color='factor(size)')) + \
        pn.geom_point() + \
        pn.scale_shape_discrete(name="level") + \
        pn.scale_color_discrete(name="size")

    pn.ggsave(g, "fig/abx_rsa_vn.pdf")
    g = pn.ggplot(data3, pn.aes(x='abx', y='rsa', shape='factor(level)',
                                color='factor(size)')) + \
        pn.geom_point() + \
        pn.scale_shape_discrete(name="level") + \
        pn.scale_color_discrete(name="size")
    pn.ggsave(g, "fig/abx_rsa3_vn.pdf")


def plot_meanpool():
    data = pd.read_csv("data/out/meanpool_rsa.csv")
    adj = adjust_for_random(data, ['rsa'])
    baseline = adj.query("size!=size & layer != 'mfcc' & layer != 'rnn0'")
    means = baseline.groupby('layer').mean().reset_index()
    means['level'] = means['layer'].map(lambda x: int(x[-1]))
    g = pn.ggplot(adj.query('size==size & layer=="rnn_top0"'), pn.aes(x='size', y='rsa', color='factor(level)')) + \
        pn.geom_point() + \
        pn.geom_smooth() + \
        pn.scale_x_continuous(trans='log2', breaks=[32, 64, 128, 256, 512, 1024]) + \
        pn.geom_hline(data=means, mapping=pn.aes(yintercept='rsa', color='factor(level)'), linetype='dashed') + \
        pn.labs(x="Codebook size",
                y=get_axislabel('rsa'),
                title=get_title('rsa'),
                shape="Mode") + \
        pn.guides(color=pn.guide_legend(title='VQ at level')) + \
        pn.theme(text=pn.element_text(size=16, family='serif'))
    pn.ggsave(g, "fig/meanpool.pdf")


def plot_size_level_vn():
    # mean duration vs size
    meandur = pd.read_csv("data/out/meandur_vn.csv")
    base_p = meandur.query('reference=="phoneme"')['mean_duration_labels'].iloc[0]
    base_w = meandur.query('reference=="word"')['mean_duration_labels'].iloc[0]
    base = pd.DataFrame.from_records([
        dict(baseline='Phoneme', mean_duration=base_p),
        dict(baseline='Word', mean_duration=base_w)])
    meandur = meandur.query('reference=="phoneme" & mode=="trained"')
    g = pn.ggplot(meandur, pn.aes(x='size',
                                  color='factor(level)',
                                  y='mean_duration')) + \
        pn. geom_point() + \
        pn.scale_x_continuous(trans="log2", breaks=[32, 64, 128, 256, 512, 1024]) + \
        pn.scale_color_discrete(name="level", guide=False) + \
        pn.geom_point(alpha=0.7, size=3) + \
        pn.geom_hline(base, pn.aes(yintercept='mean_duration',
                                   linetype='baseline')) + \
        pn.geom_smooth() + \
        pn.labs(x="Codebook size",
                y=get_axislabel('mean_duration'),
                title=get_title('mean_duration'),
                linetype="Baseline") + \
        pn.theme(text=pn.element_text(size=16, family='serif'))
    pn.ggsave(g, 'fig/mean_duration_size_vn.pdf')
    # other metrics
    data_d = load_data(source='vn')
    for metric in ['diag', 'rsa', 'abx', 'vmeasure']:
        g = plot_metric_vs_size(data_d, metric)
        pn.ggsave(g, f"fig/{metric}_size_vn.pdf")
    # word-level
    metrics = ['rsa', 'vmeasure']
    data_d = load_data(source='vn', ref='word', metrics=metrics)
    for metric in metrics:
        g = plot_metric_vs_size(data_d, metric)
        pn.ggsave(g, f"fig/{metric}_w_size_vn.pdf")


def plot_word_correspondence():
    metrics = ['rsa', 'vmeasure']
    data_vg = load_data(source='vg', ref='word', metrics=metrics)
    data_vn = load_data(source='vn', ref='word', metrics=metrics)
    data_vg["model"] = "Visually grounded"
    data_vn["model"] = "Self-supervised"
    data_vn["level"] = 0
    data = pd.concat([data_vg, data_vn])
    for metric in metrics:
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
            pn.scale_shape_manual(values=shapes, labels=labels) + \
            pn.geom_smooth() + \
            pn.geom_point(inherit_aes=True, mapping=pn.aes(fill='mode'),size=3, alpha=.7) + \
            pn.scale_fill_manual(values=fill_values) + \
            pn.labs(x="Codebook size",
                    y=get_axislabel(metric),
                    title=get_title(metric),
                    shape="Model",
                    fill="Mode",
                    linetype="Mode") + \
            pn.theme(text=pn.element_text(size=16, family='serif')) + \
            pn.guides(color=pn.guide_legend(title='Model'))
        pn.ggsave(g, f"fig/{metric}_word_correspondence.pdf")


def get_axislabel(metric):
    labels = {'rsa': "r",
              'vmeasure': 'NMI',
              'abx': 'Accuracy',
              'diag': 'Accuracy',
              'mean_duration': 'Timesteps',
              'recall10': 'Recall@10'}
    if metric in labels:
        return labels[metric]
    else:
        return metric


def get_title(metric):
    labels = {'rsa': "RSA",
              'vmeasure': 'NMI',
              'abx': 'ABX',
              'diag': 'DC',
              'mean_duration': 'Mean duration',
              'recall10': 'Image retrieval'}
    if metric in labels:
        return labels[metric]
    else:
        return metric


def plot_metric_vs_size(data, metric):
    # hide legend for color when we have only one level
    guide_color = data.level.unique().shape[0] > 1
    g = pn.ggplot(data, pn.aes(x='size',
                               color='factor(level)',
                               shape='mode',
                               linetype='mode',
                               y=metric)) + \
        pn.scale_x_continuous(trans="log2", breaks=[32, 64, 128, 256, 512, 1024]) + \
        pn.scale_color_discrete(name="level", guide=guide_color) + \
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


def metric_correlation():
    import ursa.util as U
    records = []
    # Visually-supervised
    abx = pd.read_csv("data/out/abx.csv").query("input=='triplets'")
    del abx['input']
    abx['reference'] = 'phoneme'
    data = pd.read_csv("data/out/rsa.csv").query("reference=='phoneme' & input=='trigram'")
    data = data.merge(abx).query("mode == 'trained'")
    records.append({'Model': 'VS', 'RSA input': "triplet", 'ABX input': 'triplet', 'r': U.pearson_r(data['rsa'], data['abx'])})
    data = pd.read_csv("data/out/rsa.csv").query("reference=='phoneme' & input=='complete'")
    data = data.merge(abx).query("mode == 'trained'")
    records.append({'Model': 'VS', 'RSA input': "complete", 'ABX input': 'triplet', 'r': U.pearson_r(data['rsa'], data['abx'])})
    abx = pd.read_csv("data/out/abx.csv").query("input=='fragments'")
    del abx['input']
    abx['reference'] = 'phoneme'
    data = pd.read_csv("data/out/rsa.csv").query("reference=='phoneme' & input=='complete'")
    data = data.merge(abx).query("mode == 'trained'")
    records.append({'Model': 'VS', 'RSA input': "complete", 'ABX input': 'fragments', 'r': U.pearson_r(data['rsa'], data['abx'])})
    # Self-supervised
    abx = pd.read_csv("data/out/abx_vn.csv")
    abx['reference'] = 'phoneme'
    data = pd.read_csv("data/out/rsa_vn.csv").query("reference=='phoneme' & input=='trigram'")
    data = data.merge(abx).query("mode == 'trained'")
    records.append({'Model': 'SS', 'RSA input': "triplet", 'ABX input': 'triplet', 'r': U.pearson_r(data['rsa'], data['abx'])})
    data = pd.read_csv("data/out/rsa_vn.csv").query("reference=='phoneme' & input=='complete'")
    data = data.merge(abx).query("mode == 'trained'")
    records.append({'Model': 'SS', 'RSA input': "complete", 'ABX input': 'triplet', 'r': U.pearson_r(data['rsa'], data['abx'])})

    out = pd.DataFrame.from_records(records)
    out.to_csv("data/out/metric_correlation.csv", index=False, header=True)
    out.to_latex(buf="data/out/metric_correlation.tex", float_format="%.2f", index=False)


def sim_var(filename='codes_sim_trained_codebook.pt', models=['vq', 'vn'], sizes=[32, 1024]):
    """Check variance of similarity scores."""
    import ursa.similarity as S
    from scipy.stats import skew, kurtosis
    records = []
    for input in ["trigrams", "complete"]:

        for arch in models:
            for size in sizes:
                sim = torch.load(f"experiments/{arch}-{size}{'' if arch == 'vn' else '-q1'}-r0/{'trigrams' if input=='trigrams' else ''}/{filename}")
                sim = S.triu(sim)

                records.append(dict(model=arch,
                                    input=input,
                                    size=size,
                                    prop_zero=(sim == 0).float().mean().item(),
                                    std=sim.std().item(),
                                    skew=skew(sim.numpy()),
                                    kurtosis=kurtosis(sim.numpy())))
    return pd.DataFrame.from_records(records)


def run_sim_var():
    data = sim_var(filename='codes_sim_trained_codebook.pt')
    data.to_csv("data/out/sim_var.csv", index=False, header=True)


def merge_frames(metrics, ref='phoneme'):
    data = pd.read_csv(f'data/out/{metrics[0]}.csv')
    for m in metrics[1:]:
        raw_data = pd.read_csv(f'data/out/{m}.csv')
        if m == 'abx':
            raw_data.loc[raw_data.input == 'triplets', 'input'] = 'complete'
        if 'reference' not in raw_data:
            raw_data['reference'] = 'phoneme'
        if 'input' not in raw_data:
            raw_data['input'] = 'complete'
        data = data.merge(raw_data.query(f"reference=='{ref}' & input=='complete'"))
    return data


def massage_data(source='vg', ref='phoneme',
                 metrics=['rsa', 'diag', 'abx', 'vmeasure']):
    data = load_data(source=source, ref=ref, metrics=metrics)
    return adjust_for_random(data, metrics)


def load_data(source='vg', ref='phoneme',
              metrics=['rsa', 'diag', 'abx', 'vmeasure']):
    columns = metrics
    if source == 'vn':
        metrics = [f'{m}_vn' for m in metrics]
    data = merge_frames(metrics, ref)
    data['mode'] = pd.Categorical(values=data['mode'], categories=['trained', 'random'])
    return data


def adjust_for_random(data, columns):
    data_t = data.query("mode=='trained'")
    data_r = data.query("mode=='random'")
    data_d = data_t.merge(
        data_r,
        on=['size', 'level', 'run', 'layer', 'reference'],
        suffixes=['_t', '_r'])
    for m in columns:
        data_d[m] = data_d[f'{m}_t'] - data_d[f'{m}_r']
    return data_d


if __name__ == '__main__':
    #torch.multiprocessing.set_start_method("spawn")
    main()
