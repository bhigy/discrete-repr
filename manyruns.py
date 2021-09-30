import torch
import logging
import platalea.basicvq as M
import platalea.dataset as D
import platalea.basic as B
from platalea.experiments.config import args
import os
from pathlib import Path
from torch.multiprocessing import Process


def run(size=32, level=1, runid=0, device=0):
    torch.cuda.set_device(device)
    cwd = os.getcwd()
    modeldir = "experiments/vq-{}-q{}-r{}/".format(size, level, runid)
    Path(modeldir).mkdir(parents=True, exist_ok=True)
    os.chdir(modeldir)

    logging.basicConfig(level=logging.INFO)
    logging.info('Loading data')
    data = dict(
        train=D.flickr8k_loader(
            args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
            args.audio_features_fn, split='train', batch_size=32, shuffle=True,
            downsampling_factor=args.downsampling_factor),
        val=D.flickr8k_loader(
            args.flickr8k_root, args.flickr8k_meta, args.flickr8k_language,
            args.audio_features_fn, split='val', batch_size=32, shuffle=False))
    bidi = True
    if size is not None:
        config = dict(ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024), norm=True), margin_size=0.2)
        assert level in [1, 2, 3]
        levels = 4
        config['SpeechEncoder'] = dict(
            SpeechEncoderBottom=dict(
                conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                rnn=dict(input_size=64, hidden_size=1024, num_layers=level, bidirectional=bidi, dropout=0)),
            VQEmbedding=dict(num_codebook_embeddings=size, embedding_dim=2 * 1024 if bidi else 1024, jitter=0.12),
            SpeechEncoderTop=dict(
                rnn=dict(input_size=2 * 1024 if bidi else 1024, hidden_size=1024, num_layers=levels-level,
                         bidirectional=bidi, dropout=0),
                att=dict(in_size=2048, hidden_size=128)))

        logging.info('Building model')
        net = M.SpeechImage(config)
    else:
        config = dict(
            SpeechEncoder=dict(
                conv=dict(in_channels=39, out_channels=64, kernel_size=6, stride=2, padding=0, bias=False),
                rnn=dict(input_size=64, hidden_size=1024, num_layers=4, bidirectional=True, dropout=0),
                att=dict(in_size=2048, hidden_size=128)),
            ImageEncoder=dict(linear=dict(in_size=2048, out_size=2*1024), norm=True), margin_size=0.2)

        logging.info('Building model')
        net = B.SpeechImage(config)

    run_config = dict(max_lr=2 * 1e-4, epochs=32)

    logging.info('Training')
    M.experiment(net, data, run_config)
    os.chdir(cwd)


def basic():
    procs = []
    size = None
    level = None
    for runid in [0, 1, 2]:
        p = Process(target=run, args=(size, level, runid, runid))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def debug():
    run()


def main():
    for size in [2**n for n in range(5, 11)]:
        for level in [1, 2, 3]:
            for runid in [0, 1, 2]:
                run(size, level, runid)


def main_parallel():
    for size in [2**n for n in range(5, 11)]:
        for level in [1, 2, 3]:
            procs = []
            for runid in [0, 1, 2]:
                p = Process(target=run, args=(size, level, runid, runid))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
    basic()
