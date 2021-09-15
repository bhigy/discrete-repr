import logging
import os
import numpy as np
from pathlib import Path
import pickle
from prepare_flickr8k import align2ipa, save_local_data, \
        get_audio_and_alignments


def save_data(indir, outdir, alignment_fpath='data/datasets/flickr8k/fa.json',
              use_precomputed_mfcc=True):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_global_data(indir, outdir=outdir, alignment_fpath=alignment_fpath,
                     use_precomputed_mfcc=use_precomputed_mfcc)
    save_local_data(directory=outdir, alignment_fpath=alignment_fpath)


def save_global_data(indir, outdir, alignment_fpath,
                     use_precomputed_mfcc=True):
    """Generate data for training a phoneme decoding model."""
    audio, alignments = get_audio_and_alignments(alignment_fpath,
                                                 use_precomputed_mfcc)
    audio_np = [a.numpy() for a in audio]

    # Global input
    audio_id = np.array([datum['audio_id'] for datum in alignments])
    global_input = dict(
        audio_id=audio_id,
        ipa=np.array([align2ipa(datum) for datum in alignments]),
        text=np.array([datum['transcript'] for datum in alignments]),
        audio=np.array(audio_np))
    global_input_path = Path(outdir) / 'global_input.pkl'
    pickle.dump(global_input, open(global_input_path, 'wb'), protocol=4)

    # Global activations
    for mode in ['trained', 'random']:
        encodings = []
        for sid in audio_id:
            fname = os.path.splitext(sid)[0] + '.txt'
            fpath = Path(indir) / mode / 'encodings' / fname
            encodings.append(np.loadtxt(fpath))
        path = f'{outdir}/global_{mode}_codebook.pkl'
        logging.info(f'Saving global data in {path}')
        pickle.dump({'codebook': encodings}, open(path, 'wb'), protocol=4)
