# Data preparation

All the models are evaluated using Flickr8K validation set. Flickr8K is additionally used to train the
visually-supervised models while self-supervised models are trained on the Zerospeech 2020 dataset.

We explains here how to download and prepare Flickr8K dataset. For instructions related to Zerospeech 2020,
please see the self-supervised model's [original
instructions](https://github.com/bhigy/ZeroSpeech#data-and-preprocessing).

## Dataset download

To use Flickr8K, you need to download:
* [Flickr8K](http://hockenmaier.cs.illinois.edu/Framing_Image_Description/KCCA.html) [1].
  Note that downloading from the official website seems broken at the moment.
  Alternatively, the dataset can be obtained from
  [here](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.names).
* The [Flickr Audio Caption Corpus](https://groups.csail.mit.edu/sls/downloads/flickraudio/) [2].
* Some additional [metadata file](https://surfdrive.surf.nl/files/index.php/s/EF1bA9YYfhiBxoN).

Create a folder to store the dataset (we will assume here that the folder is
`~/corpora/flickr8k`)  and move all the files you downloaded there, then
extract the content of the archives. You can now setup the environment and
start preprocessing the data.

### Configuration

Platalea uses ConfigArgParse for setting necessary input variables, including the
location of the dataset.  This means you can use either a configuration file
(config.ini or config.yml), environment variables or command line arguments to
specify the necessary configuration parameters.

To specify the location of the dataset, one option is to create a configuration
file under your home directory (`~/.platalea/config.yml`)., with
follwing content:

```
flickr8k_root   /home/<user>/corpora/flickr8k
```

The same result can be achieved with an environment variable:

```sh
export FLICKR8K_ROOT=/home/<user>/corpora/flickr8k
```

You could also specify this option directly on the command line when running
an experiment (the respective options would be `--flickr8k_root=...`).

### Preprocessing

Run the preprocessing script to extract input features:

```bash
python platalea/utils/preprocessing.py flickr8k
```

## References

[1] Hodosh, M., Young, P., & Hockenmaier, J. (2013). Framing Image Description
as a Ranking Task: Data, Models and Evaluation Metrics. Journal of Artificial
Intelligence Research, 47, 853–899. https://doi.org/10.1613/jair.3994.

[2] Harwath, D., & Glass, J. (2015). Deep multimodal semantic embeddings for
speech and images. 2015 IEEE Workshop on Automatic Speech Recognition and
Understanding (ASRU), 237–244. https://doi.org/10.1109/ASRU.2015.7404800.
