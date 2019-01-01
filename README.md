# tacotron-tensorflow
A TensorFlow implementation of DeepMind's Tacotron. A deep neural network architectures described in many papers.

Especially for English, Korean.

highly inspired by [here](https://github.com/Rayhane-mamah/Tacotron-2)

## Requirements

* Python 3.x (preferred)
* Tensorflow 1.x
* matplotlib
* librosa
* numpy
* tqdm

## Usage

> 0. Download Dataset

* [IJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/)

> 0. Install ```requirements.txt``` via ```pip```

``` python -m pip install -r requirements.txt ```

> 1. Adjust ```config.py``` (path stuff)

> 2. Just execute ```train.py```

``` python train.py ```

## DataSet

|          DataSet          |     Samples    |          Size                 |
| :-----------------------: | :------------: | :---------------------------: |
|       IJSpeech-1.1        |      13100     |   about 30GB is needed        |


## Source Tree

```
│
├── assets
│    └── images       (readme images)
├── datasets
│    ├── ljspeech.py  (LJSpeech 1.1 DataSet)
│    └── ...
├── model
│    └── log data     (readme images)
├── config.py         (whole configuration)
├── dataloader.py     (data loading stuff)
├── model.py          (lots of TTS models)
├── modules.py        (lots of modules frequently used at model)
├── synthesize.py     (inference)
├── train.py          (model training)
├── utils.py          (useful utils)
└── tfutils.py        (useful TF utils)
```

## Model Architecture

### Tacotron 1
![architecture](./assets/tacotron-1.png)

### Tacotron 2
![architecture](./assets/tacotron-2.png)

### DeepVoice v2

soon!

### DeepVoice v3

![architecture](./assets/deep_voice_3.png)

## Author

HyeongChan Kim / [@kozistr](http://kozistr.tech)
