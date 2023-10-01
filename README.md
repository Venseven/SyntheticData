## TDGAN - Synthetic Tabular data with datetime generation model
---
[![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)](https://www.python.org/downloads/release/python-380/)
[![Tensorflow 2.x](https://img.shields.io/badge/Tensorflow-TF%202.X-yellowgreen)
](https://www.tensorflow.org/)


### Installing tdgan package <br>
```bash
 pip install .
```
or
### External installation
- Create a Personal Access Token : 
```bash
https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token
```
- Run
```bash 
pip install git+https://<PERSONAL ACCESS TOKEN>@github.com/saamaresearch/TGAN_tf.git@packaging
```

### Default  MODEL HYPERPARAMETERS 
Tune the parameters at `train.py` for intensive training
```python
PARAMS = {
            "max_epoch":5,
            "steps_per_epoch":10000,
            "batch_size":128,
            "z_dim":200,
            "noise":0.2,
            "l2norm":0.00001,
            "learning_rate":0.001,
            "num_gen_rnn":100,
            "num_gen_feature":100,
            "num_dis_layers":1,
            "num_dis_hidden":100
        }
```
### Default setup configuration 
`train.py` and' `inference.py` contain a configuration `CONFIG`, where a runtime specific information should be supplied.
- `NUMERICAL_COLS` and `DATE_COLS` are the names of the numerical/continuous data columns and the date data columns, respectively.
- `DATE_DELIMITER` is the datetime delimiter of date records.
- `MODELPATH` specifies the location of the trained model.
- `GPU` if available, specify the number of GPUs, Otherwise, leave it empty.

```python
CONFIG = {
          "NUMERICAL_COLS": ["Total Costs of OSHPD Projects", "Number of OSHPD Projects"],
          "DATE_COLS": ["Data Generation Date"], 
          "DATE_DELIMITER": "/", 
          "MODELPATH": "/mnt/new/research/TGAN_tf/output/model/date_model.pkl", 
          "GPU": "0,1,2,3", 
          "final_date_columns": None
```

### Colab Notebooks
Training  - [![Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jB87qu21e6sOP-SKX3Wp0wTXvy9moGel?usp=sharing)

Inference - [![Inference](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BM49ScrIXFsrlTHhXIjOuqG1324pfEox?usp=sharing)

### Training the TDGAN model:

```bash
  python train.py \
    --datapath [TRAIN DATAFILE PATH] \ 
    --data_format [TRAIN DATAFILE FORMAT]
```

### Generating synthetic data samples from  trained TDGAN:

```bash
  python inference.py \
    --num_samples [NUMBER OF SYNTHETIC SAMPLES] \ 
    --model_path [TRAINED MODEL PATH]
```

  

