from __future__ import absolute_import
from __future__ import print_function
import os
import yaml
import argparse
import sys
import numpy as np
from flask import Flask, request, jsonify
import json
import os 
import io
from werkzeug.utils import secure_filename
import subprocess
AUDIO_STORAGE = os.path.join("/content", "audio_storage")
if not os.path.isdir(AUDIO_STORAGE):
    os.makedirs(AUDIO_STORAGE)
import timeit
from DatasetLoader import loadWAV
from SpeakerNet import *
import wget

parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int,  default=500,    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list');
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list');
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set');
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set');
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8888", help='Port for distributed training, input as text');
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args();

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))


#
#   Load Model
#
def loadParameters(path, model):
    if not os.path.isfile(path):
        url = 'http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model'
        wget.download(url, '/app/baseline_v2_ap.model')
    self_state = model.module.state_dict()
    loaded_state = torch.load(path, map_location="cpu")
    for name, param in loaded_state.items():
        origname = name
        if name not in self_state:
            name = name.replace("module.", "")

            if name not in self_state:
                print("%s is not in the model."%origname)
                continue

        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
            continue

        self_state[name].copy_(param)

def load_model():
    s = SpeakerNetCpu(**vars(args))
    s = WrappedModel(s).cpu()
    print("load model", args.initial_model)
    loadParameters(path=args.initial_model , model= s)

    pytorch_total_params = sum(p.numel() for p in s.module.__S__.parameters())
    print('Total parameters: ',pytorch_total_params)
    return s

def loadAudio(file):
    audio = loadWAV(file, args.eval_frames, evalmode=True)
    return torch.FloatTensor(audio)



# Flask
app = Flask(__name__)
s = load_model()
@app.route("/api/predict", methods=['POST'])
def api_predict():
    """
    Required params:
        audio
    """
    audio_file_1 = request.files['audio'] # Required

    if audio_file_1:
        filename_1 = os.path.join(AUDIO_STORAGE,secure_filename(audio_file_1.filename))
        start = timeit.default_timer() 
        audio_file_1.save(filename_1) # Save audio in audio_storage, path: audio_storage/filename_1        
        out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(filename_1,filename_1), shell=True)
        if out != 0:
            raise ValueError('Conversion failed %s.'%fname)     

        data = loadAudio(filename_1)
        stop = timeit.default_timer()
        print('Load file: ', stop - start)
        
        start = timeit.default_timer() 
        re = s(data).detach().numpy().tolist()
        stop = timeit.default_timer()
        
        print('Model run: ', stop - start)
        
        return json.dumps({'vector': re})
    return "please provide audio file"

def test():
    with open('/content/drive/MyDrive/colabdrive/Thesis/devices/train.txt', 'r') as f:
        lines = f.readlines()
    result = {}
    for line in lines:
        filename_1 = line.split(" ")[-1].rstrip()
        name = line.split(" ")[0]
        if name not in result:
            result[name] = []
        try: 
            data = loadAudio(filename_1)
            re = s(data).detach().numpy().tolist()
            result[name].append(re)
        except Exception as e:
            print(e)
    import json
    with open('/content/result.json', 'w') as fp:
        json.dump(result, fp)
if __name__ == '__main__':
    # app.run(host='0.0.0.0', port='6677', debug=False)
    test()
