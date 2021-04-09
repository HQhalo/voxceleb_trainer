from __future__ import absolute_import
from __future__ import print_function
import os
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

# ===========================================
#        Parse the argument
# ===========================================


parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files');

## Training details
parser.add_argument('--trainfunc',      type=str,   default="softmaxproto",     help='Loss function');

## Optimizer

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="/app/baseline_v2_ap.model",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and test data

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--model',          type=str,   default="ResNetSE34V2",     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');




parser.add_argument('--test_file_path',  type=str,   default="",     help='test file');

args = parser.parse_args()



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
        re = s(data).detach().numpy()[0].tolist()
        stop = timeit.default_timer()
        
        print('Model run: ', stop - start)
        
        return json.dumps({'vector': re})
    return "please provide audio file"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6677', debug=False)
