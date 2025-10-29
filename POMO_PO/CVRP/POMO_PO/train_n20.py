##########################################################################################
# Machine Environment Config
import torch.cuda
import transformers

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTrainer import CVRPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size': 20,
    'pomo_size': 1,
    'embd_dim': 128,
    'batch_size': 50,
    'original_size': 73,
    'num_truck': 1,
    'capacity': 1,
    'pre_train': False,
    'FLAG__use_saved_problems': True,
    'load_model': {
        'enable': False,
        'path': '/result/saved_pt_model',
        'epoch': 500,  # epoch version of pre_trained model to load
    }
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [501, ],
        'gamma': 0.1
    }
}

transformer = {
    'trans_lr': 1e-4,
    'optimizer_type': 'adamw',
    'scheduler_type': 'CosineDecay',
    'vocab_size': 1,
    'n_layer': 1,
    'embd_dim': 128,
    'prob_size': 20,
    'n_embd': 128,
    'n_head': 4,
    'n_positions': 1024,
    'resid_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'pref_attn_embd_dim': 256,
    'train_type': "sum",
    'use_weighted_sum': False,
    'warmup_steps': 500,
    'total_steps': 10000,
    'lmd': 0.5,
    'tau': 0.7,
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 100,
    'train_episodes': 50 * 2000,
    'train_batch_size': 50,
    'prev_model_path': None,
    'logging': {
        'model_save_interval': 5,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp_20.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
        'log_image_params_3': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
        'log_image_params_4': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
        'log_image_params_5': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        }
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        # 'path': './result/saved_CVRP20_model',  # directory path of pre-trained model and log files saved.
        'path': './result/20241225_170241_train_cvrp_n20_with_instNorm',
        'epoch': 85,  # epoch version of pre-trained model to laod.

    }
}

logger_params = {
    'log_file': {
        'desc': 'train_cvrp_n20_with_instNorm',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()
    torch.cuda.set_device(CUDA_DEVICE_NUM)
    # config = transformers.GPT2Config(**transformer)
    # trans = TransRewardModel(config=config, activation='relu', activation_final='relu')
    # pt = PreferenceTransformer(config, trans, **env_params)

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      # pt=pt,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
