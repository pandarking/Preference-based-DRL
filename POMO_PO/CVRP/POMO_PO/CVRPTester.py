
import torch

import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()


        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        Arc_Difference = AverageMeter()
        AD_Percent = AverageMeter()
        Route_Difference = AverageMeter()
        RD_Percent = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score, arc_difference, ad_percent, route_difference, rd_percent = self._test_one_batch(batch_size)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            Arc_Difference.update(arc_difference, batch_size)
            AD_Percent.update(ad_percent, batch_size)
            Route_Difference.update(route_difference, batch_size)
            RD_Percent.update(rd_percent, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}, arc_difference: {:.3f}, ad_percent: {:.3f}, route_difference: {:.3f}, rd_percent: {:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score, arc_difference, ad_percent, route_difference, rd_percent))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))
                self.logger.info(" Arc_Difference: {:.4f} ".format(Arc_Difference.avg))
                self.logger.info(" AD_Percent: {:.4f} ".format(AD_Percent.avg))
                self.logger.info(" Route_Difference: {:.4f} ".format(Route_Difference.avg))
                self.logger.info(" RD_Percent: {:.4f} ".format(RD_Percent.avg))

    def _test_one_batch(self, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.use_saved_problems(batch_size)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO_PO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _, encoded_last_node = self.model(state)
            # shape: (batch, pomo)
            state, reward, done, distance, arc_difference, ad_percent, route_difference, rd_percent = self.env.step(selected, encoded_last_node)

        # Return
        ###############################################
        aug_reward = distance.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value
        # print('arc difference shape:', arc_difference.shape)
        max_ad, _ = arc_difference.max(dim=-1)
        min_ad_percent, _ = ad_percent.min(dim=-1)
        min_rd, _ = route_difference.min(dim=-1)
        min_rd_percent, _ = rd_percent.min(dim=-1)
        # print('max ad shape:', max_ad.shape)
        batch_max_ad, _ = max_ad.max(dim=-1)
        batch_max_ad = -batch_max_ad
        batch_min_ad_percent, _ = min_ad_percent.min(dim=-1)
        batch_min_rd, _ = min_rd.max(dim=-1)
        batch_min_rd_percent, _ = min_rd_percent.min(dim=-1)

        return no_aug_score.item(), aug_score.item(), batch_max_ad.item(), batch_min_ad_percent.item(), batch_min_rd.item(), batch_min_rd_percent.item()
