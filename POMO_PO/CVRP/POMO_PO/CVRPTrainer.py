
import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model
from prepare_data import *
# from solve_cvrp_with_cplex import *

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 # pt,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        self.FLAG__use_saved_problems = env_params['FLAG__use_saved_problems']
        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        # self.pt = pt
        self.env = Env(**self.env_params)  # self.pt,

        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, pomo_train_loss, arc_difference, ad_percent, route_difference, rd_percent = self._train_one_epoch(epoch)  # pt_train_loss, , solution_error
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('pomo_train_loss', epoch, pomo_train_loss)
            # self.result_log.append('pt_train_loss', epoch, pt_train_loss)
            self.result_log.append('arc_difference', epoch, arc_difference)
            self.result_log.append('ad_percent', epoch, ad_percent)
            self.result_log.append('route_difference', epoch, route_difference)
            self.result_log.append('rd_percent', epoch, rd_percent)
            # self.result_log.append('solution_error', epoch, solution_error)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['pomo_train_loss'])
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_3'],
                #                                self.result_log, labels=['pt_train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_4'],
                                               self.result_log, labels=['arc_difference'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_5'],
                                               self.result_log, labels=['ad_percent'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_5'],
                                               self.result_log, labels=['route_difference'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_5'],
                                               self.result_log, labels=['rd_percent'])
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_5'],
                #                                self.result_log, labels=['solution_error'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['pomo_train_loss'])
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_3'],
                #                                self.result_log, labels=['pt_train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_4'],
                                               self.result_log, labels=['arc_difference'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_5'],
                                               self.result_log, labels=['ad_percent'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_5'],
                                               self.result_log, labels=['route_difference'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_5'],
                                               self.result_log, labels=['rd_percent'])
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_5'],
                #                                self.result_log, labels=['solution_error'])

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

            torch.cuda.empty_cache()

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        # PT_Loss = AverageMeter()
        Arc_Difference = AverageMeter()
        AD_Percent = AverageMeter()
        Route_Diffenrce = AverageMeter()
        RD_Percent = AverageMeter()
        # Solution_error = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            # import time
            # print(time.time())
            avg_score, avg_loss, arc_difference, ad_percent, route_difference, rd_percent = self._train_one_batch(batch_size)  # , pt_loss, solution_error
            # print(time.time())
            # exit(0)
            # avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            # PT_Loss.update(pt_loss, batch_size)
            Arc_Difference.update(arc_difference, batch_size)
            AD_Percent.update(ad_percent, batch_size)
            Route_Diffenrce.update(route_difference, batch_size)
            RD_Percent.update(rd_percent, batch_size)
            # Solution_error.update(solution_error, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  POMO_Loss: {:.4f}, arc_difference: {:.4f}, ad_percent: {:.4f}, route_difference: {:.4f}, rd_percent: {:.4f}'  # ,  PT_Loss: {:.6f}, Solution_error: {}
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg, Arc_Difference.avg, AD_Percent.avg, Route_Diffenrce.avg, RD_Percent.avg))  # , PT_Loss.avg, Solution_error.avg

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}, arc_difference: {:.4f}, ad_percent: {:.4f}, route_difference: {:.4f}, rd_percent: {:.4f}'  # ,  PT_Loss: {:.6f}, Solution_error: {}
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg, Arc_Difference.avg, AD_Percent.avg, Route_Diffenrce.avg, RD_Percent.avg))  # , PT_Loss.avg, Solution_error.avg

        return score_AM.avg, loss_AM.avg, Arc_Difference.avg, AD_Percent.avg, Route_Diffenrce.avg, RD_Percent.avg  # , PT_Loss.avg, Solution_error.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        if self.FLAG__use_saved_problems:
            self.env.use_saved_problems(batch_size)
        else:
            self.env.load_problems(batch_size)
        rewards = torch.empty((batch_size, 0))
        prob_lists = None
        for i in range(1):
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
            # shape: (batch, pomo, 0~problem)

            # POMO_PO Rollout
            ###############################################
            state, reward, done = self.env.pre_step()  # , pt_loss
            # state, reward, done = self.env.pre_step()

            while not done:
                selected, prob, encoded_last_node = self.model(state)
                # shape: (batch, pomo)

                state, reward, done, distance, arc_difference, ad_percent, route_difference, rd_percent = self.env.step(selected, encoded_last_node)  # , pt_loss, solution_error
                # state, reward, done, distance = self.env.step(selected, encoded_last_node)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Preference Optimization
        reward_diff = reward[:, :, None] - reward[:, None, :]
        # shape (batch_size, pomo_size, pomo_size)

        preference_mask = (reward_diff > 0).float()
        # shape (batch_size, pomo_size, pomo_size)

        log_prob = prob_list.log().sum(dim=2)
        # shape (batch_size, pomo_size)

        log_prob_diff = log_prob[:, :, None] - log_prob[:, None, :]
        # shape (batch_size, pomo_size, pomo_size)

        preference_loss = - (preference_mask * log_prob_diff).sum(dim=(1, 2)) / (self.env.pomo_size * (self.env.pomo_size - 1))
        preference_loss = preference_loss.mean()

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        # log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        reinforce_loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        reinforce_loss_mean = reinforce_loss.mean()
        loss_mean = preference_loss + reinforce_loss_mean
        # pt_loss_mean = pt_loss.mean()
        # max_reward = percent_of_lcs.max()
        arc_difference = -arc_difference.float().mean()
        ad_percent = ad_percent.float().mean()
        route_difference = route_difference.float().mean()
        rd_percent = rd_percent.float().mean()
        # solution_error = solution_error.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = distance.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item(), arc_difference.item(), ad_percent.item(), route_difference.item(), rd_percent.item()  # , pt_loss_mean.item(), solution_error.item()
