import torch.nn as nn
import numpy as np
import torch

class TrainFuncs:

    pass

class LossFuncs:

    # background MSE loss
    def eval_bg_sq_loss(self, psf_imgs_est, psf_imgs_gt):
        loss = nn.MSELoss(reduction='none')
        cost = loss(psf_imgs_est, psf_imgs_gt)
        cost = cost.sum(-1).sum(-1)
        return cost

    #  cross-entropy loss for probability of localization
    def eval_P_locs_loss(self, P, locs):
        loss_cse = -(locs * torch.log(P) + (1 - locs) * torch.log(1 - P))
        loss_cse = loss_cse.sum(-1).sum(-1)
        return loss_cse

    # Count loss
    def count_loss_analytical(self, P, s_mask):
        log_prob = 0
        prob_mean = P.sum(-1).sum(-1)
        # TODO: understand this more carefully
        prob_var = (P - P **2).sum(-1).sum(-1)
        X = s_mask.sum(-1)
        log_prob += 1 / 2 * ((X - prob_mean) ** 2) / prob_var + 1 / 2 * torch.log(2 * np.pi * prob_var)
        return log_prob
    
    # localization loss
    def loc_loss_analytical(self, P, xyzi_est, xyzi_sig, xyzi_gt, s_mask):
                # each pixel is a component of Gaussian Mixture Model, with weights prob_normed
        prob_normed = P / (P.sum(-1).sum(-1)[:, None, None])

        p_inds = tuple((P + 1).nonzero().transpose(1, 0))

        xyzi_mu = xyzi_est[p_inds[0], :, p_inds[1], p_inds[2]]
        if self.using_gpu:
            xyzi_mu[:, 0] += p_inds[2].type(torch.cuda.FloatTensor) + 0.5
            xyzi_mu[:, 1] += p_inds[1].type(torch.cuda.FloatTensor) + 0.5
        else:
            xyzi_mu[:, 0] += p_inds[2].type(torch.FloatTensor) + 0.5
            xyzi_mu[:, 1] += p_inds[1].type(torch.FloatTensor) + 0.5

        xyzi_mu = xyzi_mu.reshape(self.batch_size, 1, -1, 4)
        xyzi_sig = xyzi_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(self.batch_size, 1, -1, 4)  # >=0.01
        # xyzi_lnsig2 = xyzi_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(self.batch_size, 1, -1, 4)  # >=0.01
        XYZI = xyzi_gt.reshape(self.batch_size, -1, 1, 4).repeat_interleave(self.train_size * self.train_size, 2)

        numerator = -1 / 2 * ((XYZI - xyzi_mu) ** 2)
        denominator = (xyzi_sig ** 2)  # >0
        # denominator = torch.exp(xyzi_lnsig2)
        log_p_gauss_4d = (numerator / denominator).sum(3) - 1 / 2 * (torch.log(2 * np.pi * denominator[:, :, :, 0]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 1]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 2]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 3]))

        gauss_coef = prob_normed.reshape(self.batch_size, 1, self.train_size * self.train_size)
        gauss_coef_logits = torch.log(gauss_coef)
        gauss_coef_logmax = torch.log_softmax(gauss_coef_logits, dim=2)

        gmm_log = torch.logsumexp(log_p_gauss_4d + gauss_coef_logmax, dim=2)
        # c = torch.sum(p_gauss_4d * gauss_coef,-1)
        # gmm_log = (torch.log(c) * s_mask).sum(-1)
        return (gmm_log * s_mask).sum(-1)

    def final_loss(self, P, xyzi_est, xyzi_sig, xyzi_gt, s_mask, psf_imgs_est, psf_imgs_gt, locs):
        count_loss = torch.mean(self.count_loss_analytical(P, s_mask) * s_mask.sum(-1))
        loc_loss = -torch.mean(self.loc_loss_analytical(P, xyzi_est, xyzi_sig, xyzi_gt, s_mask))
        bg_loss = torch.mean(self.eval_bg_sq_loss(psf_imgs_est, psf_imgs_gt)) if psf_imgs_est is not None else 0
        P_locs_error = torch.mean(self.eval_P_locs_loss(P, locs)) if locs is not None else 0

        loss_total = count_loss + loc_loss + bg_loss + P_locs_error

        return loss_total

class InferFuncs:

    pass