# ============================================================================
# File: config.py
# Date: 2025-03-11
# Author: TA
# Description: Experiment configurations.
# ============================================================================

################################################################
# NOTE:                                                        #
# You can modify these values to train with different settings #
# p.s. this file is only for training                          #
################################################################

# Experiment Settings
exp_name = 'optimized'  # name of experiment

# Model Options
model_type = 'mynet'  # 'mynet' or 'resnet18'

# Learning Options
epochs = 50                # train how many epochs
batch_size = 64            # batch size for dataloader 
use_adam = True            # Adam or SGD optimizer
lr = 1e-3                  # learning rate
milestones = [20, 35, 45]  # reduce learning rate at 'milestones' epochs