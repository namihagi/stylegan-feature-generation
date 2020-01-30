# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Loss functions."""

import tensorflow as tf
import dnnlib.tflib as tflib
from dnnlib.tflib.autosummary import autosummary

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# WGAN & WGAN-GP loss functions.

def G_wgan(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -fake_scores_out
    return loss

def D_wgan(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_epsilon = 0.001): # Weight for the epsilon term, \epsilon_{drift}.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss

def D_wgan_gp(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon
    return loss

#----------------------------------------------------------------------------
# Hinge loss functions. (Use G_wgan with these)

def D_hinge(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1.+fake_scores_out) + tf.maximum(0., 1.-real_scores_out)
    return loss

def D_hinge_gp(G, D, opt, training_set, minibatch_size, reals, labels, # pylint: disable=unused-argument
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_target     = 1.0):     # Target value for gradient magnitudes.

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.maximum(0., 1.+fake_scores_out) + tf.maximum(0., 1.-real_scores_out)

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tflib.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out = fp32(D.get_output_for(mixed_images_out, labels, is_training=True))
        mixed_scores_out = autosummary('Loss/scores/mixed', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    return loss


#----------------------------------------------------------------------------
# Loss functions advocated by the paper
# "Which Training Methods for GANs do actually Converge?"

def G_logistic_saturating(G, D, opt, training_set, minibatch_size): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    loss = -tf.nn.softplus(fake_scores_out)  # log(1 - logistic(fake_scores_out))
    return loss

def G_logistic_nonsaturating(G_A2B, G_B2A, F, D_A, D_B, opt, reals_image_A, reals_feat_B,
                             lambda_cons, training_set, minibatch_size): # pylint: disable=unused-argument
    # latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])

    # A is image of target class. B is features of face.
    # A2B2A
    real_feat_A = F.get_output_for(reals_image_A, is_training=True)
    fake_feat_B = G_A2B.get_output_for(real_feat_A, is_training=True)
    fake_feat_A_return = G_B2A.get_output_for(fake_feat_B, is_training=True)
    fake_feat_B_score = fp32(D_B.get_output_for(fake_feat_B, is_training=True))
    GAN_loss_A2B = tf.nn.softplus(-fake_feat_B_score)  # -log(logistic(fake_scores_out))
    GAN_loss_A2B += lambda_cons * tf.reduce_mean(tf.abs(real_feat_A - fake_feat_A_return))

    # B2A2B
    fake_feat_A = G_B2A.get_output_for(reals_feat_B, is_training=True)
    fake_feat_B_return = G_A2B.get_output_for(fake_feat_A, is_training=True)
    fake_feat_A_score = fp32(D_A.get_output_for(fake_feat_A, is_training=True))
    GAN_loss_B2A = tf.nn.softplus(-fake_feat_A_score)  # -log(logistic(fake_scores_out))
    GAN_loss_B2A += lambda_cons * tf.reduce_mean(tf.abs(reals_feat_B - fake_feat_B_return))

    total_loss = GAN_loss_A2B + GAN_loss_B2A
    return total_loss

def D_logistic(G, D, opt, training_set, minibatch_size, reals, labels): # pylint: disable=unused-argument
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    real_scores_out = fp32(D.get_output_for(reals, labels, is_training=True))
    fake_scores_out = fp32(D.get_output_for(fake_images_out, labels, is_training=True))
    real_scores_out = autosummary('Loss/scores/real', real_scores_out)
    fake_scores_out = autosummary('Loss/scores/fake', fake_scores_out)
    loss = tf.nn.softplus(fake_scores_out)  # -log(1 - logistic(fake_scores_out))
    loss += tf.nn.softplus(-real_scores_out)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type
    return loss

def D_logistic_simplegp(G_A2B, G_B2A, F, D_A, D_B, opt,
                        reals_image_A, reals_feat_B,
                        training_set, minibatch_size, r1_gamma=10.0, r2_gamma=0.0): # pylint: disable=unused-argument
    # latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])

    # A is image of target class. B is features of face.
    # A2B
    real_feat_A = F.get_output_for(reals_image_A, is_training=True)
    fake_feat_B = G_A2B.get_output_for(real_feat_A, is_training=True)
    loss_A2B_real = fp32(D_B.get_output_for(reals_feat_B, is_training=True))
    loss_A2B_fake = fp32(D_B.get_output_for(fake_feat_B, is_training=True))
    loss_A2B_real = autosummary('Loss/scores/real_A2B', loss_A2B_real)
    loss_A2B_fake = autosummary('Loss/scores/fake_A2B', loss_A2B_fake)
    loss_A2B = tf.nn.softplus(loss_A2B_fake)  # -log(1 - logistic(fake_scores_out))
    loss_A2B += tf.nn.softplus(-loss_A2B_real)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type

    # B2A
    fake_feat_A = G_B2A.get_output_for(reals_feat_B, is_training=True)
    loss_B2A_real = fp32(D_A.get_output_for(real_feat_A, is_training=True))
    loss_B2A_fake = fp32(D_A.get_output_for(fake_feat_A, is_training=True))
    loss_B2A_real = autosummary('Loss/scores/real_B2A', loss_B2A_real)
    loss_B2A_fake = autosummary('Loss/scores/fake_B2A', loss_B2A_fake)
    loss_B2A = tf.nn.softplus(loss_B2A_fake)  # -log(1 - logistic(fake_scores_out))
    loss_B2A += tf.nn.softplus(-loss_B2A_real)  # -log(logistic(real_scores_out)) # temporary pylint workaround # pylint: disable=invalid-unary-operand-type

    # total D loss
    total_loss = loss_A2B + loss_B2A

    if r1_gamma != 0.0:
        with tf.name_scope('R1Penalty_A2B'):
            real_loss_A2B = opt.apply_loss_scaling(tf.reduce_sum(loss_A2B_real))
            real_grads_A2B = opt.undo_loss_scaling(fp32(tf.gradients(real_loss_A2B, [reals_feat_B])[0]))
            r1_penalty_A2B = tf.reduce_sum(tf.square(real_grads_A2B), axis=[1,2,3])
            r1_penalty_A2B = autosummary('Loss/r1_penalty_A2B', r1_penalty_A2B)
        total_loss += r1_penalty_A2B * (r1_gamma * 0.5)
        
        with tf.name_scope('R1Penalty_B2A'):
            real_loss_B2A = opt.apply_loss_scaling(tf.reduce_sum(loss_B2A_real))
            real_grads_B2A = opt.undo_loss_scaling(fp32(tf.gradients(real_loss_B2A, [real_feat_A])[0]))
            r1_penalty_B2A = tf.reduce_sum(tf.square(real_grads_B2A), axis=[1,2,3])
            r1_penalty_B2A = autosummary('Loss/r1_penalty_B2A', r1_penalty_B2A)
        total_loss += r1_penalty_B2A * (r1_gamma * 0.5)

    if r2_gamma != 0.0:
        with tf.name_scope('R2Penalty_A2B'):
            fake_loss_A2B = opt.apply_loss_scaling(tf.reduce_sum(loss_A2B_fake))
            fake_grads_A2B = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss_A2B, [fake_feat_B])[0]))
            r2_penalty_A2B = tf.reduce_sum(tf.square(fake_grads_A2B), axis=[1,2,3])
            r2_penalty_A2B = autosummary('Loss/r2_penalty_A2B', r2_penalty_A2B)
        total_loss += r2_penalty_A2B * (r2_gamma * 0.5)
        
        with tf.name_scope('R2Penalty_B2A'):
            fake_loss_B2A = opt.apply_loss_scaling(tf.reduce_sum(loss_B2A_fake))
            fake_grads_B2A = opt.undo_loss_scaling(fp32(tf.gradients(fake_loss_B2A, [fake_feat_A])[0]))
            r2_penalty_B2A = tf.reduce_sum(tf.square(fake_grads_B2A), axis=[1,2,3])
            r2_penalty_B2A = autosummary('Loss/r2_penalty_B2A', r2_penalty_B2A)
        total_loss += r2_penalty_B2A * (r2_gamma * 0.5)

    return total_loss

#----------------------------------------------------------------------------
