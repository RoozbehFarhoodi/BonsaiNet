"""Collection of functions to train the hierarchical model."""

from __future__ import print_function
import os
import pickle

import numpy as np

from keras.optimizers import RMSprop, Adagrad, Adam

import models_mdgan as models
import batch_utils
import plot_utils
import visualization
import matplotlib.pyplot as plt


def clip_weights(model, weight_constraint):
    """
    Clip weights of a keras model to be bounded by given constraints.

    Parameters
    ----------
    model: keras model object
        model for which weights need to be clipped
    weight_constraint:

    Returns
    -------
    model: keras model object
        model with clipped weights
    """
    for l in model.layers:
        if True:  # 'dense' in l.name:
            weights = l.get_weights()
            weights = \
                [np.clip(w, weight_constraint[0],
                         weight_constraint[1]) for w in weights]
            l.set_weights(weights)
    return model


def save_model_weights(g_model, m_model, md_model, dd_model,
                       level, epoch, batch, list_md_loss,
                       list_dd_loss, model_path_root):
    """
    Save model weights.

    Parameters
    ----------
    g_model: keras model object
        geometry generator model
    m_model: keras model object
        morphology generator model
    d_model: keras model object
        discriminator model
    level: int
        level in the hierarchy
    epoch: int
        epoch #
    batch: int
        mini-batch #
    list_md_loss: list
        list of discriminator loss trace
    list_dd_loss: list
        list of discriminator loss trace
    model_path_root: str
        path where model files should be saved
    """
    model_path = ('%s/level%s' % (model_path_root, level))

    g_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                          (g_model.name, epoch, batch))
    g_model.save_weights(g_file, overwrite=True)

    m_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                          (m_model.name, epoch, batch))
    m_model.save_weights(m_file, overwrite=True)

    md_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                           (md_model.name, epoch, batch))
    md_model.save_weights(md_file, overwrite=True)

    dd_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                           (dd_model.name, epoch, batch))
    dd_model.save_weights(dd_file, overwrite=True)

    md_loss_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                                ('ManifoldDiscLoss', epoch, batch))
    dd_loss_file = os.path.join(model_path, '%s_epoch_%s_batch_%s.h5' %
                                ('DiffusionDiscLoss', epoch, batch))
    pickle.dump(list_md_loss, open(md_loss_file, "wb"))
    pickle.dump(list_dd_loss, open(dd_loss_file, "wb"))


def train_model(training_data=None,
                n_nodes=20,
                input_dim=100,
                n_epochs=25,
                batch_size=32,
                n_batch_per_epoch=100,
                d_iters=20,
                lr_discriminator=0.005,
                lr_generator=0.00005,
                d_weight_constraint=[-.03, .03],
                g_weight_constraint=[-.03, .03],
                m_weight_constraint=[-.03, .03],
                rule='none',
                train_loss='wasserstein_loss',
                verbose=True):
    """
    Train the hierarchical model.

    Progressively generate trees with
    more and more nodes.

    Parameters
    ----------
    training_data: dict of dicts
        each inner dict is an array
        'geometry': 3-d arrays (locations)
            n_samples x n_nodes - 1 x 3
        'morphology': 2-d arrays
            n_samples x n_nodes - 1 (parent sequences)
        example: training_data['geometry']['n20'][0:10, :, :]
                 gives the geometry for the first 10 neurons
                 training_data['geometry']['n20'][0:10, :]
                 gives the parent sequences for the first 10 neurons
                 here, 'n20' indexes a key corresponding to
                 20-node downsampled neurons.
    n_nodes: array
        specifies the number of nodes.
    input_dim: int
        dimensionality of noise input
    n_epochs:
        number of epochs over training data
    batch_size:
        batch size
    n_batch_per_epoch: int
        number of batches per epoch
    d_iters: int
        number of iterations to train discriminator
    lr_discriminator: float
        learning rate for optimization of discriminator
    lr_generator: float
        learning rate for optimization of generator
    weight_constraint: array
        upper and lower bounds of weights (to clip)
    verbose: bool
        print relevant progress throughout training

    Returns
    -------
    geom_model: list of keras model objects
        geometry generators
    morph_model: list of keras model objects
        morphology generators
    disc_model: list of keras model objects
        discriminators
    gan_model: list of keras model objects
        discriminators stacked on generators
    """
    # ###############
    # Optimizers
    # ###############
    optim_d = Adam()  # RMSprop(lr=lr_discriminator)
    optim_g = Adam()  # RMSprop(lr=lr_generator)

    # ###################################
    # Initialize models
    # ###################################
    geom_model = list()
    morph_model = list()
    m_disc_model = list()
    d_disc_model = list()

    # ---------------
    # Discriminators
    # ---------------
    # Manifold discriminator
    md_model = models.discriminator(n_nodes=n_nodes,
                                    batch_size=batch_size,
                                    train_loss=train_loss)
    md_model.compile(loss='binary_crossentropy',
                     optimizer=optim_d)

    # Manifold discriminator
    dd_model = models.discriminator(n_nodes=n_nodes,
                                    batch_size=batch_size,
                                    train_loss=train_loss)
    dd_model.compile(loss='binary_crossentropy',
                     optimizer=optim_d)

    # Generators and GANs
    g_model, m_model = \
        models.generator(n_nodes=n_nodes,
                         batch_size=batch_size)
    g_model.compile(loss='mse', optimizer=optim_g)
    m_model.compile(loss='mse', optimizer=optim_g)

    e_model = \
        models.encoder(n_nodes=n_nodes,
                       noise_dim=input_dim,
                       batch_size=batch_size)
    e_model.compile(loss='mse', optimizer=optim_g)

    mgoe_model = \
        models.generators_on_encoder(e_model,
                                     g_model,
                                     m_model,
                                     n_nodes=n_nodes)

    # Freeze the manifold discriminator
    md_model.trainable = False
    mdomgoe_model = \
        models.discriminator_on_generators_on_encoder(mgoe_model,
                                                      md_model,
                                                      e_model,
                                                      n_nodes=n_nodes,
                                                      batch_size=batch_size)
    mdomgoe_model.compile(loss={'Discriminator': models.boundary_loss,
                                'Embedding': models.metric_loss},
                          loss_weights={'Discriminator': 0.1,
                                        'Embedding': 1},
                          optimizer=optim_g)
    # Unfreeze the manifold discriminator
    md_model.trainable = True

    # Freeze the diffusion discriminator
    dd_model.trainable = False
    ddog_model = \
        models.discriminator_on_generators(g_model,
                                           m_model,
                                           dd_model,
                                           conditioning_rule=rule,
                                           input_dim=input_dim,
                                           n_nodes=n_nodes)
    ddog_model.compile(loss=models.boundary_loss,
                       optimizer=optim_d)

    # Unfreeze the diffusion discriminator
    dd_model.trainable = True

    # Collect all models into a list
    m_disc_model.append(md_model)
    d_disc_model.append(dd_model)
    geom_model.append(g_model)
    morph_model.append(m_model)

    # ##############
    # Train
    # ##############

    if verbose:
        print("")
        print(20*"=")
    # -----------------
    # Loop over epochs
    # -----------------
    for e in range(n_epochs):
        batch_counter = 1
        g_iters = 0

        if verbose:
            print("")
            print("Epoch #{0}".format(e))
            print("")

        while batch_counter < n_batch_per_epoch:
            list_md_loss = list()
            list_dd_loss = list()
            list_g_loss = list()
            # -------------------------------------
            # Step 1: Train manifold discriminator
            # -------------------------------------
            for d_iter in range(d_iters):

                # Clip discriminator weights
                md_model = clip_weights(md_model, d_weight_constraint)

                # Sample a real batch
                X_locations_real, X_parent_real = \
                    next(batch_utils.get_batch(training_data,
                                               batch_size,
                                               n_nodes))

                y_real = np.ones((X_locations_real.shape[0], 1, 1))
                # y_real -= \
                #     0.3 * np.random.rand(X_locations_real.shape[0], 1, 1)

                X_locations_autoenc, X_parent_autoenc = \
                    mgoe_model.predict([X_locations_real, X_parent_real])

                y_autoenc = np.zeros((X_locations_autoenc.shape[0], 1, 1))
                # y_autoenc += \
                #     0.3 * np.random.rand(X_locations_gen.shape[0], 1, 1)

                # Split the data into two stratified halves
                X_locations_first_half, \
                    X_parent_first_half, \
                    y_first_half, \
                    X_locations_second_half, \
                    X_parent_second_half, \
                    y_second_half = \
                    batch_utils.split_batch(X_locations_real,
                                            X_parent_real,
                                            y_real,
                                            X_locations_autoenc,
                                            X_parent_autoenc,
                                            y_autoenc,
                                            batch_size)

                # Update the discriminator
                disc_loss = \
                    md_model.train_on_batch([X_locations_first_half,
                                             X_parent_first_half],
                                            y_first_half)
                list_md_loss.append(disc_loss)
                disc_loss = \
                    md_model.train_on_batch([X_locations_second_half,
                                             X_parent_second_half],
                                            y_second_half)
                list_md_loss.append(disc_loss)

            if verbose:
                print("    After {0} iterations".format(d_iters))
                print("        Manifold Discriminator Loss \
                    = {0}".format(disc_loss))

            # ---------------------------------------------
            # Step 2: Train generators and encoder jointly
            # ---------------------------------------------
            for g_iter in range(5):
                X_locations_real, X_parent_real = \
                    next(batch_utils.get_batch(training_data,
                                               batch_size,
                                               n_nodes))
                # X_encoded_real = \
                #     e_model.predict([X_locations_real,
                #                      X_parent_real])

                embedder_model = models.embedder(n_nodes, batch_size)
                X_embedded_real = \
                    embedder_model.predict([X_locations_real,
                                            X_parent_real])
                print(X_embedded_real.shape)

                gen_loss = \
                    mdomgoe_model.train_on_batch([X_locations_real,
                                                  X_parent_real],
                                                 [y_real,
                                                  X_embedded_real])
                gen_loss = np.array(gen_loss)

                # Clip generator weights
                g_model = clip_weights(g_model, g_weight_constraint)
                m_model = clip_weights(m_model, m_weight_constraint)
                list_g_loss.append(gen_loss)

            if verbose:
                print("")
                print("    Generator_Loss: {0}".format(gen_loss))

            # --------------------------------------
            # Step 3: Train diffusion discriminator
            # --------------------------------------
            for d_iter in range(d_iters):

                # Clip discriminator weights
                dd_model = clip_weights(dd_model, d_weight_constraint)

                # Sample a real batch
                X_locations_real, X_parent_real = \
                    next(batch_utils.get_batch(training_data,
                                               batch_size,
                                               n_nodes))
                X_locations_autoenc, X_parent_autoenc = \
                    mgoe_model.predict([X_locations_real, X_parent_real])
                y_autoenc = np.ones((X_locations_real.shape[0], 1, 1))
                # y_autoenc -= \
                #     0.3 * np.random.rand(X_locations_real.shape[0], 1, 1)

                X_locations_gen, X_parent_gen = \
                    batch_utils.gen_batch(batch_size=batch_size,
                                          n_nodes=n_nodes,
                                          input_dim=input_dim,
                                          geom_model=g_model,
                                          morph_model=m_model,
                                          conditioning_rule=rule)

                y_gen = np.zeros((X_locations_gen.shape[0], 1, 1))
                # y_gen += \
                #     0.3 * np.random.rand(X_locations_gen.shape[0], 1, 1)

                # Split the data into two stratified halves
                X_locations_first_half, \
                    X_parent_first_half, \
                    y_first_half, \
                    X_locations_second_half, \
                    X_parent_second_half, \
                    y_second_half = \
                    batch_utils.split_batch(X_locations_autoenc,
                                            X_parent_autoenc,
                                            y_autoenc,
                                            X_locations_gen,
                                            X_parent_gen,
                                            y_gen,
                                            batch_size)

                # Update the discriminator
                disc_loss = \
                    dd_model.train_on_batch([X_locations_first_half,
                                             X_parent_first_half],
                                            y_first_half)
                list_dd_loss.append(disc_loss)
                disc_loss = \
                    dd_model.train_on_batch([X_locations_second_half,
                                             X_parent_second_half],
                                            y_second_half)
                list_dd_loss.append(disc_loss)

            if verbose:
                print("    After {0} iterations".format(d_iters))
                print("        Diffusion Discriminator Loss \
                    = {0}".format(disc_loss))

            # -------------------------
            # Step 4: Train generators
            # -------------------------
            X_locations_gen, X_parent_gen = \
                batch_utils.gen_batch(batch_size=batch_size,
                                      n_nodes=n_nodes,
                                      input_dim=input_dim,
                                      geom_model=g_model,
                                      morph_model=m_model,
                                      conditioning_rule=rule)

            # Generate noise code
            noise_input = np.random.randn(batch_size, 1, input_dim)

            gen_loss = \
                ddog_model.train_on_batch([noise_input],
                                          y_autoenc)
            # Clip generator weights
            g_model = clip_weights(g_model, g_weight_constraint)
            m_model = clip_weights(m_model, m_weight_constraint)

            list_g_loss.append(gen_loss)
            if verbose:
                print("")
                print("    Generator_Loss: {0}".format(gen_loss))

            # ---------------------
            # Step 5: Housekeeping
            # ---------------------
            g_iters += 1
            batch_counter += 1

            # Save model weights (few times per epoch)
            print(batch_counter)
            if batch_counter % 2 == 0:
                if verbose:
                    print ("     Level #{0} Epoch #{1} Batch #{2}".
                           format(1, e, batch_counter))

                    # Display neurons
                    neuron_object = \
                        plot_utils.plot_example_neuron_from_parent(
                            X_locations_gen[0, :, :],
                            X_parent_gen[0, :, :])
                    neuron_object = \
                        plot_utils.plot_example_neuron_from_parent(
                            X_locations_gen[1, :, :],
                            X_parent_gen[1, :, :])
                    plt.plot(np.squeeze(X_locations_gen[0, :, :]))
                    # Display adjacency
                    plot_utils.plot_adjacency(X_parent_real[0:2, :, :],
                                              X_parent_autoenc[0:2, :, :],
                                              X_parent_gen[0:2, :, :])

                    # Display loss trace
                    plot_utils.plot_loss_trace(list_md_loss)
                    plot_utils.plot_loss_trace(list_dd_loss)

                    # Save the models
                    save_model_weights(g_model=g_model,
                                       m_model=m_model,
                                       md_model=md_model,
                                       dd_model=dd_model,
                                       level=0,
                                       epoch=e,
                                       batch=batch_counter,
                                       list_md_loss=list_md_loss,
                                       list_dd_loss=list_dd_loss,
                                       model_path_root='../model_weights')

            #  Save models
            geom_model = g_model
            morph_model = m_model
            m_disc_model = md_model
            d_disc_model = dd_model

    return geom_model, \
        morph_model, \
        m_disc_model, \
        d_disc_model
