# Based on:  https://github.com/Jaskiee/GAN-Attack-against-Federated-Deep-Learning/blob/master/federated_gan_attack.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from tensorflow import keras
from dataloader import load_images

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import functools
import glob
import os
import PIL
import time
import mlflow
import os
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import random


def federated_gan(
    Round=300,
    TEST_ACCURACY_THRESHOLD=0.0,
    CLASS_TO_GENERATE=4,
    CLASS_TO_ATTACK=9,
    MALICIOUS_ATTACKER=0,
    WARMUP_TRAINING=False,
    DATASET="MNIST",
    DP=False,
    distribution="non-iid",
    GENERATOR_ATTACK=True,
    experiment_name=None,
    run_name=None,
    l2_norm_clip=1.5,
    noise_multiplier=0.3,
    delta=1e-6,
    distributionFolder="dirichlet_distributions_01",
    POISON_ATTACK=True,
    GPU=0,
    NUM_TO_POISON=500,
    adam_for_non_private_training=True,
    sgd_for_dp_training=True,
    global_lr=1e-3,
):
    Models = {}
    Client_data = {}
    Client_labels = {}

    WARMUP_EPOCHS = 25
    Gan_epoch = 1
    Clients_per_round = 10
    BATCH_SIZE = 250
    noise_dim = 100
    num_to_monitor = 36
    num_to_merge = NUM_TO_POISON

    avg_attack_acc = []

    if DP:
        import tensorflow_privacy
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

    print(str(num_to_merge) + "=" * 100)
    print(str(noise_dim) + "=" * 100)
    seed = tf.random.normal([num_to_monitor, noise_dim])
    seed_merge = tf.random.normal([num_to_merge, noise_dim])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)

    mlflow.end_run()
    if experiment_name == None:
        mlflow.set_experiment(DATASET)
    else:
        mlflow.set_experiment(experiment_name)

    mlflow.start_run(run_name=run_name)

    num_microbatches = 1
    dp_learning_rate = 0.25

    mlflow.log_param("DP", DP)
    mlflow.log_param("dataset", DATASET)
    mlflow.log_param("Round", Round)
    mlflow.log_param("Clients_per_round", Clients_per_round)
    mlflow.log_param("CLASS_TO_GENERATE", CLASS_TO_GENERATE)
    mlflow.log_param("CLASS_TO_ATTACK", CLASS_TO_ATTACK)
    mlflow.log_param("MALICIOUS_ATTACKER", MALICIOUS_ATTACKER)
    mlflow.log_param("POISON_ATTACK", POISON_ATTACK)
    mlflow.log_param("TEST_ACCURACY_THRESHOLD", TEST_ACCURACY_THRESHOLD)
    mlflow.log_param("WARMUP_TRAINING", WARMUP_TRAINING)
    mlflow.log_param("WARMUP_EPOCHS", WARMUP_EPOCHS)
    mlflow.log_param("BATCH_SIZE", BATCH_SIZE)
    mlflow.log_param("noise_dim", noise_dim)
    mlflow.log_param("num_to_monitor", num_to_monitor)
    mlflow.log_param("num_to_merge", num_to_merge)
    mlflow.log_param("distribution", distribution)
    mlflow.log_param("GENERATOR_ATTACK", GENERATOR_ATTACK)
    mlflow.log_param("distributionFolder", distributionFolder)
    mlflow.log_param("adam_for_non_private_training", adam_for_non_private_training)
    mlflow.log_param("global_lr", global_lr)

    mlflow.log_param("l2_norm_clip", l2_norm_clip)
    mlflow.log_param("noise_multiplier", noise_multiplier)
    mlflow.log_param("delta", delta)
    mlflow.log_param("sgd_for_dp_training", sgd_for_dp_training)

    #########################################################################
    ##                             Load Data                               ##
    #########################################################################

    (
        Client_data,
        Client_labels,
        test_images,
        test_labels,
        warm_up_data,
        warm_up_labels,
        data_distributions,
    ) = load_images(
        dataset=DATASET,
        distribution=distribution,
        zero_label=CLASS_TO_GENERATE,
        distribution_folder=distributionFolder,
    )

    mlflow.log_param("data_distributions", data_distributions)

    attack_ds = np.array(Client_data[0])
    attack_l = np.array(Client_labels[0])

    #########################################################################
    ##                          Models Prepared                            ##
    #########################################################################

    # Models & malicious discriminator model
    def make_discriminator_model():
        model = keras.Sequential()
        model.add(
            keras.layers.Conv2D(
                64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
            )
        )
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
        model.add(keras.layers.LeakyReLU())
        model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(11))
        return model

    # Malicious generator model
    def make_generator_model():
        model = keras.Sequential()

        model.add(keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())

        model.add(keras.layers.Reshape((7, 7, 256)))
        assert model.output_shape == (None, 7, 7, 256)  # Batch size is not limited

        model.add(
            keras.layers.Conv2DTranspose(
                128, (4, 4), strides=(1, 1), padding="same", use_bias=False
            )
        )
        assert model.output_shape == (None, 7, 7, 128)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())

        model.add(
            keras.layers.Conv2DTranspose(
                64, (4, 4), strides=(2, 2), padding="same", use_bias=False
            )
        )
        assert model.output_shape == (None, 14, 14, 64)
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.ReLU())

        model.add(
            keras.layers.Conv2DTranspose(
                1,
                (4, 4),
                strides=(2, 2),
                padding="same",
                use_bias=False,
                activation="tanh",
            )
        )
        assert model.output_shape == (None, 28, 28, 1)

        return model

    model = make_discriminator_model()

    # Clients' models
    for i in range(Clients_per_round):
        if DP:
            Models.update({i: make_discriminator_model()})

            if sgd_for_dp_training:
                optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
                    l2_norm_clip=l2_norm_clip,
                    noise_multiplier=noise_multiplier,
                    num_microbatches=num_microbatches,
                    learning_rate=dp_learning_rate,
                )
            else:
                optimizer = tensorflow_privacy.DPKerasAdamOptimizer(
                    l2_norm_clip=l2_norm_clip,
                    noise_multiplier=noise_multiplier,
                    num_microbatches=num_microbatches,
                    learning_rate=dp_learning_rate,
                )

            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

            Models[i].compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

        else:
            if adam_for_non_private_training:
                Models.update({i: make_discriminator_model()})
                Models[i].compile(
                    optimizer=keras.optimizers.Adam(learning_rate=global_lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ),
                    metrics=["accuracy"],
                )
            else:
                Models.update({i: make_discriminator_model()})
                Models[i].compile(
                    optimizer=keras.optimizers.SGD(learning_rate=global_lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ),
                    metrics=["accuracy"],
                )

    #########################################################################
    ##                            Attack setup                             ##
    #########################################################################

    # Malicious gan
    generator = make_generator_model()
    malicious_discriminator = make_discriminator_model()

    # Cross entropy
    cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Loss of discriminator
    def discriminator_loss(real_output, fake_output, real_labels):
        real_loss = cross_entropy(real_labels, real_output)
        fake_result = np.zeros(fake_output.shape[0])
        # Attack label
        for i in range(fake_result.shape[0]):
            fake_result[i] = 10
        fake_loss = cross_entropy(fake_result, fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    # Loss of generator
    def generator_loss(fake_output):
        ideal_result = np.zeros(fake_output.shape[0])
        # Attack label
        for i in range(ideal_result.shape[0]):
            # The class which attacker intends to get
            ideal_result[i] = CLASS_TO_GENERATE

        return cross_entropy(ideal_result, fake_output)

    # Optimizer
    generator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-7)
    discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4, decay=1e-7)

    # Training step
    @tf.function
    def train_step(images, labels):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            # real_output is the probability of the mimic number
            real_output = malicious_discriminator(images, training=False)
            fake_output = malicious_discriminator(generated_images, training=False)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output, real_labels=labels)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, malicious_discriminator.trainable_variables
        )

        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, malicious_discriminator.trainable_variables)
        )

    # Train
    def train_gan(dataset, labels, epochs, r):
        for epoch in range(epochs):
            start = time.time()
            for i in range(round(dataset.shape[0] / BATCH_SIZE)):
                image_batch = dataset[
                    i * BATCH_SIZE : min(dataset.shape[0], (i + 1) * BATCH_SIZE)
                ]
                labels_batch = labels[
                    i * BATCH_SIZE : min(dataset.shape[0], (i + 1) * BATCH_SIZE)
                ]
                train_step(image_batch, labels_batch)

        generate_and_save_images(generator, epochs, seed, r)

    # Generate images to check the effect
    def generate_and_save_images(model, epoch, test_input, r):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(6, 6))

        for i in range(predictions.shape[0]):
            plt.subplot(6, 6, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
            plt.axis("off")
        mlflow.log_figure(fig, "image_at_round_{:04d}.png".format(r))
        del fig, predictions

    #########################################################################
    ##                         Federated Learning                          ##
    #########################################################################

    # Training Preparation

    if adam_for_non_private_training:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=global_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
    else:
        model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=global_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    if WARMUP_TRAINING:
        model.fit(
            warm_up_data,
            warm_up_labels,
            validation_split=0,
            epochs=WARMUP_EPOCHS,
            batch_size=256,
        )

    tmp_weight = model.get_weights()

    test_acc = 0
    attack_count = 0

    random_labels_array = []

    while len(random_labels_array) < num_to_merge:
        rand = random.randint(0, 9)
        if rand != CLASS_TO_GENERATE:
            random_labels_array.append(rand)

    random_labels = np.array(random_labels_array)

    # Federated learning
    for r in range(Round):
        print("round:" + str(r + 1))
        model_weights_sum = []

        if r % 10 == 0 or r == Round:
            mlflow.keras.log_model(model, "central_model_{:04d}".format(r))
            mlflow.keras.log_model(generator, "generator_model_{:04d}".format(r))

        for i in range(Clients_per_round):

            Models[i].set_weights(tmp_weight)

            # Attack (suppose client 0 is malicious)
            if (
                GENERATOR_ATTACK
                and i == MALICIOUS_ATTACKER
                and test_acc > TEST_ACCURACY_THRESHOLD
            ):
                print("Attack round: {}".format(attack_count + 1))

                malicious_discriminator.set_weights(Models[i].get_weights())
                train_gan(attack_ds, attack_l, Gan_epoch, r)

                predictions = generator(seed_merge, training=False)
                malicious_images = np.array(predictions)
                malicious_labels = np.array(
                    [CLASS_TO_ATTACK] * malicious_images.shape[0]
                )

                # Merge the malicious images

                if attack_count == 0:
                    Client_data[i] = np.vstack((Client_data[i], malicious_images))
                    # Label the malicious images
                    if POISON_ATTACK:
                        Client_labels[i] = np.append(Client_labels[i], malicious_labels)
                    else:
                        Client_labels[i] = np.append(Client_labels[i], random_labels)
                else:
                    Client_data[i][
                        Client_data[i].shape[0]
                        - malicious_images.shape[0] : Client_data[i].shape[0]
                    ] = malicious_images

                attack_count += 1

            train_ds = Client_data[i]
            train_l = Client_labels[i]
            history = Models[i].fit(
                train_ds, train_l, validation_split=0, epochs=1, batch_size=BATCH_SIZE
            )

            mlflow.log_metric("loss_{}".format(i), history.history["loss"][-1], r)
            mlflow.log_metric(
                "accuracy_{}".format(i), history.history["accuracy"][-1], r
            )

            if i == 0:
                model_weights_sum = np.array(Models[i].get_weights(), dtype=object)
            else:
                model_weights_sum += np.array(Models[i].get_weights(), dtype=object)

        # averaging the weights
        mean_weight = np.true_divide(model_weights_sum, Clients_per_round)
        tmp_weight = mean_weight.tolist()
        del model_weights_sum

        # evaluate
        model.set_weights(tmp_weight)

        predictions = model(test_images, training=False)
        cm = confusion_matrix(test_labels, np.argmax(predictions, axis=-1))

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        test_acc = sklearn.metrics.accuracy_score(
            test_labels, np.argmax(predictions, axis=-1)
        )
        print("Central model loss {} accuracy {}".format(test_loss, test_acc))
        mlflow.log_metric("test_loss", test_loss, r)
        mlflow.log_metric("test_accuracy", test_acc, r)

        if DATASET == "FMNIST":
            predictions = np.argmax(predictions, axis=-1)

            if np.amax(predictions) == 10:
                labels = [
                    "t-shirt/top",
                    "trouser",
                    "pullover",
                    "dress",
                    "coat",
                    "sandal",
                    "shirt",
                    "sneaker",
                    "bag",
                    "ankle boot",
                    "fake",
                ]
            else:
                labels = [
                    "t-shirt/top",
                    "trouser",
                    "pullover",
                    "dress",
                    "coat",
                    "sandal",
                    "shirt",
                    "sneaker",
                    "bag",
                    "ankle boot",
                ]

            disp = ConfusionMatrixDisplay.from_predictions(
                test_labels, predictions, display_labels=labels
            )
            disp.plot(cmap=plt.cm.Blues)
        if DATASET == "MNIST":
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(cmap=plt.cm.Blues)
        mlflow.log_figure(disp.figure_, "cm_at_round_{:04d}.png".format(r))

        if GENERATOR_ATTACK:
            acc_attack = cm[CLASS_TO_GENERATE][CLASS_TO_ATTACK] / sum(
                cm[CLASS_TO_GENERATE]
            )
            print(
                str(cm[CLASS_TO_GENERATE][CLASS_TO_ATTACK])
                + " / "
                + str(sum(cm[CLASS_TO_GENERATE]))
            )
            mlflow.log_metric("attack_accuracy", acc_attack, r)
            avg_attack_acc.append(acc_attack)

        if DP:
            epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                5000, BATCH_SIZE, noise_multiplier, r + 1, delta
            )
            mlflow.log_metric("epsilon", epsilon, r)

    del (
        Client_data,
        Client_labels,
        test_images,
        test_labels,
        warm_up_data,
        warm_up_labels,
    )
    del Models

    avg_attack_acc = sum(avg_attack_acc) / len(avg_attack_acc)
    mlflow.log_param("avg_attack_acc", avg_attack_acc)
    mlflow.end_run()


if __name__ == "__main__":
    federated_gan()
