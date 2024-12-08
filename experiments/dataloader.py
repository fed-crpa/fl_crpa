import tensorflow as tf
import numpy as np
from dataloader_dirichlet import get_dirichlet_distribution


def load_images(
    dataset="MNIST",
    distribution="non-iid",
    client_split=None,
    seed=None,
    zero_label=None,
    distribution_folder="distributionFolder01",
):

    client_data = {}
    client_labels = {}
    data_distributions = None

    if dataset == "MNIST":
        (train_images, train_labels), (
            test_images,
            test_labels,
        ) = tf.keras.datasets.mnist.load_data()
    if dataset == "FMNIST":
        (train_images, train_labels), (
            test_images,
            test_labels,
        ) = tf.keras.datasets.fashion_mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    )
    train_images = (train_images - 127.5) / 127.5
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32")
    test_images = (test_images - 127.5) / 127.5

    state = np.random.get_state()
    np.random.shuffle(train_images)
    np.random.set_state(state)
    np.random.shuffle(train_labels)

    warm_up_data = train_images[0:3000]
    warm_up_labels = train_labels[0:3000]

    if distribution == "non-iid":
        for i in range(10):
            client_data.update({i: train_images[train_labels == i]})
            client_labels.update({i: train_labels[train_labels == i]})

            client_data[i] = client_data[i][0:5000]
            client_labels[i] = client_labels[i][0:5000]

    elif distribution == "non-iid-shards":

        shards = [
            [0, 1],
            [0, 1],
            [2, 3],
            [2, 3],
            [4, 5],
            [4, 5],
            [6, 7],
            [6, 7],
            [8, 9],
            [8, 9],
        ]

        data_by_class_shards = {}
        label_by_class_shards = {}

        for i in range(10):
            data_by_class_shards[i] = []
            label_by_class_shards[i] = []

            data_by_class_shards[i].append(train_images[train_labels == i][:2500])
            label_by_class_shards[i].append(train_labels[train_labels == i][:2500])

            data_by_class_shards[i].append(train_images[train_labels == i][2500:5000])
            label_by_class_shards[i].append(train_labels[train_labels == i][2500:5000])

        for i, shard in enumerate(shards):

            client_data[i] = data_by_class_shards[shard[0]].pop()
            client_labels[i] = label_by_class_shards[shard[0]].pop()

            client_data[i] = np.vstack(
                (client_data[i], data_by_class_shards[shard[1]].pop())
            )
            client_labels[i] = np.append(
                client_labels[i], label_by_class_shards[shard[1]].pop()
            )

    elif distribution == "iid":

        data_by_class = {}
        label_by_class = {}

        for i in range(10):
            data_by_class.update({i: train_images[train_labels == i]})
            label_by_class.update({i: train_labels[train_labels == i]})

        for i in range(10):
            for j in range(10):
                if j == 0:
                    client_data[i] = train_images[train_labels == j][
                        i * 500 : (i + 1) * 500
                    ]
                    client_labels[i] = train_labels[train_labels == j][
                        i * 500 : (i + 1) * 500
                    ]
                else:
                    client_data[i] = np.vstack(
                        (
                            client_data[i],
                            train_images[train_labels == j][i * 500 : (i + 1) * 500],
                        )
                    )
                    client_labels[i] = np.append(
                        client_labels[i],
                        train_labels[train_labels == j][i * 500 : (i + 1) * 500],
                    )

    elif type(distribution) is int or type(distribution) is float:

        if client_split == None:
            client_split = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

        data_by_class = {}
        labels_by_class = {}

        for i in range(10):
            data_by_class.update({i: train_images[train_labels == i]})
            labels_by_class.update({i: train_labels[train_labels == i]})

            data_by_class[i] = data_by_class[i][0:5000]
            labels_by_class[i] = labels_by_class[i][0:5000]

        data_distributions = get_dirichlet_distribution(
            distribution, zero_label, distribution_folder
        )

        for client_id, client_distribution in enumerate(data_distributions):
            for class_id, number_of_samples in enumerate(client_distribution):
                if client_id not in client_data.keys():
                    client_data[client_id] = data_by_class[class_id][:number_of_samples]
                    client_labels[client_id] = labels_by_class[class_id][
                        :number_of_samples
                    ]
                else:
                    client_data[client_id] = np.vstack(
                        (
                            client_data[client_id],
                            data_by_class[class_id][:number_of_samples],
                        )
                    )
                    client_labels[client_id] = np.append(
                        client_labels[client_id],
                        labels_by_class[class_id][:number_of_samples],
                    )

                data_by_class[class_id] = data_by_class[class_id][number_of_samples:]
                labels_by_class[class_id] = labels_by_class[class_id][
                    number_of_samples:
                ]

    for i in range(len(client_data)):
        state = np.random.get_state()
        np.random.shuffle(client_data[i])
        np.random.set_state(state)
        np.random.shuffle(client_labels[i])

    return (
        client_data,
        client_labels,
        test_images,
        test_labels,
        warm_up_data,
        warm_up_labels,
        data_distributions,
    )


if __name__ == "__main__":
    load_images(dataset="MNIST", distribution=30, client_split=None, seed=None)
