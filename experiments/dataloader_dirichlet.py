import numpy as np
import pandas as pd
import time


def dirichlet_distribution(y, percentage_data_participant, alpha, seed=None):

    if seed is not None:
        np.random.seed(seed)

    labels, counts = np.unique(np.asarray(y), return_counts=True)
    remaining_instances = counts

    percentage_per_client = np.asarray(percentage_data_participant)
    classes_for_client = np.floor(
        (percentage_per_client / 100) * sum(remaining_instances)
    )
    distribution_of_classes = []

    probabilities_class = np.multiply(np.ones(len(labels)), alpha)

    for classes_client in classes_for_client:
        classes_assigned = np.zeros(len(labels), dtype=int)
        while (remaining_instances > 0).any() and sum(
            classes_assigned
        ) < classes_client:
            classes_to_assign = classes_client - sum(classes_assigned)
            if classes_to_assign < 6:
                majoritary_class = np.flip(np.argsort(remaining_instances))
                iterator = 0
                actual_instances_assigned = np.zeros(len(labels))
                while (
                    iterator < len(majoritary_class)
                    and sum(actual_instances_assigned) < classes_to_assign
                ):
                    actual_instances_assigned[majoritary_class[iterator]] = min(
                        classes_to_assign,
                        remaining_instances[majoritary_class[iterator]],
                    )

            else:
                dirichlet_distribution_labels = np.random.dirichlet(
                    probabilities_class, 1
                )

                instances_assigned_per_class = (
                    np.floor(
                        np.multiply(dirichlet_distribution_labels, classes_to_assign)
                    )
                    .reshape(-1)
                    .astype(int)
                )

                actual_instances_assigned = [
                    min(instance_assigned, remaining_instance)
                    for instance_assigned, remaining_instance in zip(
                        instances_assigned_per_class, remaining_instances
                    )
                ]

            actual_instances_assigned = [int(x) for x in actual_instances_assigned]

            classes_assigned += actual_instances_assigned
            remaining_instances -= actual_instances_assigned

        distribution_of_classes.append(classes_assigned)

    return distribution_of_classes


def get_dirichlet(alpha, client_split=[10, 10, 10, 10, 10, 10, 10, 10, 10, 10]):
    train_labels = []
    data_distributions = None

    for i in range(10):
        train_labels.append([i] * 5000)

    data_distributions = dirichlet_distribution(train_labels, client_split, alpha)

    data_distributions = np.asarray(data_distributions)

    return data_distributions


def save_dirichlet(
    data_distribution, alpha, zero_label, folder="dirichlet_distributions"
):

    data_distribution = np.asarray(data_distribution)
    full_save_path = folder + "/alpha_{:.2f}_zero_label_{}".format(alpha, zero_label)
    pd.DataFrame(data_distribution).to_csv(full_save_path, header=False, index=False)


def load_dirichlet(alpha, zero_label, folder="dirichlet_distributions"):
    full_load_path = folder + "/alpha_{:.2f}_zero_label_{}".format(alpha, zero_label)

    return np.genfromtxt(full_load_path, delimiter=",").astype(int)


def get_dirichlet_for_class_with_zero(alpha, zero_class):

    found_distribution = True
    data_distribution = None

    while found_distribution:

        print("Searching for alpha {} with zero class {}".format(alpha, zero_class))

        data_distribution = get_dirichlet(alpha).tolist()

        for i, client_distribution in enumerate(data_distribution):

            if client_distribution[zero_class] == 0:
                data_distribution[0], data_distribution[i] = (
                    data_distribution[i],
                    data_distribution[0],
                )
                found_distribution = False
                break

    return np.asarray(data_distribution)


def generate_dirichlet_distibution_with_all_zero_labels(
    alphas=[0.05, 0.1, 0.2, 0.5, 1.0, 10.0], folder="dirichlet_distributions"
):

    distribution = np.identity(10)
    distribution = distribution * 5000

    for i in range(10):
        save_dirichlet(distribution, 0.0, i, folder)

    distribution.T[[0, 1]] = distribution.T[[1, 0]]
    save_dirichlet(distribution, 0.0, 0, folder)

    for a in alphas:
        for zero_label in range(10):
            data_distributions = get_dirichlet_for_class_with_zero(a, zero_label)

            save_dirichlet(data_distributions, a, zero_label, folder)


def get_dirichlet_distribution(alpha, zero_label, folder="dirichlet_distributions"):
    distribution = load_dirichlet(alpha, zero_label, folder=folder)
    return distribution.tolist()


if __name__ == "__main__":
    generate_dirichlet_distibution_with_all_zero_labels(
        alphas=[0.05, 0.1, 0.2, 0.5, 1.0, 10.0], folder="dirichlet_distributions_09"
    )
