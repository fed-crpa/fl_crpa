import mlflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay

experimentNamesMNIST = [
    "2024_12_03_12:57:23_MNIST_attack_combination",
    "2024_12_03_12:57:23_MNIST_attack_combination",
    "2024_12_03_12:57:23_MNIST_attack_combination",
]

experimentNamesFMNIST = [
    "2024_12_03_12:57:23_FMNIST_attack_combination",
    "2024_12_03_12:57:23_FMNIST_attack_combination",
    "2024_12_03_12:57:23_FMNIST_attack_combination",
]

client = MlflowClient()

writeToMlFlow = False


def generate_cm_for_experiments(experimentNames):

    indexNames = []
    for gen in range(10):
        for att in range(10):
            if gen != att:
                indexNames.append(f"gen_{gen}_att_{att}")

    dfComplete = pd.DataFrame(index=indexNames, columns=experimentNames)

    for experimentName in experimentNames:

        experiment_object = client.get_experiment_by_name(experimentName)
        experimentID = experiment_object._experiment_id

        runsPandas = mlflow.search_runs(experiment_ids=[experimentID])

        for runID in runsPandas["run_id"]:

            runObject = client.get_run(runID)
            runMetric = client.get_metric_history(runID, "attack_accuracy")

            CLASS_TO_GENERATE = int(runObject.data.params["CLASS_TO_GENERATE"])
            CLASS_TO_ATTACK = int(runObject.data.params["CLASS_TO_ATTACK"])

            valuesList = [metricObject.value for metricObject in runMetric]

            dfComplete.at[
                f"gen_{CLASS_TO_GENERATE}_att_{CLASS_TO_ATTACK}", experimentName
            ] = sum(valuesList) / len(valuesList)

    dfComplete["mean"] = dfComplete[experimentNames].mean(axis=1)
    dfComplete["std"] = dfComplete[experimentNames].std(axis=1)

    df = dfComplete.copy()

    df["mean_diff"] = (dfComplete["mean"] - dfComplete["mean"].mean()).abs()

    display = pd.options.display
    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = 1000

    print(df)
    print(df.mean(axis=0))

    att_acc_mean_list = np.zeros((10, 10))
    for index, row in dfComplete.iterrows():
        gen = int(index[4])
        att = int(index[10])
        att_acc_mean_list[gen, att] = row["mean"]

    disp = ConfusionMatrixDisplay(confusion_matrix=att_acc_mean_list)
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel("poisoned label")
    plt.ylabel("reconstructed label")

    if "FMNIST" in experimentNames[0]:
        plt.savefig("FMNIST_confucion_matrix.png")
    else:
        plt.savefig("MNIST_confucion_matrix.png")

    np.fill_diagonal(att_acc_mean_list, -1)
    att_acc_mean_list = att_acc_mean_list.reshape((-1, 1))
    att_acc_mean_list = np.delete(att_acc_mean_list, np.where(att_acc_mean_list == -1))
    return att_acc_mean_list


att_acc_MNIST = generate_cm_for_experiments(experimentNamesMNIST)
att_acc_FMNIST = generate_cm_for_experiments(experimentNamesFMNIST)

df_MNIST = pd.DataFrame(data=att_acc_MNIST, columns=["att_acc"]).assign(dataset="MNIST")
df_FMNIST = pd.DataFrame(data=att_acc_FMNIST, columns=["att_acc"]).assign(
    dataset="FMNIST"
)

df_avg_data = pd.concat([df_MNIST, df_FMNIST], ignore_index=True)

plt.close("all")
sns.set()
sns_ax = sns.histplot(data=df_avg_data, x="att_acc", hue="dataset", kde=True)
sns_ax.set(xlabel="average attack accuracy", ylabel="count")

plt.savefig("histrogram.png")
