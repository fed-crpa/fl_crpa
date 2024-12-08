import mlflow
import numpy as np
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd

experimentName = "2024_11_08_15:36:14_MNIST_attack_combination"
metricName = "attack_accuracy"
client = MlflowClient()
runName = "gen_4_att_9"

experiment_object= client.get_experiment_by_name(experimentName)
experimentID = experiment_object._experiment_id

runsPandas = mlflow.search_runs(experiment_ids=[experimentID])

runsPandas = runsPandas.loc[runsPandas['tags.mlflow.runName'] == runName]

print(runsPandas)

for runID in runsPandas["run_id"]:

    runObject = client.get_run(runID)
    runMetric = client.get_metric_history(runID, metricName)
    
    valuesList = [metricObject.value for metricObject in runMetric]
    epochsList = [x for x in range(len(valuesList))]
    
    df = pd.DataFrame(list(zip(epochsList, valuesList)),
               columns =['epoch', 'attack_accuracy'])

    df['attack_accuracy_ewm'] = df.attack_accuracy.ewm(alpha=0.01).mean()

    sns.set_context("talk")
    plt.figure(figsize=(9,6))
    sns.lineplot(x="epoch", y="attack_accuracy", 
                data=df, label="attack accuracy")

    sns.lineplot(x="epoch", y="attack_accuracy_ewm", data=df, label="attack accuracy EMA")

    plt.xlabel("epoch")
    plt.ylabel("attack accuracy")
    plt.tight_layout() 
    plt.savefig("{}_attack_accuracy_{}.png".format(experimentName, runName))

