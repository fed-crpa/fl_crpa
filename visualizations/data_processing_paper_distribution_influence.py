import mlflow
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mlflow.tracking import MlflowClient
from sklearn.metrics import ConfusionMatrixDisplay

experimentNamesMNIST = [
"2024_12_03_12:57:23_MNIST_distribution_influence", 
"2024_12_03_12:57:23_MNIST_distribution_influence",
"2024_12_03_12:57:23_MNIST_distribution_influence"
]

experimentNamesFMNIST = [
"2024_12_03_12:57:23_FMNIST_distribution_influence", 
"2024_12_03_12:57:23_FMNIST_distribution_influence",
"2024_12_03_12:57:23_FMNIST_distribution_influence"
] 

client = MlflowClient()

writeToMlFlow = False

def generate_df_for_experiments(experimentNames):

    dfComplete = pd.DataFrame()

    for experimentName in experimentNames:

        experiment_object= client.get_experiment_by_name(experimentName)
        experimentID = experiment_object._experiment_id

        runsPandas = mlflow.search_runs(experiment_ids=[experimentID])

        for runID in runsPandas["run_id"]:

            runObject = client.get_run(runID)
            runMetric = client.get_metric_history(runID, "attack_accuracy")
            
            CLASS_TO_GENERATE = int(runObject.data.params["CLASS_TO_GENERATE"])
            CLASS_TO_ATTACK = int(runObject.data.params["CLASS_TO_ATTACK"])
            data_distributions = float(runObject.data.params["distribution"])
            combination = f"gen_{CLASS_TO_GENERATE}_att_{CLASS_TO_ATTACK}"

            valuesList = [metricObject.value for metricObject in runMetric]
            avg = sum(valuesList)/len(valuesList)

            newRowDF = pd.DataFrame.from_records([{"experimentName":experimentName, "combination":combination, "alpha": data_distributions, 'avg attack accuracy':avg}])

            dfComplete = pd.concat([dfComplete,newRowDF], ignore_index=True)

    resultMetrics  = dfComplete.drop(['experimentName'], axis=1).groupby(by=["combination", "alpha"]).agg(['min', 'max', 'mean', 'std', lambda x: max(x) - min(x)])
    display = pd.options.display
    display.max_columns = 1000
    display.max_rows = 1000
    display.max_colwidth = 199
    display.width = 1000

    

    name = "FMNIST" if "FMNIST" in experimentNames[0] else "MNIST"

    if name == "MNIST":
        hue_order = ['gen_0_att_1', 'gen_8_att_4', 'gen_4_att_9'][::-1]
    else:
        hue_order = ['gen_6_att_7', 'gen_0_att_8', 'gen_7_att_5'][::-1]

    resultMetrics = resultMetrics.reset_index()

    resultMetrics.columns = [ "_".join(pair) if "" not in pair else pair[0] for pair in resultMetrics.columns ]
    print(resultMetrics)


    seaplot = sns.catplot(x="alpha", y="avg attack accuracy_mean", hue = 'combination', hue_order=hue_order, data=resultMetrics, kind="bar", legend = False)
    plt.legend(loc='center right', title = "combination")
    plt.title(name)
    plt.savefig("distribution_"+name+'.png', bbox_inches='tight')

generate_df_for_experiments(experimentNamesMNIST)
generate_df_for_experiments(experimentNamesFMNIST)

