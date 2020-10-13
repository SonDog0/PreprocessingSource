import pandas as pd
import sys

import warnings
warnings.filterwarnings(action='ignore')


treatment_column_for_drop = "tx_name"
drug_column_for_drop = "rx_name"

full_data = 'data/0806_pet_input_joined_dataset.csv'
stat_data = 'data/0806_simple_statistics_dev.xlsx'

dog_disease_list = 'data/dog_disease_list.csv'
dog_symptom_list = 'data/dog_symptom_list.csv'

dog_disease_drug_data ='data/dog_disease_drug_data.csv'
dog_disease_treatment_data = 'data/dog_disease_treatment_data.csv'

df = pd.read_csv("featureselection/FeatureSelection_Rx_Dx_Mul.csv" ,encoding='CP949')

drop_list = [
    "birth",
    "species_code",
    "breed",
    "lifecycle",
    "sex",
    # "chart_origin_num",
    "info_origin_num",
    "isDisease",
    # "treatment_name"

    "main_category",
    "breed_size",
    "isTreatment",
    "charted"
]

### SAPRK SETTING

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

sc = SparkContext(appName = "spark_rf")

spark = SparkSession.Builder().getOrCreate()

def get_drug_from_pet(file, drop, for_drop):
    data = pd.read_csv(file, sep="," , encoding='CP949')
    data = data[data["isDisease"] == 0]
    if for_drop == drug_column_for_drop:
        data = data[data["tx_name"] != "unknown"]
    elif for_drop == treatment_column_for_drop:
        data = data[data["rx_name"] != "unknown"]

    data = data.drop(drop_list, axis=1)
    data = data.drop(for_drop, axis=1)
    data = data.dropna(axis=0)
    data = data.drop_duplicates()
    data.to_csv(drop, index=False, encoding='CP949')

def get_disease_from_pet(input ,DS):
    if DS == 'D':
        data = pd.read_excel(input, sheet_name='dog_dx_disease')
        data.to_csv('dog_disease_list.csv' , encoding='CP949' , index=False)
        print(data)
    elif DS == 'S':
        data = pd.read_excel(input, sheet_name='dog_dx_symptom')
        data.to_csv('dog_symptom_list.csv' , encoding='CP949' , index=False)
        print(data)
def reshape_data(file, i, j):
    data = pd.read_csv(file, encoding='CP949')

    # disease_list_stat = pd.read_csv(r"c:/Users/Gaion/Desktop/python-ml/data/dog/dog_list_disease.csv")
    disease_list_stat = pd.read_csv(dog_disease_list, encoding='CP949')

    # start 1 => ignore 'unknown'
    disease_list_stat_top = disease_list_stat.loc[0:i, :]['dx'].values

    impotant_rx_list = (i / 10) - 1
    impotant_rx_list = df.iloc[int(impotant_rx_list), :][:j].tolist()

    data = data[data["dx"].isin(disease_list_stat_top)]
    data = data[data["rx_name"].isin(impotant_rx_list)]

    column_list = pd.factorize(data['rx_name'])
    data['dx_factorize'] = pd.factorize(data['dx'])[0]

    reshape = data.groupby(['chart_origin_num', 'dx_factorize', 'rx_name']).size().unstack(fill_value=0)
    reshape = reshape.reset_index()

    cols = reshape.columns
    for k in range(2, len(cols)):
        reshape.loc[reshape[cols[k]] > 0, cols[k]] = 1

    reshape.to_csv('Input_Dog_D_Rx_10_10.csv' , encoding='CP949' , index =False)

    return reshape, column_list



if __name__ == "__main__":
    pd.set_option('display.max_columns' , None)
    # get_drug_from_pet(full_data , 'dog_disease_treatment_data.csv' , drug_column_for_drop)
    #
    # df = pd.read_csv('dog_disease_treatment_data.csv', encoding='CP949')

    # df = get_disease_from_pet(stat_data ,DS='D')
    # print(df)

    column_names = ["Disease ", "Rx", "Accuracy"]
    result = pd.DataFrame(columns=column_names)

    # for i in range(10, 60, 10):
    #     for j in range(10, 60, 10):
    i = 10
    j = 10
    data, drug_list = reshape_data(dog_disease_drug_data, i, j)


    data = spark.createDataFrame(data)

    # print(data.limit(5).toPandas())

    data_columns = data.columns[2:]

    # print(data.printSchema)

    from pyspark.ml.feature import VectorAssembler

    assembler = VectorAssembler() \
        .setInputCols(data_columns) \
        .setOutputCol("features")

    dataToVec = assembler.transform(data)

    # print(train_mod01.limit(2).toPandas())

    train = dataToVec.select("features", "dx_factorize")

    # print(train_mod02.limit(2).toPandas())

    from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel

    rfClassifer = RandomForestClassifier(labelCol="dx_factorize", numTrees=100)

    from pyspark.ml import Pipeline

    pipeline = Pipeline(stages=[rfClassifer])

    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    paramGrid = ParamGridBuilder() \
        .addGrid(rfClassifer.maxDepth, [12]) \
        .addGrid(rfClassifer.minInstancesPerNode, [20]) \
        .build()


    evaluator = MulticlassClassificationEvaluator(labelCol="dx_factorize", predictionCol="prediction",
                                                  metricName="accuracy")
    # evaluator_f1 = MulticlassClassificationEvaluator(labelCol="dx_factorize", predictionCol="prediction",
    #                                               metricName="f1")


    crossval= CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5)

    cvModel = crossval.fit(train)

    print(i , j , cvModel.avgMetrics)

    newlist = [i , j , cvModel.avgMetrics]
    result.loc[len(result)] = newlist


    result.to_csv('RxDx_D_RF_acc.csv', encoding='CP949' , index= False)

    sys.exit()



