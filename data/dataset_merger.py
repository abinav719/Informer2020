import numpy as np
import pandas as pd

file_name = ["ADMA_OBD_domain_train_lowspeedInformer_dataset_file.csv" , "ADMA_OBD_Test_domainshiftInformer_dataset_file.csv",
             "ADMA_OBD_2ndInformer_dataset_file.csv", "ADMA_OBD_3rdInformer_dataset_file.csv",
             "ADMA_OBD_4thInformer_dataset_file.csv","ADMA_OBD_5thInformer_dataset_file.csv"]

concatenated_df = pd.read_csv(file_name[0])
for i in range(1,len(file_name)):
    df1 = pd.read_csv(file_name[i])

    # Concatenate the dataframes based on columns
    concatenated_df = pd.concat([concatenated_df, df1], axis=0, ignore_index=True)

concatenated_df=concatenated_df.dropna()
# Save the concatenated dataframe to a new CSV file
concatenated_df.to_csv("Informer_dataset_file_fivehourdataset.csv", index=False)