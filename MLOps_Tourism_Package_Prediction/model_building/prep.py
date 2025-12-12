# for data manipulation
import pandas as pd
import numpy as np
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
# api = HfApi(token=os.getenv("HF_TOKEN"))
api = HfApi(token=os.getenv("HF_TOKEN_MLOPSTPP"))
DATASET_PATH = "hf://datasets/PPathakH18/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Convert the Categorical columns to Category Datatype
# ----------------------------------------------------

# 1.creating list of category columns that are not object type
cat_cols = ["CityTier","ProdTaken","NumberOfPersonVisiting","NumberOfChildrenVisiting","PreferredPropertyStar","Passport","PitchSatisfactionScore","OwnCar"]
df[cat_cols] = df[cat_cols].astype("category")

# 2.selecting all object datatypes and converting to category
cols = df.select_dtypes(["object"])
for i in cols.columns:
    df[i] = df[i].astype("category")

# 3.check the dataset for updated datatypes
df.info()


# Check the missing values
#--------------------------

# 1....check number of null records
df.isna().sum()

# 2....Treat Age and MonthlyIncome for missing values
# replace the missing values with median income w.r.t the customer"s designation
df["MonthlyIncome"] = df.groupby(["Designation"])["MonthlyIncome"].transform(lambda x: x.fillna(x.median()))
df["Age"] = df.groupby(["Designation"])["Age"].transform(lambda x: x.fillna(x.median()))

# 3....Treat other numerical columns for missing values
# create list of numerical columns
missing_numerical = df.select_dtypes(include=np.number).columns.tolist()

# remove Age and MonthlyIncome as we have already treated these columns
missing_numerical.remove("MonthlyIncome")
missing_numerical.remove("Age")

# function for replacing with the Median value of the attributes
medianFiller = lambda x: x.fillna(x.median())

# apply the function
df[missing_numerical] = df[missing_numerical].apply(medianFiller,axis=0)

# 4....Check the data for category columns
#create a list of categorical columns
cat_cols =  df.select_dtypes(["category"])

#get the valuecounts
for i in cat_cols.columns:
    print(cat_cols[i].value_counts())
    print("-"*50)
    print("\n")

# 5....Treat the other columns for missing values
# treating missing values in remaining categorical variables
df["TypeofContact"] = df["TypeofContact"].fillna("Self Enquiry")
df["NumberOfChildrenVisiting"] = df["NumberOfChildrenVisiting"].fillna(1.0)
df["PreferredPropertyStar"] = df["PreferredPropertyStar"].fillna(3.0)

# 6....treating error in Gender Column
df.Gender = df.Gender.replace("Fe Male","Female")

# verify the update
df.Gender.value_counts()

# Data Verification: Since the data has been treated for missing values, error and also converted for categorical etc.
#----------------------------------------------------------------------------------------------------------------------

# 1.summary of numerical columns
df.describe().T

# 2.summary of categorical columns
df.describe(include="category").T


# Customer Interaction data("PitchSatisfactionScore","ProductPitched","NumberOfFollowups","DurationOfPitch") is not relevant for our analysis and we will ignore.
# Also CustomerID column is not required
# "ProdTaken" is the target(y)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# split the columns in Dependent and Independent Set

# 1....Split into X (features) and y (target)
X = df.drop(["CustomerID","ProdTaken","PitchSatisfactionScore","ProductPitched","NumberOfFollowups","DurationOfPitch"],axis=1)
y = df["ProdTaken"]

# Encoding the categorical 'Type' column
# label_encoder = LabelEncoder()
# df['Type'] = label_encoder.fit_transform(df['Type'])

# 2....use get_dummies function to convert the categorical columns
# X = pd.get_dummies(X, drop_first=True)


# 3....Perform train-test split, use stratify to maintain the original distribution of Dependent variable as of original set
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4....creating a list of column names
feature_names = Xtrain.columns.to_list()

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="PPathakH18/Tourism-Package-Prediction",
        repo_type="dataset",
    )
