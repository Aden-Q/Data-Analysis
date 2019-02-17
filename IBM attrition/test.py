import tensorflow as tf
import numpy as np
import pandas as pd
import io
import os
import requests
import math
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

model_file = './model/model'

def feature_normalize(dataset):
    '''
    特征均一化
    '''
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def str_to_int(df):
    str_columns = df.select_dtypes(['object']).columns
    print(str_columns)
    for col in str_columns:
        df[col] = df[col].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def count_space_except_nan(x):
    if isinstance(x,str):
        return x.count(" ") + 1
    else :
        return 0
    
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        del df[each]
        df = pd.concat([df, dummies], axis=1)
    return df


df_train = pd.read_csv('/Users/qianzecheng/Downloads/contest/train.csv')
###
df_test = pd.read_csv('/Users/qianzecheng/Downloads/contest/test.csv')
###
delete_columns = ["Unnamed: 0","Name", "YearofBirth", "X", "Over18", "EmployeeCount", "StandardHours", "EmployeeNumber", "PerformanceRating"]   # waiting to be modified


def pre_processing(df):
    df.drop(delete_columns, axis=1, inplace=True)
    # Count room nubmer
    # df_train["Cabin"] = df_train["Cabin"].apply(count_space_except_nan)
    # Replace NaN with mean value
    df["DistanceFromHome"].fillna(df["DistanceFromHome"].mean(), inplace=True)
    df["LastYearTrainingTime"].fillna(df["LastYearTrainingTime"].mean(), inplace=True)
    # EducationLevel, Embarked one-hot
    df = one_hot(df, df.loc[:, ["EducationLevel"]].columns)
    df = one_hot(df, df.loc[:, ["EducationField"]].columns)
    df = one_hot(df, df.loc[:, ["Travel_For_Business"]].columns)
    df = one_hot(df, df.loc[:, ["MaritalStatus"]].columns)
    df = one_hot(df, df.loc[:, ["EnvironmentSatisfaction"]].columns)
    df = one_hot(df, df.loc[:, ["JobInvolvement"]].columns)
    df = one_hot(df, df.loc[:, ["JobSatisfaction"]].columns)
    df = one_hot(df, df.loc[:, ["RelationshipSatisfaction"]].columns)
    df = one_hot(df, df.loc[:, ["WorkLifeBalance"]].columns)
    df = one_hot(df, df.loc[:, ["JobLevel"]].columns)
    df = one_hot(df, df.loc[:, ["JobRole"]].columns)
    df = one_hot(df, df.loc[:, ["StockOptionLevel"]].columns)
    df = one_hot(df, df.loc[:, ["LastYearTrainingTime"]].columns)
    df = one_hot(df, df.loc[:, ["Gender"]].columns)
    df = one_hot(df, df.loc[:, ["Department"]].columns)
    df = one_hot(df, df.loc[:, ["OverTime"]].columns)

    # String to int
    df = str_to_int(df)
    # Age Normalization
    df["DailyRate"] = feature_normalize(df["DailyRate"])
    df["HourlyRate"] = feature_normalize(df["HourlyRate"])
    df["MonthlyRate"] = feature_normalize(df["MonthlyRate"])
    df["Age"] = feature_normalize(df["Age"])
    df["DistanceFromHome"] = feature_normalize(df["DistanceFromHome"])
    df["PercentSalaryHike"] = feature_normalize(df["PercentSalaryHike"])
    df["YearsAtCompany"] = feature_normalize(df["YearsAtCompany"])
    df["YearsWithCurrManager"] = feature_normalize(df["YearsWithCurrManager"])
    df["YearsSinceLastPromotion"] = feature_normalize(df["YearsSinceLastPromotion"])
    df["MonthlyIncome"] = feature_normalize(df["MonthlyIncome"])
    df["TotalWorkingYears"] = feature_normalize(df["TotalWorkingYears"])
    df["YearsInCurrentRole"] = feature_normalize(df["YearsInCurrentRole"])
    df["NumCompaniesWorked"] = feature_normalize(df["NumCompaniesWorked"])
    
    # stats.describe(df).variance
    return df

df_train = pre_processing(df_train)
df_train.to_csv('new.csv', index=False)
test_JobID = df_test["JobID"]
df_test = pre_processing(df_test)
features = df_train.iloc[:, 2:].values
labels = df_train.iloc[:, 1:2].values # 1100*1
rnd_indices = np.random.rand(len(features)) < 0.80 # margin waiting to be modified

train_x = features[rnd_indices]
train_y = labels[rnd_indices]

# Smote for strengthen
# sm = SMOTE(random_state=2)
# train_x, train_y = SMOTE().fit_resample(train_x, train_y.ravel())
# train_y = train_y.reshape(train_y.shape[0], 1)

real_test_x = df_test.iloc[:, 1:].values
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]

feature_count = train_x.shape[1]
label_count = train_y.shape[1]



# inputs
training_epochs = 50000
learning_rate = 1e-6
hidden_layers = feature_count-2
cost_history = np.empty(shape=[1],dtype=float)
loss_history = np.empty(shape=[1],dtype=float)
loss_history_test = np.empty(shape=[1],dtype=float)
test_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,feature_count])
Y = tf.placeholder(tf.float32,[None,label_count])
is_training=tf.Variable(True,dtype=tf.bool)


# models
initializer = tf.contrib.layers.xavier_initializer()
h0 = tf.layers.dense(X, hidden_layers, activation=tf.nn.relu, kernel_initializer=initializer)
# h0 = tf.layers.dense(X, hidden_layers, activation=None, kernel_initializer=initializer)
h0 = tf.nn.dropout(h0, 0.90)
h1 = tf.layers.dense(h0, label_count, activation=None)

# 正则化
# regularizer = tf.contrib.layers.l2_regularizer(scale=5.0/20000)
# reg_term = tf.contrib.layers.apply_regularization(regularizer)

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h1)
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h1)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)
#prediction = tf.argmax(h0, 1)
#correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predicted = tf.nn.sigmoid(h1)
correct_pred = tf.equal(tf.round(predicted), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# session

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={
                                 X: train_x, Y: train_y})
        cost_history = np.append(cost_history, acc)
        loss_history = np.append(loss_history, loss)
        loss_history_test = np.append(loss_history_test, loss)
        if step % 500 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
            
        if step % 200 == 0:
            acc, tt= sess.run([accuracy, tf.round(predicted)], feed_dict={X: test_x, Y: test_y})
            test_history = np.append(test_history, acc)
            if step % 400 == 0:
              	print("Testing:   Step: {:5}\tAcc: {:.2%}".format(
                   step, acc))
            
    # Test model and check accuracy
    print('Test Accuracy:', sess.run([accuracy, tf.round(predicted)], feed_dict={X: test_x, Y: test_y}))
    
    # Save test result
    test_predict_result = sess.run(tf.cast(tf.round(predicted), tf.int32), feed_dict={X: real_test_x})
    evaluation = test_JobID.to_frame()
    evaluation["Attrition"] = test_predict_result
    evaluation.to_csv('result.csv', index=False)

    saver.save(sess, model_file)



cost_history = list(cost_history[1:])
loss_history = list(loss_history[1:])
test_history = list(test_history[1:])
f_train = open("train.txt", "w")
f_loss_train = open("loss_train.txt", "w")
f_test = open("test.txt", "w")
f_loss_test = open("loss_test.txt", "w")

for num in cost_history:
    f_train.write(str(num))
    f_train.write('\n')
    
for num in test_history:
    f_test.write(str(num))
    f_test.write('\n')

for num in loss_history:
    f_loss_train.write(str(num))
    f_loss_train.write('\n')

for num in loss_history_test:
    f_loss_test.write(str(num))
    f_loss_test.write('\n')
