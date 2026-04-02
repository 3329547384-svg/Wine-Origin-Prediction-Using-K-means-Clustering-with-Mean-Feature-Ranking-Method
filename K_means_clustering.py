import os
os.environ['OMP_NUM_THREADS'] = '1'
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler

# load the data
wine_data = load_wine(as_frame=True)
wine_df = wine_data.frame

features=wine_df.iloc[:,0:13]
target=wine_df.iloc[:,13]
features=np.array(features)
target=np.array(target)

# Baseline accuracy
X_original = features
y = target

kf = KFold(n_splits=3, shuffle=True, random_state=42)
accuracies_original = []

for train_index, test_index in kf.split(X_original):
    X_train, X_test = X_original[train_index], X_original[test_index]
    y_train, y_test = y[train_index], y[test_index]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k = 3  # set k value for K-means clustering classifier/model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_train_scaled)  # fit the classifier/model
    # get the center and labels
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_
    unique=np.unique(cluster_labels)
    c1=[]
    c2=[]
    c3=[]
    for i in range(len(X_train_scaled)):
        if cluster_labels[i]==unique[0]:
            c1.append(y_train[i])
        elif cluster_labels[i]==unique[1]:
            c2.append(y_train[i])
        elif cluster_labels[i]==unique[2]:
            c3.append(y_train[i])
    c1t=[c1.count(0),c1.count(1),c1.count(2)]
    c2t=[c2.count(0),c2.count(1),c2.count(2)]
    c3t=[c3.count(0),c3.count(1),c3.count(2)]
    #  true cluster labels for the clusters
    c=[c1t.index(max(c1t)),c2t.index(max(c2t)),c3t.index(max(c3t))]
    pre=kmeans.predict(X_test_scaled) #  predict the labels
    for i in range(len(X_test_scaled)): #  compare prediction with the true label
        if pre[i]==unique[0]:
              pre[i]=c[0]
        elif pre[i]==unique[1]:
              pre[i]=c[1]
        elif pre[i]==unique[2]:
              pre[i]=c[2]
    count=0
    for j in range(len(X_test)):  #  calculate the accuracy
         if pre[j]==y_test[j]:
            count=count+1
    accuracy=count/len(X_test)
    accuracies_original.append(accuracy)

print('Baseline mean accuracy with all features:', np.mean(accuracies_original))


for f in range(0,13):
    features_modified = features.copy()
    features_modified[:, f] = np.mean(features_modified[:, f])
    X = features_modified
    y = target

    # divide the data into training and testing
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        k = 3  # set k value for K-means clustering classifier/model
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_train_scaled)  # fit the classifier/model
        # get the center and labels
        cluster_centers = kmeans.cluster_centers_
        cluster_labels = kmeans.labels_
        unique=np.unique(cluster_labels)
        c1=[]
        c2=[]
        c3=[]
        for i in range(len(X_train_scaled)):
            if cluster_labels[i]==unique[0]:
                c1.append(y_train[i])
            elif cluster_labels[i]==unique[1]:
                c2.append(y_train[i])
            elif cluster_labels[i]==unique[2]:
                c3.append(y_train[i])
        c1t=[c1.count(0),c1.count(1),c1.count(2)]
        c2t=[c2.count(0),c2.count(1),c2.count(2)]
        c3t=[c3.count(0),c3.count(1),c3.count(2)]
        #  true cluster labels for the clusters
        c=[c1t.index(max(c1t)),c2t.index(max(c2t)),c3t.index(max(c3t))]
        pre=kmeans.predict(X_test_scaled) #  predict the labels
        for i in range(len(X_test_scaled)): #  compare prediction with the true label
            if pre[i]==unique[0]:
                  pre[i]=c[0]
            elif pre[i]==unique[1]:
                  pre[i]=c[1]
            elif pre[i]==unique[2]:
                  pre[i]=c[2]
        count=0
        for j in range(len(X_test_scaled)):  #  calculate the accuracy
             if pre[j]==y_test[j]:
                count=count+1
        accuracy=count/len(X_test_scaled)
        accuracies.append(accuracy)
    print('Mean accuracy without feature', wine_data.feature_names[f], 'is:', np.mean(accuracies))
