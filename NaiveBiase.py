import pandas as pd

dtrain = pd.read_csv('train.csv')
print(dtrain)

labelCount = {}
labelsProbability = {}

label_aggregation = dtrain.groupby(dtrain.hotel_market).size()
print(label_aggregation)

for label, label_count in label_aggregation.iteritems():
            labelCount[label] = label_count
            labelsProbability[label] = float(label_count + 1) / float(len(dtrain) + 100)
            
probabilities = {}
for feature in list(dtrain.columns.values):
    print(feature)
    probabilities[feature] = {}
    print ('Calculating probability for feature: {}'.format(feature))
    # iterate over all values for that feature
    for feature_value in dtrain[feature].unique():
        probabilities[feature][feature_value] = {}
        # iterate over all class labels
        for class_label in labelCount:
            # count (feature=feature_value & class=class_value)
            feature_count = dtrain[
                        (dtrain[feature] == feature_value) &
                        (dtrain.hotel_market == class_label)] \
                    .groupby(feature).size()
            if not (len(feature_count) == 1):
                # print('feature: {}, value: {}, cluster: {}'.format(feature, feature_value, class_label))
                feature_count = 0
            else:
                feature_count = feature_count.iloc[0]
                
         # calculate probability (laplace correction)
            probability = float(feature_count + 1) / \
            float(labelCount[class_label] + len(labelCount))
            probabilities[feature][feature_value][class_label] = probability

dtest = pd.read_csv('test.csv')
dtest.drop(dtest.columns[[0]], axis=1, inplace=True)

columns = dtest.columns.values
predicted_labels = {}

# iterate through every row
for index, row in dtest.iterrows():
            max_prob = 0

            for class_label in labelsProbability:
                prob_product = 1
                for feature in columns:
                    feature_value = row[feature]
                    if (feature_value in probabilities[feature]):
                        prob_product *= probabilities[feature][feature_value][class_label]
                    else:
                        prob_product = 0

                # check if max prob, if so add to predicted_labels
                if prob_product > max_prob:
                    max_prob = prob_product
                    predicted_labels[index] = class_label

from pprint import pprint
pprint(predicted_labels)