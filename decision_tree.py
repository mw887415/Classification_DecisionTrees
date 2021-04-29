from sklearn import tree


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# model: decision tree classifierfds
clf_decision_tree = tree.DecisionTreeClassifier()
#and train them on our data set
clf_decision_tree = clf_decision_tree.fit(X, Y)

prediction =   clf_decision_tree.predict([[190, 70, 43]])
prediction_2 =  clf_decision_tree.predict([[153,55,39]])

print(prediction)
print(prediction_2)
