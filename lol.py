import pandas as pd

data_set = pd.read_csv('banana_quality.csv')
# print(data_set.shape)

print(data_set.columns)

y = data_set['Sweetness']
parameters = ['HarvestTime', 'Ripeness', 'Softness', 'Size', 'Weight']
X = data_set[parameters]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

def getOptimazedDepth() -> int:
    depth = [3,4,5,6,7]
    min_mae = 1.0
    optimal_depth = depth[0]
    for dv in depth:
        model = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=dv)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        result = [1.0 if banana > 0.5 else 0.0 for banana in predicted]
        mae = mean_absolute_error(y_test, result)
        if mae < min_mae:
            min_mae = mae
            optimal_depth = dv
    return optimal_depth

depth = getOptimazedDepth()

def getMaxLeafNodes() -> int:
    leaf = [10, 30, 50, 100, 300, 200, 80, 500]
    min_mae = 1.0
    optimal_leaf = leaf[0]
    for dv in leaf:
        model = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=depth, max_leaf_nodes=dv)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        result = [1.0 if banana > 0.5 else 0.0 for banana in predicted]
        mae = mean_absolute_error(y_test, result)
        if mae < min_mae:
            min_mae = mae
            optimal_leaf = dv
    return optimal_leaf

leaf = getMaxLeafNodes()

model = RandomForestRegressor(n_estimators=100, random_state=1, max_depth=depth, max_leaf_nodes=leaf)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
result = [1.0 if banana > 0.5 else 0.0 for banana in predicted]
mae = mean_absolute_error(y_test, result)

print("mistake %", mae)
print("depth", depth)
print("leaf", leaf)