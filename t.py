import pandas as pd
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae


result = 'submission.csv'
print("Input path to .csv with train data")
path_to_train = input()
data = pd.read_csv(path_to_train)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'TotRmsAbvGrd', 'FullBath', 'OverallQual', 'GarageCars', 'GarageArea']

x = data[features]
Y = data['SalePrice']

train_IN, valid_IN, train_OUT, valid_OUT = tts(x, Y, train_size=0.8)

def get_optimal_leaf_nodes(train_IN, train_OUT, valid_IN, valid_OUT) -> int:
    optimal_leaf = 10
    leaf_nodes = [10, 50, 90, 150, 210, 300]
    mae_min = -1.0
    for leaf in leaf_nodes:
        model = rfr(max_leaf_nodes=leaf, random_state=1)
        model.fit(train_IN, train_OUT)
        predicted = model.predict(valid_IN)
        error = mae(valid_OUT, predicted)
        if leaf == leaf_nodes[0]:
            mae_min = error
        if error < mae_min:
            mae_min = error
            optimal_leaf = leaf
    return optimal_leaf

model = rfr(random_state=1, max_leaf_nodes=get_optimal_leaf_nodes(train_IN=train_IN, train_OUT=train_OUT, valid_IN=valid_IN, valid_OUT=valid_OUT))
model.fit(train_IN, train_OUT)
predicted = model.predict(valid_IN)
# print("Average error is about {.1f}".format(data.describe()))
# print(data.describe())
print(mae(valid_OUT, predicted))

real_life_data = pd.read_csv('test.csv')
rld_predicted = model.predict(real_life_data[features])
output = pd.DataFrame(
                        {'Id': real_life_data['Id'],
                        'SalePrice': rld_predicted}
                    )
output.to_csv(result, index=False)

print("your data has been written successfully to {}".format(result))