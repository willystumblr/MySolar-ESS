from lstm import *
from lstm import Net
# date = [2022,8,1,'Mon']
# temperature = [26,26.5, 27.2, 27.3,27.2,26.4,25.8,25.7,26.9,27.9,28.9,29.1,28.3,29.1,28.9,29,28.8,28.7,28.4,28.3,27.9,27.5,27.4,26.7]
# humidity = [99,99,99,99,99,99,99,99,99,99,97,97,97,93,95,97,97,97,98,98,99,99,98,99]
# print(len(temperature), len(humidity))

def build_dataset3(time_series, seq_length):
    dataX = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i:i+seq_length, :]
        # print(_x, "-->",_y)
        dataX.append(_x)
    if dataX ==[]:
        dataX.append(time_series[:,:])
    return np.array(dataX)

building_num = 0
test_date = "0729"
test_date = "0620"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 설정값
data_dim = 8
hidden_dim = 128
output_dim = 2
learning_rate = 0.0001
nb_epochs = 150
seq_length = 1
batch = 128

PATH = "./model/"+str(building_num)+".pth"
name_train_set = './data_merge/train_'+str(building_num)+'.csv'
train_set = pd.read_csv(name_train_set)

scaler_x = MinMaxScaler()
scaler_x.fit(train_set.iloc[:, :-2])

scaler_y = MinMaxScaler()
scaler_y.fit(train_set.iloc[:, -2:])

test_set = pd.read_csv('./predict_test_'+test_date+'.csv')
test_set.iloc[:,:6] = scaler_x.transform(test_set.iloc[:,:6])

testX = build_dataset3(np.array(test_set), seq_length)
# testX = np.array(test_set)
# print(test_set, testX)
testX_tensor = torch.FloatTensor(testX)

# print(testX_tensor)

model = Net(data_dim, hidden_dim, seq_length, output_dim, 1)
model.load_state_dict(torch.load(PATH), strict=False)

res = []
with torch.no_grad(): 
    pred = []
    for pr in range(len(testX_tensor)):

        model.reset_hidden_state()

        predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
        predi = [t.numpy() for t in predicted.cpu()]
        pred.append(predi)

    pred_inverse = np.round(scaler_y.inverse_transform(np.array(pred).reshape(-1, 2)),1)
    res.append(pred_inverse)
print(res, len(res[0]))