from lstm import *
from lstm import Net
import csv
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


test_date = "20220629"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 설정값
data_dim = 7
hidden_dim = 100
output_dim = 2
seq_length = 24
res = []
for n in range(27):
    print("progress:", n, ".....")
    building_num = n
    PATH = "./model3/"+str(building_num)+".pth"
    name_train_set = './data_merge/train_'+str(building_num)+'.csv'
    train_set = pd.read_csv(name_train_set)

    scaler_x = MinMaxScaler()
    scaler_x.fit(train_set.iloc[:, :-2])

    scaler_y = MinMaxScaler()
    scaler_y.fit(train_set.iloc[:, -2:])

    test_set = pd.read_csv('./predict_test_'+test_date+'.csv')
    test_set.iloc[:,:7] = scaler_x.transform(test_set.iloc[:,:7])

    testX = build_dataset3(np.array(test_set), seq_length)
    # testX = np.array(test_set)
    # print(test_set, testX)
    testX_tensor = torch.FloatTensor(testX)

    ####################################
    # name_test_set = './data_merge/test_'+str(building_num)+'.csv'
    # name_train_set = './data_merge/train_'+str(building_num)+'.csv'
    # test_set = pd.read_csv(name_test_set)
    # train_set = pd.read_csv(name_train_set)

    # ###seq

    # # print(train_set, test_set)
    # # Input scale
    # scaler_x = MinMaxScaler()
    # scaler_x.fit(train_set.iloc[:, :-2])

    # train_set.iloc[:, :-2] = scaler_x.transform(train_set.iloc[:, :-2])
    # test_set.iloc[:, :-2] = scaler_x.transform(test_set.iloc[:, :-2])

    # # Output scale
    # scaler_y = MinMaxScaler()
    # scaler_y.fit(train_set.iloc[:, -2:])

    # train_set.iloc[:, -2:] = scaler_y.transform(train_set.iloc[:, -2:])
    # test_set.iloc[:, -2:] = scaler_y.transform(test_set.iloc[:, -2:])

    # ######
    # # print(train_set, test_set)
    # trainX, trainY = build_dataset2(np.array(train_set), seq_length)
    # testX, testY = build_dataset2(np.array(test_set), seq_length)

    # # 텐서로 변환
    # trainX_tensor = torch.FloatTensor(trainX).to(device)
    # trainY_tensor = torch.FloatTensor(trainY).to(device)

    # testX_tensor = torch.FloatTensor(testX).to(device)
    # testY_tensor = torch.FloatTensor(testY).to(device)
    # # print(testX_tensor, testY_tensor)

    ####################################


    # print(testX_tensor)

    model = Net(data_dim, hidden_dim, seq_length, output_dim, 1)
    model.load_state_dict(torch.load(PATH), strict=False)

    alpha = 0
    with torch.no_grad(): 
        pred = []
        for pr in range(len(testX_tensor)):

            model.reset_hidden_state()

            predicted = model(torch.unsqueeze(testX_tensor[pr].cpu(), 0))
            predi = [t.numpy() for t in predicted.cpu()]
            pred.append(predi)
        
        pred_inverse = np.round(scaler_y.inverse_transform(np.array(pred).reshape(-1, 2)),1)
        # testY_inverse = np.round(scaler_y.inverse_transform(testY_tensor.cpu()),1)
        # for v in range(len(pred_inverse)):
        #     if pred_inverse[0][v][0]
        res.append(pred_inverse)

        # print(pred_inverse, testY_inverse)
        # pred_inverse = pred_inverse+[-40,140]
        # print('\tDIFF SCORE_inv : ', diff(pred_inverse, testY_inverse))
        # print('\tMAE SCORE_inv : ', MAE(pred_inverse, testY_inverse))
        # print('\tMSE SCORE_inv : ', MSE(pred_inverse, testY_inverse))
# print(np.array(res))
print(len(res), len(res[0]),len(res[0][0]))
file1 = open('./result_power.csv', 'a', encoding='utf-8', newline='')
file2 = open('./result_energy.csv', 'a', encoding='utf-8', newline='')
wr1 = csv.writer(file1)
wr2 = csv.writer(file2)
for i in range(24):
    line1 = []
    line2 = []
    for x in range(27):
        line1.append(res[x][i][0])
        line2.append(res[x][i][1])
    wr1.writerows([line1])
    wr2.writerows([line2])
