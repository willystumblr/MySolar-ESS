import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader

###### 데이터셋 생성 함수
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i:i+seq_length, :]
        _y = time_series[i+seq_length, [-1]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)

def build_dataset2(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i:i+seq_length, :-2]
        _y = time_series[i+seq_length, [-2, -1]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)

#######
class Net(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias = True) 
        
    # 학습 초기화를 위한 함수
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                torch.zeros(self.layers, self.seq_len, self.hidden_dim))
    
    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

def train_model(model, train_df, num_epochs = None, lr = None, verbose = 10, patience = 2):
    
    criterion = nn.MSELoss().to(device)
    # criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    nb_epochs = num_epochs
    
    # epoch마다 loss 저장
    train_hist = np.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        avg_cost = 0
        total_batch = len(train_df)
        
        for batch_idx, samples in enumerate(train_df):

            x_train, y_train = samples
            # seq별 hidden state reset
            model.reset_hidden_state()
            
            # H(x) 계산
            outputs = model(x_train)
            # print(outputs)
            # cost 계산
            loss = criterion(outputs, y_train)                    
        
            # cost로 H(x) 개선
            optimizer.zero_grad()
            loss.backward()
            
            # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

            optimizer.step()
            
            avg_cost += loss/total_batch
            
        train_hist[epoch] = avg_cost        
        
        if epoch % verbose == 0:
            print('\tEpoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
            
        # patience번째 마다 early stopping 여부 확인
        if (epoch % patience == 0) & (epoch != 0):
            # loss가 커졌다면 early stop
            if train_hist[epoch-patience] < train_hist[epoch]:
                print('\t => Early Stopping')
                break
            
    return model.eval(), train_hist

####
def MAE(true, pred):
    return np.mean(np.abs(true-pred))
def MSE(true, pred):
    return np.mean((true-pred)**2)

def All_train(num):
    # 데이터 불러오기
    name_test_set = './data_merge/test_'+num+'.csv'
    name_train_set = './data_merge/train_'+num+'.csv'
    test_set = pd.read_csv(name_test_set)
    train_set = pd.read_csv(name_train_set)

    ###seq

    # print(train_set, test_set)
    # Input scale
    scaler_x = MinMaxScaler()
    scaler_x.fit(train_set.iloc[:, :-2])

    train_set.iloc[:, :-2] = scaler_x.transform(train_set.iloc[:, :-2])
    test_set.iloc[:, :-2] = scaler_x.transform(test_set.iloc[:, :-2])

    # Output scale
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_set.iloc[:, -2:])

    train_set.iloc[:, -2:] = scaler_y.transform(train_set.iloc[:, -2:])
    test_set.iloc[:, -2:] = scaler_y.transform(test_set.iloc[:, -2:])

    ######
    # print(train_set, test_set)
    trainX, trainY = build_dataset2(np.array(train_set), seq_length)
    testX, testY = build_dataset2(np.array(test_set), seq_length)

    # 텐서로 변환
    trainX_tensor = torch.FloatTensor(trainX).to(device)
    trainY_tensor = torch.FloatTensor(trainY).to(device)

    testX_tensor = torch.FloatTensor(testX).to(device)
    testY_tensor = torch.FloatTensor(testY).to(device)
    # print(testX_tensor, testY_tensor)
    # 텐서 형태로 데이터 정의
    dataset = TensorDataset(trainX_tensor, trainY_tensor)

    # 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
    dataloader = DataLoader(dataset,
                            batch_size=batch,
                            shuffle=True,  
                            drop_last=True)

    ######
    
    # 모델 학습
    net = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)  
    model, train_hist = train_model(net, dataloader, num_epochs = nb_epochs, lr = learning_rate, verbose = 10, patience = 10)

    # epoch별 손실값
    fig = plt.figure(figsize=(10, 4))
    title1 = "Training loss_"+num
    plt.plot(train_hist, label=title1)
    plt.legend()
    # plt.show()
    title1 = './result/'+title1
    plt.savefig(title1)

    # 모델 저장    
    PATH = "./model2/"+num+".pth"
    torch.save(model.state_dict(), PATH)

    # 불러오기
    model = Net(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)  
    model.load_state_dict(torch.load(PATH), strict=False)
    model.eval()

    # 예측 테스트
    with torch.no_grad(): 
        pred = []
        for pr in range(len(testX_tensor)):

            model.reset_hidden_state()

            predicted = model(torch.unsqueeze(testX_tensor[pr], 0))
            # predicted = torch.flatten(predicted[0]).item()
            predi = [t.numpy() for t in predicted.cpu()]
            pred.append(predi)

        # INVERSE
        pred_inverse = np.round(scaler_y.inverse_transform(np.array(pred).reshape(-1, 2)),1)
        testY_inverse = np.round(scaler_y.inverse_transform(testY_tensor.cpu()),1)
        print('\t', pred_inverse, testY_inverse)

        print('\tMAE SCORE : ', MAE(np.array(pred).reshape(-1, 2), np.array(testY_tensor.cpu())))
        print('\tMSE SCORE : ', MSE(np.array(pred).reshape(-1, 2), np.array(testY_tensor.cpu())))
    ######
    print('\tMAE SCORE_inv : ', MAE(pred_inverse, testY_inverse))
    print('\tMSE SCORE_inv : ', MSE(pred_inverse, testY_inverse))

    fig = plt.figure(figsize=(8,3))
    plt.plot(np.arange(len(pred_inverse)), pred_inverse, label = 'pred')
    plt.plot(np.arange(len(testY_inverse)), testY_inverse, label = 'true')
    title2 = "Loss plot_"+num
    plt.title(title2)
    # plt.show()
    title2 = './result/'+title2
    plt.savefig(title2)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 설정값
    data_dim = 6
    hidden_dim = 150
    output_dim = 2
    learning_rate = 0.0001
    nb_epochs = 150

    # -일간의 데이터가 입력으로 들어가고 batch size는 임의로 지정
    # ,date-time,day,temp,humid,active_power,active_energy
    seq_length = 24*7*5
    batch = 100

    for i in range(0,27):
        print("training building ", i,".....")
        All_train(str(i))