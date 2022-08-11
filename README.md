# MySolar-ESS
GIST 창의융합경진대회

Reference 
* https://github.com/shubhamchouksey/Power-Prediction-LSTM
## EDA
### 1. Solar Power Prediction
path: `solar_power_prediction/SolarPower EDA.ipynb` \
기상청 시간단위 데이터 중, 시간당발전량(kWh)와 상관관계가 높게 나타났던 '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)', '일조(hr)', '일사(MJ/m2)', 'Cloud', 'Hour' 등을 feature로 활용하고, 이와 비슷한 정도의 상관관계를 보이던 '외기온도', '모듈온도' 등의 feature는 overfitting 방지를 위해 feature로 활용하지 않음.
* categorical features: 'Cloud', 'Hour'
  * Cloud: 구름의 양을 1~10으로 나타내는 전운량(10분위)를 그대로 활용하고자 했으나, 약 2년치 학습데이터에는 1~10의 수치가 모두 나타나는 반면 22/6/29의 target 데이터에는 5~10 까지만 나타나 있어 그대로 활용하기 어렵다고 판단, 따라서 전운량 값이 5를 초과하면 `True`, 그렇지 않으면 `False` 값을 갖는 새로운 feature를 만듦.
  * Hour: 태양광은 일시, 특히 시간대에 영향을 가장 많이 받고 연속적으로 변하는 값이 아니므로 categorical feature로 적합하다고 판단함.
* numeric feature: '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)', '일조(hr)', '일사(MJ/m2)' 등
* `NaN` value: 일부 데이터는 시간당발전량이 존재하지 않았고(`-`), 기상청 데이터는 일조·일사·전운량·강수량 등에 비어있는 값이 많았음. 따라서, 다음과 같은 방법을 활용하여 `NaN` value를 처리함.
  * 시간당발전량: 대부분 하루치 데이터 전체가 비어있었으므로, 이같은 경우에는 해당 건물과 가장 가까운 건물의 시간당발전량 값을 활용함.
  * 일조·일사·전운량·강수량: 값이 측정되지 않아(비가 오지 않는 등의 이유로) `NaN`인 경우에는 0으로, 단순히 값이 비어있는 경우에는 바로 앞 시간대와 바로 뒤 시간대의 값을 평균하거나 바로 앞 시간대의 값을 그대로 활용하여 채움.
## Model
### 1. Solar Power Prediction
path: `solar_power_prediction/Model.ipynb` \
LSTM을 활용하여 건물별로 값을 예측함 (14개의 모델 활용). 다양한 parameter를 넣어준 후, [R^2](https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95%EA%B3%84%EC%88%98) 값이 가장 큰 것을 최종 선택하여 target 데이터 예측에 활용함.
## Prediction
### 1. Solar Power Prediction
path: `outputs_0629/*/prediction.csv` \
2022/6/29, 모델에 넣어준 feature를 통일시키기 위해 당일의 기상청 데이터를 활용하여 건물별 24시간 시간당발전량 예측값을 csv 형태로 저장함. \
