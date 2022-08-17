# MySolar-ESS
GIST 창의융합경진대회

Reference 
* https://github.com/shubhamchouksey/Power-Prediction-LSTM

## Bill
한전공식 전기요금표 참고: https://cyber.kepco.co.kr/ckepco/front/jsp/CY/E/E/CYEEHP00104.jsp \
* 적용일자 - 2021년 1월 1일 기준: 921500730원
* 적용일자 - 2022년 4월 1일 기준: 882711820원

## EDA
### 1. Solar Power Prediction
path: `solar_power_prediction/SolarPower EDA.ipynb` \
기상청 시간단위 데이터 중, 시간당발전량(kWh)와 상관관계가 높게 나타났던 '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)', '일조(hr)', '일사(MJ/m2)', 'Cloud', 'Hour' 등을 feature로 활용하고, 이와 비슷한 정도의 상관관계를 보이던 '외기온도', '모듈온도' 등의 feature는 overfitting 방지를 위해 feature로 활용하지 않음.
* categorical features: 'Cloud', 'Hour'
  * Cloud: 구름의 양을 1\~10으로 나타내는 전운량(10분위)를 그대로 활용하고자 했으나, 약 2년치 학습데이터에는 1\~10의 수치가 모두 나타나는 반면 22/6/29의 target 데이터에는 5\~10 까지만 나타나 있어 그대로 활용하기 어렵다고 판단, 따라서 전운량 값이 5를 초과하면 `True`, 그렇지 않으면 `False` 값을 갖는 새로운 feature를 만듦.
  * Hour: 태양광은 일시, 특히 시간대에 영향을 가장 많이 받고 연속적으로 변하는 값이 아니므로 categorical feature로 적합하다고 판단함.
* numeric feature: '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)', '일조(hr)', '일사(MJ/m2)' 등
* `NaN` value: 일부 데이터는 시간당발전량이 존재하지 않았고(`-`), 기상청 데이터는 일조·일사·전운량·강수량 등에 비어있는 값이 많았음. 따라서, 다음과 같은 방법을 활용하여 `NaN` value를 처리함.
  * 시간당발전량: 대부분 하루치 데이터 전체가 비어있었으므로, 이같은 경우에는 해당 건물과 가장 가까운 건물의 시간당발전량 값을 활용함.
  * 일조·일사·전운량·강수량: 값이 측정되지 않아(비가 오지 않는 등의 이유로) `NaN`인 경우에는 0으로, 단순히 값이 비어있는 경우에는 바로 앞 시간대와 바로 뒤 시간대의 값을 평균하거나 바로 앞 시간대의 값을 그대로 활용하여 채움.
  
### 2. Power Load Prediction
path: `power load/data_processing.py`, `power load/data_post_processing.py`, `power load/weather_data_processing.py`, `power load/data_merge.py` \
주어진 전력부하량 데이터와 기상청 데이터로부터 년도, 날짜, 시간, 요일(주말o/x), 기온, 습도, 강수량, 일조, 일사 등을 feature로 활용
* 년도, 달, 일, 시간: 2020년 6월 1일부터 2022년 6월 28일까지의 데이터를 활용했으며 년도는 20\~22 사이의 정수값, 달은 1\~12 사이의 정수값, 일은 1\~31 사이의 정수값, 시간은 0\~23 사이의 정수값으로 설정함.
* 요일: 평일과 주말(토/일) 사이의 전력사용량은 크게 차이나기에 평일은 1, 주말은 0으로 binary 값을 설정함.
* 기온, 습도, 강수량, 일조, 일사:  2020년 6월 1일부터 2022년 6월 28일까지의 데이터를 활용했으며 강수량, 일조, 일사의 경우 데이터가 비어있는 경우가 많아 제외하고 기온과 습도를 실수값으로 설정함.
*  `NaN` value: 마찬가지로 일부 데이터는 시간당 유효전력량이 존재하지 않았고(`-`), 기상청 데이터는 일조·일사·강수량 등에 비어있는 값이 많았음. 데이터 특성상 시간대에 연속적이지 않은 값을 가지고 있어, 전후 시간대의 평균을 활용하기보다는 데이터를 제거하는 것이 오차를 줄이기에 적합하다고 판단함.

## Model
### 1. Solar Power Prediction
path: `solar_power_prediction/Model.ipynb` \
LSTM을 활용하여 건물별로 값을 예측함 (14개의 모델 활용). 다양한 parameter를 넣어준 후, [R^2](https://ko.wikipedia.org/wiki/%EA%B2%B0%EC%A0%95%EA%B3%84%EC%88%98) 값이 가장 큰 것을 최종 선택하여 target 데이터 예측에 활용함.

### 2. Power Load Prediction
path: `power load/lstm.py` \
LSTM을 활용하여 건물별로 값을 예측함 (26개의 모델 활용). parameter 값을 달리하여 model1, model2, model3 생성하고 최종적으로 model3 사용. 각 폴더에 건물별 학습 모델 저장함.

## Prediction
### 1. Solar Power Prediction
path: `outputs_0629/*/prediction.csv` & `./aggregated_0629.csv`\
2022/6/29, 모델에 넣어준 feature를 통일시키기 위해 당일의 기상청 데이터를 활용하여 건물별 24시간 시간당발전량 예측값을 csv 형태로 저장함. 

### 2. Power Load Prediction
path: `power load/predict.py`, `output_power&energy/*.csv` \
2022/6/29, 태양광 발전량 예측과 마찬가지로 건물별 24시간 시간당발전량 예측값을 csv 형태로 저장함. "SV-2"~"고등광연구소(E)"까지의 건물을 0~26의 숫자로 각각 매핑한 결과(데이터셋에 있는 순)이며, power는 유효전력 energy는 유효전력량을 예측한 값임.
