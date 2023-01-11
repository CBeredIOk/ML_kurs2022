import pandas as pd
from colorama import Fore
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, recall_score

data = pd.read_csv('train_dataset_train.csv', encoding='utf-8', sep=',')
print(Fore.BLUE + f'{data.sample(5)}')

for col in data.columns:
    print(Fore.MAGENTA + f'{col} - {data[col].unique()}')

data.drop(columns=['id', 'line_id', 'station_id', 'entrance_id', 'ticket_id'], axis=1, inplace=True)

print(Fore.BLUE + f'{data.sample(5)}')


def change_time(data_time):
    data_time.pass_dttm = pd.to_datetime(data_time.pass_dttm)
    data_time['day'] = data_time.pass_dttm.dt.dayofweek
    data_time['hour'] = data_time.pass_dttm.dt.hour
    data_time['workday'] = data_time['day'].apply(lambda x: 0 if x == 5 or x == 6 else 1)

    def peak_time(x):
        return 0 if 0 <= x < 5 else (
            1 if 5 <= x <= 10 else (2 if 10 < x <= 18 else (3 if 18 < x <= 23 else x)))

    data_time['peak'] = data_time['hour'].apply(peak_time)

    data_time = data_time.drop(columns=['pass_dttm'])

    return data_time


data.dropna(inplace=True)

x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(data.drop(columns=['time_to_under']),
                                                                    data[['time_to_under']], test_size=0.3)

x_reg_train = change_time(x_reg_train)
x_reg_test = change_time(x_reg_test)

print(Fore.MAGENTA + f'{x_reg_train.sample(15)}')
print(Fore.BLUE + f'{x_reg_train["hour"].value_counts()}')

print(Fore.MAGENTA)

model_reg = CatBoostRegressor(iterations=500)
model_reg.fit(x_reg_train, y_reg_train,
              cat_features=['ticket_type_nm', 'entrance_nm', 'station_nm', 'line_nm'])
forecast_reg = model_reg.predict(x_reg_test)

print(Fore.BLUE)

x_class_train = data.drop(columns=['label'])[:10000]
y_class_train = data[['label']][:10000]

test_class = data.loc[data.ticket_type_nm.isin(x_class_train.ticket_type_nm) &
                      data.entrance_nm.isin(x_class_train.entrance_nm) &
                      data.station_nm.isin(x_class_train.station_nm) &
                      data.line_nm.isin(x_class_train.line_nm)]

x_class_test = test_class.drop(columns=['label'])[-3000:]
y_class_test = test_class[['label']][-3000:]

x_class_train = change_time(x_class_train)
x_class_test = change_time(x_class_test)

model_class = CatBoostClassifier(iterations=50, depth=6)
model_class.fit(x_class_train, y_class_train,
                cat_features=['ticket_type_nm', 'entrance_nm', 'station_nm', 'line_nm'])
forecast_class = model_class.predict(x_class_test)


def result(test_class, forecast_class, test_reg, forecast_reg):
    value = 0.5*r2_score(test_reg, forecast_reg) + 0.5*recall_score(test_class, forecast_class, average='micro')
    return value


print(Fore.MAGENTA + f'R2 = {r2_score(y_reg_test, forecast_reg)}')
print(Fore.MAGENTA + f'Recall = {recall_score(y_class_test, forecast_class, average="micro")}')
print(Fore.MAGENTA + f'Result = {result(y_class_test, forecast_class, y_reg_test, forecast_reg)}')
