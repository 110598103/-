# 辛普森人物辨識
# 資料來源
    https://www.kaggle.com/c/machine-learningntut-2021-autumn-classification/data
# 使用的版本
    查看 function.txt
# 資料處理
    使用 create.py 把Train分成train 和 valid
    train: 81929筆 
    valid: 15000筆
# 圖片處理
    把圖片翻轉(20度和左右翻轉)和縮放(從224*244到128*128)
![下載 (8)](https://user-images.githubusercontent.com/93694868/147255302-a7a0b81f-ed8c-4b89-b4bf-64e9bde1914f.png)
![下載 (9)](https://user-images.githubusercontent.com/93694868/147255380-93ef7c6d-212b-4906-97b6-163d47267024.png)
![下載 (10)](https://user-images.githubusercontent.com/93694868/147255310-8afd5487-d28c-45b4-9625-4df5ab2a686c.png)
![下載 (11)](https://user-images.githubusercontent.com/93694868/147255317-eee2d527-0514-4732-a4b8-911ef3f21e5b.png)
![下載 (12)](https://user-images.githubusercontent.com/93694868/147255409-d879946f-c62c-464a-80cc-3573ff35f756.png)
![下載 (13)](https://user-images.githubusercontent.com/93694868/147255422-d9da6729-3435-4fc3-976f-874398c8fdad.png)
![下載 (14)](https://user-images.githubusercontent.com/93694868/147255431-fee46534-96a7-4ade-8c9e-e054bb4e1b63.png)
# 模型

        model = Sequential()
        model.add(Conv2D(filters=16,kernel_size=5,padding='same',input_shape=(128,128,3),activation = 'relu'))
        model.add(Conv2D(filters=16,kernel_size=5,padding='same',activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Conv2D(filters=24,kernel_size=5,padding='same',activation = 'relu'))
        model.add(Conv2D(filters=24,kernel_size=5,padding='same',activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=32,kernel_size=5,padding='same',activation = 'relu'))
        model.add(Conv2D(filters=32,kernel_size=5,padding='same',activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=48,kernel_size=5,padding='same',activation = 'relu'))
        model.add(Conv2D(filters=48,kernel_size=5,padding='same',activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64,kernel_size=5,padding='same',activation = 'relu'))
        model.add(Conv2D(filters=64,kernel_size=5,padding='same',activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=96,kernel_size=5,padding='same',activation = 'relu'))
        model.add(Conv2D(filters=96,kernel_size=5,padding='same',activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units =2048, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(units =1024, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(units =256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(units =128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(50,activation='softmax'))
# 每層的權重
![下載 (1)](https://user-images.githubusercontent.com/93694868/147260794-a40f27be-39a6-4625-a2aa-96497b2bc9fd.png)
![下載 (2)](https://user-images.githubusercontent.com/93694868/147260801-bae4149b-9b00-4d87-93c1-6435f866152d.png)
![下載 (3)](https://user-images.githubusercontent.com/93694868/147260805-5558daaf-b575-428f-847a-da12c830692d.png)
![下載 (4)](https://user-images.githubusercontent.com/93694868/147260808-b7a8a99d-3255-4ee5-b61c-5df3087ef593.png)
![下載 (5)](https://user-images.githubusercontent.com/93694868/147260815-f8fe9622-27f7-48d4-a774-3e7c0597a6b7.png)
![下載 (6)](https://user-images.githubusercontent.com/93694868/147260823-1966a32a-a14d-4776-b3cd-6bfa2861e4a9.png)

# 混淆矩陣
![下載 (7)](https://user-images.githubusercontent.com/93694868/147260748-b591c46e-c637-4a20-a2ab-0aaee3cb86d5.png)
# 程式如何執行
## 模型使用
    執行CNN.py可以訓練模型，也可以用 Train.sh執行
    執行test.py可以執行使用練好的模型
    也可以用 Test.sh執行



    
