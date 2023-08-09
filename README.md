# 育秀盃創意獎第20屆佳作: 基於車流模式分析之智慧型紅綠燈
參賽組別: 數位應用類

學校/科系: 淡江大學資訊工程學系

組員: 江岱樺、楊子誼、鄭承斌

指導老師: 吳孟倫

模擬一條道路的一個路口根據不同車流量來控制紅綠燈燈號，以此減少全部車輛的平均等待時間。

## demo影片
[![IMAGE ALT TEXT](http://img.youtube.com/vi/JgcGpmFGeW8/0.jpg)](https://www.youtube.com/watch?v=JgcGpmFGeW8&ab_channel=%E9%9B%B2o%E5%B8%8C%E5%86%80 "SA204546 demo")

## 使用說明
環境及安裝軟體:
軟體在Windows10/Windows11上運行
1.	電腦需有支援cuda的nvidia GPU，並安裝Cuda和cudnn(版本須匹配)
Cuda官網: https://developer.nvidia.com/cuda-downloads
Cudnn官網: https://developer.nvidia.com/rdp/cudnn-archive

2.	安裝Anaconda3
Anaconda3官網: https://www.anaconda.com/products/distribution

3.	安裝完成Anaconda3之後，打開anaconda prompt，自行創立虛擬環境，並在該環境安裝以下套件:
安裝套件:
numpy
tensorflow>=2.0.0
pytorch>=1.0.0
opencv-python
matplotlib

4.	安裝SUMO
SUMO官網: https://sumo.dlr.de/docs/Downloads.php

### 物件偵測YOLOv7使用
先使用yolov7對於測試影片進行物件偵測並記錄每輛車經過的時間。

1.	開啟Anaconda prompt並進入虛擬環境後，cd進入” SA204546_demo\objectdetect”資料夾
```python
cd 資料夾名稱
```

3.	測試物件偵測模型，輸入以下指令:
```python
python detect.py --weights best.pt --conf 0.1 --source test.mp4
```
best.pt為模型檔案，test.mp4為測試影片。
測試影片下載鏈結: https://drive.google.com/file/d/1rTSDYg1_3LOfMmNjWy83Md4nsBEqdusG/view?usp=drive_link
模型檔案下載鏈結: https://drive.google.com/file/d/11dwylaPDtCaj_VvLnqgFoCaGIOBylJXL/view?usp=drive_link

4.	指令執行完畢後，在執行指令的資料夾中，會產生”per10.txt”及”per300
.txt”，並且” objectdetect\runs\detect\object_trackingxx”(結尾的xx為數字)資料夾底下會產生模型偵測物件的影片檔
 
### Deep-Q-Learning強化學習模型使用
1.	開啟Anaconda prompt並進入虛擬環境後，cd進入” SA204546_demo\ deep_q_learning”資料夾
```python
cd 資料夾名稱
```

2.	使用模擬器進行測試(檔案內附有相關設定檔案及訓練完成的模型檔案)，輸入以下指令:
```python
python testing_main.py
```
指令執行中會打開SUMO模擬器，模擬執行完畢之後請將軟體關閉。

3.	使用模擬器進行訓練，輸入以下指令:
```python
python training_main.py
```

### Acknowledgements
https://github.com/WongKinYiu/yolov7

https://github.com/AndreaVidali/Deep-QLearning-Agent-for-Traffic-Signal-Control


