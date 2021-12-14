#SSD-Pytorch 

## 配置安装环境
pip3 install -r requirements.txt

## 如何使用

项目要求数据集格式：COCO、VOC

+ 有自己数据集

    example: darknet-yolo数据集 --> VOC
  + 准备数据
      1. 将所有图片放入 ./data/VOCdevkit/JPEGImages

  + 生成VOC数据集
      1. 运行 ./data/VOCdevkit/darknet2voc.py -->生成xml文件
      2. 运行 ./data/VOCdevkit/split_txt.py -->生成train.txt trainval.txt test.txt val.txt


+ 没有数据集

    + 运行 ./data/scipts 中你需要的数据集


+ 开始训练

    + 修改 ./data/config.py 类别数
    + 修改 ./data/voc0712.py VOC_CLASSES
    + 修改 ./ssd.py build_ssd()的num_classes
    + 修改 ./train.py iteration(保存模型的迭代次数）、learning-rate、batch_size
    + 运行 ./train.py


+ 验证

    + 修改 ./eval.py trained_model(训练好的模型名)，运行
    + 修改 ./test.py trained_model(训练好的模型名)，运行


+ 检测图片，可视化

    + 运行 ./demo/live_img.py 获得检测框图片
    + 运行 ./demo/live_score.py 获得检测框带置信度图片 