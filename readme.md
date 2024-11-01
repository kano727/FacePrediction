# 人脸颜值评分

## 数据处理与输入
1. 输入长宽相等的人脸图片
2. 数据进行了增强，保证各个分类样本数量均衡
3. 粗略的认为颜色不影响人脸颜值评分，因此图片进行灰度化处理

## 模型
基于ResNet18实现，增加线性层，修改了输出层实现3分类

## 训练过程
1. fine-tune方式训练，先冻结除输出层外的所有参数，训练10epoch，再解冻所有参数，训练30epoch
2. 收集训练过程中分类错误的图片并移动到val_data/incorrect中
3. 对于分类错误的图片提供了方便的标签工具img_tagger

## 输出
1. 输出共3类，0为最低，2为最高
2. 根据预测结果移动图片到对应文件夹

## 准确率
在现有数据下，模型的最高准确率为98%