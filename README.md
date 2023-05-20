# yolov8在旭日x3派板子上的部署
## 主要内容如下：
1. 用修改后的yolov8训练自己的数据集
2. Pytorch到ONNX模型转换
3. ONNX到BIN转换
4. 旭日x3派上的实时检测


以下是官方对于yolov8的介绍和链接
<div align="center">
  <p>
    <a href="https://ultralytics.com/yolov8" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png"></a>
  </p>

[English](README.md) | [简体中文](README.zh-CN.md)
<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>

[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), developed by [Ultralytics](https://ultralytics.com), is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. YOLOv8 is designed to be fast, accurate, and easy to use, making it an excellent choice for a wide range of object detection, image segmentation and image classification tasks.

To request an Enterprise License please complete the form at [Ultralytics Licensing](https://ultralytics.com/license).

<img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png"></a>

<div align="center">
  <a href="https://github.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.linkedin.com/company/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-linkedin.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://twitter.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-twitter.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://youtube.com/ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-youtube.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.tiktok.com/@ultralytics" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-tiktok.png" width="2%" alt="" /></a>
  <img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="2%" alt="" />
  <a href="https://www.instagram.com/ultralytics/" style="text-decoration:none;">
    <img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-instagram.png" width="2%" alt="" /></a>
</div>
</div>

其中yolov8n比yolov5s参数量少一倍的情况下，精度可以持平yolov5s，同时与yolov6和yolov7相比也是更胜一筹。
## get started
```sh
git clone https://github.com/Hzbupahaozi/yolov8_xj3_deploy.git
cd yolov8_xj3_deploy
pip install -r requirements.txt
python setup.py install
```
## train
旭日x3派开发板的亮点在于其BPU 5T的int8算力\
BPU的高效源于其特别的网络结构————可变组卷积（VarGNet），具体可以参考<https://arxiv.org/abs/1907.05653>\
因此对原本的yolov8网络结构进行修改，即将yolov8.yaml修改为x3pi_model_config/yolov8-vargnetct.yaml\
准备好yolo格式的数据集后就可以开始训练了
```sh
yolo task=detect mode=train model=./x3pi_config/yolov8n-vargnetct.yaml data=./mydata.yaml batch=32 epochs=80 imgsz=640 workers=8 device=0
```
训练结果如下图
![x3true_yolov8n_80](https://user-images.githubusercontent.com/84694458/235343978-88fb6a4f-a916-4d31-9b95-d16d3a0d84fb.jpg)
## pytorch模型到onnx模型的转换
以下是三木大佬的原话博客：\
由于 YOLOv8 原始仓库转换的模型把预测特征解码放到了模型里面，这其实会导致两个问题：
1. 目标检测模型并不是要计算所有框并且回归的，通过置信度可以筛选超过 90% 的框，这些都不需要额外计算的，因此我提供了一个传参 x3pi=True 用来删除解码等操作。
2. 原始模型的解码操作 BPU 不能很好的加速，那索性就全用 CPU 处理好了。
转换命令：
```sh
yolo export detect model=best.pt format=onnx x3pi=True
```
## onnx到bin的转换
这里使用docker挂载天工开物的开发工具包实现\
教程参考：[BPU部署走出新手村](https://blog.csdn.net/Zhaoxi_Li/article/details/125516265?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167981185316800225598988%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=167981185316800225598988&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-1-125516265-null-null.142^v76^wechat_v2,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=BPU%E9%83%A8%E7%BD%B2%E5%8F%AB%E6%88%90&spm=1018.2226.3001.4187)\
需要注意的是要下载最新的2.5.2的容器和2.5.2的oe文档，上述教程使用版本过旧（导致我搞了好一段时间没啥进展），
```sh
docker run -it --rm -v "D:\docker\horizon_xj3_open_explorer_v2.5.2-py38_20230331":/open_explorer -v "D:\docker\BPUCodes":/data/horizon_x3/codes openexplorer/ai_toolchain_ubuntu_20_xj3_cpu:v2.5.2

cd /open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/04_detection/
mkdir ./yolov8/
cd ./yolov8/
```
为了模型的轻量化在转换为bin模型时会将模型参数通过PTQ转换为int8，需要准备校准数据集，具体脚本为make_calib.py\
然后把一些相关文件（onnx模型，转换文件my_config.yaml，校准数据集文件夹）拷贝进去
```sh
docker cp 本地文件路径 容器:/open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/04_detection/yolov8/
```
其中容器可通过以下命令查看
```
docker ps
```
然后就可以开始转换了
```sh
# check
hb_mapper checker --model-type onnx \
    --model best.onnx \
    --march bernoulli2

# export
hb_mapper makertbin --config my_config.yaml --model-type onnx
```
然后就可以看到很多日志
```
==============================================================================================================================================
Node                                                ON   Subgraph  Type                       Cosine Similarity  Threshold   In/Out DataType 
----------------------------------------------------------------------------------------------------------------------------------------------
HZ_PREPROCESS_FOR_images                            BPU  id(0)     HzSQuantizedPreprocess     0.999949           127.000000  int8/int8       
Conv_0                                              BPU  id(0)     HzSQuantizedConv           0.999869           1.502627    int8/int8       
Conv_2                                              BPU  id(0)     HzSQuantizedConv           0.999687           17.772688   int8/int8       
Conv_4                                              BPU  id(0)     HzSQuantizedConv           0.999460           16.475630   int8/int8       
Split_6                                             BPU  id(0)     Split                                                     int8/int8       
Conv_7                                              BPU  id(0)     HzSQuantizedConv           0.999311           14.938033   int8/int8       
Conv_9                                              BPU  id(0)     HzSQuantizedConv           0.999322           16.493052   int8/int8       
Conv_11                                             BPU  id(0)     HzSQuantizedConv           0.999112           19.128515   int8/int8       
Conv_13                                             BPU  id(0)     HzSQuantizedConv           0.999251           18.809891   int8/int8       
UNIT_CONV_FOR_Add_15                                BPU  id(0)     HzSQuantizedConv           0.999442           14.938033   int8/int8       
...CONV_FOR_onnx::Concat_444_0.14999_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.24_0.14999_TO_FUSE_SCALE        BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_16                                           BPU  id(0)     Concat                     0.999392           14.938033   int8/int8       
Conv_17                                             BPU  id(0)     HzSQuantizedConv           0.999088           19.048702   int8/int8       
Conv_19                                             BPU  id(0)     HzSQuantizedConv           0.999166           11.896363   int8/int8       
Conv_21                                             BPU  id(0)     HzSQuantizedConv           0.999127           13.429939   int8/int8       
Split_23                                            BPU  id(0)     Split                                                     int8/int8       
Conv_24                                             BPU  id(0)     HzSQuantizedConv           0.996399           13.270452   int8/int8       
Conv_26                                             BPU  id(0)     HzSQuantizedConv           0.996832           9.714583    int8/int8       
Conv_28                                             BPU  id(0)     HzSQuantizedConv           0.995260           6.936894    int8/int8       
Conv_30                                             BPU  id(0)     HzSQuantizedConv           0.996167           5.460835    int8/int8       
UNIT_CONV_FOR_Add_32                                BPU  id(0)     HzSQuantizedConv           0.998495           13.270452   int8/int8       
Conv_33                                             BPU  id(0)     HzSQuantizedConv           0.996459           8.126689    int8/int8       
Conv_35                                             BPU  id(0)     HzSQuantizedConv           0.996397           5.057761    int8/int8       
Conv_37                                             BPU  id(0)     HzSQuantizedConv           0.995583           4.751214    int8/int8       
Conv_39                                             BPU  id(0)     HzSQuantizedConv           0.995153           6.242482    int8/int8       
UNIT_CONV_FOR_Add_41                                BPU  id(0)     HzSQuantizedConv           0.998127           8.126689    int8/int8       
...CONV_FOR_onnx::Concat_469_0.09842_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.88_0.09842_TO_FUSE_SCALE        BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.124_0.09842_TO_FUSE_SCALE       BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_42                                           BPU  id(0)     Concat                     0.998541           13.270452   int8/int8       
Conv_43                                             BPU  id(0)     HzSQuantizedConv           0.997992           12.499084   int8/int8       
Conv_45                                             BPU  id(0)     HzSQuantizedConv           0.999056           6.938571    int8/int8       
Conv_47                                             BPU  id(0)     HzSQuantizedConv           0.999112           6.857578    int8/int8       
Split_49                                            BPU  id(0)     Split                                                     int8/int8       
Conv_50                                             BPU  id(0)     HzSQuantizedConv           0.998487           6.594507    int8/int8       
Conv_52                                             BPU  id(0)     HzSQuantizedConv           0.998260           6.571420    int8/int8       
Conv_54                                             BPU  id(0)     HzSQuantizedConv           0.997727           4.126614    int8/int8       
Conv_56                                             BPU  id(0)     HzSQuantizedConv           0.996096           4.693301    int8/int8       
UNIT_CONV_FOR_Add_58                                BPU  id(0)     HzSQuantizedConv           0.998729           6.594507    int8/int8       
Conv_59                                             BPU  id(0)     HzSQuantizedConv           0.997990           6.840492    int8/int8       
Conv_61                                             BPU  id(0)     HzSQuantizedConv           0.997279           4.396142    int8/int8       
Conv_63                                             BPU  id(0)     HzSQuantizedConv           0.996712           3.522115    int8/int8       
Conv_65                                             BPU  id(0)     HzSQuantizedConv           0.995933           4.622054    int8/int8       
UNIT_CONV_FOR_Add_67                                BPU  id(0)     HzSQuantizedConv           0.998451           6.840492    int8/int8       
...CONV_FOR_onnx::Concat_507_0.05330_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.188_0.05330_TO_FUSE_SCALE       BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.224_0.05330_TO_FUSE_SCALE       BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_68                                           BPU  id(0)     Concat                     0.998793           6.594507    int8/int8       
Conv_69                                             BPU  id(0)     HzSQuantizedConv           0.998581           6.768975    int8/int8       
Conv_71                                             BPU  id(0)     HzSQuantizedConv           0.999188           4.436905    int8/int8       
Conv_73                                             BPU  id(0)     HzSQuantizedConv           0.999031           4.389203    int8/int8       
Split_75                                            BPU  id(0)     Split                                                     int8/int8       
Conv_76                                             BPU  id(0)     HzSQuantizedConv           0.998980           4.278706    int8/int8       
Conv_78                                             BPU  id(0)     HzSQuantizedConv           0.998558           5.475333    int8/int8       
Conv_80                                             BPU  id(0)     HzSQuantizedConv           0.998531           3.166869    int8/int8       
Conv_82                                             BPU  id(0)     HzSQuantizedConv           0.998872           4.244014    int8/int8       
UNIT_CONV_FOR_Add_84                                BPU  id(0)     HzSQuantizedConv           0.999180           4.278706    int8/int8       
...CONV_FOR_onnx::Concat_545_0.03857_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.288_0.03857_TO_FUSE_SCALE       BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_85                                           BPU  id(0)     Concat                     0.999096           4.278706    int8/int8       
Conv_86                                             BPU  id(0)     HzSQuantizedConv           0.998955           4.898155    int8/int8       
Conv_88                                             BPU  id(0)     HzSQuantizedConv           0.998856           4.396799    int8/int8       
MaxPool_90                                          BPU  id(0)     HzQuantizedMaxPool         0.999222           6.915278    int8/int8       
MaxPool_91                                          BPU  id(0)     HzQuantizedMaxPool         0.999414           6.915278    int8/int8       
MaxPool_92                                          BPU  id(0)     HzQuantizedMaxPool         0.999525           6.915278    int8/int8       
Concat_93                                           BPU  id(0)     Concat                     0.999353           6.915278    int8/int8       
Conv_94                                             BPU  id(0)     HzSQuantizedConv           0.999698           6.915278    int8/int8       
ConvTranspose_96                                    BPU  id(0)     HzSQuantizedConvTranspose  0.999779           3.459090    int8/int8       
UNIT_CONV_FOR_onnx::Conv_538_0.04263_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_98                                           BPU  id(0)     Concat                     0.999320           5.414181    int8/int8       
Conv_99                                             BPU  id(0)     HzSQuantizedConv           0.999117           5.414181    int8/int8       
Split_101                                           BPU  id(0)     Split                                                     int8/int8       
Conv_102                                            BPU  id(0)     HzSQuantizedConv           0.998763           5.000547    int8/int8       
Conv_104                                            BPU  id(0)     HzSQuantizedConv           0.998287           5.897664    int8/int8       
Conv_106                                            BPU  id(0)     HzSQuantizedConv           0.997791           4.177810    int8/int8       
Conv_108                                            BPU  id(0)     HzSQuantizedConv           0.997375           4.131814    int8/int8       
...CONV_FOR_onnx::Concat_580_0.03904_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.372_0.03904_TO_FUSE_SCALE       BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_110                                          BPU  id(0)     Concat                     0.998700           5.000547    int8/int8       
Conv_111                                            BPU  id(0)     HzSQuantizedConv           0.998120           4.958530    int8/int8       
ConvTranspose_113                                   BPU  id(0)     HzSQuantizedConvTranspose  0.998221           5.232376    int8/int8       
UNIT_CONV_FOR_onnx::Conv_500_0.05410_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_115                                          BPU  id(0)     Concat                     0.998037           6.871005    int8/int8       
Conv_116                                            BPU  id(0)     HzSQuantizedConv           0.997310           6.871005    int8/int8       
Split_118                                           BPU  id(0)     Split                                                     int8/int8       
Conv_119                                            BPU  id(0)     HzSQuantizedConv           0.996089           5.632505    int8/int8       
Conv_121                                            BPU  id(0)     HzSQuantizedConv           0.995166           5.500845    int8/int8       
Conv_123                                            BPU  id(0)     HzSQuantizedConv           0.993983           6.010270    int8/int8       
Conv_125                                            BPU  id(0)     HzSQuantizedConv           0.993902           7.229157    int8/int8       
...CONV_FOR_onnx::Concat_604_0.04773_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.436_0.04773_TO_FUSE_SCALE       BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_127                                          BPU  id(0)     Concat                     0.996177           5.632505    int8/int8       
Conv_128                                            BPU  id(0)     HzSQuantizedConv           0.994840           6.062278    int8/int8       
Conv_130                                            BPU  id(0)     HzSQuantizedConv           0.996329           7.191900    int8/int8       
...R_onnx::ConvTranspose_597_0.05140_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_132                                          BPU  id(0)     Concat                     0.997258           6.528210    int8/int8       
Conv_133                                            BPU  id(0)     HzSQuantizedConv           0.996429           6.528210    int8/int8       
Split_135                                           BPU  id(0)     Split                                                     int8/int8       
Conv_136                                            BPU  id(0)     HzSQuantizedConv           0.994795           6.808836    int8/int8       
Conv_138                                            BPU  id(0)     HzSQuantizedConv           0.994417           6.371107    int8/int8       
Conv_140                                            BPU  id(0)     HzSQuantizedConv           0.993589           6.077781    int8/int8       
Conv_142                                            BPU  id(0)     HzSQuantizedConv           0.993910           6.707007    int8/int8       
...CONV_FOR_onnx::Concat_629_0.05270_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.504_0.05270_TO_FUSE_SCALE       BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_144                                          BPU  id(0)     Concat                     0.995724           6.808836    int8/int8       
Conv_145                                            BPU  id(0)     HzSQuantizedConv           0.995444           6.693432    int8/int8       
Conv_147                                            BPU  id(0)     HzSQuantizedConv           0.993256           6.435788    int8/int8       
...R_onnx::ConvTranspose_573_0.03540_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_149                                          BPU  id(0)     Concat                     0.998485           4.496263    int8/int8       
Conv_150                                            BPU  id(0)     HzSQuantizedConv           0.997724           4.496263    int8/int8       
Split_152                                           BPU  id(0)     Split                                                     int8/int8       
Conv_153                                            BPU  id(0)     HzSQuantizedConv           0.997050           4.260736    int8/int8       
Conv_155                                            BPU  id(0)     HzSQuantizedConv           0.996641           5.090493    int8/int8       
Conv_157                                            BPU  id(0)     HzSQuantizedConv           0.994862           3.914743    int8/int8       
Conv_159                                            BPU  id(0)     HzSQuantizedConv           0.992631           4.832358    int8/int8       
...CONV_FOR_onnx::Concat_654_0.03318_TO_FUSE_SCALE  BPU  id(0)     HzSQuantizedConv                                          int8/int8       
UNIT_CONV_FOR_input.572_0.03318_TO_FUSE_SCALE       BPU  id(0)     HzSQuantizedConv                                          int8/int8       
Concat_161                                          BPU  id(0)     Concat                     0.995999           4.260736    int8/int8       
Conv_162                                            BPU  id(0)     HzSQuantizedConv           0.995250           4.214143    int8/int8       
Conv_164                                            BPU  id(0)     HzSQuantizedConv           0.995858           7.191900    int8/int8       
Mul_166                                             BPU  id(0)     HzLut                      0.996182           7.299980    int8/int8       
Conv_167                                            BPU  id(0)     HzSQuantizedConv           0.997211           7.295052    int8/int8       
Mul_169                                             BPU  id(0)     HzLut                      0.997212           7.081635    int8/int8       
Conv_170                                            BPU  id(0)     HzSQuantizedConv           0.998338           7.075689    int8/int32      
Conv_172                                            BPU  id(0)     HzSQuantizedConv           0.995082           7.191900    int8/int8       
Mul_174                                             BPU  id(0)     HzLut                      0.993972           6.291241    int8/int8       
Conv_175                                            BPU  id(0)     HzSQuantizedConv           0.994323           6.279607    int8/int8       
Mul_177                                             BPU  id(0)     HzLut                      0.994310           10.864396   int8/int8       
Conv_178                                            BPU  id(0)     HzSQuantizedConv           0.999579           10.864189   int8/int32      
Conv_180                                            BPU  id(0)     HzSQuantizedConv           0.996984           6.435788    int8/int8       
Mul_182                                             BPU  id(0)     HzLut                      0.996698           6.473208    int8/int8       
Conv_183                                            BPU  id(0)     HzSQuantizedConv           0.997236           6.463228    int8/int8       
Mul_185                                             BPU  id(0)     HzLut                      0.996391           7.075638    int8/int8       
Conv_186                                            BPU  id(0)     HzSQuantizedConv           0.998851           7.069662    int8/int32      
Conv_188                                            BPU  id(0)     HzSQuantizedConv           0.995746           6.435788    int8/int8       
Mul_190                                             BPU  id(0)     HzLut                      0.995045           5.152664    int8/int8       
Conv_191                                            BPU  id(0)     HzSQuantizedConv           0.995736           5.123033    int8/int8       
Mul_193                                             BPU  id(0)     HzLut                      0.995775           6.424686    int8/int8       
Conv_194                                            BPU  id(0)     HzSQuantizedConv           0.999827           6.414289    int8/int32      
Conv_196                                            BPU  id(0)     HzSQuantizedConv           0.994093           3.806277    int8/int8       
Mul_198                                             BPU  id(0)     HzLut                      0.993954           4.152104    int8/int8       
Conv_199                                            BPU  id(0)     HzSQuantizedConv           0.994331           4.087798    int8/int8       
Mul_201                                             BPU  id(0)     HzLut                      0.994222           4.594001    int8/int8       
Conv_202                                            BPU  id(0)     HzSQuantizedConv           0.999736           4.548010    int8/int32      
Conv_204                                            BPU  id(0)     HzSQuantizedConv           0.998809           3.806277    int8/int8       
Mul_206                                             BPU  id(0)     HzLut                      0.998360           3.759366    int8/int8       
Conv_207                                            BPU  id(0)     HzSQuantizedConv           0.999285           3.673772    int8/int8       
Mul_209                                             BPU  id(0)     HzLut                      0.999141           4.446995    int8/int8       
Conv_210                                            BPU  id(0)     HzSQuantizedConv           0.999996           4.395507    int8/int32
2023-04-30 12:46:11,222 INFO [Sun Apr 30 12:46:11 2023] End to Horizon NN Model Convert.
2023-04-30 12:46:11,268 INFO start convert to *.bin file....
2023-04-30 12:46:11,299 INFO ONNX model output num : 6
2023-04-30 12:46:11,302 INFO ############# model deps info #############
2023-04-30 12:46:11,303 INFO hb_mapper version   : 1.15.5
2023-04-30 12:46:11,303 INFO hbdk version        : 3.44.1
2023-04-30 12:46:11,303 INFO hbdk runtime version: 3.15.17.0
2023-04-30 12:46:11,303 INFO horizon_nn version  : 0.16.3
2023-04-30 12:46:11,304 INFO ############# model_parameters info #############
2023-04-30 12:46:11,304 INFO onnx_model          : /open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/04_detection/myyolov8/best.onnx
2023-04-30 12:46:11,304 INFO BPU march           : bernoulli2
2023-04-30 12:46:11,304 INFO layer_out_dump      : False
2023-04-30 12:46:11,305 INFO log_level           : DEBUG
2023-04-30 12:46:11,305 INFO working dir         : /open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/04_detection/myyolov8/model_outputs
2023-04-30 12:46:11,305 INFO output_model_file_prefix: yolov8_horizon
2023-04-30 12:46:11,305 INFO ############# input_parameters info #############
2023-04-30 12:46:11,305 INFO ------------------------------------------
2023-04-30 12:46:11,306 INFO ---------input info : images ---------
2023-04-30 12:46:11,306 INFO input_name          : images
2023-04-30 12:46:11,306 INFO input_type_rt       : nv12
2023-04-30 12:46:11,306 INFO input_space&range   : regular
2023-04-30 12:46:11,306 INFO input_layout_rt     : NHWC
2023-04-30 12:46:11,307 INFO input_type_train    : rgb
2023-04-30 12:46:11,307 INFO input_layout_train  : NCHW
2023-04-30 12:46:11,307 INFO norm_type           : data_scale
2023-04-30 12:46:11,307 INFO input_shape         : 1x3x640x640
2023-04-30 12:46:11,307 INFO input_batch         : 1
2023-04-30 12:46:11,308 INFO scale_value         : 0.003921568627451,
2023-04-30 12:46:11,308 INFO cal_data_dir        : /open_explorer/ddk/samples/ai_toolchain/horizon_model_convert_sample/04_detection/myyolov8/calib_640_f32
2023-04-30 12:46:11,308 INFO cal_data_type       : float32
2023-04-30 12:46:11,308 INFO ---------input info : images end -------
2023-04-30 12:46:11,309 INFO ------------------------------------------
2023-04-30 12:46:11,309 INFO ############# calibration_parameters info #############
2023-04-30 12:46:11,309 INFO preprocess_on       : False
2023-04-30 12:46:11,309 INFO calibration_type:   : default
2023-04-30 12:46:11,309 INFO ############# compiler_parameters info #############
2023-04-30 12:46:11,310 INFO hbdk_pass_through_params: --O3 --core-num 2 --fast
2023-04-30 12:46:11,310 INFO input-source        : {'images': 'pyramid', '_default_value': 'ddr'}
2023-04-30 12:46:11,316 INFO Convert to runtime bin file successfully!
2023-04-30 12:46:11,316 INFO End Model Convert
```
我们可以看到所有算子都是在BPU上跑了，所以可以起到加速推理的作用，同时也说明对yolov8模型的修改是有效的！因为我尝试过使用最原本的yolov8n结构，结果还是有一部分不能在bpu上跑\
模型的输出结果在model_output中，其中的yolov8_horizon.bin就是我们需要的模型
## 最后一步 上板！
首先我们可以测试以下地平线预装模型yolov5_672x672_nv12.bin，yolov5s_672x672_nv12.bin的效果
```
hrt_model_exec perf \
  --model_file yolov5s_672x672_nv12.bin \
  --model_name="" \
  --core_id=0 \
  --frame_count=500 \
  --perf_time=0 \
  --thread_num=1 \
  --profile_path="."

hrt_model_exec perf \
  --model_file fcos_512x512_nv12.bin \
  --model_name="" \
  --core_id=0 \
  --frame_count=500 \
  --perf_time=0 \
  --thread_num=1 \
  --profile_path="."
```
结果如下：
```
Running condition:
  Thread number is: 1
  Frame count   is: 500
  Program run time: 36346.591000 ms
Perf result:
  Frame totally latency is: 36325.519531 ms
  Average    latency    is: 72.651039 ms
  Frame      rate       is: 13.756448 FPS

Running condition:
  Thread number is: 1
  Frame count   is: 500
  Program run time: 36308.297000 ms
Perf result:
  Frame totally latency is: 36286.542969 ms
  Average    latency    is: 72.573090 ms
  Frame      rate       is: 13.770957 FPS
```
然后就是我们模型的效果
```
hrt_model_exec perf \
  --model_file yolov8s-vargnetct-det.bin \
  --model_name="" \
  --core_id=0 \
  --frame_count=500 \
  --perf_time=0 \
  --thread_num=1 \
  --profile_path="."
 ```
 日志如下：
 ```
Running condition:
  Thread number is: 1
  Frame count   is: 500
  Program run time: 20303.004000 ms
Perf result:
  Frame totally latency is: 20181.697266 ms
  Average    latency    is: 40.36.3396 ms
  Frame      rate       is: 24.626898 FPS
 ```
强！
最后就是调用usb摄像头进行实时检测，脚本为camera.py\
代码中 72-144 行是解析模型输出的关键内容，由于这部分在 CPU 跑，因此我主要使用 numpy 高效并行优化了一下后处理中的框回归计算，然后使用 opencv-python 提供的 NMS 接口对多余重复框去重复\
代码中 170-176 首先推理了 10 次空图做了一下预热，让模型运行更稳定\
代码中 199-202 打印了推理过程中，除了画框部分的耗时情况\
执行代码后，会打印如下日志：

![微信图片_20230430171956](https://user-images.githubusercontent.com/84694458/235345658-9380b5cc-de71-4c53-9663-299cc87a58b9.png)\
前处理耗时：7 ms\
模型推理： 44 ms\
后处理耗时: 8 ms\
整体耗时： 59 ms

### 前前后后耗时将近一个月，中间一段时间将近放弃部署直接在电脑上跑，这两天看了一些资料加班了一下最后总算是搞完了，最后感谢以下地平线论坛里的工作人员，还有最最感谢的就是三木大佬，本次参考[triple-Mu/yolov8](https://github.com/triple-Mu/yolov8/tree/triplemu/x3pi)

后续再记录一下，实现了在本地的使用onnxruntime进行推理，在inference文件夹中，其中包括
1. 只导出backbone+neck的分别是自制数据集和coco数据集的onnx模型的推理\
2. 正常export后onnx模型的推理\
![vargnet_backbone+neck的onnx模型推理](https://github.com/Hzbupahaozi/yolov8_xj3_deploy/assets/84694458/b0604474-2430-4e2b-ad4f-747fe2183ec2)\
![origin_backbone+neck的onnx模型推理](https://github.com/Hzbupahaozi/yolov8_xj3_deploy/assets/84694458/938977cd-4aa7-4ed6-8106-3ad4a1b32d7a)\
可以看到旭日x3派经过BPU部署后的性能和笔记本的cpu推理的性能差不多

### To Do List
1. python的多线程推理
2. C++封装python
3. C++推理


