
# 目录
[一、介绍](#一介绍)

[二、数据分析以及预处理](#二数据分析以及预处理)
<br>[1、灰度化](#1灰度化)
<br>[2、图像锐化](#2图像锐化)
<br>[3、图像降噪](#3图像降噪)
<br>[4、8邻域降噪](#48邻域降噪)
<br>[5、二值化](#5二值化)

[三、实现过程](#三、实现过程)
<br>[1、图像分割](#1图像分割)
<br>[2、BP神经网络](#2BP神经网络)
    <br>[前向传播](#前向传播)
    <br>[反向传播](#反向传播)

[四、项目结果](#四、项目结果) 

[参考文献](#参考文献) 

# 一、介绍
给定4000个验证码数据，每个验证码数据由10个数字或者字母组成，字母区分大小写，编写一个判别器，将这4000个验证码数据进行输入训练后，能完成基本的验证码识别工作。如下图所示能够正确的判别验证码内的数据：

![](https://i.niupic.com/images/2020/06/24/8jlH.png)

该4000组验证码的数据大致分为四大类：
1. 正常的没有进行模糊处理和添加噪声的验证码

![](https://i.niupic.com/images/2020/06/24/8jlI.png)

2. 经过模糊化处理之后的验证码

![](https://i.niupic.com/images/2020/06/24/8jlN.png)

3. 添加了很多噪声的验证码

![](https://i.niupic.com/images/2020/06/24/8jlM.png)

4. 经过模糊处理和添加了大量噪声的验证码

![](https://i.niupic.com/images/2020/06/24/8jlO.png)

我对于该验证码识别的大致思路是先对数据进行预处理降噪来去除椒盐噪声，图像增强来将模糊的部分进行清晰化，然后将10个数字或者是字母进行分离，最后通过一个BP神经网络进行训练，后续进行识别的时候通过该验证码进行分割，进入到该BP神经网络进行分类输出，最后组合所有的分割之后识别的数据为最后的识别结果。

# 二、数据分析以及预处理
## 1、灰度化
为了简化后续的运算，我们先将图像进行灰度化，将图像灰度化在OpenCV内可以不用显示的转化，可以在读入图像数据的时候就指定图像的格式为灰度图像，反映到代码层面为：`Mat mat = imread("path_to_image", CV_8UC1);`，最后一个参数即为将读入的图像作为灰度图像。
## 2、图像锐化
我们可以使用空间滤波进行图像锐化操作，使得被模糊化的图像更加清晰一些。这里我们使用的滤波器为：

![](https://math.jianshu.com/math?formula=%5cbegin%7bpmatrix%7d+++++0+%26+-1+%26+0+%5c%5c+++++-1+%26+5+%26+-1+%5c%5c++++0+%26+-1+%26+0%5cend%7bpmatrix%7d)

对于像素点，我们可以基于这个滤波器的简单算法主要是遍历图像中的像素点，根据其邻域像素点的值来确定其锐化后的值，计算公式为：`sharpened_pixel = 5 * current – left – right – up – down`，反应到C++的OpenCV内的代码如下：

```C++
//算子
Mat Kernel(3, 3, CV_8SC1);
short scalar[3][3] = {
        {0,  -1, 0},
        {-1, 5,  -1},
        {0,  -1, 0}
};
//赋值
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
        Kernel.at<short>(i, j) = scalar[i][j];
```
然后我们使用OpenCV内的`filter2D(mat, mat, -1, Kenel)`函数即可完成图像与算子的卷积操作，即进行图像锐化的处理。锐化前后对比图如下：

![](https://i.niupic.com/images/2020/06/24/8jlU.png)

可以看到我们经过算子的卷积操作之后图像相比原图像是清晰很多的。
## 3、图像降噪
同样的，我们也使用空间滤波进行图像的降噪处理，而中值滤波和均值滤波算法对于降噪来说是效果比较好的一种算法，但是我们这次验证码的噪声是椒盐噪声，而中值滤波作为一种非线性滤波能够很好的消除椒盐噪声，而均值滤波作为一种线性滤波器对椒盐噪声的消除效果不是很好，所以我们在这里使用的降噪算法为中值滤波。

中值滤波，顾名思义就是这些像素点的中间值，当我们的滤波器的大小给到3时，如下为中间像素点的周围像素点：

![](https://i.niupic.com/images/2020/06/24/8jm5.png)

我们需要将这9个像素点从高到低（或者从低到高）进行排序，然后取他们的中位数，即中值作为我们该像素的值。

中值滤波在OpenCV内的函数为`medianBlur(mat, mat, 3);`，最后的参数3代表的是滤波器的大小，我们这里选择的大小为3，即3*3的滤波器。我们使用一张带有椒盐噪声的验证码图像使用中值滤波进行降噪，运行结果如下：

![](https://i.niupic.com/images/2020/06/24/8jm9.png)

可以看到我们的经过之前的锐化和这次的降噪（主要是降噪）操作之后我们图像的噪声相比原图像之后已经减少很多了。
## 4、8邻域降噪
8邻域降噪作为在验证码内降噪比较常用的一种算法，其前提是将图片灰度化，在灰度图像内越接近白色的点像素越接近255，越接近黑色的点像素越接近0，而验证码字符肯定是非白色的。对于其中噪点大部分都是孤立的小点的，而且字符都是串联在一起的。

8邻域降噪的原理就是依次遍历图中所有非白色的点，计算其周围8个点中属于非白色点的个数，如果数量小于一个固定值，那么这个点就是噪点。对于不同类型的验证码这个阈值是不同的，所以可以在程序中配置，不断尝试找到最佳的阈值。

经过测试发现8邻域降噪对于小的噪点的去除是比较有效的，而且计算量不大，下图是阈值设置为4去噪后的结果：

![](https://i.niupic.com/images/2020/06/24/8jmp.png)

其中8邻域降噪法的代码实现如下：
```C++
//计算邻域非白色的个数
size_t calculateNoiseCount(Mat &img, size_t indexI, size_t indexJ) {
    size_t count = 0;
    size_t rows = img.rows;
    size_t cols = img.cols;
    for (size_t i = indexI - 1; i < indexI + 1; i++) {
        if (i < 0 || i >= rows)
            continue;
        for (size_t j = indexJ - 1; j < indexJ + 1; j++) {
            if (j < 0 || j >= cols)
                continue;
            if (img.at<uchar>(i, j) < 255)
                count++;
        }
    }
    return count;
}

//8邻域降噪，如果该点为黑，但是周围黑点小于4，那么则认为这个点是噪声
void noiseReduction(Mat &img, int k = 4) {
    std::vector<std::pair<size_t, size_t>> indexes;
    for (size_t i = 0; i < img.rows; i++) {
        for (size_t j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) < 255) {
                if (calculateNoiseCount(img, i, j) < 4)
                    indexes.emplace_back(i, j);
            }
        }
    }
    for (std::pair<size_t, size_t> pair:indexes) {
        img.at<uchar>(pair.first, pair.second) = 255;
    }
}
```
在后续操作中我会将8邻域降噪法和之前的中值滤波一起使用，即先进行中值滤波，再进行8邻域降噪（后进行8邻域降噪的原因是因为8邻域降噪法会将验证码字符的轨迹变细，会影响中值滤波降噪的操作）。将二者进行结合之后的图像降噪之后的效果如下图所示：

![](https://i.niupic.com/images/2020/06/24/8jmz.png)

可以看到效果比之前的单纯的中值滤波降噪和8邻域降噪的效果要好一些。
### 5、二值化
二值化，这一步主要是将上图中周围有一些灰度的去除，在代码内实现就比较简单了，OpenCV内有专门的二值化函数，我这里的代码如下：

```C++
threshold(mat, mat, 0, 255, THRESH_OTSU);
mat = 255 - mat;
```
最后经过上述的步骤处理结果如下图所示：

![](https://i.niupic.com/images/2020/06/24/8jmK.png)

我们选择一张既经过模糊化处理，又经过添加噪声之后的图像进行上述的处理，结果如下：

![](https://i.niupic.com/images/2020/06/24/8jmQ.png)

可以看到虽然结果还是不尽如人意，但是相比较原始的图像数据来说已经能看清大致了。

# 三、实现过程
## 1、图像分割
我们的验证码识别的第一步是将我们的验证码的每一个字符都分割出来，然后再通过BP神经网络进行训练识别，所以第一步的图像分割还是比较重要的。这里我们使用的是OpenCV自带的最小外接矩形算法，其函数声明为：

```C++
RotatedRect minAreaRect(InputArray points);
```
其中输入参数points是所要求最小外结矩的点集数组或向量。所以如果我们需要使用这个函数，那么我们就需要找到验证码图片内每一个字符所构成的点集或者是寻找这个字符的构成的轮廓。在OpenCV内有一个函数专门是可以来寻找轮廓的，该函数为`findContours()`，该函数的原型如下：
```C++
findContours( InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset=Point());
```
第一个参数：为Mat图像矩阵。

第二个参数：contours，是一个双重向量，向量内每个元素保存了一组由连续的Point点构成的点的集合的向量，每一组Point点集就是一个轮廓。

第三个参数：hierarchy，向量hiararchy内的元素和轮廓向量contours内的元素是一一对应的，向量的容量相同。

第四个参数：int型的mode，定义轮廓的检索模式，这里我们只用到其一个类型：`RETR_EXTERNAL`，表示只检测最外围轮廓。

第五个参数：int型的method，定义轮廓的近似方法，这里我们也只用其一种类型的参数`CHAIN_APPROX_NONE`，表示保存物体边界上所有连续的轮廓点到contours向量内。

第六个参数：Point偏移量，所有的轮廓信息相对于原始图像对应点的偏移量，相当于在每一个检测出的轮廓点上加上该偏移量，这个参数我们是用不上的。

所以在我们的代码里面的该函数的使用方法如下：
```C++
std::vector<std::vector<Point>> contours;
std::vector<Vec4i> hierarchy;
findContours(mat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
```
接着我们就可以通过获得的这些字符的轮廓信息，然后使用之前的最小外接矩形函数来进行验证码的分割处理。具体的代码如下：
```C++
std::vector<Rect> rectMat;
for (int i = 0; i < contours.size(); i++) {
    //求出最小外结矩形
    RotatedRect rect = minAreaRect(contours[i]);
    Point2f P[4];
    rect.points(P);
    //绘制最小外结矩形
    for(int j=0;j<=3;j++)
    {
        line(src,P[j],P[(j+1)%4],Scalar(255),2);
    }
}
```
我们一张没有进行模糊和降噪的图像运行上述代码的结果如下图所示：

![](https://i.niupic.com/images/2020/06/25/8jw2.png)

可以看到我们基本已经找到了每个字符的外接矩形，并用矩形给它标注了出来。

选取一张加入了椒盐噪声的图像，进行分割字符：

![](https://i.niupic.com/images/2020/06/25/8jw4.png)

可以看到矩形也能正确的包围起来我们所需要的字符，再选取一张经过模糊处理过的字符：

![](https://i.niupic.com/images/2020/06/25/8jw5.png)

可以看到这个时候就有一些问题了，中间那个V没有很好的包围起来。最后测试一个有噪声和模糊化处理的图像：

![](https://i.niupic.com/images/2020/06/25/8jw7.png)

可以看到这个时候效果就很差了，它是将所有的验证码全部识别成了一个全部相连的物体，并且使用最外层一个大矩形将其包围了起来。

接下来我们就需要取出我们的每个字符。可以看到我们之前使用的OpenCV的最小外接矩形函数算出来的矩形是有可能有旋转的情况的，这时候我们比较简单的做法是取这个旋转矩形的外接矩形作为我们的结果，然后再在原图像内取出我们所需要的这个字符，然后再进行BP神经网络的训练。我们取的过程还有可能会遇到一些噪声也被当做了字符，这时候区分的比较简单的方法是是我们的每一个验证码字符的高度不可能小于10（宽度有可能，如小写字母l和数字1）。而且还有可能碰到粘连的情况，我这里处理粘连的情况就比较简单了，直接每35个像素点宽度取一个字符（因为我们的验证码图像宽度是350，有10个字符，我们直接取平均值）。这部分的代码如下：
```C++
//对每个轮廓
for (int i = 0; i < contours.size(); i++) {
    //绘制轮廓的最小外结矩形
    RotatedRect rect = minAreaRect(contours[i]);
    //最小包围矩形的外接矩形，处理有旋转的情况
    Rect bound = rect.boundingRect();
    //检测有效性，不能下标越界了
    AssertRect(bound, mat.cols, mat.rows);
    //验证码字符的高度不可能小于10
    if (bound.height < 10)
        continue;
    if (bound.width > 50) {
        //如果宽度大于50，则认为有粘连的情况，按照每35个像素宽度进行取一个矩形块
        //35个像素宽度是因为一张验证码有10个字符，宽度为350个像素点
        int subSize = floor(bound.width / 35.0);
        int averageWidth = bound.width / subSize;
        for (int j = 0; j < subSize; j++) {
            Rect subRect(bound.x + averageWidth * j, bound.y, averageWidth, bound.height);
            rectMat.push_back(subRect);
        }
    } else {
        rectMat.push_back(bound);
    }
}
 ```
 接下来还有一步需要做的便是我们的这些外接矩形不是从左到右的顺序进行排列的，所以我们还得将其排序（之前是保存了每个外接矩形的数据信息），排序完毕后再在原图上取对应的矩形内的数据进行作为我们所需要的验证码字符数据。排序代码如下：
 ```C++
//按照顺序进行排序字母和数字
std::sort(rectMat.begin(), rectMat.end(), [](const Rect &rect1, const Rect &rect2) {
    return rect1.x < rect2.x;
});
//取得所有的字母和数字
std::vector<Mat> resultMat;
for (Rect &subRect:rectMat) {
    Mat subMat = mat(subRect);
    resultMat.push_back(subMat);
}
return resultMat;
 ```
 我们选取一张带有椒盐噪声的图像，使用上述的代码取出我们的所有验证码字符如下：

![](https://i.niupic.com/images/2020/06/25/8jx7.png)

![](https://i.niupic.com/images/2020/06/25/8jx5.png)

![](https://i.niupic.com/images/2020/06/25/8jx6.png)

可以看到效果还是不错的，接下来进行BP神经网络的训练。
## 2、BP神经网络
BP神经网络主要分为三块：输入层，隐藏层和输出层。

![](https://imgconvert.csdnimg.cn/aHR0cDovL2ltYWdlcy5jbml0YmxvZy5jb20vYmxvZy81NzEyMjcvMjAxNDExLzIzMTQyOTMyNDM3NzYwNS5wbmc)

在我们这个实验内我们的输入层的大小为28 * 28=784，表示我们输入的图像要重新调整大小到28 * 28，隐藏层大小我们将其定为128，输出层就和验证码的数字和字母的大小相等，为62。

BP（Back Propagation）神经网络主要分为两个过程：工作信号正向传递子过程和误差信号反向传递子过程。
### 前向传播 
对于每一个的神经元，先对输入数据加权求和加偏置：

![](https://math.jianshu.com/math?formula=x%3d%5csum_%7bi%3d0%7d%5e%7bn%7d%7bw_ix_i%7d%2bb)

其中w为权重，b为偏置值。

然后再使用激活函数进行激活，我们这里选择的激活函数是Sigmod，公式计算如下

![](https://math.jianshu.com/math?formula=f(x)%3d%5cfrac%7b1%7d%7b1%2be%5e%7b-x%7d%7d)

正向传播就是输入数据经过一层一层的神经元运算、输出的过程，最后一层输出值作为算法预测值y'。在代码内的实现如下：
```C++
//sigmod激活函数
double sigmoidal(double x) {
    return 1 / (1 + exp(-x));
}
//前向传播
void layerForward() {
    int i, j;
    double temp;
    for (i = 1; i <= hidden_n; i++)//input->hidden
    {
        temp = 0.0;
        for (j = 0; j <= input_n; j++)
            temp += input_units[j] * input_weights[j][i];
        hidden_units[i] = sigmoidal(temp);
    }
    for (i = 1; i <= output_n; i++)//hidden->out
    {
        temp = 0.0;
        for (j = 0; j <= hidden_n; j++)
            temp += hidden_units[j] * hidden_weights[j][i];
        output_units[i] = sigmoidal(temp);
    }
}
```
### 反向传播
反向传播的过程其实就是进行修正我们之前的权重。具体的公式计算步骤如下：
1. 先计算损失函数L(y',y)：

![](https://math.jianshu.com/math?formula=J(w%2Cb)%3D-%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%5By%5E%7B(i)%7Dlogy%27%5E%7B(i)%7D%2B(1-y%5E%7B(i)%7D)log(1-y%27%5E%7B(i)%7D)%5D)

2. 然后基于梯度下降原理更新参数：

![](https://math.jianshu.com/math?formula=w%27_j%3Dw_j-%20%5Calpha%20%5Cfrac%7Bd%7D%7Bdw_j%7D%20J(w%2Cb))

其中这里的α是梯度下降学习率。

反应到代码内的计算过程如下：
```C++
//反向传播调节权重
void adjustWeights() {
    int i, j;
    for (i = 0; i < hidden_n + 1; i++)             //out->hidden
        for (j = 1; j <= output_n; j++) {
            hidden_weights[i][j] += eta * output_delta[j] * hidden_units[i] + momentum * hidden_prev_weights[i][j];
            hidden_prev_weights[i][j] =
                    eta * output_delta[j] * hidden_units[i] + momentum * hidden_prev_weights[i][j];
        }
    for (i = 0; i <= input_n; i++)
        for (j = 1; j <= hidden_n; j++)    //hidden->input
        {
            input_weights[i][j] += eta * hidden_delta[j] * input_units[i] + momentum * input_prev_weights[i][j];
            input_prev_weights[i][j] = eta * hidden_delta[j] * input_units[i] + momentum * input_prev_weights[i][j];
        }
}
```
然后我们进行训练时就读入`data_train.txt`，然后对每一张图像都进行之前的图像分割，然后对每个分割后的字符数字进行前向传播和反向传播调整权值。具体的训练的代码如下：
```C++
double Etotal = 0.0;
fstream file_in;
file_in.open("../verification_code_dataset/data_train.txt");
std::string s;
//4000个验证码数据
for (int i = 0; i < MAX_TRAIN_TIMES; i++) {
    int data[10];
    std::string imageName;
    file_in >> imageName;
    //读入数据
    for (int j = 0; j < 10; j++)
        file_in >> data[j];
    //读入图片
    Mat mat = imread("../verification_code_dataset/train_images/" + imageName, CV_8UC1);
    //预处理
    Pretreatment(mat);
    //分割
    std::vector<Mat> splitMats = SplitLetterAndDigit(mat);
    std::size_t size = splitMats.size() == 10 ? 10 : splitMats.size();
    if (size != 10)
        continue;
    //对每个字符进行训练
    for (int j = 0; j < size; j++) {
        Mat trainMat = splitMats[j].clone();
        //resize到28*28
        resize(trainMat, trainMat, Size(28, 28));
        for (int k = 1; k <= 28; k++) {
            for (int kk = 1; kk <= 28; kk++) {
                input_units[k * 28 + kk] = trainMat.at<uchar>(k - 1, kk - 1) / 255.0;
            }
        }
        target[data[j] + 1] = 1;
        layerForward();     //前项传播
        Etotal += getOutputError();   //计算总的误差
        getHiddenError();   //计算中间层误差
        adjustWeights();    //反向传播调节
        target[data[j] + 1] = 0;
    }
}
file_in.close();
```
我对这4000组数据训练了50次，最后基于BP神经网络的误差

![](https://math.jianshu.com/math?formula=E_%7btotal%7d%3d%5csum%7b%5cfrac%7b1%7d%7b2%7d(target-output)%5e2%7d)

降低到了300左右。我们在后续进行验证码识别的时候我们只需要和训练的过程类似，将该验证码分割成字符，然后对每个字符进行一次前向传播，然后找出输出层最大的那个位置，作为我们的输出结果。反应到代码内如下：
```C++
layerForward();
int index = 1;
double max_num = output_units[1];        //找到输出层数组元素最大的值的元素位置作为目标输出
for (int i = 1; i <= output_n; i++)
    if (max_num < output_units[i]) {
        max_num = output_units[i];
        index = i;
    }
//输出结果
std::cout << letters[index - 1] << " ";
```
# 四、项目结果
我们将四种不同类型的验证码都测试一下识别效果：

![](https://i.niupic.com/images/2020/06/25/8jDZ.png)

原始未经过处理的验证码（f识别成了T，小写z识别成了Z，F识别成了m）

![](https://i.niupic.com/images/2020/06/25/8jDX.png)

经过模糊化处理的验证码（O识别成了U，后面几个都错了）

![](https://i.niupic.com/images/2020/06/25/8jEf.png)

加入噪声的验证码，好像大部分都错了

![](https://i.niupic.com/images/2020/06/25/8jEj.png)

加入模糊化和噪声的验证码，好像也是大部分都错了

最后测得本次所有的4000个验证码，一共40000个字符，识别率大致在60%左右，其中大部分问题出在字符的大小写问题以及同时经过模糊化和噪声处理的图像，这种情况下误识别率还是挺高的。

现在行业内大致有这几种验证码识别的方法：
1. OCR软件，OCR识别引擎Tesseract
   该方法的优点是：开发量少；比较通用，适合于各种变形较少的验证码；对于扭曲不严重的字母和数字识别率高。缺点也很明显：对于扭曲的字母和数字识别率大大降低；对于字符间有粘连的验证码几乎难以正确识别；很难针对特定网站的验证码做定制开发。 
2. 模板库匹配
   通常的做法是使用汉明距离或编辑距离定义相似度，并用KNN方法得到K个最相似的字符，最后从K个字符中选取出现次数最多的那个作为匹配结果。该方法的优点是：原理简单直观；可以针对不同网站定制优化；对于扭曲的字母和数字识别率较高。缺点是：开发量大，需要定制开发；需要收集大量的字符图片库；字符变化很多的情况，匹配次数增加速度下降；对于字符有粘连的图片识别率低.
3. 支持向量机   
   验证码识别问题实际上是其中单个字符识别问题，而在字符可穷举的情况下，比如只有英文字符和数字，单个字符识别问题其实是一个分类问题。一个英文字母或数字表示一类，而验证码中切分后得到的单个字符需要被机器自动分到某一类该方法的优点是：无需设计快速的图像匹配算法；只要图片切分方法合适，对于扭曲倾斜的字母和数字识别率也较高；并且可以针对不同类型的验证码做定制优化。缺点是：支持向量机原理比较复杂，无法直观解释，需要了解支持向量机等机器学习方法。
4. 神经网络  
   以上验证码识别都依赖于字符切分，切分的好坏几乎直接决定识别的准确程度。而对于有字符粘连的图片，往往识别率就会低很多。目前验证码识别最先进的是谷歌在识别“街景”图像中门牌号码中使用的一套的算法。该算法采用一种“深度卷积神经网络”（deep convolutional neural network）方法进行识别，准确率可以达到99%以上。

# 五、项目总结
自己的验证码识别和这些顶尖的二维码识相比较，技术和识别率还是相差很多的，问题主要出在BP神经网络好像不能很好的判断字母的大小，以及如果有噪声加入的话，效果也会很差，以后改进的话就不使用BP神经网络了，改用其他效果比较好的神经网络，可能最后的测试结果也会好很多。

Github链接：[https://github.com/MyLovePoppet/digital_letter_verification_code_recognition](https://github.com/MyLovePoppet/digital_letter_verification_code_recognition)
# 参考文献

[1].BP神经网络的梳理-巴拉巴拉9515
[https://www.jianshu.com/p/9037890c9b65](https://www.jianshu.com/p/9037890c9b65)

[2].常用验证码的识别方法-网易云社区
[https://juejin.im/post/5bea7d786fb9a049b77fe6f2](https://juejin.im/post/5bea7d786fb9a049b77fe6f2)

[3].Python3 识别验证码（opencv-python）-整合侠[https://www.cnblogs.com/lizm166/p/9969647.html](https://www.cnblogs.com/lizm166/p/9969647.html)