#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;
#define INPUT_N 784
#define HIDDEN_N 128
#define OUTPUT_N 62
#define MAX_TRAIN_TIMES 1000
short scalar[3][3] = {
        {0,  -1, 0},
        {-1, 5,  -1},
        {0,  -1, 0}
};
const Mat Kernel(3, 3, CV_8SC1, scalar);
const string letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
const int imageNums[] = {103, 101, 94, 91, 97, 92, 103, 78, 85, 90, 105, 99, 88, 104, 104, 98, 89, 113, 101, 110, 117,
                         81, 87,
                         106, 76, 102,

                         107, 100, 114, 87, 115, 110, 119, 95, 95, 125, 120, 83, 92, 93, 102, 86, 98, 103, 97, 112, 97,
                         111,
                         24, 106, 98, 87,

                         100, 95, 103, 119, 110, 95, 108, 114, 85, 111};

const size_t charLength = 62;

template<class T>
void clamp(T &x, T min, T max) {
    if (x < min)
        x = min;
    else if (x > max)
        x = max;
}

void AssertRect(Rect &bound, int cols, int rows) {
    clamp(bound.x, 0, cols);
    clamp(bound.y, 0, rows);
    clamp(bound.height, 0, rows - bound.y);
    clamp(bound.width, 0, cols - bound.x);
}

//最小外接矩形算法
vector<Mat> SplitLetterAndDigit(Mat &mat) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    //找轮廓
    findContours(mat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
    vector<Rect> rectMats;
    for (int i = 0; i < contours.size(); i++) {
        //绘制轮廓的最小外结矩形
        RotatedRect rect = minAreaRect(contours[i]);
        //最小包围矩形的外接矩形
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
                rectMats.push_back(subRect);
            }
        } else {
            rectMats.push_back(bound);
        }
    }
    //按照顺序进行排序字母和数字
    sort(rectMats.begin(), rectMats.end(), [](const Rect &rect1, const Rect &rect2) {
        return rect1.x < rect2.x;
    });
    //取得所有的字母和数字
    vector<Mat> resultMat;
    for (Rect &subRect:rectMats) {
        Mat subMat = mat(subRect);
        resultMat.push_back(subMat);
    }
    return resultMat;
}

//图像锐化，使图像更加清晰
void sharpen(Mat &img, const Mat &kenel = Kernel) {
    filter2D(img, img, -1, kenel);
}

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
    vector<pair<size_t, size_t>> indexes;
    for (size_t i = 0; i < img.rows; i++) {
        for (size_t j = 0; j < img.cols; j++) {
            if (img.at<uchar>(i, j) < 255) {
                if (calculateNoiseCount(img, i, j) < 4)
                    indexes.emplace_back(i, j);
            }
        }
    }
    for (pair<size_t, size_t> pair:indexes) {
        img.at<uchar>(pair.first, pair.second) = 255;
    }
}

//预处理
void Pretreatment(Mat &mat) {

    //拉普拉斯算子锐化
    sharpen(mat);
    //中值滤波
    medianBlur(mat, mat, 3);

    //8邻域降噪
    noiseReduction(mat);
    //二值化
    threshold(mat, mat, 0, 255, THRESH_OTSU);
    mat = 255 - mat;

}

/* Returns length of LCS for X[0..m-1], Y[0..n-1] */
int lcs(string X, string Y, int m, int n) {
    if (m == 0 || n == 0)
        return 0;
    if (X[m - 1] == Y[n - 1])
        return 1 + lcs(X, Y, m - 1, n - 1);
    else
        return max(lcs(X, Y, m, n - 1), lcs(X, Y, m - 1, n));
}

//参考代码，来自于https://www.cnblogs.com/ronny/p/opencv_road_more_01.html
void Train() {
    // 512 x 512 零矩阵
    int width = 512, height = 512;
    Mat img = Mat::zeros(height, width, CV_8UC3);

    // 训练样本
    float train_data[6][2] = {{500, 60},
                              {245, 40},
                              {480, 250},
                              {160, 380},
                              {400, 25},
                              {55,  400}};
    float labels[6] = {0, 0, 0, 1, 0, 1};  // 每个样本数据对应的输出
    Mat train_data_mat(6, 2, CV_32FC1, train_data);
    Mat labels_mat(6, 1, CV_32FC1, labels);

    // BP 模型创建和参数设置
    Ptr<ml::ANN_MLP> bp = ml::ANN_MLP::create();

    Mat layers_size = (Mat_<int>(1, 3) << 2, 6, 1); // 2维点，1维输出
    bp->setLayerSizes(layers_size);

    bp->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.1, 0.1);
    bp->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM);
    bp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 10000, /*FLT_EPSILON*/1e-6));
    // 保存训练好的神经网络参数
    bool trained = bp->train(train_data_mat, ml::ROW_SAMPLE, labels_mat);
    if (trained) {
        bp->save("bp_param");
    }

    // 创建训练好的神经网络
//    Ptr<ml::ANN_MLP> bp = ml::ANN_MLP::load("bp_param");

    // 显示分类的结果
    Vec3b green(0, 255, 0), blue(255, 0, 0);
    for (auto i = 0; i < img.rows; ++i) {
        for (auto j = 0; j < img.cols; ++j) {
            Mat sample_mat = (Mat_<float>(1, 2) << j, i);
            Mat response_mat;
            bp->predict(sample_mat, response_mat);
            float response = response_mat.ptr<float>(0)[0];
            if (response > 0.5) {
                img.at<Vec3b>(i, j) = green;
            } else if (response < 0.5) {
                img.at<Vec3b>(i, j) = blue;
            }
        }
    }

    // 画出训练样本数据
    int thickness = -1;
    int lineType = 8;
    circle(img, Point(500, 60), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(img, Point(245, 40), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(img, Point(480, 250), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(img, Point(160, 380), 5, Scalar(0, 0, 255), thickness, lineType);
    circle(img, Point(400, 25), 5, Scalar(255, 255, 255), thickness, lineType);
    circle(img, Point(55, 400), 5, Scalar(0, 0, 255), thickness, lineType);

    imwrite("result.png", img);        // 保存训练的结果
    imshow("BP Simple Example", img);

    waitKey(0);
}

//分割字符到文件夹分类
void SplitCharToFile() {
    int charSize[charLength] = {0};
    fstream file_in;
    file_in.open("../verification_code_dataset/data_train.txt");
    for (int i = 0; i < MAX_TRAIN_TIMES; i++) {
        int data[10];
        string imageName;
        file_in >> imageName;
        for (int j = 0; j < 10; j++)
            file_in >> data[j];
        Mat mat = imread("../verification_code_dataset/train_images/" + imageName, CV_8UC1);
        Pretreatment(mat);
        vector<Mat> splitMats = SplitLetterAndDigit(mat);
        size_t size = splitMats.size() == 10 ? 10 : splitMats.size();
        if (size != 10)
            continue;
        for (int j = 0; j < size; j++) {
            Mat trainMat = splitMats[j].clone();
            resize(trainMat, trainMat, Size(28, 28));
            string fileName = "D:/shuqy-package/OperatingSystem/digital-letter-verification-code-recognition/verification_code_dataset/Split/";
            char c = letters[data[j]];
            if (isupper(c)) {
                fileName.push_back('-');
                fileName.push_back(c);
            } else {
                fileName.push_back(c);
            }
            fileName.push_back('/');
            fileName.append(to_string(charSize[data[j]]++) + ".jpg");
            imwrite(fileName, trainMat);
        }
        cout << imageName + "OK!" << endl;
    }
    for (size_t i = 0; i < letters.length(); i++) {
        cout << charSize[i] << " ";
    }
    cout << endl;
    file_in.close();
}

//创建训练数据
void createTrainMat(Mat &train_data_mat, Mat &labels_mat) {
    size_t imageSize = 0;
    for (int i:imageNums) {
        imageSize += i;
    }
    size_t currentImageIndex = 0;
    train_data_mat = Mat(imageSize, 28 * 28, CV_32FC1);
    labels_mat = Mat::zeros(imageSize, 62, CV_32FC1);
    fstream file_in;
    file_in.open("../verification_code_dataset/data_train.txt");
    for (int i = 0; i < MAX_TRAIN_TIMES; i++) {
        int data[10];
        string imageName;
        file_in >> imageName;
        for (int j = 0; j < 10; j++)
            file_in >> data[j];
        Mat mat = imread("../verification_code_dataset/train_images/" + imageName, CV_8UC1);
        //预处理
        Pretreatment(mat);
        vector<Mat> splitMats = SplitLetterAndDigit(mat);
        size_t size = splitMats.size() == 10 ? 10 : splitMats.size();
        if (size != 10)
            continue;
        for (int j = 0; j < size; j++, currentImageIndex++) {
            Mat trainMat = splitMats[j].clone();
            resize(trainMat, trainMat, Size(28, 28));
            trainMat.convertTo(trainMat, CV_32FC1, 1.0 / 255.0);
            //训练数据
            for (int ii = 0; ii < 28; ii++)
                for (int jj = 0; jj < 28; jj++) {
                    train_data_mat.at<float>(currentImageIndex, ii * 28 + jj) = trainMat.at<float>(ii, jj);
                }
            //label数据
            labels_mat.at<float>(currentImageIndex, data[j]) = 1.0;
        }
    }
    file_in.close();
}

//开始训练
void MyTrain() {
    Mat train_data_mat;
    Mat labels_mat;
    createTrainMat(train_data_mat, labels_mat);

    // BP 模型创建和参数设置
    Ptr<ml::ANN_MLP> bp = ml::ANN_MLP::create();
    //784*128*62
    Mat layers_size = (Mat_<int>(1, 3) << INPUT_N, HIDDEN_N, OUTPUT_N);
    bp->setLayerSizes(layers_size);
    //SIGMOD函数激活
    bp->setActivationFunction(ml::ANN_MLP::ActivationFunctions::SIGMOID_SYM);
    bp->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.05, 0.05);
    bp->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 100, /*FLT_EPSILON*/ 0.0001));
    // 保存训练好的神经网络参数
    bool trained = bp->train(train_data_mat, ml::ROW_SAMPLE, labels_mat);
    if (trained) {
        bp->save("bp_param");
    }
}

//预测一张验证码数据
void Predict(Mat &mat) {
    Ptr<ml::ANN_MLP> bp = ml::ANN_MLP::load("bp_param");
    Pretreatment(mat);
    vector<Mat> mats = SplitLetterAndDigit(mat);
    Mat result;
    for (Mat &_mat:mats) {
        //调整大小到28*28
        resize(_mat, _mat, Size(28, 28));
        _mat.convertTo(_mat, CV_32FC1, 1.0 / 255);
        //将28*28的图像转化为1*(28*28)，准备Predict预测操作
        Mat A_mat(1, _mat.rows * _mat.cols, CV_32FC1);
        for (int i = 0; i < _mat.rows; i++)
            for (int j = 0; j < _mat.cols; j++) {
                A_mat.at<float>(0, i * _mat.rows + j) = _mat.at<float>(i, j);
            }
        bp->predict(A_mat, result);
        //找到最大值的位置
        Point maxLoc;
        //将该字符分类给该标签的置信值
        double maxVal = 0;
        minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);
        cout << "标签位置：" << maxLoc << " 字符: " << letters[maxLoc.x] << " 置信值: " << maxVal << endl;
    }
}

//预测所有的验证码数据，并计算正确率
void Predict() {
    size_t correct = 0;
    Ptr<ml::ANN_MLP> bp = ml::ANN_MLP::load("bp_param");
    fstream file_in;
    file_in.open("../verification_code_dataset/data_train.txt");
    for (int i = 0; i < MAX_TRAIN_TIMES; i++) {
        int data[10];
        string imageName;
        //真实的结果
        string offer;
        file_in >> imageName;
        for (int j = 0; j < 10; j++) {
            file_in >> data[j];
            offer.push_back(letters[data[j]]);
        }
        Mat mat = imread("../verification_code_dataset/train_images/" + imageName, CV_8UC1);
        Pretreatment(mat);
        vector<Mat> splitMats = SplitLetterAndDigit(mat);
        size_t size = splitMats.size() == 10 ? 10 : splitMats.size();
        //预测的结果
        string predict;
        for (int j = 0; j < size; j++) {
            Mat trainMat = splitMats[j].clone();
            resize(trainMat, trainMat, Size(28, 28));
            trainMat.convertTo(trainMat, CV_32FC1, 1.0 / 255.0);
            //生成训练的1*(28*28)数据
            Mat A_mat(1, trainMat.rows * trainMat.cols, CV_32FC1);
            for (int ii = 0; ii < trainMat.rows; ii++)
                for (int jj = 0; jj < trainMat.cols; jj++) {
                    A_mat.at<float>(0, ii * trainMat.rows + jj) = trainMat.at<float>(ii, jj);
                }
            Mat result;
            //预测
            bp->predict(A_mat, result);
            Point maxLoc;
            double maxVal = 0;
            minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);
            predict.push_back(letters[maxLoc.x]);
        }
        cout << imageName << "\t原始数据:" + offer << "\t预测数据:" << predict << endl;
        correct += lcs(offer, predict, offer.length(), predict.length());
    }
    //正确率
    cout << "correct rate: " << correct * 1.0 / (1000 * 10) << endl;
    file_in.close();
}

int main() {
    //SplitCharToFile();
    //MyTrain();
    Mat mat = imread("../verification_code_dataset/train_images/3.jpg", CV_8UC1);
    imshow("mat", mat);
    Predict(mat);
    waitKey(0);
    return 0;
}