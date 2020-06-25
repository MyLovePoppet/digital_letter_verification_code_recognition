#include <iostream>
#include <cmath>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;
#define INPUT_N 784
#define HIDDEN_N 128
#define OUTPUT_N 62
#define MAX_TRAIN_TIMES 4000
Mat Kernel(3, 3, CV_8SC1);
const char letters[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

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

//��С��Ӿ����㷨
std::vector<Mat> SplitLetterAndDigit(Mat &mat) {
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    //������
    findContours(mat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
    std::vector<Rect> rectMats;
    for (int i = 0; i < contours.size(); i++) {
        //������������С������
        RotatedRect rect = minAreaRect(contours[i]);
        //��С��Χ���ε���Ӿ���
        Rect bound = rect.boundingRect();
        //�����Ч�ԣ������±�Խ����
        AssertRect(bound, mat.cols, mat.rows);
        //��֤���ַ��ĸ߶Ȳ�����С��10
        if (bound.height < 10)
            continue;
        if (bound.width > 50) {
            //�����ȴ���50������Ϊ��ճ�������������ÿ35�����ؿ�Ƚ���ȡһ�����ο�
            //35�����ؿ������Ϊһ����֤����10���ַ������Ϊ350�����ص�
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
    //����˳�����������ĸ������
    std::sort(rectMats.begin(), rectMats.end(), [](const Rect &rect1, const Rect &rect2) {
        return rect1.x < rect2.x;
    });
    //ȡ�����е���ĸ������
    int i=0;
    std::vector<Mat> resultMat;
    for (Rect &subRect:rectMats) {
        Mat subMat = mat(subRect);
        resultMat.push_back(subMat);
    }
    return resultMat;
}

//ͼ���񻯣�ʹͼ���������
void sharpen(Mat &img, const Mat &kenel = Kernel) {
    filter2D(img, img, -1, kenel);
}

//��������ǰ�ɫ�ĸ���
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

//8�����룬����õ�Ϊ�ڣ�������Χ�ڵ�С��4����ô����Ϊ�����������
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

void Pretreatment(Mat &mat) {
    //������˹������
    sharpen(mat);
    //��ֵ�˲�
    medianBlur(mat, mat, 3);
    //8������
    noiseReduction(mat);
    //��ֵ��
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

class CBPNN {
private:
    int input_n;
    int hidden_n;
    int output_n;

    double *input_units;
    double *hidden_units;
    double *output_units;

    double *hidden_delta;
    double *output_delta;

    double *target;

    double **input_weights;
    double **hidden_weights;

    double **input_prev_weights;
    double **hidden_prev_weights;
    double eta;
    double momentum;

public:
    CBPNN() {
        int i, j;
        input_n = INPUT_N;
        hidden_n = HIDDEN_N;
        output_n = OUTPUT_N;


        input_units = new double[input_n + 1];
        hidden_units = new double[hidden_n + 1];
        output_units = new double[output_n + 1];


        hidden_delta = new double[hidden_n + 1];
        output_delta = new double[output_n + 1];

        target = new double[output_n + 1];

        input_weights = new double *[input_n + 1];
        for (i = 0; i < input_n + 1; i++)
            input_weights[i] = new double[hidden_n + 1];
        hidden_weights = new double *[hidden_n + 1];
        for (i = 0; i < hidden_n + 1; i++)
            hidden_weights[i] = new double[output_n + 1];

        input_prev_weights = new double *[input_n + 1];
        for (i = 0; i < input_n + 1; i++)
            input_prev_weights[i] = new double[hidden_n + 1];
        for (i = 0; i < input_n + 1; i++)
            for (j = 0; j < hidden_n + 1; j++)
                input_prev_weights[i][j] = 0.0;
        hidden_prev_weights = new double *[hidden_n + 1];
        for (i = 0; i < hidden_n + 1; i++)
            hidden_prev_weights[i] = new double[output_n + 1];
        for (i = 0; i < hidden_n + 1; i++)
            for (j = 0; j < output_n; j++)
                hidden_prev_weights[i][j] = 0.0;
        hidden_units[0] = 1.0;
        input_units[0] = 1.0;
        eta = 0.3;
        momentum = 0.3;
    }

    void initSeed() {
        srand(time(NULL));
    }

    void initWeights()   //��ʼ��weights
    {
        int i, j;
        initSeed();
        for (i = 0; i < input_n + 1; i++)
            for (j = 1; j <= hidden_n; j++)
                input_weights[i][j] = (double) (rand()) / (32767 / 2) - 1;
        for (i = 0; i < hidden_n + 1; i++)
            for (j = 1; j <= output_n; j++)
                hidden_weights[i][j] = (double) (rand()) / (32767 / 2) - 1;
    }

    void saveWeights()   //����weights
    {
        int i, j;
        fstream file_out;
        file_out.open("input_weights_data.txt", ios::out);
        if (!file_out.is_open())
            exit(-1);
        for (i = 0; i < input_n + 1; i++)
            for (j = 1; j <= hidden_n; j++)
                file_out << input_weights[i][j] << endl;
        file_out.close();
        file_out.open("hidden_weights_data.txt", ios::out);
        if (!file_out.is_open())
            exit(-1);
        for (i = 0; i < hidden_n + 1; i++)
            for (j = 1; j <= output_n; j++)
                file_out << hidden_weights[i][j] << endl;
        file_out.close();
    }

    void readData() {
        int K = 10;
        while (K-- > 0) {
            double Etotal = 0.0;
            fstream file_in;
            file_in.open("../verification_code_dataset/data_train.txt");
            std::string s;
            for (int i = 0; i < MAX_TRAIN_TIMES; i++) {
                int data[10];
                std::string imageName;
                file_in >> imageName;
                for (int j = 0; j < 10; j++)
                    file_in >> data[j];
                Mat mat = imread("../verification_code_dataset/train_images/" + imageName, CV_8UC1);
                Pretreatment(mat);
                std::vector<Mat> splitMats = SplitLetterAndDigit(mat);
                std::size_t size = splitMats.size() == 10 ? 10 : splitMats.size();
                if (size != 10)
                    continue;
                for (int j = 0; j < size; j++) {
                    Mat trainMat = splitMats[j].clone();
                    resize(trainMat, trainMat, Size(28, 28));
                    for (int k = 1; k <= 28; k++) {
                        for (int kk = 1; kk <= 28; kk++) {
                            input_units[k * 28 + kk] = trainMat.at<uchar>(k - 1, kk - 1) / 255.0;
                        }
                    }
                    target[data[j] + 1] = 1;
                    layerForward();     //ǰ���
                    Etotal += getOutputError();   //�����ܵ����
                    getHiddenError();   //�����м�����
                    adjustWeights();    //���򴫲�����
                    target[data[j] + 1] = 0;
                }
            }
            file_in.close();
            cout << 50 - K << ": " << Etotal << endl;
        }
    }

    void readWeights() //��ȡweights
    {
        int i, j;
        fstream file_in;
        file_in.open("input_weights_data.txt", ios::in);
        if (!file_in.is_open())
            exit(-1);
        for (i = 0; i < input_n + 1; i++)
            for (j = 1; j <= hidden_n; j++)
                file_in >> input_weights[i][j];
        file_in.close();
        file_in.open("hidden_weights_data.txt", ios::in);
        if (!file_in.is_open())
            exit(-1);
        for (i = 0; i < hidden_n + 1; i++)
            for (j = 1; j <= output_n; j++)
                file_in >> hidden_weights[i][j];
        file_in.close();
    }

    //sigmod�����
    double sigmoidal(double x) {
        return 1 / (1 + exp(-x));
    }

    //ǰ�򴫲�
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

    double getOutputError() {
        int i;
        double Etotal = 0.0;  //calculate the error
        for (i = 1; i <= output_n; i++) {
            output_delta[i] = output_units[i] * (1.0 - output_units[i]) * (target[i] - output_units[i]);
            Etotal += fabs(output_delta[i]);
        }
        return Etotal;
    }

    double getHiddenError() {
        int i, j;
        double Etotal = 0.0;
        for (i = 1; i <= hidden_n; i++) {
            double temp = 0.0;
            for (j = 1; j <= output_n; j++)
                temp += output_delta[j] * hidden_weights[i][j];
            hidden_delta[i] = hidden_units[i] * (1.0 - hidden_units[i]) * temp;
            Etotal += fabs(hidden_delta[i]);
        }
        return Etotal;
    }

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

    void test() {
        int sum_correct;
        fstream file_in;
        file_in.open("../verification_code_dataset/data_train.txt");
        std::string s;
        for (int i = 0; i < MAX_TRAIN_TIMES; i++) {
            int data[10];
            std::string imageName;
            file_in >> imageName;
            string correct;
            string predict;
            for (int j = 0; j < 10; j++) {
                file_in >> data[j];
                correct.push_back(letters[data[j]]);
            }

            Mat mat = imread("../verification_code_dataset/train_images/" + imageName, CV_8UC1);
            Pretreatment(mat);
            std::vector<Mat> splitMats = SplitLetterAndDigit(mat);
            std::size_t size = splitMats.size() == 10 ? 10 : splitMats.size();
            for (int j = 0; j < size; j++) {
                Mat trainMat = splitMats[j].clone();
                resize(trainMat, trainMat, Size(28, 28));
                for (int k = 1; k <= 28; k++) {
                    for (int kk = 1; kk <= 28; kk++) {
                        input_units[k * 28 + kk] = trainMat.at<uchar>(k - 1, kk - 1) / 255.0;
                    }
                }
                layerForward();     //ǰ���
                int index = 1;
                double max_num = output_units[1];        //�ҵ����������Ԫ����ӽ�1��Ԫ��λ����ΪĿ�����
                for (int kk = 1; kk <= output_n; kk++)
                    if (max_num < output_units[kk]) {
                        max_num = output_units[kk];
                        index = kk;
                    }
                predict.push_back(letters[index - 1]);
            }
            cout << correct << "\t" << predict << endl;
            int c = lcs(correct, predict, correct.length(), predict.length());
            sum_correct += c;
        }
        file_in.close();
        std::cout << "��ȷ�ʣ�" << sum_correct * 1.0 / (4000 * 10) << std::endl;
    }

    void test(Mat &mat) {
        imshow("ԭʼͼ��",mat);
        string predict;
        Pretreatment(mat);
        std::vector<Mat> splitMats = SplitLetterAndDigit(mat);
        std::size_t size = splitMats.size() == 10 ? 10 : splitMats.size();
        for (int j = 0; j < size; j++) {
            Mat trainMat = splitMats[j].clone();
            resize(trainMat, trainMat, Size(28, 28));
            for (int k = 1; k <= 28; k++) {
                for (int kk = 1; kk <= 28; kk++) {
                    input_units[k * 28 + kk] = trainMat.at<uchar>(k - 1, kk - 1) / 255.0;
                }
            }
            layerForward();     //ǰ���
            int index = 1;
            double max_num = output_units[1];        //�ҵ����������Ԫ����ӽ�1��Ԫ��λ����ΪĿ�����
            for (int kk = 1; kk <= output_n; kk++)
                if (max_num < output_units[kk]) {
                    max_num = output_units[kk];
                    index = kk;
                }
            predict.push_back(letters[index - 1]);
        }
        std::cout << "result: " << predict << std::endl;
    }

    ~CBPNN() {
        int i;
//        delete[]input_units;
//        delete[]hidden_units;
//        delete[]output_units;
//
//        delete[]hidden_delta;
//        delete[]output_delta;
//
//        delete[]target;
//
//        for (i = 0; i < input_n + 1; i++)
//            delete[]input_weights[i];
//        delete[]input_weights;
//
//        for (i = 0; i < hidden_n + 1; i++)
//            delete[]hidden_weights[i];
//        delete[]hidden_weights;
//
//        for (i = 0; i < input_n + 1; i++)
//            delete[]input_prev_weights[i];
//        delete[]input_prev_weights;
//        for (i = 0; i < hidden_n + 1; i++)
//            delete[]hidden_prev_weights[i];
//        delete[]hidden_prev_weights;
    }
};

int main() {
    short scalar[3][3] = {
            {0,  -1, 0},
            {-1, 5,  -1},
            {0,  -1, 0}
    };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Kernel.at<short>(i, j) = scalar[i][j];
    Mat mat=imread("../verification_code_dataset/train_images/1591854438_2914968.jpg",CV_8UC1);
    CBPNN BPNN;
    //BPNN.initWeights();
    BPNN.readWeights();
    //BPNN.readData();            //��ȡ���� ͬʱ����ǰ��������򴫲�������weights
    //BPNN.saveWeights();          //ѵ����� ����weights����
    BPNN.test(mat);                 //������ȷ�ʲ���
    waitKey(0);
    return 0;
}