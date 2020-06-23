#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<ctime>
#include<math.h>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
#define INPUT_N 784
#define HIDDEN_N 128
#define OUTPUT_N 62
#define MAX_TRAIN_TIMES 4000
#define MAX_TEST_TIMES 10000
using namespace cv;
//最小外接矩形算法
std::vector<Mat> SplitLetterAndDigit(Mat &src) {
    Mat mat = src.clone();
    blur(mat, mat, Size(3, 3));
    threshold(mat, mat, 0, 255, THRESH_OTSU);
    mat = 255 - mat;
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(mat, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
    Mat imageContours = Mat::zeros(mat.size(), CV_8UC1);

    std::vector<Rect> rectMat;
    for (int i = 0; i < contours.size(); i++) {
        //绘制轮廓
        drawContours(imageContours, contours, i, Scalar(255), 1, 8, hierarchy);
        //绘制轮廓的最小外结矩形
        RotatedRect rect = minAreaRect(contours[i]);
        Point2f P[4];
        rect.points(P);
        if (P[2].x < 0)
            P[2].x = 0;
        if (P[2].y < 0)
            P[2].y = 0;
        if (P[0].x > mat.cols)
            P[0].x = mat.cols;
        if (P[0].y > mat.rows)
            P[0].y = mat.rows;
        Rect subRect(P[2], P[0]);
        //如果面积小于64或者是大于1024个像素点则认为这不是一个有效的字符或者是数字
        //小于可能是噪声，大于可能是没有完全分割
        double area=subRect.width*subRect.height;
        if (area < 64 || area > 4096)
            continue;
        if(area>1024){
            Rect rect1(subRect.x,subRect.y,subRect.width/2,subRect.height);
            Rect rect2(subRect.x+subRect.width/2,subRect.y,subRect.width/2,subRect.height);
            rectMat.push_back(rect1);
            rectMat.push_back(rect2);
        }
        else
            rectMat.push_back(subRect);
    }
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
}

class CBPNN {
private:
    int input_n;                  // ?????????????
    int hidden_n;                 // ??????????????
    int output_n;                 // ?????????????

    double *input_units;          // ?????????????(input_units[i], i=1,2,...,input_n),????input_units[0]=1.0
    double *hidden_units;         // ?????????????(hidden_units[i], i=1,2,...,hidden_n),????hidden_units[0]=1.0
    double *output_units;         // ?????????????(output_units[i], i=1,2,...,output_n)

    double *hidden_delta;         // ???????????????(hidden_delta[i],i=1,...,hidden_n)
    double *output_delta;         // ???????????????(output_delta[i],i=1,...,output_n)

    double *target;               // ???????(target[i],i=1,...,output_n)

    double **input_weights;       // ???????????????????? input_weights[i][j]?????i??????????j?????????????????,input_weights[0][j]?????????j??????????
    double **hidden_weights;      // ???????????????????? hidden_weights[i][j]?????i??????????j?????????????????,hidden_weights[0][j]?????????j??????????

    // ?????????????????
    double **input_prev_weights;  // ???????????????????
    double **hidden_prev_weights; // ???????????????????
    double eta;                   // ??????,????0.3, hidden_weights = hidden_weights + eta*output_delta*hidden_units + momentum*hidden_prev_weights
    //hidden_prev_weights = eta*output_delta*hidden_units + momentum*hidden_prev_weights
    double momentum;              // ???????,????0.3, input_weights = input_weights + eta*hidden_delta*input_units + momentum*input_prev_weights
    //input_prev_weights = eta*hidden_delta*input_units + momentum*input_prev_weights

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

    void initWeights()   //初始化weights
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

    void saveWeights()   //保存weights
    {
        int i, j;
        FILE *fpWrite = fopen("input_weights_data.txt", "w");
        if (fpWrite == NULL)
            exit(-1);
        for (i = 0; i < input_n + 1; i++)
            for (j = 1; j <= hidden_n; j++)
                fprintf(fpWrite, "%lf\n", input_weights[i][j]);
        fclose(fpWrite);
        FILE *fqWrite = fopen("hidden_weights_data.txt", "w");
        if (fqWrite == NULL)
            exit(-1);
        for (i = 0; i < hidden_n + 1; i++)
            for (j = 1; j <= output_n; j++)
                fprintf(fqWrite, "%lf\n", hidden_weights[i][j]);
        fclose(fpWrite);
    }

    void readData() {
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
            std::cout<<"Current training image:"<<imageName<<std::endl;
            Mat mat = imread("../verification_code_dataset/train_images/" + imageName, CV_8UC1);
            std::vector<Mat> splitMats = SplitLetterAndDigit(mat);
            std::size_t size = splitMats.size() > 10 ? 10 : splitMats.size();
            for (int j = 0; j < size; j++) {
                Mat trainMat = splitMats[j].clone();
                resize(trainMat, trainMat, Size(28, 28));
                for (int k = 1; k <= 28; k++) {
                    for (int kk = 1; kk <= 28; kk++) {
                        input_units[k * 28 + kk] = trainMat.at<uchar>(k - 1, kk - 1) / 255.0;
                    }
                }
                //memset(target,0,sizeof(double)*OUTPUT_N);
                target[data[j]+1]=1;
                layerForward();     //前项传播
                Etotal += getOutputError();   //计算总的误差
                getHiddenError();   //计算中间层误差
                adjustWeights();    //反向传播调节
                target[data[j]+1]=0;
            }


        }
        file_in.close();
        cout << Etotal << endl;
            /*
            FILE *fpRead = fopen("train_sample.txt", "r");
            if (fpRead == NULL)
                exit(-1);
            times = MAX_TRAIN_TIMES;
            while (times--)   //采用了每次读入一组数据 便进行传播和调节weights
            {

                for (i = 1; i < input_n + output_n + 1; i++) {
                    if (i < input_n + 1)
                        fscanf(fpRead, "%lf", &input_units[i]);
                    else
                        fscanf(fpRead, "%lf", &target[i - input_n]);
                }
                layerForward();     //前项传播
                Etotal += getOutputError();   //计算总的误差
                getHiddenError();   //计算中间层误差
                adjustWeights();    //反向传播调节
            }
            cout << Etotal << endl;
            fclose(fpRead);
             */
        //}
    }

    void readWeights() //读取weights
    {
        int i, j;
        {
            FILE *fpRead = fopen("input_weights_data.txt", "r");
            if (fpRead == NULL)
                exit(-1);
            for (i = 0; i < input_n + 1; i++)
                for (j = 1; j <= hidden_n; j++)
                    fscanf(fpRead, "%lf", &input_weights[i][j]);
            fclose(fpRead);
        }
        {
            FILE *fpRead = fopen("hidden_weights_data.txt", "r");
            if (fpRead == NULL)
                exit(-1);
            for (i = 0; i < hidden_n + 1; i++)
                for (j = 1; j <= output_n; j++)
                    fscanf(fpRead, "%lf", &hidden_weights[i][j]);
            fclose(fpRead);
        }
    }

    double sigmoidal(double x) {
        return 1 / (1 + exp(-x));
    }

    void layerForward() {
        int i, j, k;
        double temp, Etotal;
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
            /*output_delta[i]=0.5*(output_units[i]-target[i])*(output_units[i]-target[i]);
            Etotal+=fabs(output_delta[i]);*/
            output_delta[i] = output_units[i] * (1.0 - output_units[i]) * (target[i] - output_units[i]);
            Etotal += fabs(output_delta[i]);
        }

        return Etotal;
    }

    double getHiddenError() {
        int i, j;
        /*double temp;
        for(i=1;i<=hidden_n;i++)
        {
            temp=0.0;
            for(j=1;j<=output_n;j++)
                temp+=output_delta[j]*hidden_weights[i][j];
            hidden_delta[i]=temp*hidden_units[i]*(1-hidden_units[i]);
        }*/
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
        int i, j, times = MAX_TEST_TIMES;
        int target_output, sum_correct = 0, index;
        double temp, Etotal, max_num;
        char a[28][28];
        FILE *fpRead = fopen("test_sample.txt", "r");
        if (fpRead == NULL)
            exit(-1);
        while (times--) {
            for (i = 1; i <= input_n + 1; i++) {
                if (i < input_n + 1)
                    fscanf(fpRead, "%lf", &input_units[i]);
                else
                    fscanf(fpRead, "%d", &target_output);
            }
            for (i = 0; i < 28; i++)
                for (j = 0; j < 28; j++) {
                    if (input_units[j + i * 28 + 1] > 0)
                        a[i][j] = 0 + '0';
                    else
                        a[i][j] = ' ';
                }
            layerForward();
            max_num = output_units[1];        //找到输出层数组元素最接近1的元素位置作为目标输出
            for (i = 1; i <= output_n; i++)
                if (max_num < output_units[i])
                    max_num = output_units[i];
            for (i = 1; i <= output_n; i++)
                if (fabs(max_num - output_units[i]) < 1e-5)
                    break;
            index = i;
            }
        cout << "ending..." << endl;
        cout << "correct Rate:" << sum_correct * 1.0 / MAX_TEST_TIMES << endl;//正确率
        fclose(fpRead);
    }

    ~CBPNN() {
        int i;
        delete[]input_units;
        delete[]hidden_units;
        delete[]output_units;

        delete[]hidden_delta;
        delete[]output_delta;

        delete[]target;

        for (i = 0; i < input_n + 1; i++)
            delete[]input_weights[i];
        delete[]input_weights;

        for (i = 0; i < hidden_n + 1; i++)
            delete[]hidden_weights[i];
        delete[]hidden_weights;

        for (i = 0; i < input_n + 1; i++)
            delete[]input_prev_weights[i];
        delete[]input_prev_weights;
        for (i = 0; i < hidden_n + 1; i++)
            delete[]hidden_prev_weights[i];
        delete[]hidden_prev_weights;
    }
};

int main() {

    CBPNN BPNN;
    BPNN.initWeights();
    BPNN.readData();            //读取数据 同时进行前项传播，反向传播，调节weights
    BPNN.saveWeights();          //训练完成 保存weights数据
    BPNN.readWeights();
    //BPNN.test();                 //进行正确率测试
    //system("pause");

//    Mat mat=imread("../verification_code_dataset/train_images/1591854365_812406.jpg",CV_8UC1);
//    std::vector<Mat>result=SplitLetterAndDigit(mat);
//    for(Mat& inMat:result){
//        resize(inMat,inMat,Size(28,28));
//        imshow("41",inMat);
//        waitKey(0);
//    }
    return 0;
}
/*
/*  void initSeed(int seed) { srand(seed); }     // 设置随机数生成器种子
	void readBPNN(SBPNN *, char* filename);    // 读取保存的网络配置及加权系数
	void saveBPNN(SBPNN *, char *filename);    // 保存当前的网络配置及加权系数
	SBPNN* createBPNN(int n_in,int n_hidden,int n_out);  //创建一个BP网络，并初始化权值(随机化intput_weidhts和hidden_weights在-0.05到0.05之间, 清零input_prev_weights和hidden_prev_weights)
	void freeBPNN(SBPNN *);
	void test(SBPNN *, double *input_unit,int input_num,double *target,int target_num);  //测试, 给定输入input_unit, 返回前向传播后得到的输出target
	void train(SBPNN *, double *input_unit,int input_num, double *target,int target_num, double *eo, double *eh); //训练样本中的某个输入向量input_unit和期望输出向量target,eo为本次训练输出层误差,eh为隐含层误差
	void adjustWeights(double *delta, int ndelta, double *ly, int nly, double** w, double **oldw, double eta, double momentum); //更新加权系数,delta是隐藏层或输入层的反向误差, ly是隐藏层或输入层的输出, w是隐藏层或输入层的加权系数, oldw是隐藏层或输入层上一次更新的加权系数
	void getHiddenError(double* delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err);  //计算反向调节时的隐藏层误差delta_h,delta_o是输出层误差,who是隐藏层到输出层的加权系数, hidden是隐藏层的实际输出,err是隐藏层各节点误差绝对值的总和
	void getOutputError(double *delta, double *target, double *output, int nj, double *err);  //计算反向调节时的输出层误差delta,target是输出层期望输出,output是输出层实际输出,err是输出层各节点误差绝对值的总和
	void layerforward(double *l1, double *l2, double **conn, int n1, int n2); //执行一次l1到l2的前向传播, conn是连接的加权系数
	double sigmoidal(double x);
*/