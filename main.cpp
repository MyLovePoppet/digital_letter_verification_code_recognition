#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

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

int main() {
    Mat mat = imread("../verification_code_dataset/train_images/1591854364_6424382.jpg", CV_8UC1);
    Mat mat_copy=mat.clone();
    if (mat.empty()) {
        std::cout << "Image path error!" << std::endl;
        return -1;
    }
    //cvtColor(mat,mat,COLOR_RGB2GRAY);
    blur(mat,mat,Size(3,3));
    threshold(mat,mat,0,255,THRESH_OTSU);
    mat=255-mat;

    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(mat,contours,hierarchy,RETR_EXTERNAL, CHAIN_APPROX_NONE,Point());
    Mat imageContours=Mat::zeros(mat.size(),CV_8UC1);
    std::cout<<contours.size()<<std::endl;
    for(int i=0;i<contours.size();i++)
    {
        //绘制轮廓
        drawContours(imageContours,contours,i,Scalar(255),1,8,hierarchy);

        //绘制轮廓的最小外结矩形
        RotatedRect rect=minAreaRect(contours[i]);
        Point2f P[4];
        rect.points(P);
        for(int j=0;j<=3;j++)
        {
            line(imageContours,P[j],P[(j+1)%4],Scalar(255),2);
            line(mat_copy,P[j],P[(j+1)%4],Scalar(255),2);
        }

    }
    imshow("MinAreaRect",mat_copy);
    waitKey(0);
    return 0;
}
