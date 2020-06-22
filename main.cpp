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
        if (rect.size.area() < 64 || rect.size.area() > 4096)
            continue;
        if(rect.size.area()>1024){
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
    for (Rect &rect3:rectMat) {
        std::cout << rect3 << std::endl;
    }
    //取得所有的字母和数字
    std::vector<Mat> resultMat;
    for (Rect &subRect:rectMat) {
        Mat subMat = mat(subRect);
        imshow("11",subMat);
        waitKey(0);
        resultMat.push_back(subMat);
    }
    return resultMat;
}

int main() {
    Mat mat = imread("../verification_code_dataset/train_images/1591854437_698722.jpg", CV_8UC1);
    //resize(mat,mat,Size(mat.rows*2,mat.rows*2));
    if (mat.empty()) {
        std::cout << "Image path error!" << std::endl;
        return -1;
    }
    std::vector<Mat> lettersAndDigits = SplitLetterAndDigit(mat);
    return 0;
}


