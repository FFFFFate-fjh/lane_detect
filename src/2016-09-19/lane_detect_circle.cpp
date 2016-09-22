#include<iostream>
#include<fstream>
#include<string>
#include <unistd.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "MSAC.h"

#define USE_PPHT
#define MAX_NUM_LINES	300

using namespace std;
using namespace cv;

Mat src, src_gray, src_bak;
//Mat dst, detected_edges;

vector<Mat> img;
vector<string> list;
int edgeThresh = 1;
int lowThreshold=100;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

int ROI_x=400;
int ROI_y=220;

int ema_T=5;
float ema_a = 2.0/(ema_T+1.0);
float ema_current = 0;
float ema_last = 0;
float ema_init = 0;
float ema_init_count = 50;
float ema_init_i = 0;
int ema_init_total = 0;
bool ema_init_flag = true;

vector<vector<cv::Point> > lineSegments;
vector<cv::Point> aux;

Mat sobel(Mat src);  //sobel计算
Mat set_ROI(Mat src, int ROI_L, int ROI_R, int width, int height);   //设置ROI大小
int autoCanny(Mat &srcmat, float type);
void hough_p_compute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec4i> &lines,
                    int thres, int hlen, int minpoints, int ROI_x, int ROI_y);
void houghcompute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec2f> &lines,
                    int thres, int hlen, int minpoints,int ROI_x, int ROI_y);
void vanishedpoint(Mat &src );

float ema_1(int ema_input);

int main( int argc, char** argv )
{
    string filename = "data/21/image_0/list.txt"; //使用正则表达生成图片名字的list
    ifstream ifile;
    ifile.open(filename.c_str());

    int count=0;

    string tmp;
    while( getline(ifile,tmp) )
    {
        list.push_back(tmp);
        cout << "Read from file: " << list[count] << endl;
        count++;
        tmp.clear();
    }
    ifile.close();

  int img_num=list.size()-4;
  cout<<"img num: "<<img_num<<endl;
  int img_begin=3;
  int img_end=list.size()-2;

  cout<<"img_end: "<<img_end<<endl;

  for(int i=img_begin;i<=img_end;i++)  //读图
  {
      char *addr_buf;
      int len = list[i].length();
      addr_buf =new char[len+1];
      strcpy(addr_buf,list[i].c_str());
      char img_buf[100];
      sprintf(img_buf, "./data/21/image_0/%s", addr_buf);
      string img_addr = img_buf;
      cout<<img_addr<<endl;
      Mat src=imread(img_addr,0);
      if( !src.data )
      { return -1; }
      Mat src_heq;
      equalizeHist( src, src_heq );
      img.push_back(src_heq);
//      img.push_back(src);
  }

  for(int i=0;i<img.size(); i++)
  {
     cout<<i<<endl;

     Mat src_heq ;
     equalizeHist( img[i], src_heq );

     Mat ROI_img2 = set_ROI(src_heq,200, 150, 800, 369-150);
     Mat ROI_img = set_ROI(src_heq,ROI_x, ROI_y, 450, 369-ROI_y);

     Mat grad, abs_grad_x;
     medianBlur(ROI_img,grad,3);
     abs_grad_x = sobel(grad);

     /// Canny detector
     int thres1,thres2,thres3,thresmax;
     thres1=autoCanny(src_heq, 1.0);
     thres2=autoCanny(ROI_img, 1.0);
     thres3=autoCanny(ROI_img2, 1.0);
     thresmax=max(thres1,thres2);
     thresmax=max(thresmax,thres3);
     thresmax = (int)ema_1(thresmax);

     Mat detected_edges;
     Canny( abs_grad_x, detected_edges, thresmax, thresmax*3, kernel_size );

     Mat hdst, roi_show;
     cvtColor(detected_edges, hdst, CV_GRAY2BGR);
     cvtColor(src_heq, roi_show, CV_GRAY2BGR);

     vector<Vec4i> lines;
     hough_p_compute(detected_edges, roi_show, hdst,lines,75, 10, 20,ROI_x,ROI_y);

     vanishedpoint(roi_show);

     imshow("source", detected_edges);
     imshow("detected lines", hdst);
     imshow("detected lines on ROI", roi_show);

        char img_buf[100];
        sprintf(img_buf, "./output/21/image_0/%d.png", i);
        string img_addr = img_buf;
        imwrite(img_addr, roi_show);
        waitKey(50);
  }

  return 0;
  }

Mat sobel(Mat src)
{
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    Mat grad_x;
    Mat abs_grad_x;

    Sobel( src, grad_x, ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    return abs_grad_x;
}

Mat set_ROI(Mat src, int ROI_L, int ROI_R, int width, int height)
{
//    Rect roi(ROI_x, ROI_y, 450, 369-ROI_y);
    Rect roi(ROI_L, ROI_R, width, height);
    Mat roi_of_image = src(roi);
    Mat dst= src.clone();
    rectangle(dst,roi,Scalar(0,0,255),3);
    imshow("RECT",dst);
    return roi_of_image;
}

int autoCanny(Mat &srcmat, float type)
{
    // Check input Mat type is CV_8UC1
    if (srcmat.type() != CV_8UC1)
        CV_Error(CV_StsUnsupportedFormat, "");

    int total_piexl = 0;
    int nr = srcmat.rows;
    int nc = srcmat.cols;

    // If the input and output mat is store continuous in memory, then loop
    // the Mat just in one rows will be much more quickly.
    if (srcmat.isContinuous()) {
        nr = 1;
        nc = nc * srcmat.rows;
    }

    // Means gray level in image.
    for (int i = 0; i < nr; i++) {
        const uchar *src_data = srcmat.ptr<uchar>(i);
        for (int j = 0; j < nc; j++) {
            total_piexl += *src_data;
        }
    }

    total_piexl /= nr * nc; // means gray level

    // the following threshould value is calc from experience.
    int thres1 = int(type * total_piexl);
    int thres2 = int(type * 3 * total_piexl);
    cout<<"thres1  "<<thres1<<"  "<<"thres2  "<<thres2<<endl;

    return thres1;
}

void hough_p_compute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec4i> &lines, int thres, int hlen, int minpoints, int ROI_x, int ROI_y)
{
    lineSegments.clear();

    int houghThreshold = 10;
    if(src.cols*src.rows < 400*400)
        houghThreshold = 20;

    cv::HoughLinesP(src, lines, 1, CV_PI/180, houghThreshold, 10,10);

    while(lines.size() > MAX_NUM_LINES)
    {
        lines.clear();
        houghThreshold += 1;
        cv::HoughLinesP(src, lines, 1, CV_PI/180, houghThreshold, 10, 10);
    }

    cout<<"lines.size():  "<<lines.size()<<endl;

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        float k = ((l[0]-l[2])*1.0)/(l[1]-l[3]);
        if(k>0.75 || k <-0.75)
        {
            line( roi_img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
            l[0]=l[0]+ROI_x;
            l[2]=l[2]+ROI_x;
            l[1]=l[1]+ROI_y;
            l[3]=l[3]+ROI_y;
            line( dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
            Point pt1, pt2;
            pt1.x = lines[i][0];
            pt1.y = lines[i][1];
            pt2.x = lines[i][2];
            pt2.y = lines[i][3];
            aux.clear();
            aux.push_back(pt1);
            aux.push_back(pt2);
            lineSegments.push_back(aux);
        }
    }
}

void houghcompute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec2f> &lines,
                  int thres, int hlen, int minpoints,int ROI_x, int ROI_y)
{
    HoughLines(src, lines, 1, CV_PI/180, thres, hlen, minpoints );

    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        float k = ((pt1.x-pt2.x)*1.0)/(pt1.y-pt2.y);
        if(k<1 && k >-1)
        {
            line( roi_img, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
            pt1.x=pt1.x+ROI_x;
            pt2.x=pt2.x+ROI_x;
            pt1.y=pt1.y+ROI_y;
            pt2.y=pt2.y+ROI_y;
            line( dst, pt1, pt2, Scalar(0,0,255), 1, CV_AA);
        }

    }
}

void vanishedpoint(Mat &src)
{
    // Multiple vanishing points
    std::vector<cv::Mat> vps;			// vector of vps: vps[vpNum], with vpNum=0...numDetectedVps
    std::vector<std::vector<int> > CS;	// index of Consensus Set for all vps: CS[vpNum] is a vector containing indexes of lineSegments belonging to Consensus Set of vp numVp
    std::vector<int> numInliers;

    std::vector<std::vector<std::vector<cv::Point> > > lineSegmentsClusters;

    Size procSize = src.size();
    int mode = MODE_LS;
    bool verbose = false;
    MSAC msac;
    msac.init(mode, procSize, verbose);
//    cerr<<"lineSegments.size(): "<<lineSegments.size()<<endl;
    msac.multipleVPEstimation(lineSegments, lineSegmentsClusters, numInliers, vps, 1);

    for(size_t v=0; v<vps.size(); v++)
    {
        printf("VP %d (%.3f, %.3f, %.3f)", v, vps[v].at<float>(0,0), vps[v].at<float>(1,0), vps[v].at<float>(2,0));
        fflush(stdout);

        vps[v].at<float>(1,0)=vps[v].at<float>(1,0)+ROI_y;
        vps[v].at<float>(0,0)=vps[v].at<float>(0,0)+ROI_x;

        double vpNorm = cv::norm(vps[v]);
        if(fabs(vpNorm - 1) < 0.001)
        {
            printf("(INFINITE)");
            fflush(stdout);
        }
        printf("\n");
    }
    // Draw line segments according to their cluster
    msac.drawCS(src, lineSegmentsClusters, vps);
}

float ema_1(int ema_input)
{
    float ema_input_f = ema_input*1.0;

    if(ema_init_flag)
    {
        if(ema_init_count!=0)
        {
            ema_init_count--;
            ema_init_i++;
            ema_init_total += ema_input_f;
        }
        else if(ema_init_count==0)
        {
            ema_init = ema_init_total/ema_init_i;
            ema_last = ema_init;
//            if(ema_init<200)
//                ema_last=200;
            ema_init_flag=false;
        }
        return 250;
    }

    ema_current = ema_a*ema_input_f + (1.0-ema_a)*ema_last;
    float ema_output = ema_current;
        cout<<"ema_a:  "<<ema_a<<endl;
    cout<<"ema_init:  "<<ema_init<<endl;
    cout<<"ema_current:  "<<ema_current<<endl;
    ema_current = ema_last;
    return ema_output;
}
