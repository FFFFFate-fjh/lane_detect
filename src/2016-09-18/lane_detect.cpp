#include<iostream>
#include<string>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "MSAC.h"

#define USE_PPHT
#define MAX_NUM_LINES	200

using namespace std;
using namespace cv;

Mat src, src_gray,src_heq;
Mat dst;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;

int ROI_x=400;
int ROI_y=220;

vector<vector<cv::Point> > lineSegments;
vector<cv::Point> aux;

void CannyThreshold(int, void*);
Mat lane_perspective(Mat src);
Mat sobel(Mat src);
Mat set_ROI(Mat src, int ROI_L, int ROI_R);
int autoCanny(Mat &srcmat, float type);
void hough_p_compute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec4i> &lines,
                     int thres, int hlen, int minpoints, int ROI_x, int ROI_y);
void houghcompute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec2f> &lines,
                  int thres, int hlen, int minpoints,int ROI_x, int ROI_y);
void vanishedpoint(Mat &src );

/** @function main */
int main( int argc, char** argv )
{
    /// Load an image
    src = imread( "./data/04/image_0/000020.png" );

    if( !src.data )
    { return -1; }

    /// Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );

    /// Convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );

    equalizeHist( src_gray, src_heq );

    /// Create a window
    namedWindow( "Edge Map", CV_WINDOW_AUTOSIZE );

    /// Create a Trackbar for user to enter threshold
    createTrackbar( "Min Threshold:", "Edge Map", &lowThreshold, max_lowThreshold, CannyThreshold );

    /// Show the image
    CannyThreshold(0, 0);

    /// Wait until user exit program by pressing a key
    waitKey(0);

    return 0;
}


void CannyThreshold(int, void*)
{
    Mat ROI_img = set_ROI(src_heq,ROI_x,ROI_y);

    Mat grad, abs_grad_x;
    medianBlur(ROI_img,grad,3);
    abs_grad_x = sobel(grad);

    /// Canny detector
    int thres;
    thres=autoCanny(ROI_img, 1.0);

    Mat detected_edges;
    Canny( abs_grad_x, detected_edges, thres, thres*3, kernel_size );

    Mat hdst, roi_show;
    cvtColor(detected_edges, hdst, CV_GRAY2BGR);
    cvtColor(src_heq, roi_show, CV_GRAY2BGR);

    vector<Vec4i> lines;
    hough_p_compute(detected_edges, roi_show, hdst,lines,75, 10, 20,ROI_x,ROI_y);

    //  vector<Vec2f> linesf;
    //  houghcompute(detected_edges_l, roi_show, hdst_l, linesf_l,35, 20, 20,ROI_x_l,ROI_y_l);

    vanishedpoint(roi_show);

    imshow("source", detected_edges);
    imshow("detected lines", hdst);
    imshow("detected lines on ROI", roi_show);
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

Mat set_ROI(Mat src, int ROI_L, int ROI_R)
{
    Rect roi(ROI_x, ROI_y, 450, 369-ROI_y);
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
    int houghThreshold = 20;
    if(src.cols*src.rows < 400*400)
        houghThreshold = 30;

    cv::HoughLinesP(src, lines, 1, CV_PI/180, houghThreshold, 10,10);
    cout<<lines.size()<<endl;
    while(lines.size() > MAX_NUM_LINES)
    {
        cout<<lines.size()<<endl;
        lines.clear();
        houghThreshold += 10;
        cv::HoughLinesP(src, lines, 1, CV_PI/180, houghThreshold, 10, 10);
    }

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        float k = ((l[0]-l[2])*1.0)/(l[1]-l[3]);
        if(k<1 && k >-1)
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
