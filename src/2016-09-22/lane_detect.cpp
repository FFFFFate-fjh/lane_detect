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

float klc=0;
float krc=0;
float blc=0;
float brc=0;

int ROI_x=400;
int ROI_y=220;

int img_width;
int img_height;

vector<vector<cv::Point>> lineSegments;
vector<cv::Point> aux;

void CannyThreshold(int, void*);
Mat lane_perspective(Mat src);
Mat sobel(Mat src);
Mat set_ROI(Mat src, int ROI_L, int ROI_R);
Mat opt_ROI(Mat src, float kl,float kr,float bl, float br );
int autoCanny(Mat &srcmat, float type);
void hough_p_compute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec4i> &lines,
                     int thres, int hlen, int minpoints, int ROI_x, int ROI_y);
void houghcompute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec2f> &lines,
                  int thres, int hlen, int minpoints,int ROI_x, int ROI_y);
void vanishedpoint(Mat &src );
float kmeans_kb(Mat inputpoints);
void compute_stddev(vector<float> data, float &mean, float &stddev);
float opt_stddev(vector<float> data,float thres);


/** @function main */
int main( int argc, char** argv )
{
    /// Load an image
    src = imread( "./data/04/image_0/000058.png" );

    if( !src.data )
    { return -1; }

    /// Create a matrix of the same type and size as src (for dst)
    dst.create( src.size(), src.type() );

    img_width = src.cols;
    img_height = src.rows;

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

    Mat roi_cut;
    if(klc > 10e-3 || klc < -10e-3)
    roi_cut = opt_ROI(abs_grad_x, klc,krc,blc, brc);
    cerr<<roi_cut.size()<<endl;

    Mat detected_edges;
    if(!roi_cut.empty())
        Canny( roi_cut, detected_edges, thres, thres*3, kernel_size );
    else
        Canny( abs_grad_x, detected_edges, thres, thres*3, kernel_size );

    Mat hdst, roi_show;
    cvtColor(detected_edges, hdst, CV_GRAY2BGR);
    cvtColor(src_heq, roi_show, CV_GRAY2BGR);

    vector<Vec4i> lines;
    hough_p_compute(detected_edges, roi_show, hdst,lines,75, 10, 20,ROI_x,ROI_y);

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
//    int ddepth = CV_8U;

    /// Generate grad_x and grad_y
    Mat grad_x;
    Mat abs_grad_x;

    Sobel( src, grad_x, ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    return abs_grad_x;
}

Mat set_ROI(Mat src, int ROI_L, int ROI_R)
{
    Rect roi(ROI_x, ROI_y, 450, img_height-ROI_y);
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
        houghThreshold = 20;

    cv::HoughLinesP(src, lines, 1, CV_PI/180, houghThreshold, 10,10);
    while(lines.size() > MAX_NUM_LINES)
    {
        lines.clear();
        houghThreshold += 2;
        cv::HoughLinesP(src, lines, 1, CV_PI/180, houghThreshold, 10, 10);
    }
    cout<<"lines.size():  "<<lines.size()<<endl;

    vector<float> kp_points_vec;
    vector<float> bp_points_vec;
    float kp_points_max = 0;
    float kp_points_min = 99;
    vector<float> kn_points_vec;
    vector<float> bn_points_vec;
    float kn_points_max = 0;
    float kn_points_min = 99;

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        float k = ((l[1]-l[3])*1.0)/(l[0]-l[2]);
        float b = l[1]-k*l[0];
        if(k>0.5 || k<-0.5)
        {
            line( roi_img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
            l[0]=l[0]+ROI_x;
            l[2]=l[2]+ROI_x;
            l[1]=l[1]+ROI_y;
            l[3]=l[3]+ROI_y;

            float k2 = ((l[1]-l[3])*1.0)/(l[0]-l[2]);
            float b2 = l[1]-k2*l[0];

            if(k2>0.5 && k2< 10)
            {
                cout<<"kp:  "<<k2<<endl;
                kp_points_vec.push_back(k2);
                bp_points_vec.push_back(b2);
//                kp_points_max = max(kp_points_max,k2);
//                kp_points_min = min(kp_points_min,k2);
            }

            if(k2<-0.5 && k2> -10)
            {
                cout<<"kn:  "<<k2<<endl;
                kn_points_vec.push_back(k2);
                bn_points_vec.push_back(b2);
//                kn_points_max = max(kn_points_max,k2);
//                kn_points_min = min(kn_points_min,k2);
            }

            Vec4i lines_kb;
            lines_kb[0]=1000;
            lines_kb[1]=k2*1000+b2;
            lines_kb[2]=0;
            lines_kb[3]=b2;
            line( dst, Point(lines_kb[0], lines_kb[1]), Point(lines_kb[2], lines_kb[3]), Scalar(0,0,255), 1, CV_AA);

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

    Mat kp_points(kp_points_vec.size(), 1, CV_32FC2);
    for(int i=0; i<kp_points_vec.size(); i++)
    {
        kp_points.at<Vec2f>(i,0)[0] = kp_points_vec[i];
        kp_points.at<Vec2f>(i,0)[1] = 1;
    }
    Mat kn_points(kn_points_vec.size(), 1, CV_32FC2);
    for(int i=0; i<kn_points_vec.size(); i++)
    {
        kn_points.at<Vec2f>(i,0)[0] = kn_points_vec[i];
        kn_points.at<Vec2f>(i,0)[1] = 1;
    }

    Mat bp_points(bp_points_vec.size(), 1, CV_32FC2);
    for(int i=0; i<bp_points_vec.size(); i++)
    {
        bp_points.at<Vec2f>(i,0)[0] = bp_points_vec[i];
        bp_points.at<Vec2f>(i,0)[1] = 1;
    }
    Mat bn_points(bn_points_vec.size(), 1, CV_32FC2);
    for(int i=0; i<bn_points_vec.size(); i++)
    {
        bn_points.at<Vec2f>(i,0)[0] = bn_points_vec[i];
        bn_points.at<Vec2f>(i,0)[1] = 1;
    }

    float mean,stddev;
    float kl,kr,bl,br;

//    compute_stddev(kp_points_vec, mean, stddev);

//    if(stddev>0.2)
//       kr=kmeans_kb(kp_points);
//    else
//        kr = mean;

//    compute_stddev(kn_points_vec, mean, stddev);
//    if(stddev>0.2)
//        kl = kmeans_kb(kn_points);
//    else
//        kl = mean;

    kl = opt_stddev(kn_points_vec,0.2);
    kr = opt_stddev(kp_points_vec,0.2);

    cout<<"kl: "<<kl<<endl;
    cout<<"kr: "<<kr<<endl;

    klc = kl;
    krc = kr;

    br = opt_stddev(bp_points_vec,50);
    bl = opt_stddev(bn_points_vec,50);

    cout<<"bl: "<<bl<<endl;
    cout<<"br: "<<br<<endl;

    blc = bl;
    brc = br;

    Vec4i lines_fl;
    lines_fl[0]=1000;
    lines_fl[1]=kl*1000+bl;
    lines_fl[2]=0;
    lines_fl[3]=bl;
    line( dst, Point(lines_fl[0], lines_fl[1]), Point(lines_fl[2], lines_fl[3]), Scalar(0,255,0), 3, CV_AA);

    Vec4i lines_fr;
    lines_fr[0]=1000;
    lines_fr[1]=kr*1000+br;
    lines_fr[2]=0;
    lines_fr[3]=br;
    line( dst, Point(lines_fr[0], lines_fr[1]), Point(lines_fr[2], lines_fr[3]), Scalar(0,255,0), 3, CV_AA);
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

float kmeans_kb(Mat inputpoints)
{
    int clusterCount = 2;
    Mat kmeans_labels,labels_k,labels_kp,labels_kn;
    Mat centers(clusterCount, 1, inputpoints.type());    //用来存储聚类后的中心点

//    while
    kmeans(inputpoints, clusterCount, labels_k,
           TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10e5, 1.0),
           3, KMEANS_PP_CENTERS, centers);  //聚类3次，取结果最好的那次，聚类的初始化采用PP特定的随机算法。

    int labels_k_count[clusterCount];
    labels_k_count[0]=0;
    labels_k_count[1]=0;
    vector<Vec2f> cluster_kp;
    vector<Vec2f> cluster_kn;

    vector<float> cluster_kpf;
    vector<float> cluster_knf;


    for( int i = 0; i < inputpoints.rows; i++ )
    {
        cout<<i<<endl;
        int clusterIdx = labels_k.at<int>(i);
                cout<<"clusterIdx: "<<clusterIdx<<endl;
        Vec2f vec = inputpoints.at<Vec2f>(i);
        Point ipt = inputpoints.at<Point2f>(i);
//        cout<<vec<<endl;

        switch(clusterIdx)
        {
           case 0:
            if(ipt.y==1 )
            {
                labels_k_count[0]++;
                cluster_kp.push_back(vec);
                cluster_kpf.push_back(inputpoints.at<float>(i,0));
            }
            break;
           case 1:
            if(ipt.y==1 )
            {
                labels_k_count[1]++;
                cluster_kn.push_back(vec);
                cluster_knf.push_back(inputpoints.at<float>(i,0));
            }
            break;
        }

    }

//    cout<<"@@@@@@@@@@@@@@"<<endl;
    for(int i=0;i<cluster_kp.size();i++)
    {
        cout<<cluster_kp[i]<<endl;
    }
//    cout<<"@@@@@@@@@@@@@@"<<endl;
//    cout<<"cluster_kn.size(): "<<cluster_kn.size()<<endl;
    for(int i=0;i<cluster_kn.size();i++)
    {
        cout<<cluster_kn[i]<<endl;
    }

    float res,mean,stddev;
    if(cluster_kpf.size()>cluster_knf.size())
    {
        compute_stddev(cluster_kpf,mean,stddev);
        res = mean;
    }
    else
    {
        compute_stddev(cluster_knf,mean,stddev);
        res = mean;
    }

    return res;

}

void compute_stddev(vector<float> data, float &mean, float &stddev)
{
    float sum= 0.0;
    for(int i=0;i<data.size();i++)
        sum+= data[i];
    mean =  sum / data.size(); //均值
//    cout<<"mean: "<<mean<<endl;
    float accum  = 0.0;
    for(int i=0;i<data.size();i++)
        accum  += (data[i]-mean)*(data[i]-mean);

    stddev = sqrt(accum/(data.size()-1));
//    cout<<"stddev: "<<stddev<<endl;
}

float opt_stddev(vector<float> data,float thres)
{
    float mean,stddev;
    vector<float> dev_tmp1;
    vector<float> dev_tmp2;

    compute_stddev(data, mean, stddev);

    if(stddev<thres)
        return mean;
    else
    {
        for(int i=0;i<data.size();i++)
        {
            if(data[i]<mean+stddev && data[i]>mean-stddev)
            dev_tmp1.push_back(data[i]);
        }
        while(stddev>thres && dev_tmp1.size()!=0)
        {
            compute_stddev(dev_tmp1, mean, stddev);
            if(stddev<thres)
            {
                return mean;
            }
            for(int i=0;i<dev_tmp1.size();i++)
            {
                if(dev_tmp1[i]<mean+stddev && dev_tmp1[i]>mean-stddev)
                dev_tmp2.push_back(dev_tmp1[i]);
            }
            if(dev_tmp2.size()!=0)
            {
                dev_tmp1.clear();
                for(int i=0;i<dev_tmp2.size();i++)
                    dev_tmp1.push_back(dev_tmp2[i]);
            }
            else
            {
                return mean;
            }
        }
    }

}

Mat opt_ROI(Mat src, float kl,float kr,float bl, float br )
{
    Mat dst;
    if (src.type() != CV_8UC1)
    {
         cerr<<"not CV_8UC1"<<endl;
    }
    dst = src.clone();
//    src.convertTo(tmp,CV_8UC1);
//    cout<<src.rows<<"  "<<src.cols<<endl;
//    Scalar intensity = src.at<uchar>(0, 1);
    Scalar int_zero;
    int_zero.all(0);
//    cout<<int_zero<<endl;
//    cout<<intensity<<endl;
//    cout<<src.row(0).col(0)<<endl;
//    src.at<uchar>(0, 0)=intensity.val[0];
//        cout<<src.row(0).col(0)<<endl;
//    cerr<<tmp.size()<<endl;
    for(int y=0; y<img_height-ROI_y; y++)
    {
        float xl,xr;
        xl = ((float)(y+ROI_y)-bl)/kl-ROI_x;
        xr = ((float)(y+ROI_y)-br)/kr-ROI_x;
//        xl =(int)xl;
//        xr =(int)xr;
//        cout<<xl<<"  "<<xr<<endl;
        for(int x=(int)xl+50; x<(int)xr-50; x++)
        {
            dst.at<uchar>(y,x)=int_zero.val[0];
        }
    }
//    tmp.convertTo(dst,src.type());
    imshow("opt_ROI",dst);
//    waitKey(0);
    return dst;
}
