#include<iostream>
#include<fstream>
#include<string>
#include <unistd.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "MSAC.h"

#define USE_PPHT
#define MAX_NUM_LINES	200

using namespace std;
using namespace cv;

Mat src, src_gray;
vector<Mat> img,img_rgb;
vector<string> list;

float klc=0;
float krc=0;
float blc=0;
float brc=0;

int img_width;
int img_height;

int ROI_x=200;
int ROI_y=400;

int ema_T=5;
float ema_a = 2.0/(ema_T+1.0);
int ema_init_limit = 30;

typedef struct ema_1
{
    float last = 0;
    float current = 0;
    int init_count = ema_init_limit;
    float init_total = 0;
    float init =0;
}ema_1;

ema_1 canny_ema,kl_ema,kr_ema,br_ema,bl_ema;

vector<vector<cv::Point> > lineSegments;
vector<cv::Point> aux;

Mat sobel(Mat src);  //sobel计算
Mat set_ROI(Mat src, int ROI_L, int ROI_R, int width, int height);   //设置ROI大小
Mat opt_ROI(Mat src, float kl,float kr,float bl, float br );
int autoCanny(Mat &srcmat, float type);
void hough_p_compute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec4i> &lines,
                     int thres, int hlen, int minpoints, int ROI_x, int ROI_y);
void houghcompute(Mat &src, Mat &dst, Mat &roi_img, vector<Vec2f> &lines,
                  int thres, int hlen, int minpoints,int ROI_x, int ROI_y);
void vanishedpoint(Mat &src );
float ema1_init(float ema_input, int &ema_init_count, float &ema_init_total, float &ema_last, float &ema_init);
float ema1_compute(float ema_input, float &ema_last, float &ema_current, float &ema_init);
float kmeans_kb(Mat inputpoints);
void compute_stddev(vector<float> data, float &mean, float &stddev);
float opt_stddev(vector<float> data,float thres, float &dev);


int main( int argc, char** argv )
{
//    string filename = "data/04/image_0/list.txt"; //使用正则表达生成图片名字的list
      string filename = "/home/fung/workspace/data/road/highway.Korea.45_p1/list.txt"; //使用正则表达生成图片名字的list
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
//        sprintf(img_buf, "./data/04/image_0/%s", addr_buf);
        sprintf(img_buf, "/home/fung/workspace/data/road/highway.Korea.45_p1/%s", addr_buf);
        string img_addr = img_buf;
        cout<<img_addr<<endl;
        Mat src=imread(img_addr,CV_LOAD_IMAGE_COLOR );
        if( !src.data )
        { return -1; }
        Mat resrc,srcg;
        resize(src, resrc, Size(960, 540));
        img_rgb.push_back(resrc);
        cvtColor(resrc, srcg, CV_BGR2GRAY);
        img.push_back(srcg);
        img_width = resrc.cols;
        img_height = resrc.rows;
    }

    for(int i=0;i<img.size(); i++)
    {
        cout<<i<<endl;

        Mat src_heq ;
        equalizeHist( img[i], src_heq );

//        Mat ROI_img2 = set_ROI(src_heq,200, 150, 800, 369-150);
//        Mat ROI_img = set_ROI(src_heq,ROI_x, ROI_y, 600, 369-ROI_y);
        Mat ROI_img2 = set_ROI(src_heq,75, 230, img_width-75, img_height-230);
        Mat ROI_img = set_ROI(src_heq,ROI_x, ROI_y, 600, img_height-ROI_y);
        Mat grad, abs_grad_x;
        medianBlur(ROI_img,grad,5);
        abs_grad_x = sobel(grad);
//cerr<<"!!!!!!"<<endl;
        Mat roi_cut;
        if((klc > 10e-3 || klc < -10e-3)&&(canny_ema.last>10e-3))
        roi_cut = opt_ROI(abs_grad_x, klc,krc,blc, brc);
        cerr<<roi_cut.size()<<endl;

        /// Canny detector
        int thres1,thres2,thres3,thresmax;
        thres1=autoCanny(src_heq, 1.0);
        thres2=autoCanny(ROI_img, 1.0);
        thres3=autoCanny(ROI_img2, 1.0);
        thresmax=max(thres1,thres2);
        thresmax=max(thresmax,thres3);

        if(canny_ema.init_count)
            thresmax = ema1_init((float)thresmax,canny_ema.init_count,canny_ema.init_total,canny_ema.last,canny_ema.init);
        else
            thresmax = ema1_compute((float)thresmax, canny_ema.last, canny_ema.current,canny_ema.init);

        Mat detected_edges;
//        Canny( abs_grad_x, detected_edges, thresmax, thresmax*3, 3 );
        if(!roi_cut.empty())
            Canny( roi_cut, detected_edges, thresmax, thresmax*3, 3 );
        else
            Canny( abs_grad_x, detected_edges, thresmax, thresmax*3, 3 );


        Mat hdst, roi_show;
        cvtColor(detected_edges, hdst, CV_GRAY2BGR);
//        cvtColor(src_heq, roi_show, CV_GRAY2BGR);
        roi_show = img_rgb[i].clone();

        vector<Vec4i> lines;
        hough_p_compute(detected_edges, roi_show, hdst,lines,75, 10, 20,ROI_x,ROI_y);

        vanishedpoint(roi_show);

        imshow("source", detected_edges);
        imshow("detected lines", hdst);
        imshow("detected lines on ROI", roi_show);

        char img_buf[100];
        sprintf(img_buf, "/home/fung/workspace/output/road/highway.Korea.45_p1/%d.png", i);
        string img_addr = img_buf;
        imwrite(img_addr, roi_show);
        waitKey(33);
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
        houghThreshold = 10;

    cv::HoughLinesP(src, lines, 1, CV_PI/180, houghThreshold, 10,10);

    while(lines.size() > MAX_NUM_LINES)
    {
        lines.clear();
        houghThreshold += 1;
        cv::HoughLinesP(src, lines, 1, CV_PI/180, houghThreshold, 10, 10);
    }

    cout<<"lines.size():  "<<lines.size()<<endl;

    vector<float> kp_points_vec;
    vector<float> bp_points_vec;
    vector<float> kn_points_vec;
    vector<float> bn_points_vec;

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        float k = ((l[1]-l[3])*1.0)/(l[0]-l[2]);
        float b = l[1]-k*l[0];
        if(k>0.25 || k<-0.25)
        {
            line( roi_img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);
            l[0]=l[0]+ROI_x;
            l[2]=l[2]+ROI_x;
            l[1]=l[1]+ROI_y;
            l[3]=l[3]+ROI_y;
            line( dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 1, CV_AA);

            float k2 = ((l[1]-l[3])*1.0)/(l[0]-l[2]);
            float b2 = l[1]-k2*l[0];
            if(k2>0.25 && k2< 5)
            {
                kp_points_vec.push_back(k2);
            }
            if(k2<-0.25 && k2> -5)
            {
                kn_points_vec.push_back(k2);
            }

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

    float kl,kr,bl,br;
    float krout,brout,klout,blout;
    float kldev,krdev,bldev,brdev;
    kldev=krdev=0.5;
    bldev=brdev=50;

    kl = opt_stddev(kn_points_vec,0.2,kldev);
    kr = opt_stddev(kp_points_vec,0.2,krdev);

    if(kl_ema.init_count)
    {
        ema1_init(kl,kl_ema.init_count,kl_ema.init_total,kl_ema.last,kl_ema.init);
        klout = kl;
    }
    else
        klout = ema1_compute(kl, kl_ema.last, kl_ema.current,kl_ema.init);

    if(kr_ema.init_count)
    {
        ema1_init(kr,kr_ema.init_count,kr_ema.init_total,kr_ema.last,kr_ema.init);
        krout = kr;
    }
    else
        krout = ema1_compute(kr, kr_ema.last, kr_ema.current,kr_ema.init);
//        krout = kr;        klout = kl;
    cout<<"kl_ema_init： "<<kl_ema.init<<endl;
    cout<<"kr_ema_init： "<<kr_ema.init<<endl;
    cout<<"kl: "<<klout<<endl;
    cout<<"kr: "<<krout<<endl;

    klc = klout;
    krc = krout;

    for( size_t i = 0; i < lines.size(); i++ )
    {
        Vec4i l = lines[i];
        l[0]=l[0]+ROI_x;
        l[2]=l[2]+ROI_x;
        l[1]=l[1]+ROI_y;
        l[3]=l[3]+ROI_y;
        float k = ((l[1]-l[3])*1.0)/(l[0]-l[2]);
        float b = l[1]-k*l[0];
        if(k>kl-kldev && k<kl+kldev)
        {
            bn_points_vec.push_back(b);
        }
        if(k>kr-krdev && k<kr+krdev)
        {
            bp_points_vec.push_back(b);
        }
    }

    br = opt_stddev(bp_points_vec,25,bldev);
    bl = opt_stddev(bn_points_vec,25,brdev);

    if(bl_ema.init_count)
    {
        ema1_init(bl,bl_ema.init_count,bl_ema.init_total,bl_ema.last,bl_ema.init);
        blout = bl;
    }
    else
    {
        blout = ema1_compute(bl, bl_ema.last, bl_ema.current,bl_ema.init);
    }

    if(br_ema.init_count)
    {
        ema1_init(br,br_ema.init_count,br_ema.init_total,br_ema.last,br_ema.init);
        brout = br;
    }
    else
    {
        brout = ema1_compute(br, br_ema.last, br_ema.current,br_ema.init);
    }
//        brout = br;        blout = bl;
    cout<<"bl_ema_init： "<<bl_ema.init<<endl;
    cout<<"br_ema_init： "<<br_ema.init<<endl;
    cout<<"bl: "<<blout<<endl;
    cout<<"br: "<<brout<<endl;

    blc = blout;
    brc = brout;

    Vec4i lines_fl;
    lines_fl[0]=1000;
    lines_fl[1]=klout*1000+blout;
    lines_fl[2]=0;
    lines_fl[3]=blout;
    line( dst, Point(lines_fl[0], lines_fl[1]), Point(lines_fl[2], lines_fl[3]), Scalar(0,255,0), 3, CV_AA);

    Vec4i lines_fr;
    lines_fr[0]=1000;
    lines_fr[1]=krout*1000+brout;
    lines_fr[2]=0;
    lines_fr[3]=brout;
    line( dst, Point(lines_fr[0], lines_fr[1]), Point(lines_fr[2], lines_fr[3]), Scalar(0,255,0), 3, CV_AA);
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

float ema1_init(float ema_input, int &ema_init_count, float &ema_init_total, float &ema_last, float &ema_init)
{
    float ema_input_f = ema_input*1.0;

    if(ema_input_f>10e5 || ema_input_f<-10e5 || (ema_input_f<10e-5)&&(ema_input_f > -10e-5)) //没有值的时候
        return ema_init_total/(float)(ema_init_limit-ema_init_count);

    if ( isnan( ema_input_f ) )  //nan的情况
        return ema_init_total/(float)(ema_init_limit-ema_init_count);

    if(ema_init_count!=0)
    {
        ema_init_count--;
        ema_init_total += ema_input_f;
    }
    if(ema_init_count==0)
    {
        ema_last = ema_init_total/(float)ema_init_limit;
        ema_init = ema_last;
    }

    return ema_init_total/(float)(ema_init_limit-ema_init_count);
}

float ema1_compute(float ema_input, float &ema_last, float &ema_current, float &ema_init)
{
    float ema_input_f = ema_input*1.0;

    if(ema_input_f>10e5 || ema_input_f<-10e5 || (ema_input_f<10e-5)&&(ema_input_f > -10e-5)) //没有值的时候
    {
       ema_input_f = ema_init;
    }

    if(isnan( ema_input_f ))
    {
       ema_input_f = ema_init;
    }

    ema_current = ema_a*ema_input_f + (1.0-ema_a)*ema_last;
    float ema_output = ema_current;
//    cout<<"ema_a:  "<<ema_a<<endl;
    cout<<"ema_init:  "<<ema_init<<endl;
    cout<<"ema_current:  "<<ema_current<<endl;
    ema_current = ema_last;
    return ema_output;
}

float kmeans_kb(Mat inputpoints)
{
    int clusterCount = 2;
    Mat kmeans_labels,labels_k,labels_kp,labels_kn;
    Mat centers(clusterCount, 1, inputpoints.type());    //用来存储聚类后的中心点

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

    for(int i=0;i<cluster_kp.size();i++)
    {
        cout<<cluster_kp[i]<<endl;
    }

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

float opt_stddev(vector<float> data,float thres, float &dev)
{
    if(!data.size())
        return 9.9e108;

    float mean,stddev;
    vector<float> dev_tmp1;
    vector<float> dev_tmp2;

    compute_stddev(data, mean, stddev);

    if(stddev<thres)
    {
        dev=thres;
        return mean;
    }
    else
    {
        for(int i=0;i<data.size();i++)
        {
            if(data[i]<(mean+stddev) && data[i]>(mean-stddev))
            dev_tmp1.push_back(data[i]);
        }

        int count=30;
        while(stddev>thres && dev_tmp1.size()!=0)
        {
            compute_stddev(dev_tmp1, mean, stddev);
            if(stddev<thres)
            {
                dev=stddev;
                return mean;
            }

            dev_tmp2.clear();
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
                dev=stddev;
                return mean;
            }

            count--;
            if(!count)
                return mean;
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
    Scalar int_zero;
    int_zero.all(0);
    cerr<<dst.size()<<endl;

    for(int y=0; y<img_height-ROI_y; y++)
    {
        float xl,xr;
        xl = ((float)(y+ROI_y)-bl)/kl-ROI_x;
        xr = ((float)(y+ROI_y)-br)/kr-ROI_x;

        xl = max((int)xl, 50);
        xr = min((int)xr, src.cols-50);

        if((int)xl==50 || (int)xr == src.cols-50)
        for(int x=(int)xl+100; x<(int)xr-100; x++)
        {
            dst.at<uchar>(y,x)=int_zero.val[0];
        }
        else
        for(int x=(int)xl+50; x<(int)xr-50; x++)
        {
            dst.at<uchar>(y,x)=int_zero.val[0];
        }
    }
    imshow("opt_ROI",dst);
    return dst;
}
