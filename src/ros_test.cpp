#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include <boost/thread.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <eigen3/Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

#include <ros/ros.h>
#include <ros/spinner.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <kinect2_bridge/kinect2_definitions.h>

class Receiver{
    boost::mutex lock;

    const std::string topicColor, topicDepth;
    bool updateCloud;
    bool running;
    bool imageReady;
    bool updated;
    size_t frame;
    unsigned int queueSize;

    cv::Mat color, depth;
    cv::Mat cameraMatrixColor, cameraMatrixDepth;
    cv::Mat lookupX, lookupY;
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    

    struct frame_t{
      cv::Mat desp;
      std::vector<cv::KeyPoint> kps;
      cv::Mat depth;
    };
    
    Eigen::Isometry3d T;


    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> ExactSyncPolicy;
    std::vector<frame_t*> frames;
    std::vector<int> params;
    ros::NodeHandle nh;
    ros::AsyncSpinner spinner;
    image_transport::ImageTransport it;
    image_transport::SubscriberFilter *subImageColor, *subImageDepth;
    message_filters::Subscriber<sensor_msgs::CameraInfo> *subCameraInfoColor, *subCameraInfoDepth;
    message_filters::Synchronizer<ExactSyncPolicy> *syncExact;

    boost::thread get_image_thread;
    boost::thread process_thread;

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud;
    pcl::PCDWriter writer;

    std::ostringstream oss;
    public:

    Receiver(const std::string &topicColor, const std::string &topicDepth)
        :topicColor(topicColor), topicDepth(topicDepth),updateCloud(false),running(false),imageReady(false),updated(true),frame(0),nh("~"), spinner(0), it(nh){
            detector = cv::FeatureDetector::create("SIFT");
            descriptor = cv::DescriptorExtractor::create("SIFT");
            matcher = cv::DescriptorMatcher::create("FlannBased");
            queueSize = 5;
            cameraMatrixColor = cv::Mat::zeros(3, 3, CV_64F);
            cameraMatrixDepth = cv::Mat::zeros(3, 3, CV_64F);
            params.push_back(cv::IMWRITE_JPEG_QUALITY);
            params.push_back(100);
            params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            params.push_back(1);
            params.push_back(cv::IMWRITE_PNG_STRATEGY);
            params.push_back(cv::IMWRITE_PNG_STRATEGY_RLE);
            params.push_back(0);
        }

    ~Receiver(){}

    void run(){
        start();
        stop();
    }

    private:

    void start()
    {
        running = true;

        std::string topicCameraInfoColor = topicColor.substr(0, topicColor.rfind('/')) + "/camera_info";
        std::string topicCameraInfoDepth = topicDepth.substr(0, topicDepth.rfind('/')) + "/camera_info";

        image_transport::TransportHints hints("raw");
        subImageColor = new image_transport::SubscriberFilter(it, topicColor, queueSize, hints);
        subImageDepth = new image_transport::SubscriberFilter(it, topicDepth, queueSize, hints);
        subCameraInfoColor = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoColor, queueSize);
        subCameraInfoDepth = new message_filters::Subscriber<sensor_msgs::CameraInfo>(nh, topicCameraInfoDepth, queueSize);

        syncExact = new message_filters::Synchronizer<ExactSyncPolicy>(ExactSyncPolicy(queueSize), *subImageColor, *subImageDepth, *subCameraInfoColor, *subCameraInfoDepth);
        syncExact->registerCallback(boost::bind(&Receiver::callback, this, _1, _2, _3, _4));

        spinner.start();
        while(!imageReady)
        {
            if(!ros::ok())
            {
                return;
            }
            boost::this_thread::sleep(boost::posix_time::milliseconds(1));
        }

        cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
        cloud->height = color.rows;
        cloud->width = color.cols;
        cloud->is_dense = false;
        cloud->points.resize(cloud->height * cloud->width);
        createLookup(this->color.cols, this->color.rows);
        cloud->points.resize(color.rows*color.cols);
        process_thread = boost::thread(&Receiver::process, this);
        cloudViewer();
    }
    void stop()
    {
        spinner.stop();

        delete syncExact;

        delete subImageColor;
        delete subImageDepth;
        delete subCameraInfoColor;
        delete subCameraInfoDepth;
      //  get_image_thread.join();
        running = false;
    }
    void callback(const sensor_msgs::Image::ConstPtr imageColor, const sensor_msgs::Image::ConstPtr imageDepth,
            const sensor_msgs::CameraInfo::ConstPtr cameraInfoColor, const sensor_msgs::CameraInfo::ConstPtr cameraInfoDepth)
    {
        if(!updated){
            return;
        }
        cv::Mat color, depth;
        readCameraInfo(cameraInfoColor, cameraMatrixColor);
        readCameraInfo(cameraInfoDepth, cameraMatrixDepth);
        readImage(imageColor, color);
        readImage(imageDepth, depth);

        // IR image input
        // TODO: Maybe Useful for Image type conversion
        if(color.type() == CV_16U)
        {
            cv::Mat tmp;
            color.convertTo(tmp, CV_8U, 0.02);
            cv::cvtColor(tmp, color, CV_GRAY2BGR);
        }
        lock.lock();
        this->color = color;
        this->depth = depth;
        imageReady = true;
        lock.unlock();

    }
    
    void process()
    {
        OUT_INFO("ready for process");
        for(; running && ros::ok();)
        {
            if(!imageReady){
                continue;
            }
            cv::Mat color, depth;
            lock.lock();
            color = this->color;
            depth = this->depth;
            imageReady = false;
            frame_t *frame = new frame_t;
            lock.unlock();
            cv::imshow("Image Viewer",color);
            //int key = cv::waitKey(0);

            /*if ((key&0xff) == 's' ){
              OUT_INFO("Start processing");
            }else{
              OUT_INFO("Skip this frame");
              continue;
            }*/
            //get feature of new frame
            detector->detect(color,frame->kps);
            descriptor->compute(color,frame->kps,frame->desp);
            frame->depth = depth;
            if(frames.size() == 0){
                frames.push_back(frame);
                OUT_INFO("Initialize first frame!");
                continue;
            }
            
            std::vector<cv::DMatch> matches;
            matcher->match(frames.back()->desp,frame->desp,matches);
            std::vector<cv::DMatch> good_matches;
            double min_dist = std::numeric_limits<double>::max();
            for (size_t i = 0;i<matches.size();++i){
                if(matches[i].distance < min_dist){
                    min_dist = matches[i].distance;
                }
            }
            min_dist = 4*min_dist;
            for (size_t i = 0;i<matches.size();++i){
                if(matches[i].distance < min_dist){
                    good_matches.push_back(matches[i]);          
                }
            }

            if (good_matches.size() < 10){
                OUT_INFO("Can not find enough matches to do RANSAC");
                continue;
            }
            frame_t *f1 = frames.back();
            frame_t *f2 = frame;

            std::vector<cv::Point3f> pts_obj;
            std::vector<cv::Point2f> pts_img;
            const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
            const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
            const float cx = cameraMatrixColor.at<double>(0, 2);
            const float cy = cameraMatrixColor.at<double>(1, 2);
            for (size_t i = 0; i < good_matches.size(); ++i) {
                cv::Point2f p = f1->kps[good_matches[i].queryIdx].pt;
                ushort d = f1->depth.ptr<ushort>(int(p.y))[int(p.x)];
                if(d == 0){
                    continue;
                }
                pts_img.push_back(cv::Point2f(f2->kps[good_matches[i].trainIdx].pt));
                cv::Point3f pt(p.x, p.y, d);
                cv::Point3f pd;
                pd.z = double(pt.z) / 1000.0;
                pd.x = (pt.x - cx)*pd.z / fx;
                pd.y = (pt.y - cy)*pd.z / fy;
                pts_obj.push_back(pd);
            }
            if(pts_obj.size() < 5 || pts_img.size() < 5){
                std::cout<<"too little pts_obj size: "<<pts_obj.size()<<std::endl;
                continue;
            }
            cv::Mat rvec, tvec, inliers;
            cv::solvePnPRansac( pts_obj, pts_img, cameraMatrixColor, cv::Mat(), rvec, tvec, false, 100, 1.0, 100, inliers );
            cv::Mat R;
            cv::Rodrigues(rvec, R);
            Eigen::Matrix3d r;
            cv::cv2eigen(R, r);
            Eigen::AngleAxisd angle(r);
            lock.lock();
            T = Eigen::Isometry3d::Identity();
            T = angle;
            T(0,3) = tvec.at<double>(0,0); 
            T(1,3) = tvec.at<double>(0,1); 
            T(2,3) = tvec.at<double>(0,2);
            updateCloud = true;
            lock.unlock();
            frames.push_back(frame);
        }
    }

    void cloudViewer()
    {
        cv::Mat color, depth;
        pcl::visualization::PCLVisualizer::Ptr visualizer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
        const std::string cloudName = "rendered";


        lock.lock();
        color = this->color;
        depth = this->depth;
        updateCloud = false;
        lock.unlock();

        createCloud(depth, color, cloud);

        visualizer->addPointCloud(cloud, cloudName);
        visualizer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, cloudName);
        visualizer->initCameraParameters();
        visualizer->setBackgroundColor(0, 0, 0);
        //visualizer->setPosition(0, 0);
        //visualizer->setSize(color.cols, color.rows);
        //visualizer->setShowFPS(true);
        visualizer->setCameraPosition(0, 0, 0, 0, -1, 0);
        visualizer->registerKeyboardCallback(&Receiver::keyboardEvent, *this);

        for(; running && ros::ok();)
        {
            if(updateCloud)
            {
                lock.lock();
                color = this->color;
                depth = this->depth;
                updateCloud = false;
                lock.unlock();
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());

                //create cloud
                pcl::PointCloud<pcl::PointXYZRGBA>::Ptr new_cloud = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
                new_cloud->height = color.rows;
                new_cloud->width = color.cols;
                new_cloud->is_dense = false;
                new_cloud->points.resize(color.rows * color.cols);
                createCloud(depth, color, new_cloud);
                bool init = frames.size() == 1 ? true:false;
                if(!init){
                    pcl::transformPointCloud(*cloud, *output, T.matrix());
                }
                OUT_INFO("before cloud size:"<<cloud->points.size());
                *new_cloud += *output;

                if(!init){
                    static pcl::VoxelGrid<pcl::PointXYZRGBA> voxel;
                    double gridsize = 0.01;
                    voxel.setLeafSize(gridsize, gridsize, gridsize);
                    voxel.setInputCloud(new_cloud);
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tmp  = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBA>());
                    voxel.filter(*tmp);
                    cloud->swap(*tmp);
                }
                else{
                    cloud->swap(*new_cloud);
                }
                cloud->swap(*new_cloud);
                OUT_INFO("after cloud size:"<<cloud->points.size());
                visualizer->updatePointCloud(cloud, cloudName);
            }
            visualizer->spinOnce(10);
            //pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);
        }
        visualizer->close();
    }

void keyboardEvent(const pcl::visualization::KeyboardEvent &event, void *)
  {
    if(event.keyUp())
    {
      switch(event.getKeyCode())
      {
      case 27:
      case 'q':
        running = false;
        break;
      case ' ':
      case 's':
        pcl::io::savePCDFileASCII ("test_pcd.pcd", *cloud);
        OUT_INFO("Saved " << cloud->points.size () << " data points to test_pcd.pcd.");
        break;
      }
    }
  }


    void createLookup(size_t width, size_t height)
    {
        const float fx = 1.0f / cameraMatrixColor.at<double>(0, 0);
        const float fy = 1.0f / cameraMatrixColor.at<double>(1, 1);
        const float cx = cameraMatrixColor.at<double>(0, 2);
        const float cy = cameraMatrixColor.at<double>(1, 2);
        float *it;

        lookupY = cv::Mat(1, height, CV_32F);
        it = lookupY.ptr<float>();
        for(size_t r = 0; r < height; ++r, ++it)
        {
            *it = (r - cy) * fy;
        }

        lookupX = cv::Mat(1, width, CV_32F);
        it = lookupX.ptr<float>();
        for(size_t c = 0; c < width; ++c, ++it)
        {
            *it = (c - cx) * fx;
        }
    }

    void createCloud(const cv::Mat &depth, const cv::Mat &color, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud) const
    {
        const float badPoint = std::numeric_limits<float>::quiet_NaN();

#pragma omp parallel for
        for(int r = 0; r < depth.rows; ++r)
        {
            pcl::PointXYZRGBA *itP = &cloud->points[r*depth.cols];
            const uint16_t *itD = depth.ptr<uint16_t>(r);
            const cv::Vec3b *itC = color.ptr<cv::Vec3b>(r);
            const float y = lookupY.at<float>(0, r);
            const float *itX = lookupX.ptr<float>();
            for(size_t c = 0; c < (size_t)depth.cols; ++c,++itD, ++itP, ++itC, ++itX)
            {
                //pcl::PointXYZRGBA itP;

                register const float depthValue = *itD / 1000.0f;
                // Check for invalid measurements
                if(*itD == 0)
                {
                    // not valid
                    itP->x = itP->y = itP->z = badPoint;
                    itP->rgba = 0;
                    continue;
                }
                itP->z = depthValue;
                itP->x = *itX * depthValue;
                itP->y = y * depthValue;
                itP->b = itC->val[0];
                itP->g = itC->val[1];
                itP->r = itC->val[2];
                itP->a = 255;
                //OUT_INFO("z,x,y,b,g,r " <<itP.z<<" "<<itP.x<<" "<<itP.y<<" "<<itP.b<<" "<<itP.g<<" "<<itP.r<<" ");
                //new_cloud->points.push_back(itP);
            }
        }
        //cloud->swap(*new_cloud);
    }

    void readImage(const sensor_msgs::Image::ConstPtr msgImage, cv::Mat &image) const
    {
        cv_bridge::CvImageConstPtr pCvImage;
        pCvImage = cv_bridge::toCvShare(msgImage, msgImage->encoding);
        pCvImage->image.copyTo(image);
    }

    void readCameraInfo(const sensor_msgs::CameraInfo::ConstPtr cameraInfo, cv::Mat &cameraMatrix) const
    {
        double *itC = cameraMatrix.ptr<double>(0, 0);
        for(size_t i = 0; i < 9; ++i, ++itC)
        {
            *itC = cameraInfo->K[i];
        }
    }

    void saveImages(const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depthColored)
    {
        oss.str("");
        oss << "./" << std::setfill('0') << std::setw(4) << frame;
        const std::string baseName = oss.str();
        const std::string cloudName = baseName + "_cloud.pcd";
        const std::string colorName = baseName + "_color.jpg";
        const std::string depthName = baseName + "_depth.png";
        const std::string depthColoredName = baseName + "_depth_colored.png";

        OUT_INFO("saving color: " << colorName);
        cv::imwrite(colorName, color, params);
        OUT_INFO("saving depth: " << depthName);
        cv::imwrite(depthName, depth, params);
        OUT_INFO("saving depth: " << depthColoredName);
        cv::imwrite(depthColoredName, depthColored, params);
        OUT_INFO("saving complete!");
        ++frame;
    }
};


int main(int argc, char **argv)
{
    cv::initModule_nonfree();
#if EXTENDED_OUTPUT
    ROSCONSOLE_AUTOINIT;
    if(!getenv("ROSCONSOLE_FORMAT"))
    {
        ros::console::g_formatter.tokens_.clear();
        ros::console::g_formatter.init("[${severity}] ${message}");
    }
#endif

    ros::init(argc, argv, "kinect2_viewer", ros::init_options::AnonymousName);

    if(!ros::ok())
    {
        return 0;
    }

    std::string ns = K2_DEFAULT_NS;
    std::string topicColor = K2_TOPIC_QHD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
    std::string topicDepth = K2_TOPIC_QHD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
    /*
     *"sd"
     *    topicColor = K2_TOPIC_SD K2_TOPIC_IMAGE_COLOR K2_TOPIC_IMAGE_RECT;
     *    topicDepth = K2_TOPIC_SD K2_TOPIC_IMAGE_DEPTH K2_TOPIC_IMAGE_RECT;
     */

    topicColor = "/" + ns + topicColor;
    topicDepth = "/" + ns + topicDepth;
    OUT_INFO("topic color: " FG_CYAN << topicColor << NO_COLOR);
    OUT_INFO("topic depth: " FG_CYAN << topicDepth << NO_COLOR);

    Receiver receiver(topicColor, topicDepth);

    OUT_INFO("starting receiver...");
    receiver.run();

    ros::shutdown();
    return 0;
}
