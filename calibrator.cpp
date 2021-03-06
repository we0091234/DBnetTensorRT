#include <iostream>
#include <iterator>
#include <fstream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include "calibrator.h"
#include "cuda_runtime_api.h"
#include "utils.h"
#include "dirent.h"
// #include "common.hpp"
static inline cv::Mat preprocess_img1(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    
 

     return out;
}

int read_files_in_dir1(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }
    closedir(p_dir);
    return 0;
}

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache)
    : batchsize_(batchsize)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , input_blob_name_(input_blob_name)
    , read_cache_(read_cache)
{
    input_count_ = 3 * input_w * input_h * batchsize;
    CUDA_CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    read_files_in_dir1(img_dir, img_files_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    CUDA_CHECK(cudaFree(device_input_));
}

int Int8EntropyCalibrator2::getBatchSize() const
{
    return batchsize_;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    if (img_idx_ + batchsize_ > (int)img_files_.size()) {
        return false;
    }

    std::vector<cv::Mat> input_imgs_;
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::Mat pr_img = preprocess_img1(temp, input_w_, input_h_);
        // cv::Mat pr_img = temp;
        input_imgs_.push_back(pr_img);
    }
    img_idx_ += batchsize_;
    cv::Mat blob = cv::dnn::blobFromImages(input_imgs_, 1.0/(255.0*0.225) , cv::Size(input_w_, input_h_), cv::Scalar(103.53, 116.28, 123.675 ), true, false);

    CUDA_CHECK(cudaMemcpy(device_input_, blob.ptr<float>(0), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], input_blob_name_));
    bindings[0] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length)
{
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length)
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}

