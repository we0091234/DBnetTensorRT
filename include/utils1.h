#ifndef UTILS_H_
#define UTILS_H_
#include "TrtClassificer.h"
#include <immintrin.h>
#include <dirent.h>
#include <sys/stat.h>
using namespace std;
// #ifdef use_aligned
// #define my_mm256_load_ps _mm256_load_ps
// #define my_mm256_store_ps _mm256_store_ps
// #define my_mm256_load_si256 _mm256_load_si256
// #define my_mm256_store_si256 _mm256_store_si256
// #else
// #define my_mm256_load_ps _mm256_loadu_ps
// #define my_mm256_store_ps _mm256_storeu_ps
// #define my_mm256_load_si256 _mm256_loadu_si256
// #define my_mm256_store_si256 _mm256_storeu_si256
// #endif

std::string getHouZhui(std::string fileName)
{
    //    std::string fileName="/home/xiaolei/23.jpg";
    int pos=fileName.find_last_of(std::string("."));
    std::string houZui=fileName.substr(pos+1);
    return houZui;
}
// bool is_dir(const char* path) {
// 	struct _stat buf = { 0 };
// 	_stat(path, &buf);
// 	return buf.st_mode & _S_IFDIR;
// }
int readFileList(char *basePath,std::vector<std::string> &fileList,std::vector<std::string> fileType)
{
    DIR *dir;
    struct dirent *ptr;
    char base[1000];

    if ((dir=opendir(basePath)) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        else if(ptr->d_type == 8)
        {    ///file
            if (fileType.size())
            {
            std::string houZui=getHouZhui(std::string(ptr->d_name));
            for (auto &s:fileType)
            {
            if (houZui==s)
            {
            fileList.push_back(basePath+std::string("/")+std::string(ptr->d_name));
            break;
            }
            }
            }
            else
            {
                fileList.push_back(basePath+std::string("/")+std::string(ptr->d_name));
            }
        }
        else if(ptr->d_type == 10)    ///link file
            printf("d_name:%s/%s\n",basePath,ptr->d_name);
        else if(ptr->d_type == 4)    ///dir
        {
            memset(base,'\0',sizeof(base));
            strcpy(base,basePath);
            strcat(base,"/");
            strcat(base,ptr->d_name);
            readFileList(base,fileList,fileType);
        }
    }
    closedir(dir);
    return 1;
}

int findMax(float *prob, int OUTPUT_SIZE)
{
	int i;
	int max = 0;
	for (i = 1; i < OUTPUT_SIZE; i++)
		if (prob[max] < prob[i])
			max = i;
	return max;
}

void setMean(cv::Mat & im, float *mean_data, float *&pdata, int CHANNELS, int INPUT_H, int INPUT_W)
{
	for (int c = 0; c < CHANNELS; ++c)
	{
		for (int h = 0; h < INPUT_H; ++h)
		{
			for (int w = 0; w < INPUT_W; ++w)
			{
				*pdata++ = float(im.at<cv::Vec3b>(h, w)[c] - mean_data[c]);
			}
		}
	}
}

void imageProcess(char *picName, float *data,float *mean_data, int CHANNELS, int INPUT_H, int INPUT_W)
{
	cv::Mat im = cv::imread(picName);
	cv::resize(im, im, cv::Size(INPUT_W, INPUT_H));
	//float mean_data[] = { 99.95327338, 96.27925874, 86.54154894 }; //均值
	//float mean_data[] = { 97.59758647 , 99.04790283, 104.8204798 };
	float *pdata = data;
	setMean(im, mean_data, pdata, CHANNELS, INPUT_H, INPUT_W);
}

float getSimilarity(std::vector<float> &lhs, std::vector<float>& rhs,int n)
{

	float tmp = 0.0;  //内积
	for (int i = 0; i < n; ++i)
		tmp += lhs[i] * rhs[i];
	return tmp;
}
float getSimilarity1(float *lhs, float * rhs, int n)
{

	float tmp = 0.0;  //内积
	for (int i = 0; i < n; ++i)
		tmp += lhs[i] * rhs[i];
	return tmp;
}



void normalizex(int n,  float * data)
{
	float sum = 0.0;
	for (int i = 0; i < n; ++i)
		sum += data[i] * data[i];
	for (int i = 0; i < n; i++)
		data[i] = data[i] / sqrt(sum);
}

// int dot_short_SSE(int len, const short* v1, const short* v2)
// {
// 	int sumi;
// #ifdef use_aligned
// 	int tmp[8 + 8], *q = (int*)(((long long)tmp + 8) >> 5 << 5);
// #else
// 	int q[8];
// 	// __declspec(align(16)) int q[8];
// #endif

// 	__m256i result = _mm256_setzero_si256();
// 	__m256i val1, val2;
// 	// __declspec(align(32)) __m256i val1, val2;

// 	int i = 0;
// 	const __m256i* vv1 = (const __m256i*)v1;
// 	const __m256i* vv2 = (const __m256i*)v2;
// 	for (; i < len; i += 16)
// 	{
// 		val1 = my_mm256_load_si256(vv1++);
// 		val2 = my_mm256_load_si256(vv2++);
// 		val1 = _mm256_madd_epi16(val1, val2);
// 		result = _mm256_add_epi32(result, val1);
// 	}
// 	my_mm256_store_si256((__m256i*)q, result);
// 	sumi = q[0] + q[1] + q[2] + q[3] + q[4] + q[5] + q[6] + q[7];
// 	/*for (; i < len; i++)
// 	{
// 	sumi += (int)v1[i] * (int)v2[i];
// 	}*/

// 	return sumi;
// }

bool judge(const pair<string, float> a, const pair<string, float> b) {
	return a.second>b.second;
}

string  getPersonId(string & fileName)
{
	int pos = fileName.find_last_of("\\");
	string subStr = fileName.substr(pos + 1);
	int pos2 = subStr.find_first_of("_");
	string personId = subStr.substr(0, pos2);
	return personId;
}

int getLabel(string &file)
{
	int pos1 = file.find_last_of("-");
	int pos2 = file.find_last_of(".");
	string substr = file.substr(pos1 + 1, pos2 - pos1 - 1);
	return atoi(substr.c_str());
}



int  getNameAndNumber(const string &txtfileName, vector<string>&outPutNameArray, int attrTagArray[])
{
	//string  txtFileName = "H:/svn1/hz_object/project/hz_objectPedestrain/model/PedestrainGlobal.prototxt";
	ifstream  infile;
	infile.open(txtfileName, ios::in);
	vector<string> myStrArray;
	char myStr[1000];
	string outPutStr = "top: \"prob";
	string outPutStr1 = "num_output";
	string InnerProduct = "inner_product_param";
	int index = 0;
	while (infile.getline(myStr, 1000))
	{
		myStrArray.push_back(myStr);
		//cout << myStr << endl;	
	}
	infile.close();
	for (int i = 0; i < myStrArray.size(); i++)
	{
		string::size_type idx;
		string::size_type idx_new;
		idx = myStrArray[i].find(outPutStr);
		idx_new = myStrArray[i].find(outPutStr1);
		if (idx != string::npos)
		{
			int pos1 = myStrArray[i].find("\"");
			string  substr1 = myStrArray[i].substr(pos1 + 1);
			int pos2 = substr1.find("\"");
			string substr2 = substr1.substr(0, substr1.length() - 1);
			outPutNameArray.push_back(substr2);
		}
		if (idx_new != string::npos)
		{
			string::size_type idx2;
			idx2 = myStrArray[i - 1].find(InnerProduct);
			if (idx2 != string::npos)
			{
				//cout << myStrArray[i] << endl;
				int pos1 = myStrArray[i].find(":");
				string  substr1 = myStrArray[i].substr(pos1 + 2);
				attrTagArray[index++] = atoi(substr1.c_str());
			}
		}
	}
	return  outPutNameArray.size();

}

void  getNumOfAttribute(const string &txtfileName, int *&ppAttrTagArray)
{
	//string  txtFileName = "H:/svn1/hz_object/project/hz_objectPedestrain/model/PedestrainGlobal.prototxt";
	ifstream  infile;
	infile.open(txtfileName, ios::in);
	vector<string> myStrArray;
	char myStr[1000];
	string outPutStr = "top: \"prob";  //通过判断最后的prob 数量，来判断有多少个属性
	int index = 0;
	while (infile.getline(myStr, 1000))
	{
		myStrArray.push_back(myStr);
		//cout << myStr << endl;	
	}
	infile.close();
	for (int i = 0; i < myStrArray.size(); i++)
	{
		string::size_type idx;
		idx = myStrArray[i].find(outPutStr);
		if (idx != string::npos)
		{
			index++;
		}
	}
	ppAttrTagArray = new int[index];

	/**ppAttrTagArray = pArray;*/

	//return  index;
}


#endif

