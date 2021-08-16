#ifndef  _MULTITRTCLASSIFICER_
#define  _MULTITRTCLASSIFICER_

#include "TrtClassificer.h"

class  multiClassifier:public TrtClassificer
{
public :
	multiClassifier(int INPUT_H, int INPUT_W, int CHANNELS, const char * INPUT_NAME, const char *OUTPUT_NAME, int outputSize, int numAttribute, int *numOutPut,std::vector<std::string>& outPutName);
	void doInferenceMultiOutPut(float* input, float **&output, int batchSize);
	int m_numAttribute;
	int *m_numOutPut;
	
	std::vector<std::string> m_outPutName;

	~multiClassifier()
	{
		if (m_numOutPut)
		{
			std::cout << "multiTrtClassificer is running" << std::endl;
			delete m_numOutPut;
		}
	}

};

#endif
