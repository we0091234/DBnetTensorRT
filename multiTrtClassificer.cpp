#include "multiTrtClassificer.h"


multiClassifier::multiClassifier(int INPUT_H, int INPUT_W, int CHANNELS, const char * INPUT_NAME, const char *OUTPUT_NAME, int outputSize, int numAttribute, int *numOutPut, std::vector<std::string>& outPutName):\
TrtClassificer(INPUT_H, INPUT_W, CHANNELS, INPUT_NAME, OUTPUT_NAME, outputSize)
{
	this->m_numAttribute = numAttribute;
	this->m_numOutPut = new int[this->m_numAttribute];
	for (int i = 0; i < m_numAttribute; i++)
	{
		m_numOutPut[i] = numOutPut[i];
	}
	m_outPutName = outPutName;
	
}

void multiClassifier::doInferenceMultiOutPut(float* input, float **&output, int batchSize)
{
	//int numAttribute = 11;
	std::vector<std::string> haha = { "prob_1","prob_2","prob_3", "prob_4" ,"prob_5" ,"prob_6" ,"prob_7" ,"prob_8" ,"prob_9" ,"prob_a" ,"prob_b" };
	//int numAttri[] = { 2,11,11,2,3,2,4,3,4,2,4 };
	const ICudaEngine& engine = (*context).getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	// std::cout<<engine.getNbBindings()<<std::endl;
	assert(engine.getNbBindings() == m_numAttribute+1); //numofAttribute  and   input  layer
	void* buffers[11];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(this->_inputName);
	/*outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME),
	outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1);*/
	int opIndex[11];
	for (int i = 0; i < m_numAttribute; i++)
	{
		opIndex[i] = engine.getBindingIndex(m_outPutName[i].c_str());
	}

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize *this->_channel *this->_input_h * this->_input_w * sizeof(float)));
	for (int i = 0; i < m_numAttribute; i++)
	{
		CHECK(cudaMalloc(&buffers[opIndex[i]], batchSize * m_numOutPut[i] * sizeof(float)));
	}
	/*CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float)));*/
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * this->_channel *this->_input_h * this->_input_w * sizeof(float), cudaMemcpyHostToDevice, stream));
	(*context).enqueue(batchSize, buffers, stream, nullptr);

	for (int i = 0; i < m_numAttribute; i++)
	{
		CHECK(cudaMemcpyAsync(output[i], buffers[opIndex[i]], batchSize *m_numOutPut[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));
	}
	/*CHECK(cudaMemcpyAsync(output[0], buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(output[1], buffers[outputIndex1], batchSize * OUTPUT_SIZE1 * sizeof(float), cudaMemcpyDeviceToHost, stream));*/
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	for (int i = 0; i < m_numAttribute; i++)
	{
		/*CHECK(cudaMemcpyAsync(output[i], buffers[opIndex[i]], batchSize *numAttri[i] * sizeof(float), cudaMemcpyDeviceToHost, stream));*/
		CHECK(cudaFree(buffers[opIndex[i]]));
	}

}