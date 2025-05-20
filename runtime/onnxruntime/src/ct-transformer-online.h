/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#pragma once 

namespace funasr {
class CTTransformerOnline : public PuncModel {
/**
 * Author: Speech Lab of DAMO Academy, Alibaba Group
 * CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection
 * https://arxiv.org/pdf/2003.01309.pdf
*/

private:

	CTokenizer m_tokenizer;
	std::vector<std::string> m_strInputNames, m_strOutputNames;
	std::vector<const char*> m_szInputNames, m_szOutputNames;

	std::shared_ptr<Ort::Session> m_session;
    Ort::Env env_;
    Ort::SessionOptions session_options;
public:

	CTTransformerOnline();
	~CTTransformerOnline();
	void InitPunc(const std::string &punc_model, const std::string &punc_config, const std::string &token_file, int thread_num);
	std::vector<int>  Infer(std::vector<int32_t> input_data, int nCacheSize);
	std::string AddPunc(const char* sz_input, std::vector<std::string> &arr_cache, std::string language="zh-cn");
	void Transport(std::vector<float>& In, int nRows, int nCols);
	void VadMask(int size, int vad_pos, std::vector<float>& Result);
	void Triangle(int text_length, std::vector<float>& Result);
};
} // namespace funasr