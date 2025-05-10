/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// websocket server for asr engine
// take some ideas from https://github.com/k2-fsa/sherpa-onnx
// online-websocket-server-impl.cc, thanks. The websocket server has two threads
// pools, one for handle network data and one for asr decoder.
// now only support offline engine.

#include "websocket-server-2pass.h"

#include <thread>
#include <utility>
#include <vector>

extern std::unordered_map<std::string, int> hws_map_;
extern int fst_inc_wts_;
extern float global_beam_, lattice_beam_, am_scale_;

nlohmann::json handle_result(FUNASR_RESULT result) {
  nlohmann::json jsonresult;
  jsonresult["text"] = "";

  std::string tmp_online_msg = FunASRGetResult(result, 0);
  if (tmp_online_msg != "") {
    LOG(INFO) << "online_res :" << tmp_online_msg;
    jsonresult["text"] = tmp_online_msg;
    jsonresult["mode"] = "2pass-online";
  }
  std::string tmp_tpass_msg = FunASRGetTpassResult(result, 0);
  if (tmp_tpass_msg != "") {
    LOG(INFO) << "offline results : " << tmp_tpass_msg;
    jsonresult["text"] = tmp_tpass_msg;
    jsonresult["mode"] = "2pass-offline";
  }

  std::string tmp_stamp_msg = FunASRGetStamp(result);
  if (tmp_stamp_msg != "") {
    LOG(INFO) << "offline stamps : " << tmp_stamp_msg;
    jsonresult["timestamp"] = tmp_stamp_msg;
  }

  std::string tmp_stamp_sents = FunASRGetStampSents(result);
  if (tmp_stamp_sents != "") {
    try{
      nlohmann::json json_stamp = nlohmann::json::parse(tmp_stamp_sents);
      LOG(INFO) << "offline stamp_sents : " << json_stamp;
      jsonresult["stamp_sents"] = json_stamp;
    }catch (std::exception const &e)
    {
      LOG(ERROR)<< tmp_stamp_sents << e.what();
      jsonresult["stamp_sents"] = "";
    }
  }

  return jsonresult;
}
// feed buffer to asr engine for decoder
void WebSocketServer::do_decoder(
    std::vector<char>& buffer, 
    WebSocketChannelPtr hdl,
    FUNASR_MESSAGE::Ptr ptr,
    std::vector<std::vector<std::string>>& punc_cache,
    std::vector<std::vector<float>> &hotwords_embedding,
    bool is_final) {
    FUNASR_HANDLE tpass_online_handle = ptr->tpass_online_handle;
    FUNASR_DEC_HANDLE decoder_handle = ptr->decoder_handle;
  // lock for each connection
  if (!tpass_online_handle) {
    LOG(INFO) << "tpass_online_handle is free, return";
    ptr->addAccessNum(-1);
	  return;
  }
  try {
    std::string wav_name = ptr->msg["wav_name"];
    std::string modetype = ptr->msg["mode"];
    std::string wav_format = ptr->msg["wav_format"];
    std::string svs_lang = ptr->msg["svs_lang"];
    int audio_fs = ptr->msg["audio_fs"];
    bool itn = ptr->msg["itn"];
    bool sys_itn = ptr->msg["svs_itn"];

    FUNASR_RESULT Result = nullptr;
    int asr_mode_ = 2;
    if (modetype == "offline") {
      asr_mode_ = 0;
    } else if (modetype == "online") {
      asr_mode_ = 1;
    } else if (modetype == "2pass") {
      asr_mode_ = 2;
    }

    while (buffer.size() >= 800 * 2 && !ptr->is_eof) {
      std::vector<char> subvector = {buffer.begin(), buffer.begin() + 800 * 2};
      buffer.erase(buffer.begin(), buffer.begin() + 800 * 2);

      try {
        if (tpass_online_handle) {
          Result = FunTpassInferBuffer(tpass_handle, tpass_online_handle,
                                       subvector.data(), subvector.size(),
                                       punc_cache, false, audio_fs,
                                       wav_format, (ASR_TYPE)asr_mode_,
                                       hotwords_embedding, itn, decoder_handle,
                                       svs_lang, sys_itn);

        } else {
          ptr->addAccessNum(-1);
          return;
        }
      } catch (std::exception const& e) {
        LOG(ERROR) << e.what();
        ptr->addAccessNum(-1);
        return;
      }
      if (Result) {
        nlohmann::json jsonresult = handle_result(Result);
        jsonresult["wav_name"] = wav_name;
        jsonresult["is_final"] = false;
        if (jsonresult["text"] != "") {
          if (hdl->isConnected()) {
            hdl->write(jsonresult.dump());
          }
        }
        FunASRFreeResult(Result);
      }
    }
    if (is_final && !ptr->is_eof) {
      try {
        if (tpass_online_handle) {
          Result = FunTpassInferBuffer(tpass_handle, tpass_online_handle,
                                       buffer.data(), buffer.size(), punc_cache,
                                       is_final, audio_fs,
                                       wav_format, (ASR_TYPE)asr_mode_,
                                       hotwords_embedding, itn, decoder_handle,
                                       svs_lang, sys_itn);
        } else {
          ptr->addAccessNum(-1);
          return;
        }
      } catch (std::exception const& e) {
        LOG(ERROR) << e.what();
        ptr->addAccessNum(-1);
        return;
      }
      if(punc_cache.size()>0){
        for (auto& vec : punc_cache) {
          vec.clear();
        }
      }
      if (Result) {
        nlohmann::json jsonresult = handle_result(Result);
        jsonresult["wav_name"] = wav_name;
        jsonresult["is_final"] = true;
        if (hdl->isConnected()) {
          hdl->write(jsonresult.dump());
        }
        FunASRFreeResult(Result);
      }else{
        if(wav_format != "pcm" && wav_format != "PCM"){
          nlohmann::json jsonresult;
          jsonresult["text"] = "ERROR. Real-time transcription service ONLY SUPPORT PCM stream.";
          jsonresult["wav_name"] = wav_name;
          jsonresult["is_final"] = true;
          if (hdl->isConnected()) {
            hdl->write(jsonresult.dump());
          }
        }
      }
    }

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  ptr->addAccessNum(-1); 
}

void WebSocketServer::on_open(const WebSocketChannelPtr& channel, const HttpRequestPtr& req) {
    // put a new data vector for new connection
    auto data_msg = channel->newContextPtr<FUNASR_MESSAGE>();

    data_msg->decoder_handle = FunASRWfstDecoderInit(tpass_handle, ASR_TWO_PASS, global_beam_, lattice_beam_, am_scale_);
  	data_msg->strand_ =	io_decoder_->loop();

}

FUNASR_MESSAGE::FUNASR_MESSAGE() {
  samples = std::make_shared<std::vector<char>>();

  msg = nlohmann::json::parse("{}");
  msg["wav_format"] = "pcm";
  msg["wav_name"] = "wav-default-id";
  msg["mode"] = "2pass";
  msg["itn"] = true;
  msg["audio_fs"] = 16000; // default is 16k
  msg["access_num"] = 0; // the number of access for this object, when it is 0, we can free it saftly
  is_eof=false; // if this connection is closed
  msg["svs_lang"]="auto";
  msg["svs_itn"]=true;
  punc_cache.resize(2);
  decoder_handle = nullptr;
  tpass_online_handle = nullptr;
  strand_ = nullptr;
}

FUNASR_MESSAGE::~FUNASR_MESSAGE(){
  if(decoder_handle){
    FunWfstDecoderUnloadHwsRes(decoder_handle);
    FunASRWfstDecoderUninit(decoder_handle);
    decoder_handle = nullptr;
  }
  if(tpass_online_handle) {
    FunTpassOnlineUninit(tpass_online_handle);
    tpass_online_handle = nullptr;
  } 
}


void WebSocketServer::on_close(const WebSocketChannelPtr& channel) {
  std::shared_ptr<FUNASR_MESSAGE> data_msg = channel->getContextPtr<FUNASR_MESSAGE>();
  if (data_msg) {
    data_msg->setEof();
    LOG(INFO) << "on_close, active connections: ";// << data_map.size();
  }
}
 
void WebSocketServer::on_message(const WebSocketChannelPtr& channel, const std::string& payload) {
  // find the sample data vector according to one connection
  std::shared_ptr<FUNASR_MESSAGE> msg_data = channel->getContextPtr<FUNASR_MESSAGE>();
  if (!msg_data || msg_data->is_eof) {
    return;
  }

  std::shared_ptr<std::vector<char>> sample_data_p = msg_data->samples;
  if (sample_data_p == nullptr) {
    LOG(INFO) << "error when fetch sample data vector";
    return;
  }

  unique_lock guard_decoder(msg_data->thread_lock); // mutex for one connection
  switch (channel->opcode) {
    case WS_OPCODE_TEXT: {
      nlohmann::json jsonresult;
      try{
        jsonresult = nlohmann::json::parse(payload);
      }catch (std::exception const &e)
      {
        LOG(ERROR)<<e.what();
        msg_data->is_eof=true;
        guard_decoder.unlock();
        return;
      }

      if (jsonresult.contains("wav_name")) {
        msg_data->msg["wav_name"] = jsonresult["wav_name"];
      }
      if (jsonresult.contains("mode")) {
        msg_data->msg["mode"] = jsonresult["mode"];
      }
      if (jsonresult.contains("wav_format")) {
        msg_data->msg["wav_format"] = jsonresult["wav_format"];
      }

      // hotwords: fst/nn
      if(msg_data->hotwords_embedding.size()){
        std::unordered_map<std::string, int> merged_hws_map;
        std::string nn_hotwords = "";

        if (jsonresult["hotwords"] != nullptr) {
          std::string json_string = jsonresult["hotwords"];
          if (!json_string.empty()){
            try{
              nlohmann::json json_fst_hws = nlohmann::json::parse(json_string);
              if(json_fst_hws.type() == nlohmann::json::value_t::object){
                // fst
                try{
                  std::unordered_map<std::string, int> client_hws_map = json_fst_hws;
                  merged_hws_map.insert(client_hws_map.begin(), client_hws_map.end());
                } catch (const std::exception& e) {
                  LOG(INFO) << e.what();
                }
              }
            } catch (std::exception const &e)
            {
              LOG(ERROR)<<e.what();
              // nn
              std::string client_nn_hws = jsonresult["hotwords"];
              nn_hotwords += " " + client_nn_hws;
              // LOG(INFO) << "nn hotwords: " << client_nn_hws;
            }
          }
        }
        merged_hws_map.insert(hws_map_.begin(), hws_map_.end());

        // fst
        LOG(INFO) << "hotwords: ";
        for (const auto& pair : merged_hws_map) {
            nn_hotwords += " " + pair.first;
            LOG(INFO) << pair.first << " : " << pair.second;
        }
        FunWfstDecoderLoadHwsRes(msg_data->decoder_handle, fst_inc_wts_, merged_hws_map);

        // nn
        msg_data->hotwords_embedding = CompileHotwordEmbedding(tpass_handle, nn_hotwords, ASR_TWO_PASS);
      }

      if (jsonresult.contains("audio_fs")) {
        msg_data->msg["audio_fs"] = jsonresult["audio_fs"];
      }
      if (jsonresult.contains("chunk_size")) {
        if (msg_data->tpass_online_handle == nullptr) {
          std::vector<int> chunk_size_vec = jsonresult["chunk_size"].get<std::vector<int>>();
          // check chunk_size_vec
          if(chunk_size_vec.size() == 3 && chunk_size_vec[1] != 0){
            FUNASR_HANDLE tpass_online_handle = FunTpassOnlineInit(tpass_handle, chunk_size_vec);
            msg_data->tpass_online_handle = tpass_online_handle;
          }else{
            LOG(ERROR) << "Wrong chunk_size!";
            break;
          }
        }
      }
      if (jsonresult.contains("itn")) {
        msg_data->msg["itn"] = jsonresult["itn"];
      }
      if (jsonresult.contains("svs_lang")) {
        msg_data->msg["svs_lang"] = jsonresult["svs_lang"];
      }
      if (jsonresult.contains("svs_itn")) {
        msg_data->msg["svs_itn"] = jsonresult["svs_itn"];
      }
      LOG(INFO) << "jsonresult=" << jsonresult << ", msg_data->msg=" << msg_data->msg;
      if ((jsonresult["is_speaking"] == false ||
          jsonresult["is_finished"] == true) && 
          !msg_data->is_eof &&
          msg_data->hotwords_embedding.size()) {
        LOG(INFO) << "client done";

        // if it is in final message, post the sample_data to decode
        try{
		  
          std::vector<std::vector<float>> hotwords_embedding_(msg_data->hotwords_embedding);
          msg_data->strand_->runInLoop(
              std::bind(&WebSocketServer::do_decoder, this,
                        std::move(*(sample_data_p.get())), channel,
                        msg_data, msg_data->punc_cache,
                        std::move(hotwords_embedding_),
                        true));
		      msg_data->msg["access_num"]=(int)(msg_data->msg["access_num"])+1;
        }
        catch (std::exception const &e)
        {
            LOG(ERROR)<<e.what();
        }
      }
      break;
    }
    case WS_OPCODE_BINARY: {
      // recived binary data
      const auto* pcm_data = static_cast<const char*>(payload.data());
      int32_t num_samples = payload.size();

      if (isonline) {
        sample_data_p->insert(sample_data_p->end(), pcm_data,
                              pcm_data + num_samples);
        int setpsize = 800 * 2;  // TODO, need get from client
        // if sample_data size > setpsize, we post data to decode
        if (sample_data_p->size() > setpsize) {
          int chunksize = floor(sample_data_p->size() / setpsize);
          // make sure the subvector size is an integer multiple of setpsize
          std::vector<char> subvector = {
              sample_data_p->begin(),
              sample_data_p->begin() + chunksize * setpsize};
          // keep remain in sample_data
          sample_data_p->erase(sample_data_p->begin(),
                               sample_data_p->begin() + chunksize * setpsize);

          try{
            // post to decode
            if (!msg_data->is_eof && msg_data->hotwords_embedding.size()) {
              std::vector<std::vector<float>> hotwords_embedding_(msg_data->hotwords_embedding);
              msg_data->strand_->runInLoop(
                        std::bind(&WebSocketServer::do_decoder, this,
                                  std::move(subvector), channel,
                                  msg_data,
                                  msg_data->punc_cache,
                                  std::move(hotwords_embedding_),
                                  false));
              msg_data->msg["access_num"]=(int)(msg_data->msg["access_num"])+1;
            }
          }
          catch (std::exception const &e)
          {
            LOG(ERROR)<<e.what();
          }
        }
      } else {
        sample_data_p->insert(sample_data_p->end(), pcm_data,
                              pcm_data + num_samples);
      }
      break;
    }
    default:
      break;
  }
  guard_decoder.unlock();
}

// init asr model
void WebSocketServer::initAsr(std::map<std::string, std::string>& model_path, int thread_num) {
  try {
    tpass_handle = FunTpassInit(model_path, thread_num);
    if (!tpass_handle) {
      LOG(ERROR) << "FunTpassInit init failed";
      exit(-1);
    }

  } catch (const std::exception& e) {
    LOG(INFO) << e.what();
  }
}
