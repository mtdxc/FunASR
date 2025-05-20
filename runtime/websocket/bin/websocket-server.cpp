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

#include "websocket-server.h"

#include <thread>
#include <utility>
#include <vector>

extern std::unordered_map<std::string, int> hws_map_;
extern int fst_inc_wts_;
extern float global_beam_, lattice_beam_, am_scale_;

FUNASR_MESSAGE::FUNASR_MESSAGE() {
  samples = std::make_shared<std::vector<char>>();
  msg = nlohmann::json::parse("{}");
  msg["wav_format"] = "pcm";
  msg["wav_name"] = "wav-default-id";
  msg["itn"] = true;
  msg["audio_fs"] = 16000; // default is 16k
  msg["svs_lang"]="auto";
  msg["svs_itn"]=true;
}

FUNASR_MESSAGE::~FUNASR_MESSAGE() {
  if (decoder_handle != nullptr) {
    FunWfstDecoderUnloadHwsRes(decoder_handle);
    FunASRWfstDecoderUninit(decoder_handle);
    decoder_handle = nullptr;
  }
}

void FUNASR_MESSAGE::config(nlohmann::json& jsonresult, FUNASR_HANDLE asr_handle){
  if (jsonresult["wav_name"] != nullptr) {
    this->msg["wav_name"] = jsonresult["wav_name"];
  }
  if (jsonresult["wav_format"] != nullptr) {
    this->msg["wav_format"] = jsonresult["wav_format"];
  }

  // hotwords: fst/nn
  if(this->hotwords_embedding.empty()){
    std::unordered_map<std::string, int> merged_hws_map;
    std::string nn_hotwords = "";

    if (jsonresult["hotwords"] != nullptr) {
      std::string json_string = jsonresult["hotwords"];
      if (!json_string.empty()){
        nlohmann::json json_fst_hws;
        try{
          json_fst_hws = nlohmann::json::parse(json_string);
          if(json_fst_hws.type() == nlohmann::json::value_t::object){
            // fst
            std::unordered_map<std::string, int> client_hws_map = json_fst_hws;
            merged_hws_map.insert(client_hws_map.begin(), client_hws_map.end());
          }
        } catch (std::exception const &e) {
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
    FunWfstDecoderLoadHwsRes(this->decoder_handle, fst_inc_wts_, merged_hws_map);

    // nn
    this->hotwords_embedding = CompileHotwordEmbedding(asr_handle, nn_hotwords);
  }
  if (jsonresult.contains("audio_fs")) {
    this->msg["audio_fs"] = jsonresult["audio_fs"];
  }
  if (jsonresult.contains("itn")) {
    this->msg["itn"] = jsonresult["itn"];
  }
  if (jsonresult.contains("svs_lang")) {
    this->msg["svs_lang"] = jsonresult["svs_lang"];
  }
  if (jsonresult.contains("svs_itn")) {
    this->msg["svs_itn"] = jsonresult["svs_itn"];
  }
}

bool FUNASR_MESSAGE::decode(const std::vector<char>& buffer, nlohmann::json& jsonresult, FUNASR_HANDLE asr_handle){
  try {
    int num_samples = buffer.size();  // the size of the buf
    std::string wav_name = msg["wav_name"];
    std::string wav_format = msg["wav_format"];
    std::string svs_lang = msg["svs_lang"];
    int audio_fs = msg["audio_fs"];
    bool itn = msg["itn"];
    bool sys_itn = msg["svs_itn"];
    if (!buffer.empty() && hotwords_embedding.size() > 0) {
      std::string asr_result="";
      std::string stamp_res="";
      std::string stamp_sents="";
      try{
        FUNASR_RESULT Result = FunOfflineInferBuffer(
            asr_handle, buffer.data(), buffer.size(), RASR_NONE, nullptr, 
            hotwords_embedding, audio_fs, wav_format, itn, decoder_handle,
            svs_lang, sys_itn);
        if (Result != nullptr) {
          asr_result = FunASRGetResult(Result, 0);  // get decode result
          stamp_res = FunASRGetStamp(Result);
          stamp_sents = FunASRGetStampSents(Result);
          FunASRFreeResult(Result);
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(20));
          LOG(ERROR) << "FUNASR_RESULT is nullptr.";
        }
      } catch (std::exception const& e) {
        LOG(ERROR) << e.what();
      }

      jsonresult["text"] = asr_result;  // put result in 'text'
      jsonresult["mode"] = "offline";
	    jsonresult["is_final"] = false;
      if(stamp_res != ""){
        jsonresult["timestamp"] = stamp_res;
      }
      if(stamp_sents != ""){
        try{
          nlohmann::json json_stamp = nlohmann::json::parse(stamp_sents);
          jsonresult["stamp_sents"] = json_stamp;
        } catch (std::exception const &e) {
          LOG(ERROR)<<e.what();
          jsonresult["stamp_sents"] = "";
        }
      }
      jsonresult["wav_name"] = wav_name;

      // send the json to client
      LOG(INFO) << "buffer.size=" << buffer.size() << ",result json=" << jsonresult.dump();
    }else{
      LOG(INFO) << "Sent empty msg";
      jsonresult["text"] = "";  // put result in 'text'
      jsonresult["mode"] = "offline";
	    jsonresult["is_final"] = false;
      jsonresult["wav_name"] = wav_name;
    }

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  addAccessNum(-1);
  return true;
}


void WebSocketServer::on_open(const WebSocketChannelPtr& channel, const HttpRequestPtr& req) {
  auto data_msg = channel->newContextPtr<FUNASR_MESSAGE>();
  data_msg->decoder_handle = FunASRWfstDecoderInit(asr_handle, ASR_OFFLINE, global_beam_, lattice_beam_, am_scale_);
  //data_map.emplace(hdl, data_msg);
  //LOG(INFO) << "on_open, active connections: " << data_map.size();
}

void WebSocketServer::on_close(const WebSocketChannelPtr& channel) {
  auto data_msg = channel->getContextPtr<FUNASR_MESSAGE>();
  if (data_msg) {
    data_msg->setEof();
    LOG(INFO) << "on_close, active connections: ";// << data_map.size();
  }
}

void WebSocketServer::on_message(const WebSocketChannelPtr& channel, const std::string& payload) {
  auto msg_data = channel->getContextPtr<FUNASR_MESSAGE>();
  if (!msg_data || msg_data->is_eof) return;

  std::shared_ptr<std::vector<char>> sample_data_p = msg_data->samples;
  if (sample_data_p == nullptr) {
    LOG(INFO) << "error when fetch sample data vector";
    return;
  }

  unique_lock guard_decoder(msg_data->thread_lock); // mutex for one connection
  switch (channel->opcode) {
    case WS_OPCODE_TEXT: {
      nlohmann::json jsonresult;
      try {
        jsonresult = nlohmann::json::parse(payload);
      }
      catch (std::exception const &e)
      {
        LOG(ERROR) << e.what();
        msg_data->is_eof = true;
        return;
      }

      msg_data->config(jsonresult, asr_handle);
      if ((jsonresult["is_speaking"] == false || jsonresult["is_finished"] == true) 
          && !msg_data->is_eof && msg_data->hotwords_embedding.size()) {
        LOG(INFO) << "client done";
        // for offline, send all receive data to decoder engine
        io_decoder->loop()->runInLoop([=]() {
          nlohmann::json resp;
          if (channel->isConnected() && msg_data->decode(*sample_data_p, resp, asr_handle)) {
            channel->send(resp.dump());
          }
        });
        msg_data->addAccessNum(1);
      }
      break;
    }
    case WS_OPCODE_BINARY: {
      // recived binary data
      const auto* pcm_data = static_cast<const char*>(payload.data());
      int32_t num_samples = payload.size();

      if (isonline) {
        // TODO
      } else {
        // for offline, we add receive data to end of the sample data vector
        sample_data_p->insert(sample_data_p->end(), pcm_data, pcm_data + num_samples);
      }
      break;
    }
    default:
      break;
  }
  guard_decoder.unlock();
}
