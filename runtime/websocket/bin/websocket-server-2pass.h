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

#ifndef WEBSOCKET_SERVER_H_
#define WEBSOCKET_SERVER_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#define ASIO_STANDALONE 1  // not boost
#include <glog/logging.h>
#include "util/text-utils.h"

#include <fstream>
#include <functional>
#include <hv/WebSocketServer.h>
#include <hv/EventLoopThreadPool.h>
#include <mutex>
#include "com-define.h"
#include "funasrruntime.h"
//#include "nlohmann/json.hpp"
#include "tclap/CmdLine.h"

using scoped_lock = std::lock_guard<std::recursive_mutex>;
using unique_lock = std::unique_lock<std::recursive_mutex>;

struct FUNASR_MESSAGE {
  using Ptr = std::shared_ptr<FUNASR_MESSAGE>;
  FUNASR_MESSAGE();
  ~FUNASR_MESSAGE();
  nlohmann::json msg;
  std::shared_ptr<std::vector<char>> samples;
  std::vector<std::vector<std::string>> punc_cache;
  std::vector<std::vector<float>> hotwords_embedding;
  std::recursive_mutex thread_lock; // lock for each connection
  FUNASR_HANDLE tpass_online_handle=nullptr;
  std::string online_res = "";
  std::string tpass_res = "";
  hv::EventLoopPtr strand_ = nullptr; // for data execute in order
  FUNASR_DEC_HANDLE decoder_handle=nullptr;

  bool is_eof=false;
  void setEof() {
    unique_lock guard_decoder(thread_lock);
    is_eof=true;
  }
  int access_num=0;
  void addAccessNum(int delta = 1) {
    unique_lock guard_decoder(thread_lock);
    access_num += delta;
  }

  void config(nlohmann::json& json, FUNASR_HANDLE tpass_handle);
  bool decode(std::vector<char>& buffer, bool is_final,
              nlohmann::json& jsonresult,
              FUNASR_HANDLE tpass_handle);
};

// See https://wiki.mozilla.org/Security/Server_Side_TLS for more details about
// the TLS modes. The code below demonstrates how to implement both the modern
enum tls_mode { MOZILLA_INTERMEDIATE = 1, MOZILLA_MODERN = 2 };
class WebSocketServer : public WebSocketService {
 public:
   WebSocketServer(hv::EventLoopThreadPool* pool, FUNASR_HANDLE asr) : io_decoder_(pool), tpass_handle(asr) {
      // set message handle
      onopen = std::bind(&WebSocketServer::on_open, this, std::placeholders::_1, std::placeholders::_2);
      onmessage = std::bind(&WebSocketServer::on_message, this, std::placeholders::_1, std::placeholders::_2);
      onclose = std::bind(&WebSocketServer::on_close, this, std::placeholders::_1);
  }

  void on_message(const WebSocketChannelPtr& channel, const std::string& msg);
  void on_open(const WebSocketChannelPtr& channel, const HttpRequestPtr& req);
  void on_close(const WebSocketChannelPtr& channel);

 private:
  hv::EventLoopThreadPool* io_decoder_;  // threads for asr decoder
  // std::ofstream fout;
  // FUNASR_HANDLE asr_handle;  // asr engine handle
  FUNASR_HANDLE tpass_handle=nullptr;
  bool isonline = true;  // online or offline engine, now only support offline
};

#endif  // WEBSOCKET_SERVER_H_
