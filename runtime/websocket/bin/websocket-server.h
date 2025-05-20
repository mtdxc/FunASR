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
#include <unordered_map>
#define ASIO_STANDALONE 1  // not boost
#include <glog/logging.h>
#include "util/text-utils.h"

#include <fstream>
#include <functional>
#include <hv/WebSocketServer.h>
#include <hv/EventLoopThreadPool.h>
#include "com-define.h"
#include "funasrruntime.h"
//#include "nlohmann/json.hpp"
#include "tclap/CmdLine.h"
#include <mutex>

using scoped_lock = std::lock_guard<std::recursive_mutex>;
using unique_lock = std::unique_lock<std::recursive_mutex>;

struct FUNASR_MESSAGE {
  using Ptr = std::shared_ptr<FUNASR_MESSAGE>;
  FUNASR_MESSAGE();
  ~FUNASR_MESSAGE();

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
  void config(nlohmann::json& json, FUNASR_HANDLE asr_handle);
  bool decode(const std::vector<char>& buffer, nlohmann::json& resp, FUNASR_HANDLE asr_handle);
  nlohmann::json msg;
  std::shared_ptr<std::vector<char>> samples;
  std::vector<std::vector<float>> hotwords_embedding;
  std::recursive_mutex thread_lock; // lock for each connection
  FUNASR_DEC_HANDLE decoder_handle=nullptr;
};


class WebSocketServer : public WebSocketService {
 public:
  WebSocketServer(hv::EventLoopThreadPool* pool, FUNASR_HANDLE asr) : io_decoder(pool), asr_handle(asr) {
      // set message handle
      onopen = std::bind(&WebSocketServer::on_open, this, std::placeholders::_1, std::placeholders::_2);
      onmessage = std::bind(&WebSocketServer::on_message, this, std::placeholders::_1, std::placeholders::_2);
      onclose = std::bind(&WebSocketServer::on_close, this, std::placeholders::_1);
  }

  void on_message(const WebSocketChannelPtr& channel, const std::string& msg);
  void on_open(const WebSocketChannelPtr& channel, const HttpRequestPtr& req);
  void on_close(const WebSocketChannelPtr& channel);
 private:
  // std::ofstream fout;
  FUNASR_HANDLE asr_handle;  // asr engine handle
  bool isonline = false;  // online or offline engine, now only support offline
  hv::EventLoopThreadPool* io_decoder;
  // use map to keep the received samples data from one connection in offline
  // engine. if for online engline, a data struct is needed(TODO)
};

// std::unordered_map<std::string, int>& hws_map, int fst_inc_wts, std::string& nn_hotwords
#endif  // WEBSOCKET_SERVER_H_
