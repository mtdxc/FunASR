/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
/* 2022-2023 by zhaomingwork */

// client for websocket, support multiple threads
// ./funasr-wss-client  --server-ip <string>
//                     --port <string>
//                     --wav-path <string>
//                     [--thread-num <int>]
//                     [--is-ssl <int>]  [--]
//                     [--version] [-h]
// example:
// ./funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path test.wav
// --thread-num 1 --is-ssl 1

#define ASIO_STANDALONE 1
#include <glog/logging.h>
#include "portaudio.h" 

#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <hv/WebSocketClient.h>
#include "util.h"
#include "audio.h"
//#include "nlohmann/json.hpp"
#include "tclap/CmdLine.h"
#include "microphone.h"
using namespace hv;

// template for tls or not config
class MyWebSocketClient : public WebSocketClient {
 public:
  bool is_record_;
  std::shared_ptr<funasr::Audio> audio;
  std::vector<float> buffer;
  bool sendBuffer() {
    int len = buffer.size();
    if (!len) return true;
    short* iArray = new short[len];
    for (size_t i = 0; i < len; ++i) {
      iArray[i] = (short)(buffer[i] * 32768);
    }
    buffer.clear();

    int ec = send((const char*)iArray, len * sizeof(short));
    if (ec < 0) {
      LOG(ERROR) << "Send Error: " << ec;
    }
    delete[] iArray;
    return true;
  }

  std::string wav_format;
  nlohmann::json jsonbegin;
  hv::TimerID tmSend = 0;
  MyWebSocketClient(hv::EventLoopPtr loop = NULL) : WebSocketClient(loop) {
    // Bind the handlers we are using
    onopen = [this]() {
      const HttpResponsePtr& resp = getHttpResponse();
      LOG(INFO) << "onopen " << resp->body;

      send(jsonbegin.dump());
      startSend();
    };
    onmessage = [this](const std::string& msg) {
      on_message(msg);
    };
    onclose = [this]() {
        LOG(INFO) << "onclose";
        stopSend();
    };
  }

  void startSend(){
    stopSend();
    tmSend = setInterval(20, [this](TimerID id) {
      bool end = false;
      if (is_record_) {
        end = !sendBuffer();
      }
      else {
        end = !send_wav_frame();
      }
      if (end) {
        sendEnd();
        stopSend();
      }
    });
  }
  void stopSend() {
    if (tmSend) {
      killTimer(tmSend);
      tmSend = 0;
    }
  }

  ~MyWebSocketClient() {
    stopSend();
    if (stream) {
      int err = Pa_CloseStream(stream);
      if (err != paNoError) {
        LOG(INFO) << "portaudio error: " << Pa_GetErrorText(err);
      }
    }
  }

  void on_message(const std::string& payload) {
    switch (opcode()) {
      case WS_OPCODE_TEXT:
        nlohmann::json jsonresult = nlohmann::json::parse(payload);
        LOG(INFO) << "on_message = " << payload;

        if (jsonresult["is_final"] == true) {
          close();
        }
    }
  }

  // This method will block until the connection is complete
  void run(const std::string& uri, const std::string& wav_list,
           const std::string& wav_ids, int audio_fs, std::string asr_mode,
           std::vector<int> chunk_size, const std::unordered_map<std::string, int>& hws_map,
           bool is_record=false, int use_itn=1, int svs_itn=1) {
    // Create a new connection to the given URI
    is_record_ = is_record;
    bool ret = false;
    if (is_record) {
      ret = send_rec_data(asr_mode, chunk_size, hws_map, use_itn, svs_itn);
    }else{
      ret = send_wav_data(wav_list, wav_ids, audio_fs, asr_mode, chunk_size, hws_map, use_itn, svs_itn);
    }
    open(uri.c_str());
  }

  // send wav to server
  bool send_wav_data(string wav_path, string wav_id, int audio_fs, std::string asr_mode,
    std::vector<int> chunk_vector, const std::unordered_map<std::string, int>& hws_map,
    int use_itn, int svs_itn) {
    uint64_t count = 0;
    std::stringstream val;

    audio.reset(new funasr::Audio(1));
    int32_t sampling_rate = audio_fs;
    wav_format = "pcm";
    if (funasr::IsTargetFile(wav_path.c_str(), "wav")) {
      if (!audio->LoadWav(wav_path.c_str(), &sampling_rate, false))
        return false;
    }
    else if (funasr::IsTargetFile(wav_path.c_str(), "pcm")) {
      if (!audio->LoadPcmwav(wav_path.c_str(), &sampling_rate, false)) return false;
    }
    else {
      wav_format = "others";
      if (!audio->LoadOthers2Char(wav_path.c_str())) return false;
    }

    nlohmann::json chunk_size = nlohmann::json::array();
    chunk_size.push_back(chunk_vector[0]);
    chunk_size.push_back(chunk_vector[1]);
    chunk_size.push_back(chunk_vector[2]);
    jsonbegin["mode"] = asr_mode;
    jsonbegin["chunk_size"] = chunk_size;
    jsonbegin["wav_name"] = wav_id;
    jsonbegin["wav_format"] = wav_format;
    jsonbegin["audio_fs"] = sampling_rate;
    jsonbegin["is_speaking"] = true;
    jsonbegin["itn"] = true;
    jsonbegin["svs_itn"] = true;
    if (use_itn == 0) {
      jsonbegin["itn"] = false;
    }
    if (svs_itn == 0) {
      jsonbegin["svs_itn"] = false;
    }
    if (!hws_map.empty()) {
      LOG(INFO) << "hotwords: ";
      for (const auto& pair : hws_map) {
        LOG(INFO) << pair.first << " : " << pair.second;
      }
      nlohmann::json json_map(hws_map);
      std::string json_map_str = json_map.dump();
      jsonbegin["hotwords"] = json_map_str;
    }
    return true;
  }

  int offset = 0;
  bool send_wav_frame() {
    float* buff;
    int len, ec;
    int flag = 0;

    // fetch wav data use asr engine api
    if (wav_format == "pcm") {
      if (audio->Fetch(buff, len, flag) > 0) {
        short* iArray = new short[len];
        for (size_t i = 0; i < len; ++i) {
          iArray[i] = (short)(buff[i] * 32768);
        }

        // send data to server
        int offset = 0;
        int block_size = 10240;
        while (offset < len) {
          int send_block = 0;
          if (offset + block_size <= len) {
            send_block = block_size;
          } else {
            send_block = len - offset;
          }
          ec = send((const char*)(iArray + offset), send_block * sizeof(short));
          offset += send_block;
        }

        LOG(INFO) << "sended data len=" << len * sizeof(short);
        if (ec < 0) {
          LOG(ERROR) << "Send Error: " + ec;
        }
        delete[] iArray;
        return true;
      }
    } else {
      int block_size = 204800;
      len = audio->GetSpeechLen();
      char* others_buff = audio->GetSpeechChar();

      if (offset < len) {
        int send_block = 0;
        if (offset + block_size <= len) {
          send_block = block_size;
        } else {
          send_block = len - offset;
        }
        ec = send((const char*)(others_buff + offset), send_block);
        offset += send_block;
        return true;
      }

      LOG(INFO) << "sended data len=" << len;
      if (ec < 0) {
        LOG(ERROR) << "Send Error: " << ec;
      }
    }
    return false;
  }

  void sendEnd() {
    nlohmann::json jsonresult;
    jsonresult["is_speaking"] = false;
    send(jsonresult.dump());
  }

  static int RecordCallback(const void* inputBuffer, void* outputBuffer,
      unsigned long framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo,
      PaStreamCallbackFlags statusFlags, void* userData)
  {
      std::vector<float>* buffer = static_cast<std::vector<float>*>(userData);
      const float* input = static_cast<const float*>(inputBuffer);
      for (unsigned int i = 0; i < framesPerBuffer; i++)
      {
          buffer->push_back(input[i]);
      }

      return paContinue;
  }

  PaStream* stream = nullptr;
  bool send_rec_data(std::string asr_mode, std::vector<int> chunk_vector, 
                     const std::unordered_map<std::string, int>& hws_map, int use_itn, int svs_itn) {
   
    float sample_rate = 16000;
    nlohmann::json chunk_size = nlohmann::json::array();
    chunk_size.push_back(chunk_vector[0]);
    chunk_size.push_back(chunk_vector[1]);
    chunk_size.push_back(chunk_vector[2]);
    jsonbegin["mode"] = asr_mode;
    jsonbegin["chunk_size"] = chunk_size;
    jsonbegin["wav_name"] = "record";
    jsonbegin["wav_format"] = "pcm";
    jsonbegin["audio_fs"] = sample_rate;
    jsonbegin["is_speaking"] = true;
    jsonbegin["itn"] = true;
    jsonbegin["svs_itn"] = true;
    if(use_itn == 0){
      jsonbegin["itn"] = false;
    }
    if(svs_itn == 0){
        jsonbegin["svs_itn"] = false;
    }
    if(!hws_map.empty()){
        LOG(INFO) << "hotwords: ";
        for (const auto& pair : hws_map) {
            LOG(INFO) << pair.first << " : " << pair.second;
        }
        nlohmann::json json_map(hws_map);
        std::string json_map_str = json_map.dump();
        jsonbegin["hotwords"] = json_map_str;
    }

    // mic
    Microphone mic;
    PaDeviceIndex num_devices = Pa_GetDeviceCount();
    LOG(INFO) << "Num devices: " << num_devices;

    PaStreamParameters param;

    param.device = Pa_GetDefaultInputDevice();
    if (param.device == paNoDevice) {
      LOG(INFO) << "No default input device found";
      return false;
    }
    LOG(INFO) << "Use default device: " << param.device;

    const PaDeviceInfo *info = Pa_GetDeviceInfo(param.device);
    LOG(INFO) << "  Name: " << info->name;
    LOG(INFO) << "  Max input channels: " << info->maxInputChannels;

    param.channelCount = 1;
    param.sampleFormat = paFloat32;

    param.suggestedLatency = info->defaultLowInputLatency;
    param.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                      sample_rate,
                      0,          // frames per buffer
                      paClipOff,  // we won't output out of range samples
                                  // so don't bother clipping them
                      RecordCallback, &buffer);
    if (err != paNoError) {
      LOG(ERROR) << "portaudio error: " << Pa_GetErrorText(err);
      return false;
    }

    err = Pa_StartStream(stream);
    LOG(INFO) << "Started: ";

    if (err != paNoError) {
      LOG(ERROR) << "portaudio error: " << Pa_GetErrorText(err);
      return false;
    }
    return true;
  }

 private:
  int total_num = 0;
};
using MyWebSocketClientPtr = std::shared_ptr<MyWebSocketClient>;

int main(int argc, char* argv[]) {
#ifdef _WIN32
  #include <windows.h>
  SetConsoleOutputCP(65001);
#endif
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  TCLAP::CmdLine cmd("funasr-wss-client-2pass", ' ', "1.0");
  TCLAP::ValueArg<std::string> server_ip_("", "server-ip", "server-ip", false, "127.0.0.1", "string");
  TCLAP::ValueArg<std::string> port_("", "port", "port", false, "10095", "string");
  TCLAP::ValueArg<std::string> wav_path_("", "wav-path",
      "the input could be: wav_path, e.g.: asr_example.wav; pcm_path, e.g.: "
      "asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)",
      false, "", "string");
  TCLAP::ValueArg<std::int32_t> audio_fs_("", "audio-fs", "the sample rate of audio", false, 16000, "int32_t");
  TCLAP::ValueArg<int> record_("", "record", "record is 1 means use record", false, 0,
      "int");
  TCLAP::ValueArg<std::string> asr_mode_("", ASR_MODE, "offline, online, 2pass",
                                         false, "2pass", "string");
  TCLAP::ValueArg<std::string> chunk_size_("", "chunk-size", "chunk_size: 5-10-5 or 5-12-5",
                                           false, "5-10-5", "string");
  TCLAP::ValueArg<int> thread_num_("", "thread-num", "thread-num", false, 1, "int");
  TCLAP::ValueArg<int> is_ssl_("", "is-ssl",
      "is-ssl is 1 means use wss connection, or use ws connection", false, 1,
      "int");
  TCLAP::ValueArg<int> use_itn_("", "use-itn",
      "use-itn is 1 means use itn, 0 means not use itn", false, 1,
      "int");
  TCLAP::ValueArg<int> svs_itn_("", "svs-itn",
      "svs-itn is 1 means use itn and punc, 0 means not use", false, 1, "int");
  TCLAP::ValueArg<std::string> hotword_("", HOTWORD,
      "the hotword file, one hotword perline, Format: Hotword Weight (could be: 阿里巴巴 20)", false, "", "string");

  cmd.add(server_ip_);
  cmd.add(port_);
  cmd.add(wav_path_);
  cmd.add(audio_fs_);
  cmd.add(asr_mode_);
  cmd.add(record_);
  cmd.add(chunk_size_);
  cmd.add(thread_num_);
  cmd.add(is_ssl_);
  cmd.add(use_itn_);
  cmd.add(svs_itn_);
  cmd.add(hotword_);
  cmd.parse(argc, argv);

  std::string server_ip = server_ip_.getValue();
  std::string port = port_.getValue();
  std::string wav_path = wav_path_.getValue();
  std::string asr_mode = asr_mode_.getValue();
  std::string chunk_size_str = chunk_size_.getValue();
  int use_itn = use_itn_.getValue();
  int svs_itn = svs_itn_.getValue();
  // get chunk_size
  std::vector<int> chunk_size;
  std::stringstream ss(chunk_size_str);
  std::string item;
  while (std::getline(ss, item, '-')) {
    try {
      chunk_size.push_back(stoi(item));
    } catch (const invalid_argument&) {
      LOG(ERROR) << "Invalid argument: " << item;
      exit(-1);
    }
  }

  int threads_num = thread_num_.getValue();
  int is_ssl = is_ssl_.getValue();
  int is_record = record_.getValue();

  std::string uri = "";
  if (is_ssl == 1) {
    uri = "wss://" + server_ip + ":" + port;
  } else {
    uri = "ws://" + server_ip + ":" + port;
  }

  // hotwords
  std::string hotword_path = hotword_.getValue();
  unordered_map<string, int> hws_map;
  if(!hotword_path.empty()){
      LOG(INFO) << "hotword path: " << hotword_path;
      funasr::ExtractHws(hotword_path, hws_map);
  }

  int audio_fs = audio_fs_.getValue();
  auto loop_thread = std::make_shared<EventLoopThread>();
  loop_thread->start();

  std::map<int, MyWebSocketClientPtr> clients;
  if(is_record == 1){
      auto client = std::make_shared<MyWebSocketClient>(loop_thread->loop());
      client->run(uri, "", "", audio_fs, asr_mode, chunk_size, hws_map, true, use_itn, svs_itn);
      clients[0] = client;
  }else{
    // read wav_path
    std::vector<string> wav_list;
    std::vector<string> wav_ids;
    string default_id = "wav_default_id";
    if (funasr::IsTargetFile(wav_path, "scp")) {
      ifstream in(wav_path);
      if (!in.is_open()) {
        printf("Failed to open scp file");
        return 0;
      }
      string line;
      while (getline(in, line)) {
        istringstream iss(line);
        string column1, column2;
        iss >> column1 >> column2;
        wav_list.emplace_back(column2);
        wav_ids.emplace_back(column1);
      }
      in.close();
    } else {
      wav_list.emplace_back(wav_path);
      wav_ids.emplace_back(default_id);
    }

    for (size_t wav_i = 0; wav_i < wav_list.size(); wav_i = wav_i + threads_num) {
      for (size_t i = 0; i < threads_num; i++) {
        if (wav_i + i >= wav_list.size()) {
          break;
        }

        auto client = std::make_shared<MyWebSocketClient>(loop_thread->loop());
        client->run(uri, wav_list[wav_i + i], wav_ids[wav_i + i], audio_fs, asr_mode, chunk_size, hws_map, false, use_itn, svs_itn);
        clients[i] = client;
      }
    }
  }

  // press Enter to stop
  printf("press enter key to exit\n");
  while (getchar() != '\n');
  loop_thread->stop();
  loop_thread->join();
}
