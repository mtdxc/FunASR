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
// ./funasr-wss-client --server-ip 127.0.0.1 --port 10095 --wav-path test.wav --thread-num 1 --is-ssl 1

#define ASIO_STANDALONE 1
#include <hv/WebSocketClient.h>
#include <fstream>
#include <atomic>
#include <thread>
#include <glog/logging.h>
#include "util.h"
#include "audio.h"
#include <condition_variable>
//#include "nlohmann/json.hpp"
#include "tclap/CmdLine.h"
using namespace hv;

// template for tls or not config
class MyWebsocketClient : public WebSocketClient {
  public:
    MyWebsocketClient(hv::EventLoopPtr loop = NULL) : WebSocketClient(loop), audio(1) {
      onopen = [this]() {
        const HttpResponsePtr& resp = getHttpResponse();
        LOG(INFO) << "onopen " << resp->body; 
        send_wav_data(true);
      };
      onmessage = [this](const std::string& msg) {
        on_message(msg);
      };
      onclose = [this]() {
        LOG(INFO) << "onclose";
        clearTimer();
      };
    }

    ~MyWebsocketClient() {
      clearTimer();
    }

    hv::TimerID tmSend = 0;
    void clearTimer() {
      if (tmSend) {
        killTimer(tmSend);
        tmSend = 0;
      }
    }
    void on_message(const std::string& payload) {
        switch (opcode()) {
        case WS_OPCODE_TEXT:
            total_recv=total_recv+1;
            LOG(INFO)<< "total_recv=" << total_recv <<", on_message = " << payload;
            total_send++;
            if (total_send >= wav_list.size()) {
              LOG(INFO) << "close client thread";
              this->close();
            }
            else {
              send_wav_data(false);
            }
            break;
        }
    }
    int wav_index = 0;
    std::vector<string> wav_list, wav_ids;
    int audio_fs, use_itn, svs_itn;
    std::unordered_map<std::string, int> hws_map;
    // This method will block until the connection is complete  
    void run(const std::vector<string>& wav_list, const std::vector<string>& wav_ids, 
             int audio_fs, const std::unordered_map<std::string, int>& hws_map, 
             int use_itn=1, int svs_itn=1) {
        this->wav_list = wav_list;
        this->wav_ids = wav_ids;
        this->audio_fs = audio_fs;
        this->hws_map = hws_map;
        this->use_itn = use_itn;
        this->svs_itn = svs_itn;
    }

    funasr::Audio audio;
    std::string wav_format;
    // send wav to server
    void send_wav_data(bool send_hotword) {
      uint64_t count = 0;
      std::stringstream val;
      std::string wav_path = wav_list[wav_index];
      std::string wav_id = wav_ids[wav_index];
      int32_t sampling_rate = audio_fs;
      wav_format = "pcm";
      if (funasr::IsTargetFile(wav_path, "wav")) {
          if (!audio.LoadWav(wav_path.c_str(), &sampling_rate))
              return;
      }
      else if (funasr::IsTargetFile(wav_path, "pcm")) {
        if (!audio.LoadPcmwav(wav_path.c_str(), &sampling_rate, false))
          return;
      }
      else {
        wav_format = "others";
        if (!audio.LoadOthers2Char(wav_path.c_str()))
          return;
      }

      nlohmann::json jsonbegin;
      nlohmann::json chunk_size = nlohmann::json::array();
      chunk_size.push_back(5);
      chunk_size.push_back(10);
      chunk_size.push_back(5);
      jsonbegin["chunk_size"] = chunk_size;
      jsonbegin["chunk_interval"] = 10;
      jsonbegin["wav_name"] = wav_id;
      jsonbegin["wav_format"] = wav_format;
      jsonbegin["audio_fs"] = sampling_rate;
      jsonbegin["itn"] = true;
      jsonbegin["svs_itn"] = true;
      if (use_itn == 0) {
        jsonbegin["itn"] = false;
      }
      if (svs_itn == 0) {
        jsonbegin["svs_itn"] = false;
      }
      jsonbegin["is_speaking"] = true;
      if (send_hotword) {
        if (!hws_map.empty()) {
          LOG(INFO) << "hotwords: ";
          for (const auto& pair : hws_map) {
            LOG(INFO) << pair.first << " : " << pair.second;
          }
          nlohmann::json json_map(hws_map);
          std::string json_map_str = json_map.dump();
          jsonbegin["hotwords"] = json_map_str;
        }
      }

      send(jsonbegin.dump());
      offset = 0;
      clearTimer();
      tmSend = setInterval(20, [this](hv::TimerID tid) {
        if (!send_frame()) {
          sendEof();
          clearTimer();
        }
      });
    }
    int offset = 0;
    bool send_frame() {
      int ec;
      float* buff;
      int len;
      int flag = 0;
      // fetch wav data use asr engine api
      if (wav_format == "pcm") {
        if (audio.Fetch(buff, len, flag) > 0) {
          short* iArray = new short[len];
          for (size_t i = 0; i < len; ++i) {
            iArray[i] = (short)(buff[i] * 32768);
          }

          // send data to server
          int block_size = 102400;
          while (offset < len) {
            int send_block = 0;
            if (offset + block_size <= len) {
              send_block = block_size;
            }
            else {
              send_block = len - offset;
            }
            ec = send((const char*)(iArray + offset), send_block * sizeof(short));
            offset += send_block;
          }

          LOG(INFO) << "Thread: " << this_thread::get_id() << ", sended data len=" << len * sizeof(short);
          // The most likely error that we will get is that the connection is
          // not in the right state. Usually this means we tried to send a
          // message to a connection that was closed or in the process of
          // closing. While many errors here can be easily recovered from,
          // in this simple example, we'll stop the data loop.
          if (ec < 0) {
            printf("Send Error: " + ec);
          }
          delete[] iArray;
          // WaitABit();
          return true;
        }
      }
      else {
        int block_size = 204800;
        len = audio.GetSpeechLen();
        char* others_buff = audio.GetSpeechChar();

        if (offset < len) {
          int send_block = 0;
          if (offset + block_size <= len) {
            send_block = block_size;
          }
          else {
            send_block = len - offset;
          }
          ec = send((const char*)(others_buff + offset), send_block);
          offset += send_block;
          return true;
        }

        LOG(INFO) << "sended data len=" << len;
        // The most likely error that we will get is that the connection is
        // not in the right state. Usually this means we tried to send a
        // message to a connection that was closed or in the process of
        // closing. While many errors here can be easily recovered from,
        // in this simple example, we'll stop the data loop.
      }
      return false;
    }

    void sendEof(){
        nlohmann::json jsonresult;
        jsonresult["is_speaking"] = false;
        send(jsonresult.dump());
    }

  private:
    bool close_client=false;
    int total_send=0;
    int total_recv=0;
};

using MyWebSocketClientPtr = std::shared_ptr<MyWebsocketClient>;

int main(int argc, char* argv[]) {
#ifdef _WIN32
    #include <windows.h>
    SetConsoleOutputCP(65001);
#endif
    google::InitGoogleLogging(argv[0]);
    FLAGS_logtostderr = true;

    TCLAP::CmdLine cmd("funasr-wss-client", ' ', "1.0");
    TCLAP::ValueArg<std::string> server_ip_("", "server-ip", "server-ip", false, "127.0.0.1", "string");
    TCLAP::ValueArg<std::string> port_("", "port", "port", false, "10095", "string");
    TCLAP::ValueArg<std::string> wav_path_("", "wav-path", 
        "the input could be: wav_path, e.g.: asr_example.wav; pcm_path, e.g.: asr_example.pcm; wav.scp, kaldi style wav list (wav_id \t wav_path)", 
        true, "", "string");
    TCLAP::ValueArg<std::int32_t> audio_fs_("", "audio-fs", "the sample rate of audio", false, 16000, "int32_t");
    TCLAP::ValueArg<int> thread_num_("", "thread-num", "thread-num",
        false, 1, "int");
    TCLAP::ValueArg<int> is_ssl_(
        "", "is-ssl", "is-ssl is 1 means use wss connection, or use ws connection", 
        false, 1, "int");
    TCLAP::ValueArg<int> use_itn_(
        "", "use-itn",
        "use-itn is 1 means use itn, 0 means not use itn", false, 1, "int");
    TCLAP::ValueArg<int> svs_itn_(
        "", "svs-itn",
        "svs-itn is 1 means use itn and punc, 0 means not use", false, 1, "int");
    TCLAP::ValueArg<std::string> hotword_("", HOTWORD,
        "the hotword file, one hotword perline, Format: Hotword Weight (could be: 阿里巴巴 20)", false, "", "string");

    cmd.add(server_ip_);
    cmd.add(port_);
    cmd.add(wav_path_);
    cmd.add(audio_fs_);
    cmd.add(thread_num_);
    cmd.add(is_ssl_);
    cmd.add(use_itn_);
    cmd.add(svs_itn_);
    cmd.add(hotword_);
    cmd.parse(argc, argv);

    std::string server_ip = server_ip_.getValue();
    std::string port = port_.getValue();
    std::string wav_path = wav_path_.getValue();
    int threads_num = thread_num_.getValue();
    int is_ssl = is_ssl_.getValue();
    int use_itn = use_itn_.getValue();
    int svs_itn = svs_itn_.getValue();

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

    // read wav_path
    std::vector<string> wav_list;
    std::vector<string> wav_ids;
    string default_id = "wav_default_id";
    if(funasr::IsTargetFile(wav_path, "scp")){
        ifstream in(wav_path);
        if (!in.is_open()) {
            printf("Failed to open scp file");
            return 0;
        }
        string line;
        while(getline(in, line))
        {
            istringstream iss(line);
            string column1, column2;
            iss >> column1 >> column2;
            wav_list.emplace_back(column2);
            wav_ids.emplace_back(column1);
        }
        in.close();
    }else{
        wav_list.emplace_back(wav_path);
        wav_ids.emplace_back(default_id);
    }
   
    int audio_fs = audio_fs_.getValue();
    auto loop_thread = std::make_shared<EventLoopThread>();
    loop_thread->start();

    std::map<int, MyWebSocketClientPtr> clients;
    for (int i = 0; i < threads_num; ++i) {
        auto client = std::make_shared<MyWebsocketClient>(loop_thread->loop());
        client->run(wav_list, wav_ids, audio_fs, hws_map, use_itn, svs_itn);
        client->open(uri.c_str());
        clients[i] = client;
    }

    // press Enter to stop
    printf("press enter key to exit\n");
    while (getchar() != '\n');
    loop_thread->stop();
    loop_thread->join();
}
