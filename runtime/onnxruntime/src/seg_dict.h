/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#ifndef SEG_DICT_H
#define SEG_DICT_H

#include <stdint.h>
#include <string>
#include <vector>
#include <map>

namespace funasr {
class SegDict {
  private:
    std::map<std::string, std::vector<std::string>> seg_dict;

  public:
    SegDict(const char *filename);
    ~SegDict();
    std::vector<std::string> GetTokensByWord(const std::string &word);
};

} // namespace funasr
#endif
