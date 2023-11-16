#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>
#include <vector>

std::vector<std::string> SplitByChar(const std::string &s, char c, bool allow_empty = false);

#endif // FILE_UTILS_H