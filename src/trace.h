#pragma once
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

// An operation dependency DAG. HBM controller internals are opaque timed spans.
class Trace {
 public:
  using Time = std::uint64_t;
  struct Node {
    std::string kind;
    int device;
    Time start, end;
    std::vector<std::size_t> parents;
    Time gemm_flops = 0, attention_flops = 0;
  };
  std::vector<Node> nodes{{"origin", -1, 0, 0, {}}};
  Time epoch = 0;
  Time end(std::size_t id) const { return nodes.at(id).end + epoch; }
  std::size_t add(std::string kind, int device, Time start, Time finish,
                  std::vector<std::size_t> parents) {
    if (start < epoch || finish < start) throw std::logic_error("invalid trace interval");
    for (auto id : parents)
      if (end(id) > start) throw std::logic_error("trace dependency travels backwards");
    nodes.push_back({std::move(kind), device, start - epoch, finish - epoch,
                     std::move(parents)});
    return nodes.size() - 1;
  }
  std::size_t join(const std::vector<std::size_t>& parents) {
    Time at = epoch;
    for (auto id : parents) at = std::max(at, end(id));
    return add("join", -1, at, at, parents);
  }
  void report() const {
    std::cout << ",\"trace\":[";
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      const auto& n = nodes[i];
      if (i) std::cout << ',';
      std::cout << "{\"id\":" << i << ",\"kind\":\"" << n.kind
                << "\",\"device\":" << n.device << ",\"start_ps\":" << n.start
                << ",\"end_ps\":" << n.end << ",\"gemm_flops\":" << n.gemm_flops
                << ",\"attention_flops\":" << n.attention_flops << ",\"parents\":[";
      for (std::size_t j = 0; j < n.parents.size(); ++j) {
        if (j) std::cout << ',';
        std::cout << n.parents[j];
      }
      std::cout << "]}";
    }
    std::cout << ']';
  }
};
