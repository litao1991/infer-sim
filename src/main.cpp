#include "hbfsim/integration/residency.h"
#include "hbmsim/kernel/event_queue.h"
#include "trace.h"
#include <map>
#include <sstream>
#include <set>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tiered_kv.h"

namespace {
using Time = std::uint64_t;

struct Request {
  Time arrival = 0;
  std::uint32_t prompt = 0, output = 0;
  std::uint32_t prefill_left = 0, generated = 0, kv_reserved_tokens = 0;
  Time first_token = 0, last_token = 0, done = 0, gap_sum = 0;
  std::uint32_t gaps = 0;
  bool admitted = false;
  std::size_t replica = 0;
  std::size_t lane = 0, trace_node = 0;
  Time queued_ps = 0, queue_wait_ps = 0;
  std::vector<Time> token_gaps;
};
struct Config {
  std::uint32_t layers = 0, hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0;
  std::uint32_t dtype = 0, devices = 0, tp = 0, pp = 0, dp = 0;
  std::uint32_t max_batch_tokens = 0, max_seqs = 0, prefill_chunk = 0, kv_block = 0;
  double peak_tflops = 0, compute_efficiency = 0, nvlink_gbps = 0;
  std::string fabric_topology = "nvswitch";
  std::string hardware_kind = "nvidia_gpu";
  std::string hbm_profile = "hbm3_6400";
  Time hbm_capacity = 0, launch_ps = 0, nvlink_latency_ps = 0;
  std::size_t hbm_queue_capacity = 0;
  bool hbf = false;
  bool prefetch = true;
  int writeback_layer = -1;
  Time cache_bytes = 0, slot_bytes = 0;
  bool kv_offload = false, explicit_storage = false;
  std::string kv_policy = "lru";
  Time kv_cache_bytes = 0, kv_logical_capacity = 0, hbf_capacity = 0, kv_hot_blocks = 1;
  std::vector<std::uint32_t> weight_hbm_layers, weight_hbm_experts, kv_hbm_layers;
  std::uint32_t ep = 1, experts = 0, top_k = 1, shared_experts = 0;
  std::uint64_t routing_seed = 1;
  std::vector<double> expert_weights;
  std::vector<std::uint32_t> expert_placement, placement, device_nodes, node_racks;
  std::string collective = "ring";
  double nic_gbps = 100, rack_gbps = 100, cross_rack_gbps = 100;
  Time nic_latency = 10000, rack_latency = 10000, cross_rack_latency = 10000;
  double gemm_efficiency = 0, attention_efficiency = 0, grouped_efficiency = 0;
  double gemm_ps_per_flop = 0, attention_ps_per_flop = 0, grouped_ps_per_flop = 0;
  Time gemm_launch = 0, attention_launch = 0, grouped_launch = 0;
  Time sram_bytes = 0, dma_tile_bytes = 0, dma_latency = 0;
  double dma_GBps = 0;
  std::vector<Request> requests;
};
std::unordered_map<std::string, std::string> read_config(const std::string& path,
                                                         Config& config) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open scenario");
  std::unordered_map<std::string, std::string> fields;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) continue;
    const auto pos = line.find('=');
    if (pos == std::string::npos) throw std::invalid_argument("invalid scenario line");
    const auto key = line.substr(0, pos), value = line.substr(pos + 1);
    if (key == "request") {
      Request r;
      const auto first = value.find(','), second = value.find(',', first + 1);
      if (first == std::string::npos || second == std::string::npos)
        throw std::invalid_argument("invalid request");
      r.arrival = std::stoull(value.substr(0, first));
      r.prompt = std::stoul(value.substr(first + 1, second - first - 1));
      r.output = std::stoul(value.substr(second + 1));
      r.prefill_left = r.prompt;
      config.requests.push_back(r);
    } else {
      if (!fields.emplace(key, value).second)
        throw std::invalid_argument("duplicate scenario field: " + key);
    }
  }
  return fields;
}
Config parse(const std::string& path) {
  Config c;
  const auto f = read_config(path, c);
  auto u = [&](const std::string& key) { return std::stoull(f.at(key)); };
  auto d = [&](const std::string& key) { return std::stod(f.at(key)); };
  c.layers = u("model.layers"); c.hidden = u("model.hidden");
  c.heads = u("model.heads"); c.kv_heads = u("model.kv_heads");
  c.ffn = u("model.ffn"); c.dtype = u("model.dtype_bytes");
  c.devices = u("hardware.devices"); c.peak_tflops = d("hardware.peak_tflops");
  c.compute_efficiency = d("hardware.compute_efficiency");
  c.hbm_capacity = u("hardware.hbm_capacity_bytes");
  c.launch_ps = u("hardware.kernel_launch_ps");
  c.nvlink_gbps = d("hardware.nvlink_gbps");
  c.nvlink_latency_ps = u("hardware.nvlink_latency_ps");
  if (auto it = f.find("hardware.kind"); it != f.end()) c.hardware_kind = it->second;
  if (auto it = f.find("hardware.hbm_profile"); it != f.end()) c.hbm_profile = it->second;
  if (auto it = f.find("hardware.fabric_topology"); it != f.end())
    c.fabric_topology = it->second;
  if (auto it = f.find("hardware.hbm_queue_capacity"); it != f.end())
    c.hbm_queue_capacity = std::stoull(it->second);
  c.tp = u("parallel.tp"); c.pp = u("parallel.pp"); c.dp = u("parallel.dp");
  c.max_batch_tokens = u("serving.max_batch_tokens");
  c.max_seqs = u("serving.max_seqs");
  c.prefill_chunk = u("serving.prefill_chunk_tokens");
  c.kv_block = u("serving.kv_block_tokens");
  if (auto it = f.find("storage.hbf"); it != f.end()) c.hbf = it->second == "1";
  if (auto it = f.find("storage.prefetch"); it != f.end()) c.prefetch = it->second == "1";
  if (auto it = f.find("storage.weights"); it != f.end()) c.hbf = it->second == "hbf";
  c.explicit_storage = f.count("storage.weights") || f.count("storage.kv");
  if (c.hbf) {
    c.cache_bytes = u("storage.cache_bytes_per_device");
    c.slot_bytes = u("storage.slot_bytes");
    if (auto it = f.find("storage.writeback_layer"); it != f.end())
      c.writeback_layer = std::stoi(it->second);
  }
  auto optional_u = [&](const std::string& key, auto& out) {
    if (auto it = f.find(key); it != f.end()) out = std::stoull(it->second);
  };
  auto optional_d = [&](const std::string& key, double& out) {
    if (auto it = f.find(key); it != f.end()) out = std::stod(it->second);
  };
  auto list = [&](const std::string& key, auto& out) {
    if (auto it = f.find(key); it != f.end()) {
      std::stringstream stream(it->second); std::string item;
      while (std::getline(stream, item, ',')) out.push_back(std::stod(item));
    }
  };
  if (auto it = f.find("storage.kv"); it != f.end()) c.kv_offload = it->second == "hbf";
  if (auto it = f.find("storage.kv_cache_policy"); it != f.end()) c.kv_policy = it->second;
  optional_u("storage.kv_cache_bytes_per_device", c.kv_cache_bytes);
  optional_u("storage.kv_capacity_bytes_per_device", c.kv_logical_capacity);
  optional_u("storage.hbf_capacity_bytes", c.hbf_capacity);
  optional_u("storage.kv_hot_blocks", c.kv_hot_blocks);
  list("storage.weight_hbm_layers", c.weight_hbm_layers);
  list("storage.weight_hbm_experts", c.weight_hbm_experts);
  list("storage.kv_hbm_layers", c.kv_hbm_layers);
  c.head_dim = c.heads ? c.hidden / c.heads : 0;
  optional_u("model.head_dim", c.head_dim);
  optional_u("parallel.ep", c.ep);
  if (auto it = f.find("model.kind"); it != f.end() && it->second == "moe") {
    c.experts = u("model.experts"); c.top_k = u("model.top_k");
    optional_u("model.shared_experts", c.shared_experts);
  }
  optional_u("model.routing_seed", c.routing_seed);
  list("model.expert_weights", c.expert_weights);
  list("parallel.expert_placement", c.expert_placement);
  list("parallel.placement", c.placement);
  list("hardware.device_nodes", c.device_nodes);
  list("hardware.node_racks", c.node_racks);
  if (c.expert_weights.empty()) c.expert_weights.assign(c.experts, 1.0);
  if (c.expert_placement.empty())
    for (std::uint32_t e = 0; e < c.experts; ++e) c.expert_placement.push_back(e % c.ep);
  if (c.placement.empty())
    for (std::uint32_t i = 0; i < c.devices; ++i) c.placement.push_back(i);
  if (c.device_nodes.empty()) c.device_nodes.assign(c.devices, 0);
  if (c.node_racks.empty()) c.node_racks.assign(1, 0);
  if (auto it = f.find("hardware.collective"); it != f.end()) c.collective = it->second;
  optional_d("hardware.nic_gbps", c.nic_gbps);
  optional_d("hardware.rack_gbps", c.rack_gbps);
  optional_d("hardware.cross_rack_gbps", c.cross_rack_gbps);
  optional_u("hardware.nic_latency_ps", c.nic_latency);
  optional_u("hardware.rack_latency_ps", c.rack_latency);
  optional_u("hardware.cross_rack_latency_ps", c.cross_rack_latency);
  c.gemm_efficiency = c.attention_efficiency = c.grouped_efficiency = c.compute_efficiency;
  optional_d("hardware.gemm_efficiency", c.gemm_efficiency);
  optional_d("hardware.attention_efficiency", c.attention_efficiency);
  optional_d("hardware.grouped_gemm_efficiency", c.grouped_efficiency);
  optional_d("hardware.gemm_ps_per_flop", c.gemm_ps_per_flop);
  optional_d("hardware.attention_ps_per_flop", c.attention_ps_per_flop);
  optional_d("hardware.grouped_gemm_ps_per_flop", c.grouped_ps_per_flop);
  c.gemm_launch = c.grouped_launch = c.launch_ps;
  optional_u("hardware.gemm_launch_ps", c.gemm_launch);
  optional_u("hardware.attention_launch_ps", c.attention_launch);
  optional_u("hardware.grouped_gemm_launch_ps", c.grouped_launch);
  optional_u("hardware.sram_bytes", c.sram_bytes);
  optional_u("hardware.dma_tile_bytes", c.dma_tile_bytes);
  optional_u("hardware.dma_latency_ps", c.dma_latency);
  optional_d("hardware.dma_GBps", c.dma_GBps);
  if (!c.layers || !c.hidden || !c.heads || !c.kv_heads || !c.tp || !c.pp || !c.dp || !c.ep ||
      c.tp * c.pp * c.dp * c.ep != c.devices || c.layers % c.pp ||
      !c.head_dim || c.heads % c.kv_heads || c.requests.empty())
    throw std::invalid_argument("invalid model or parallel dimensions");
  return c;
}

Time ceil_div(Time a, Time b) { return (a + b - 1) / b; }
Time checked_add(Time a, Time b) {
  if (b > std::numeric_limits<Time>::max() - a) throw std::overflow_error("timeline overflow");
  return a + b;
}
Time aligned(Time a, Time b) { return ceil_div(a, b) * b; }
Time to_time(long double value) {
  if (!std::isfinite(value) || value < 0 ||
      value > static_cast<long double>(std::numeric_limits<Time>::max()))
    throw std::overflow_error("invalid simulated duration");
  return static_cast<Time>(std::ceil(value));
}

struct Batch {
  std::size_t replica = 0;
  std::vector<std::pair<std::size_t, std::uint32_t>> work;
  std::uint32_t tokens = 0;
  Time attention_positions = 0;
  std::size_t cursor = 0;
  std::vector<Time> lane_tokens, lane_positions;
};
struct Op {
  bool compute_done = false, memory_done = false, finished = false;
  std::function<void()> on_done;
  void finish() {
    if (compute_done && memory_done && !finished) {
      finished = true;
      on_done();
    }
  }
};
struct Device {
  std::unique_ptr<hbmsim::HbmSystem> hbm;
  std::unique_ptr<hbfsim::ResidencyManager> residency;
  std::unique_ptr<TieredKV> kv;
  std::deque<std::function<void()>> waiting;
};
struct Replica {
  std::deque<std::size_t> queue;
  std::vector<bool> stage_busy;
  std::vector<std::deque<std::pair<std::shared_ptr<Batch>, std::uint32_t>>> stage_waiting;
  std::size_t inflight = 0;
  Time kv_used = 0, peak_kv = 0;
  std::uint64_t batches = 0;
  std::vector<Time> lane_kv, lane_commitment, lane_hbf_commitment;
  std::vector<std::size_t> stage_trace;
};

class Engine {
 public:
  explicit Engine(Config config) : c_(std::move(config)) {
    const Time query_dim = static_cast<Time>(c_.heads) * c_.head_dim;
    const Time kv_dim = static_cast<Time>(c_.kv_heads) * c_.head_dim;
    attention_params_ = c_.hidden * (query_dim + 2 * kv_dim) + query_dim * c_.hidden;
    expert_params_ = 3ULL * c_.hidden * c_.ffn;
    params_layer_ = attention_params_ + (c_.experts ?
      expert_params_ * c_.shared_experts + static_cast<Time>(c_.hidden) * c_.experts : expert_params_);
    weight_bytes_layer_ = params_layer_ * c_.dtype;
    kv_bytes_token_layer_ = 2ULL * c_.kv_heads * c_.head_dim * c_.dtype;
    objects_per_layer_ = c_.experts ? c_.experts + 1 : 1;
    Time max_experts = 0;
    for (std::size_t lane = 0; lane < c_.ep; ++lane)
      max_experts = std::max<Time>(max_experts, std::count(c_.expert_placement.begin(),
                                                         c_.expert_placement.end(), lane));
    weight_bytes_device_ = ceil_div((weight_bytes_layer_ + max_experts * expert_params_ * c_.dtype)
                                    * c_.layers, c_.tp * c_.pp);
    scratch_bytes_device_ = ceil_div(static_cast<Time>(c_.max_batch_tokens) *
                                     std::max<Time>(c_.hidden, query_dim) * c_.dtype * (c_.experts ? 1 + c_.top_k : 1), c_.tp);
    weight_backing_.resize(c_.devices);
    resident_weights_ = 0;
    for (std::size_t stage = 0; stage < c_.pp; ++stage) {
      Time hbm_layers = 0;
      for (std::uint32_t layer = stage * (c_.layers / c_.pp); layer < (stage + 1) * (c_.layers / c_.pp); ++layer)
        if (!offloaded_kv(layer)) ++hbm_layers;
      kv_hbm_layers_ = std::max(kv_hbm_layers_, hbm_layers);
      kv_hbf_layers_ = std::max(kv_hbf_layers_, c_.layers / c_.pp - hbm_layers);
      for (std::size_t lane = 0; lane < c_.ep; ++lane) {
        Time resident = 0;
        for (std::uint32_t layer = stage * (c_.layers / c_.pp); layer < (stage + 1) * (c_.layers / c_.pp); ++layer)
          for (std::uint32_t index = 0; index < objects_per_layer_; ++index) {
            if (index && c_.expert_placement[index - 1] != lane) continue;
            const auto object = static_cast<std::uint32_t>(base_object(layer) + index);
            if (!hbf_weight(object)) { resident += object_bytes(object); continue; }
            max_hbf_object_bytes_ = std::max(max_hbf_object_bytes_, aligned(object_bytes(object), 4096));
            for (std::size_t dp = 0; dp < c_.dp; ++dp)
              for (std::size_t rank = 0; rank < c_.tp; ++rank) {
                weight_backing_[device_id(dp, stage, rank, lane)][object] = hbf_seed_bytes_;
                hbf_seed_bytes_ += aligned(object_bytes(object), 4096);
              }
          }
        resident_weights_ = std::max(resident_weights_, resident);
      }
    }
    kv_page_bytes_ = aligned(ceil_div(c_.kv_block * kv_bytes_token_layer_, c_.tp), 4096);
    memory_reserved_ = resident_weights_ + (c_.hbf ? c_.cache_bytes : 0) +
                       (c_.kv_offload ? c_.kv_cache_bytes : 0) + scratch_bytes_device_;
    if (memory_reserved_ > c_.hbm_capacity || (memory_reserved_ == c_.hbm_capacity && kv_hbm_layers_))
      throw std::invalid_argument("weights/cache leave no room for KV cache");
    kv_capacity_ = c_.hbm_capacity - memory_reserved_;
    if (c_.hbf && (c_.slot_bytes < max_hbf_object_bytes_ ||
                   c_.cache_bytes < c_.slot_bytes))
      throw std::invalid_argument("HBF slot cannot hold one layer's weights");
    replicas_.resize(c_.dp);
    for (auto& replica : replicas_) {
      replica.stage_busy.resize(c_.pp);
      replica.stage_waiting.resize(c_.pp);
      replica.lane_kv.resize(c_.ep);
      replica.lane_commitment.resize(c_.ep);
      replica.lane_hbf_commitment.resize(c_.ep);
      replica.stage_trace.resize(c_.pp);
    }
    if (c_.hbf || c_.kv_offload) {
      auto cfg = hbfsim::Config::for_profile(hbfsim::SimulationProfile::HbfV07);
      cfg.stacks = cfg.dies_per_stack = cfg.planes_per_die = 1;
      cfg.banks_per_die = cfg.host_channels_per_stack = cfg.hbf_channel_count = 1;
      cfg.ports_per_stack = cfg.max_active_planes_per_die = cfg.max_active_planes_per_stack = 1;
      cfg.blocks_per_plane = 4;
      const auto physical_bytes = c_.hbf_capacity ? c_.hbf_capacity : hbf_seed_bytes_ + c_.slot_bytes;
      c_.hbf_capacity = physical_bytes;
      if (physical_bytes < hbf_seed_bytes_) throw std::invalid_argument("HBF capacity cannot hold weight backing");
      cfg.pages_per_block = std::max<Time>(16, ceil_div(physical_bytes, 4 * 4096));
      cfg.initialization_mode = hbfsim::InitializationMode::Empty;
      cfg.channel_media_policy = hbfsim::ChannelMediaPolicy::Linear;
      cfg.strict_media_validation = true;
      cfg.read_cache_enabled = false;
      hbf_ = std::make_unique<hbfsim::Simulator>(cfg);
      hbf_->retain_responses(false);
      std::size_t seeded = 0;
      const auto source_bytes = hbf_seed_bytes_;
      for (Time address = 0; address < source_bytes; address += 4096) {
        auto token = hbf_->submit({hbf_->next_submission_time_ns(), hbfsim::OpType::Write, address, 4096, 0},
                                  [&seeded](const auto& result) {
          if (!result.ok()) throw std::runtime_error("HBF weight seeding failed");
          ++seeded;
        });
        if (!token) throw std::runtime_error("HBF weight seed was rejected");
        hbf_->run();  // Bound startup admission and preserve sequential program order.
      }
      hbf_->run();
      if (seeded != source_bytes / 4096)
        throw std::runtime_error("incomplete HBF weight seed");
      hbf_seed_bytes_ = source_bytes;
      epoch_ps_ = hbf_->now_ps();
      hbf_scheduler_ = std::make_unique<hbfsim::HbmSimSchedulerAdapter>(*hbf_);
      scheduler_ = hbf_scheduler_.get();
    } else scheduler_ = &events_;
    trace_.epoch = epoch_ps_;
    if (c_.kv_offload) kv_flash_ = std::make_unique<KvFlashLog>(*hbf_, trace_, hbf_seed_bytes_, c_.hbf_capacity);
    for (const auto* tier : {"card_interconnect", "nic", "intra_rack", "cross_rack"}) tiers_[tier];
    devices_.resize(c_.devices);
    dma_ready_.resize(c_.devices);
    compute_ready_.resize(c_.devices);
    device_compute_ps_.resize(c_.devices);
    expert_tokens_.resize(c_.experts);
    expert_device_tokens_.resize(c_.devices);
    for (std::size_t i = 0; i < devices_.size(); ++i) {
      auto hbm_cfg = c_.hbm_profile == "hbm2_2000" ? hbmsim::HbmConfig::hbm2_2000() :
                     c_.hbm_profile == "hbm4_8000" ? hbmsim::HbmConfig::hbm4_8000() :
                     hbmsim::HbmConfig::hbm3_6400();
      hbm_cfg.controller.read_queue_capacity = c_.hbm_queue_capacity;
      hbm_cfg.controller.write_queue_capacity = c_.hbm_queue_capacity;
      hbm_cfg.simulation.detailed_stats = false;
      if (c_.hbf) {
        hbfsim::ResidencyConfig rc;
        rc.capacity_bytes = c_.cache_bytes;
        rc.slot_bytes = c_.slot_bytes;
        auto& memory = devices_[i].residency =
          std::make_unique<hbfsim::ResidencyManager>(*hbf_, hbm_cfg, rc);
        for (auto [object, backing] : weight_backing_[i])
          memory->register_object({object, backing, aligned(object_bytes(object), 4096), std::nullopt, false});
        memory->add_capacity_listener([this, i] { retry(i); });
      } else {
        devices_[i].hbm = std::make_unique<hbmsim::HbmSystem>(hbm_cfg, *scheduler_);
        devices_[i].hbm->set_completion_callback([this, i](const auto& completion) {
          auto it = memory_callbacks_.find(completion.id);
          if (it == memory_callbacks_.end()) throw std::logic_error("unknown HBM completion");
          auto callback = std::move(it->second);
          memory_callbacks_.erase(it);
          callback();
          (void)i;
        });
        devices_[i].hbm->set_capacity_callback([this, i](const auto&) { retry(i); });
      }
    }
    if (c_.kv_offload)
      for (std::size_t device = 0; device < c_.devices; ++device)
        devices_[device].kv = std::make_unique<TieredKV>(*kv_flash_, kv_page_bytes_, c_.kv_cache_bytes,
          kv_bytes_token_layer_, c_.tp, c_.kv_block, c_.kv_policy, c_.kv_hot_blocks, device,
          [this, device](bool write, Time bytes, Time address, const std::string& kind, std::size_t origin, Done done) {
            const auto begin = now();
            submit_memory(device, write ? hbmsim::HbmOp::Write : hbmsim::HbmOp::Read, bytes, address,
              [this, device, kind, origin, begin, done] {
                done(trace_.add(kind, device, begin, now(), {origin}));
              });
          }, [this] { return now(); });
    for (std::size_t i = 0; i < c_.requests.size(); ++i) {
      c_.requests[i].replica = i % c_.dp;
      c_.requests[i].lane = (i / c_.dp) % c_.ep;
      c_.requests[i].arrival = checked_add(c_.requests[i].arrival, epoch_ps_);
      c_.requests[i].queued_ps = c_.requests[i].arrival;
      c_.requests[i].trace_node = trace_.add("arrival", -1, epoch_ps_, c_.requests[i].arrival, {0});
      scheduler_->schedule_at(c_.requests[i].arrival, [this, i] {
        const auto replica = c_.requests[i].replica;
        replicas_[replica].queue.push_back(i);
        pump(replica);
      });
    }
  }

  void run() {
    if (hbf_) hbf_->run(); else events_.run();
    for (const auto& r : c_.requests)
      if (!r.done) throw std::runtime_error("simulation stalled before all requests completed");
    for (const auto& device : devices_) {
      if (device.kv && device.kv->live_bytes()) throw std::logic_error("KV pages remain after request completion");
      const auto& stats = device.hbm ? device.hbm->stats() : device.residency->hbm_stats();
      service_hbm_read_bytes_ += stats.read_bytes;
      service_hbm_write_bytes_ += stats.write_bytes;
      if (device.residency) service_hbf_fill_bytes_ += device.residency->stats().fill_bytes;
    }
    if (c_.writeback_layer >= 0) {
      writeback_begin_ps_ = now();
      start_writeback();
      hbf_->run();
      if (!writeback_done_) throw std::runtime_error("writeback did not complete");
    }
    report();
  }

 private:
  Time now() const { return scheduler_->now(); }
  std::size_t device_id(std::size_t dp, std::size_t pp, std::size_t tp, std::size_t lane = 0) const {
    return c_.placement.at(((dp * c_.pp + pp) * c_.ep + lane) * c_.tp + tp);
  }
  bool hbf_weight(std::uint32_t object) const {
    const auto layer = object / objects_per_layer_, index = object % objects_per_layer_;
    return c_.hbf && std::find(c_.weight_hbm_layers.begin(), c_.weight_hbm_layers.end(), layer) == c_.weight_hbm_layers.end()
      && (!index || std::find(c_.weight_hbm_experts.begin(), c_.weight_hbm_experts.end(), index - 1) == c_.weight_hbm_experts.end());
  }
  bool offloaded_kv(std::uint32_t layer) const {
    return c_.kv_offload && std::find(c_.kv_hbm_layers.begin(), c_.kv_hbm_layers.end(), layer) == c_.kv_hbm_layers.end();
  }
  Time kv_bytes_for_tokens(Time tokens) const {
    return ceil_div(tokens * kv_bytes_token_layer_ * kv_hbm_layers_, c_.tp);
  }
  Time hbf_kv_budget(Time tokens) const {
    return ceil_div(tokens, c_.kv_block) * kv_page_bytes_ * kv_hbf_layers_;
  }
  void pump(std::size_t replica) {
    auto& state = replicas_[replica];
    if (state.stage_busy[0] || state.queue.empty()) return;
    auto batch = std::make_shared<Batch>();
    batch->replica = replica;
    batch->lane_tokens.resize(c_.ep);
    batch->lane_positions.resize(c_.ep);
    std::vector<std::size_t> ready_nodes;
    const auto pending = state.queue.size();
    for (std::size_t count = 0; count < pending; ++count) {
      auto id = state.queue.front(); state.queue.pop_front();
      auto& r = c_.requests[id];
      if (batch->work.size() >= c_.max_seqs || batch->tokens >= c_.max_batch_tokens) {
        state.queue.push_back(id); continue;
      }
      const auto commitment = kv_bytes_for_tokens(aligned(static_cast<Time>(r.prompt) + r.output, c_.kv_block));
      const auto hbf_commitment = hbf_kv_budget(static_cast<Time>(r.prompt) + r.output);
      if (!r.admitted && (state.lane_commitment[r.lane] + commitment > kv_capacity_ ||
          (c_.kv_offload && state.lane_hbf_commitment[r.lane] + hbf_commitment > c_.kv_logical_capacity))) {
        state.queue.push_back(id); continue;
      }
      if (!r.admitted) {
        r.admitted = true; state.lane_commitment[r.lane] += commitment;
        state.lane_hbf_commitment[r.lane] += hbf_commitment;
      }
      const auto budget = c_.max_batch_tokens - batch->tokens;
      const auto chunk = r.prefill_left ? std::min({r.prefill_left, c_.prefill_chunk, budget})
                                        : std::min<std::uint32_t>(1, budget);
      const auto consumed = r.prompt - r.prefill_left + r.generated;
      const auto new_reserved = aligned(consumed + chunk, c_.kv_block);
      const auto delta_bytes = kv_bytes_for_tokens(new_reserved) -
                               kv_bytes_for_tokens(r.kv_reserved_tokens);
      if (state.lane_kv[r.lane] + delta_bytes > kv_capacity_) {
        state.queue.push_back(id); continue;
      }
      state.lane_kv[r.lane] += delta_bytes;
      state.peak_kv = std::max(state.peak_kv, state.lane_kv[r.lane]);
      r.queue_wait_ps += now() - r.queued_ps;
      ready_nodes.push_back(trace_.add("serving_queue", -1, r.queued_ps, now(), {r.trace_node}));
      r.kv_reserved_tokens = new_reserved;
      batch->work.emplace_back(id, chunk);
      batch->tokens += chunk;
      batch->lane_tokens[r.lane] += chunk;
      batch->lane_positions[r.lane] += static_cast<Time>(chunk) * consumed +
                                     static_cast<Time>(chunk) * (chunk + 1) / 2;
      batch->attention_positions += static_cast<Time>(chunk) * consumed +
                                    static_cast<Time>(chunk) * (chunk + 1) / 2;
    }
    if (batch->work.empty()) {
      if (!state.queue.empty() && !state.inflight)
        throw std::runtime_error("KV capacity deadlock");
      return;
    }
    batch->cursor = trace_.join(ready_nodes);
    ++state.inflight;
    ++state.batches;
    run_layer(batch, 0);
  }

  void release_stage(std::size_t replica, std::size_t stage) {
    auto& state = replicas_[replica];
    if (!state.stage_busy[stage]) throw std::logic_error("releasing idle pipeline stage");
    state.stage_busy[stage] = false;
    if (!state.stage_waiting[stage].empty()) {
      auto [batch, layer] = std::move(state.stage_waiting[stage].front());
      state.stage_waiting[stage].pop_front();
      scheduler_->schedule_at(now(), [this, batch, layer] { run_layer(batch, layer); });
    }
    if (stage == 0 && c_.pp > 1)
      scheduler_->schedule_at(now(), [this, replica] { pump(replica); });
  }

  void retry(std::size_t device) {
    auto callbacks = std::move(devices_[device].waiting);
    devices_[device].waiting.clear();
    for (auto& callback : callbacks)
      scheduler_->schedule_at(now(), std::move(callback));
  }

  void submit_memory(std::size_t device, hbmsim::HbmOp operation,
                     Time bytes, Time address,
                     std::function<void()> callback) {
    auto attempt = std::make_shared<std::function<void()>>();
    std::weak_ptr<std::function<void()>> weak = attempt;
    *attempt = [this, device, operation, bytes, address,
                callback = std::move(callback), weak] {
      hbmsim::HbmTransaction tx;
      tx.id = next_memory_id_++; tx.op = operation;
      tx.address = address; tx.size_bytes = aligned(std::max<Time>(bytes, 64), 64);
      tx.arrival_time = now(); tx.client = device;
      hbmsim::SubmitResult result;
      if (devices_[device].hbm) result = devices_[device].hbm->try_submit_now(tx);
      else result = devices_[device].residency->try_submit_hbm(
        operation, tx.address, tx.size_bytes, {}, [callback](const auto&) { callback(); });
      if (result.accepted()) {
        if (devices_[device].hbm) memory_callbacks_.emplace(tx.id, callback);
        hbm_requested_bytes_ += tx.size_bytes;
      } else if (result.status == hbmsim::SubmitStatus::Backpressure) {
        ++hbm_backpressure_;
        auto retry = weak.lock();
        devices_[device].waiting.emplace_back([retry] { (*retry)(); });
      } else throw std::runtime_error("HBM admission: " + result.message);
    };
    (*attempt)();
  }

  using Done = std::function<void(std::size_t)>;
  Time object_bytes(std::uint32_t object) const {
    const bool expert = c_.experts && object % objects_per_layer_ != 0;
    return ceil_div((expert ? expert_params_ * c_.dtype : weight_bytes_layer_), c_.tp);
  }
  std::size_t base_object(std::uint32_t layer) const { return layer * objects_per_layer_; }
  void dma(std::size_t device, Time bytes, std::size_t origin, Done done) {
    const auto tile = std::min(c_.sram_bytes, c_.dma_tile_bytes);
    const auto tiles = ceil_div(bytes, tile);
    auto parent = trace_.join({origin, dma_ready_[device]});
    const auto start = std::max(now(), trace_.end(parent));
    const Time duration = to_time(tiles * static_cast<long double>(c_.dma_latency) + bytes * 1000.0L / c_.dma_GBps);
    const auto node = trace_.add("dma", device, start, checked_add(start, duration), {parent});
    dma_ready_[device] = node;
    dma_bytes_ += bytes; dma_tiles_ += tiles; dma_busy_ps_ += duration;
    sram_peak_ = std::max(sram_peak_, std::min(bytes, tile));
    scheduler_->schedule_at(checked_add(start, duration), [done, node] { done(node); });
  }
  void compute(std::size_t device, Time gemm_flops, Time attention_flops,
               bool grouped, std::size_t origin, Done done) {
    const auto gemm_slope = grouped ? c_.grouped_ps_per_flop : c_.gemm_ps_per_flop;
    const auto gemm_eff = grouped ? c_.grouped_efficiency : c_.gemm_efficiency;
    const auto gemm_launch = grouped ? c_.grouped_launch : c_.gemm_launch;
    const Time duration = to_time(gemm_launch + static_cast<long double>(attention_flops ? c_.attention_launch : 0) +
      gemm_flops * (gemm_slope > 0 ? gemm_slope : 1.0L / (c_.peak_tflops * gemm_eff)) +
              attention_flops * (c_.attention_ps_per_flop > 0 ? c_.attention_ps_per_flop :
                                1.0L / (c_.peak_tflops * c_.attention_efficiency)));
    auto parent = trace_.join({origin, compute_ready_[device]});
    const auto start = std::max(now(), trace_.end(parent));
    auto node = trace_.add(grouped ? "grouped_gemm" : "compute", device,
                           start, checked_add(start, duration), {parent});
    trace_.nodes[node].gemm_flops = gemm_flops;
    trace_.nodes[node].attention_flops = attention_flops;
    compute_ready_[device] = node;
    compute_flops_ += gemm_flops + attention_flops;
    compute_ps_sum_ += duration; device_compute_ps_[device] += duration;
    scheduler_->schedule_at(checked_add(start, duration), [done, node] { done(node); });
  }
  void read_weights(std::size_t device, std::vector<std::uint32_t> objects,
                     std::uint32_t layer, std::size_t origin,
                     std::function<void(std::size_t)> ready, Done done) {
    const auto begin = now();
    Time logical_bytes = 0;
    for (auto object : objects) logical_bytes += object_bytes(object);
    weights_hbm_read_bytes_ += logical_bytes;
    auto original_done = done;
    done = [this, begin, original_done](std::size_t node) { weights_wait_ps_ += now() - begin; original_done(node); };
    if (std::none_of(objects.begin(), objects.end(), [this](auto object) { return hbf_weight(object); })) {
      Time bytes = 0;
      for (auto object : objects) bytes += object_bytes(object);
      ready(origin);
      submit_memory(device, hbmsim::HbmOp::Read, bytes, 0x50000000ULL + layer * 1024ULL * 1024,
        [this, device, origin, begin, done] {
          done(trace_.add("hbm_weights", device, begin, now(), {origin}));
        });
      return;
    }
    // Stream objects one by one so a single cache slot cannot deadlock a group.
    auto sequence = std::make_shared<std::function<void(std::size_t)>>();
    std::weak_ptr<std::function<void(std::size_t)>> weak_sequence = sequence;
    *sequence = [this, device, objects, layer, origin, begin, ready, done,
                 weak_sequence](std::size_t index) {
      if (index == objects.size()) {
        auto node = trace_.add("hbf_weights", device, begin, now(), {origin});
        if (objects.size() > 1) ready(node);
        done(node); return;
      }
      const auto current = objects[index];
      if (!hbf_weight(current)) {
        submit_memory(device, hbmsim::HbmOp::Read, object_bytes(current), 0x50000000ULL + current * 1024ULL * 1024,
          [sequence = weak_sequence.lock(), index] { (*sequence)(index + 1); });
        return;
      }
      auto acquire = std::make_shared<std::function<void()>>();
      std::weak_ptr<std::function<void()>> weak = acquire;
      *acquire = [this, device, objects, layer, origin, begin, index, ready,
                  sequence = weak_sequence.lock(), weak] {
        const auto object = objects[index];
        auto status = devices_[device].residency->try_acquire(object,
          [this, device, objects, layer, origin, begin, index, object, ready, sequence](const auto& r) {
            if (!r.resident) throw std::runtime_error("HBF fill failed");
            if (objects.size() == 1) {
              auto node = trace_.add("hbf_fill_wait", device, begin, now(), {origin});
              ready(node);
            }
            if (c_.prefetch && layer + 1 < c_.layers && weight_backing_[device].count(object + objects_per_layer_))
              devices_[device].residency->prefetch(object + objects_per_layer_);
            auto attempt = std::make_shared<std::function<void()>>();
            std::weak_ptr<std::function<void()>> weak_read = attempt;
            *attempt = [this, device, object, index, sequence, weak_read] {
              auto result = devices_[device].residency->try_access(object, hbmsim::HbmOp::Read,
                [this, device, object, index, sequence](const auto&) {
                  devices_[device].residency->release(object);
                  (*sequence)(index + 1);
                });
              if (result.status == hbmsim::SubmitStatus::Backpressure) {
                ++hbm_backpressure_;
                devices_[device].waiting.emplace_back([retry = weak_read.lock()] { (*retry)(); });
              } else if (!result.accepted()) throw std::runtime_error("resident HBM access rejected");
            };
            (*attempt)();
          });
        if (status == hbfsim::ResidencyAdmission::Backpressure)
          devices_[device].waiting.emplace_back([retry = weak.lock()] { (*retry)(); });
      };
      (*acquire)();
    };
    (*sequence)(0);
  }
  void process_kv(std::shared_ptr<Batch> batch, std::size_t lane, std::size_t device,
                  std::uint32_t layer, std::size_t origin, Done done) {
    struct Access { TieredKV::Key key; Time before, after, positions, latest; };
    auto accesses = std::make_shared<std::vector<Access>>();
    for (auto [id, chunk] : batch->work) {
      const auto& r = c_.requests[id];
      if (r.lane != lane) continue;
      const Time consumed = r.prompt - r.prefill_left + r.generated;
      const Time end = consumed + chunk, blocks = ceil_div(end, c_.kv_block);
      for (Time block = 0; block < blocks; ++block) {
        const Time start = block * c_.kv_block;
        const Time before = consumed > start ? std::min<Time>(c_.kv_block, consumed - start) : 0;
        const Time after = std::min<Time>(c_.kv_block, end - start);
        Time positions = 0;
        for (Time query = consumed + 1; query <= end; ++query)
          if (query > start) positions += std::min<Time>(c_.kv_block, query - start);
        accesses->push_back({{id, layer, block}, before, after, positions, blocks - 1});
      }
    }
    auto sequence = std::make_shared<std::function<void(std::size_t, std::size_t)>>();
    std::weak_ptr<std::function<void(std::size_t, std::size_t)>> weak = sequence;
    *sequence = [this, accesses, device, layer, done, weak](std::size_t index, std::size_t node) {
      if (index == accesses->size()) { done(node); return; }
      const auto& a = accesses->at(index);
      if (!offloaded_kv(layer)) {
        const auto bytes = ceil_div(a.positions * kv_bytes_token_layer_, c_.tp);
        const auto delta = ceil_div((a.after - a.before) * kv_bytes_token_layer_, c_.tp);
        kv_hbm_read_logical_ += bytes; kv_hbm_append_logical_ += delta;
        const auto address = 0x60000000ULL + std::get<0>(a.key) * 1024ULL * 1024 + layer * 65536ULL + std::get<2>(a.key) * 4096;
        auto read = [this, device, bytes, address, sequence = weak.lock(), index](std::size_t ready) {
          const auto begin = now();
          submit_memory(device, hbmsim::HbmOp::Read, bytes, address,
            [this, device, sequence, index, begin, ready] {
              (*sequence)(index + 1, trace_.add("hbm_kv_read", device, begin, now(), {ready}));
            });
        };
        if (!delta) read(node);
        else {
          const auto begin = now();
          submit_memory(device, hbmsim::HbmOp::Write, delta, address,
            [this, device, begin, node, read] { read(trace_.add("hbm_kv_append", device, begin, now(), {node})); });
        }
        return;
      }
      devices_[device].kv->access(a.key, a.before, a.after, a.positions, a.latest, node,
        [sequence = weak.lock(), index](std::size_t ready) { (*sequence)(index + 1, ready); });
    };
    (*sequence)(0, origin);
  }

  // Offloaded KV uses explicit QKV production -> block append/read -> attention.
  // Each KV block is streamed through bounded HBM slots; attention compute is
  // charged after the stream (conservative non-overlapped aggregate service).
  void launch_kv_work(std::shared_ptr<Batch> batch, std::size_t lane, std::size_t device,
                      std::uint32_t layer, Time tokens, Time positions, Time gemm_flops,
                      std::size_t origin, Done done) {
    if (!tokens) { done(origin); return; }
    const auto query = static_cast<Time>(c_.heads) * c_.head_dim;
    const auto kv = static_cast<Time>(c_.kv_heads) * c_.head_dim;
    const auto qkv_flops = ceil_div(2ULL * c_.hidden * (query + 2 * kv) * tokens, c_.tp);
    const auto attention = ceil_div(4 * query * positions, c_.tp);
    const auto kv_read = ceil_div(kv_bytes_token_layer_ * positions, c_.tp);
    const auto kv_output = ceil_div(kv_bytes_token_layer_ * tokens, c_.tp);
    const auto activation = ceil_div(tokens * c_.hidden * c_.dtype, c_.tp);
    auto remaining = std::make_shared<std::size_t>(2);
    auto nodes = std::make_shared<std::vector<std::size_t>>();
    auto loaded = [this, batch, lane, device, layer, gemm_flops, qkv_flops, attention,
                   kv_read, kv_output, activation, remaining, nodes, done](std::size_t node) {
      nodes->push_back(node);
      if (--*remaining) return;
      auto project = [this, batch, lane, device, layer, gemm_flops, qkv_flops, attention,
                      kv_read, kv_output, done](std::size_t input) {
        compute(device, qkv_flops, 0, false, input,
          [this, batch, lane, device, layer, gemm_flops, qkv_flops, attention, kv_read, kv_output, done](std::size_t projected) {
            auto stream = [this, batch, lane, device, layer, gemm_flops, qkv_flops, attention, kv_read, done](std::size_t ready) {
              process_kv(batch, lane, device, layer, ready,
                [this, device, gemm_flops, qkv_flops, attention, kv_read, done](std::size_t streamed) {
                  auto attend = [this, device, gemm_flops, qkv_flops, attention, done](std::size_t input) {
                    compute(device, gemm_flops - qkv_flops, attention, false, input, done);
                  };
                  if (c_.dma_GBps > 0) dma(device, kv_read, streamed, attend); else attend(streamed);
                });
            };
            if (c_.dma_GBps > 0) dma(device, kv_output, projected, stream); else stream(projected);
          });
      };
      const auto loaded_node = trace_.join(*nodes);
      if (c_.dma_GBps > 0) dma(device, object_bytes(base_object(layer)) + activation, loaded_node, project);
      else project(loaded_node);
    };
    read_weights(device, {static_cast<std::uint32_t>(base_object(layer))}, layer, origin,
                 [](std::size_t) {}, loaded);
    const auto begin = now();
    submit_memory(device, hbmsim::HbmOp::Read, activation, 0x20000000ULL + layer * 1024ULL * 1024,
      [this, device, begin, origin, loaded] { loaded(trace_.add("hbm_activation", device, begin, now(), {origin})); });
  }

  void launch_work(std::size_t device, std::uint32_t layer, Time tokens, Time positions,
                    Time gemm_flops, bool grouped, std::vector<std::uint32_t> objects,
                    std::size_t origin, Done done) {
    if (!tokens) { done(origin); return; }
    const auto begin = now();
    const auto activation = ceil_div(tokens * c_.hidden * c_.dtype, c_.tp);
    const auto kv_read = grouped ? 0 : ceil_div(kv_bytes_token_layer_ * positions, c_.tp);
    const auto output_bytes = grouped ? activation : ceil_div(tokens * kv_bytes_token_layer_, c_.tp);
    if (!grouped) { kv_hbm_read_logical_ += kv_read; kv_hbm_append_logical_ += output_bytes; }
    const auto attention_flops = grouped ? 0 : ceil_div(4ULL * c_.heads * c_.head_dim * positions, c_.tp);
    Time weights = 0;
    for (auto object : objects) weights += object_bytes(object);
    const bool staged = c_.dma_GBps > 0;
    auto memory_nodes = std::make_shared<std::vector<std::size_t>>();
    auto compute_node = std::make_shared<std::size_t>(origin);
    auto op = std::make_shared<Op>();
    op->on_done = [this, done, memory_nodes, compute_node] {
      auto parents = *memory_nodes; parents.push_back(*compute_node);
      done(trace_.join(parents));
    };
    auto start_compute = [this, device, gemm_flops, attention_flops, grouped,
                          op, compute_node](std::size_t parent) {
      compute(device, gemm_flops, attention_flops, grouped, parent,
        [op, compute_node](std::size_t node) {
          *compute_node = node; op->compute_done = true; op->finish();
        });
    };
    auto pending = std::make_shared<std::size_t>(staged ? 3 : 4);
    auto memory_done = [this, device, layer, activation, kv_read, output_bytes, weights,
                        gemm_flops, attention_flops, grouped, staged, pending, memory_nodes,
                        op, compute_node](std::size_t node) {
      memory_nodes->push_back(node);
      if (--*pending) return;
      if (!staged) { op->memory_done = true; op->finish(); return; }
      const auto input = trace_.join(*memory_nodes);
      dma(device, weights + activation + kv_read, input,
        [this, device, layer, output_bytes, gemm_flops, attention_flops, grouped,
         op, compute_node, memory_nodes](std::size_t ready) {
          compute(device, gemm_flops, attention_flops, grouped, ready,
            [this, device, layer, output_bytes, op, compute_node, memory_nodes](std::size_t computed) {
              *compute_node = computed; op->compute_done = true;
              dma(device, output_bytes, computed,
                [this, device, layer, output_bytes, op, memory_nodes](std::size_t ready_out) {
                  const auto begin_out = now();
                  submit_memory(device, hbmsim::HbmOp::Write, output_bytes,
                    0x30000000ULL + layer * 1024ULL * 1024,
                    [this, device, ready_out, begin_out, op, memory_nodes] {
                      memory_nodes->push_back(trace_.add("hbm_output", device,
                                                        begin_out, now(), {ready_out}));
                      op->memory_done = true; op->finish();
                    });
                });
            });
        });
    };
    auto memory = [this, device, layer, begin, origin, memory_done]
                  (hbmsim::HbmOp operation, Time bytes, Time base, const std::string& name) {
      if (!bytes) { memory_done(origin); return; }
      submit_memory(device, operation, bytes, base + layer * 1024ULL * 1024,
        [this, device, begin, origin, memory_done, name] {
          memory_done(trace_.add(name, device, begin, now(), {origin}));
        });
    };
    read_weights(device, objects, layer, origin,
                  staged ? Done([](std::size_t) {}) : Done(start_compute), memory_done);
    memory(hbmsim::HbmOp::Read, kv_read, 0x10000000ULL, "hbm_kv");
    memory(hbmsim::HbmOp::Read, activation, 0x20000000ULL, "hbm_activation");
    if (!staged) memory(hbmsim::HbmOp::Write, output_bytes, 0x30000000ULL, "hbm_output");
  }

  std::size_t reserve_link(const std::vector<std::string>& resources,
                           const std::string& tier, Time bytes, double gbps, Time latency,
                           std::size_t origin, int device = -1) {
    std::vector<std::size_t> parents{origin};
    for (const auto& resource : resources) parents.push_back(link_ready_[resource]);
    const auto joined = trace_.join(parents);
    const auto start = trace_.end(joined);
    const Time duration = to_time(latency + bytes * 8000.0L / gbps);
    auto node = trace_.add(tier, device, start, checked_add(start, duration), {joined});
    for (const auto& resource : resources) link_ready_[resource] = node;
    auto& stats = tiers_[tier];
    stats.bytes += bytes; stats.busy += duration; stats.wait += start - trace_.end(origin);
    fabric_bytes_ += bytes; fabric_busy_ps_ += duration;
    fabric_queue_wait_ps_ += start - trace_.end(origin);
    return node;
  }
  std::size_t transfer(std::size_t source, std::size_t destination,
                        Time bytes, std::size_t origin) {
    if (source == destination || !bytes) return origin;
    const auto source_node = c_.device_nodes.at(source), dest_node = c_.device_nodes.at(destination);
    const auto source_rack = c_.node_racks.at(source_node), dest_rack = c_.node_racks.at(dest_node);
    auto label = [](const std::string& prefix, std::size_t id) { return prefix + std::to_string(id); };
    if (source_node == dest_node) {
      const std::vector<std::string> links = c_.fabric_topology == "nvswitch" ?
        std::vector<std::string>{label("nvout", source), label("nvin", destination)} :
        std::vector<std::string>{label("mesh", source) + ":" + std::to_string(destination)};
      return reserve_link(links, "card_interconnect", bytes, c_.nvlink_gbps, c_.nvlink_latency_ps,
                           origin, source);
    }
    auto cursor = reserve_link({label("nicout", source_node)}, "nic", bytes,
                               c_.nic_gbps, c_.nic_latency, origin, source);
    if (source_rack == dest_rack) {
      cursor = reserve_link({label("rack", source_rack)}, "intra_rack", bytes,
                             c_.rack_gbps, c_.rack_latency, cursor);
    } else {
      cursor = reserve_link({label("rackout", source_rack)}, "intra_rack", bytes,
                             c_.rack_gbps, c_.rack_latency, cursor);
      cursor = reserve_link({label("rackuplinkout", source_rack),
                             label("rackuplinkin", dest_rack)}, "cross_rack", bytes,
                             c_.cross_rack_gbps, c_.cross_rack_latency, cursor);
      cursor = reserve_link({label("rackin", dest_rack)}, "intra_rack", bytes,
                             c_.rack_gbps, c_.rack_latency, cursor);
    }
    return reserve_link({label("nicin", dest_node)}, "nic", bytes,
                         c_.nic_gbps, c_.nic_latency, cursor, destination);
  }
  using Message = std::pair<std::size_t, std::size_t>;
  void messages(const std::vector<Message>& pairs, Time bytes, std::size_t origin, Done done) {
    std::vector<std::size_t> nodes{origin};
    for (const auto [src, dst] : pairs) nodes.push_back(transfer(src, dst, bytes, origin));
    const auto node = trace_.join(nodes);
    scheduler_->schedule_at(trace_.end(node), [node, done] { done(node); });
  }
  void ring(std::vector<std::size_t> ranks, Time bytes, std::size_t origin, Done done) {
    if (ranks.size() <= 1) { done(origin); return; }
    auto step = std::make_shared<std::function<void(std::size_t, std::size_t)>>();
    std::weak_ptr<std::function<void(std::size_t, std::size_t)>> weak = step;
    *step = [this, ranks, bytes, done, weak](std::size_t round, std::size_t parent) {
      if (round == 2 * (ranks.size() - 1)) { done(parent); return; }
      std::vector<Message> pairs;
      for (std::size_t i = 0; i < ranks.size(); ++i)
        pairs.emplace_back(ranks[i], ranks[(i + 1) % ranks.size()]);
      messages(pairs, ceil_div(bytes, ranks.size()), parent,
                [next = weak.lock(), round](std::size_t node) { (*next)(round + 1, node); });
    };
    (*step)(0, origin);
  }
  void collective(std::vector<std::size_t> ranks, Time bytes, std::size_t origin, Done done) {
    if (c_.collective == "ring" || ranks.size() <= 1) { ring(ranks, bytes, origin, done); return; }
    std::map<std::uint32_t, std::vector<std::size_t>> nodes, racks;
    for (auto rank : ranks) nodes[c_.device_nodes[rank]].push_back(rank);
    std::vector<Message> local, rack;
    for (const auto& [id, group] : nodes) {
      racks[c_.node_racks[id]].push_back(group.front());
      for (std::size_t j = 1; j < group.size(); ++j) local.emplace_back(group[j], group.front());
    }
    std::vector<std::size_t> leaders;
    for (const auto& [id, group] : racks) {
      (void)id; leaders.push_back(group.front());
      for (std::size_t j = 1; j < group.size(); ++j) rack.emplace_back(group[j], group.front());
    }
    messages(local, bytes, origin, [this, rack, local, leaders, bytes, done](std::size_t n1) {
      messages(rack, bytes, n1, [this, rack, local, leaders, bytes, done](std::size_t n2) {
        ring(leaders, bytes, n2, [this, rack, local, bytes, done](std::size_t n3) mutable {
          for (auto& pair : rack) std::swap(pair.first, pair.second);
          for (auto& pair : local) std::swap(pair.first, pair.second);
          messages(rack, bytes, n3, [this, local, bytes, done](std::size_t n4) {
            messages(local, bytes, n4, done);
          });
        });
      });
    });
  }
  void allreduce_lanes(std::shared_ptr<Batch> batch, std::uint32_t layer,
                        std::vector<Time> lane_tokens, std::size_t origin, Done done) {
    auto nodes = std::make_shared<std::vector<std::size_t>>();
    auto remaining = std::make_shared<std::size_t>(c_.ep);
    for (std::size_t lane = 0; lane < c_.ep; ++lane) {
      std::vector<std::size_t> ranks;
      for (std::size_t rank = 0; rank < c_.tp; ++rank)
        ranks.push_back(device_id(batch->replica, layer / (c_.layers / c_.pp), rank, lane));
      auto completed = [this, nodes, remaining, done](std::size_t node) {
        nodes->push_back(node);
        if (--*remaining == 0) done(trace_.join(*nodes));
      };
      if (!lane_tokens[lane]) completed(origin);
      else collective(ranks, lane_tokens[lane] * c_.hidden * c_.dtype, origin, completed);
    }
  }
  static std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }
  struct Routing {
    std::vector<Time> counts, lane_tokens;
    std::vector<std::vector<Time>> messages;
  };
  Routing route(const Batch& batch, std::uint32_t layer) {
    Routing result;
    result.counts.resize(c_.experts); result.lane_tokens.resize(c_.ep);
    result.messages.assign(c_.ep, std::vector<Time>(c_.ep));
    for (auto [id, chunk] : batch.work) {
      const auto& r = c_.requests[id];
      const Time consumed = r.prompt - r.prefill_left + r.generated;
      for (Time token = consumed; token < consumed + chunk; ++token) {
        std::vector<std::pair<double, std::size_t>> scores;
        for (std::size_t e = 0; e < c_.experts; ++e) {
          auto hash = mix(c_.routing_seed ^ mix(id) ^ mix(token + 0x100000ULL) ^
                           mix(layer + 0x200000ULL) ^ mix(e + 0x300000ULL));
          const double uniform = ((hash >> 11) + 1.0) / 9007199254740993.0;
          const double score = c_.expert_weights[e] > 0 ?
            -std::log(uniform) / c_.expert_weights[e] : std::numeric_limits<double>::infinity();
          scores.emplace_back(score, e);
        }
        std::sort(scores.begin(), scores.end());
        for (std::size_t k = 0; k < c_.top_k; ++k) {
          const auto e = scores[k].second;
          const auto lane = c_.expert_placement[e];
          ++result.counts[e]; ++result.lane_tokens[lane]; ++result.messages[r.lane][lane];
          ++expert_tokens_[e]; ++routed_tokens_;
        }
        ++moe_token_visits_;
      }
    }
    return result;
  }
  void alltoall(std::shared_ptr<Batch> batch, std::uint32_t layer, const Routing& routing,
                 bool combine, std::size_t origin, Done done) {
    std::vector<std::size_t> nodes{origin};
    const auto stage = layer / (c_.layers / c_.pp);
    for (std::size_t src = 0; src < c_.ep; ++src)
      for (std::size_t dst = 0; dst < c_.ep; ++dst) {
        auto bytes = ceil_div(routing.messages[src][dst] * c_.hidden * c_.dtype, c_.tp);
        if (!bytes || src == dst) continue;
        for (std::size_t rank = 0; rank < c_.tp; ++rank) {
          auto source = device_id(batch->replica, stage, rank, combine ? dst : src);
          auto dest = device_id(batch->replica, stage, rank, combine ? src : dst);
          nodes.push_back(transfer(source, dest, bytes, origin));
          (combine ? combine_bytes_ : dispatch_bytes_) += bytes;
        }
      }
    const auto node = trace_.join(nodes);
    scheduler_->schedule_at(trace_.end(node), [done, node] { done(node); });
  }
  void moe_experts(std::shared_ptr<Batch> batch, std::uint32_t layer,
                    Routing routing, std::size_t origin, Done done) {
    alltoall(batch, layer, routing, false, origin,
      [this, batch, layer, routing, done](std::size_t dispatched) {
        auto remaining = std::make_shared<std::size_t>(c_.ep * c_.tp);
        auto nodes = std::make_shared<std::vector<std::size_t>>();
        for (std::size_t lane = 0; lane < c_.ep; ++lane) {
          std::vector<std::uint32_t> objects;
          for (std::size_t e = 0; e < c_.experts; ++e)
            if (c_.expert_placement[e] == lane && routing.counts[e])
              objects.push_back(base_object(layer) + 1 + e);
          const auto tokens = routing.lane_tokens[lane];
          for (std::size_t rank = 0; rank < c_.tp; ++rank) {
            const auto device = device_id(batch->replica, layer / (c_.layers / c_.pp), rank, lane);
            expert_device_tokens_[device] += tokens;
            launch_work(device, layer, tokens, 0, ceil_div(2 * expert_params_ * tokens, c_.tp),
                         true, objects, dispatched,
              [this, batch, layer, routing, done, remaining, nodes, dispatched](std::size_t node) {
                nodes->push_back(node);
                if (--*remaining) return;
                Time earliest = std::numeric_limits<Time>::max(), latest = 0;
                for (auto n : *nodes) {
                  // Idle expert ranks also join the barrier and expose skew.
                  earliest = std::min(earliest, trace_.end(n)); latest = std::max(latest, trace_.end(n));
                }
                (void)dispatched;
                moe_straggler_ps_ += latest - earliest;
                allreduce_lanes(batch, layer, routing.lane_tokens, trace_.join(*nodes),
                  [this, batch, layer, routing, done](std::size_t reduced) {
                    alltoall(batch, layer, routing, true, reduced, done);
                  });
              });
          }
        }
      });
  }
  void finish_layer(std::shared_ptr<Batch> batch, std::uint32_t layer, std::size_t node) {
    batch->cursor = node;
    const auto stage_size = c_.layers / c_.pp;
    const auto stage = layer / stage_size;
    if ((layer + 1) % stage_size == 0) {
      replicas_[batch->replica].stage_trace[stage] = node;
      release_stage(batch->replica, stage);
    }
    if (layer + 1 == c_.layers || (layer + 1) % stage_size) {
      run_layer(batch, layer + 1); return;
    }
    std::vector<std::size_t> nodes{node};
    for (std::size_t lane = 0; lane < c_.ep; ++lane)
      for (std::size_t rank = 0; rank < c_.tp; ++rank)
        nodes.push_back(transfer(device_id(batch->replica, stage, rank, lane),
                                 device_id(batch->replica, stage + 1, rank, lane),
                                 ceil_div(batch->lane_tokens[lane] * c_.hidden * c_.dtype, c_.tp), node));
    batch->cursor = trace_.join(nodes);
    scheduler_->schedule_at(trace_.end(batch->cursor), [this, batch, layer] { run_layer(batch, layer + 1); });
  }
  void run_layer(std::shared_ptr<Batch> batch, std::uint32_t layer) {
    if (layer == c_.layers) { finish_batch(batch); return; }
    const auto stage_size = c_.layers / c_.pp;
    const auto stage = layer / stage_size;
    if (layer % stage_size == 0) {
      auto& state = replicas_[batch->replica];
      if (state.stage_busy[stage]) { state.stage_waiting[stage].emplace_back(batch, layer); return; }
      state.stage_busy[stage] = true;
      auto wait = trace_.add("pipeline_queue", -1, trace_.end(batch->cursor), now(), {batch->cursor});
      batch->cursor = trace_.join({wait, state.stage_trace[stage]});
    }
    auto remaining = std::make_shared<std::size_t>(c_.tp * c_.ep);
    auto nodes = std::make_shared<std::vector<std::size_t>>();
    for (std::size_t lane = 0; lane < c_.ep; ++lane)
      for (std::size_t rank = 0; rank < c_.tp; ++rank) {
        const auto device = device_id(batch->replica, stage, rank, lane);
        auto launch = [this, batch, lane](auto device, auto layer, auto tokens, auto positions, auto flops,
                                         bool grouped, auto objects, auto origin, Done done) {
          if (c_.explicit_storage) launch_kv_work(batch, lane, device, layer, tokens, positions, flops, origin, done);
          else launch_work(device, layer, tokens, positions, flops, grouped, objects, origin, done);
        };
        launch(device, layer, batch->lane_tokens[lane], batch->lane_positions[lane],
                     ceil_div(2ULL * params_layer_ * batch->lane_tokens[lane], c_.tp), false,
                     std::vector<std::uint32_t>{static_cast<std::uint32_t>(base_object(layer))}, batch->cursor,
          [this, batch, layer, remaining, nodes](std::size_t node) {
            nodes->push_back(node);
            if (--*remaining) return;
            allreduce_lanes(batch, layer, batch->lane_tokens, trace_.join(*nodes),
              [this, batch, layer](std::size_t reduced) {
                auto done = [this, batch, layer](std::size_t finished) { finish_layer(batch, layer, finished); };
                if (c_.experts) moe_experts(batch, layer, route(*batch, layer), reduced, done);
                else done(reduced);
              });
          });
      }
  }

  void finish_batch(const std::shared_ptr<Batch>& batch) {
    auto& state = replicas_[batch->replica];
    for (const auto [id, chunk] : batch->work) {
      auto& r = c_.requests[id];
      if (r.prefill_left) r.prefill_left -= chunk;
      else {
        ++r.generated;
        if (!r.first_token) r.first_token = now();
        if (r.last_token) {
          r.gap_sum += now() - r.last_token; ++r.gaps;
          r.token_gaps.push_back(now() - r.last_token);
        }
        r.last_token = now();
      }
      r.trace_node = batch->cursor;
      r.queued_ps = now();
      if (r.generated == r.output) {
        r.done = now();
        if (c_.kv_offload) for (auto& device : devices_) device.kv->release_request(id);
        state.lane_hbf_commitment[r.lane] -= hbf_kv_budget(static_cast<Time>(r.prompt) + r.output);
        state.lane_kv[r.lane] -= kv_bytes_for_tokens(r.kv_reserved_tokens);
        state.lane_commitment[r.lane] -= kv_bytes_for_tokens(
            aligned(static_cast<Time>(r.prompt) + r.output, c_.kv_block));
      } else state.queue.push_back(id);
    }
    if (!state.inflight) throw std::logic_error("missing in-flight batch");
    --state.inflight;
    pump(batch->replica);
  }
  void start_writeback() {
    auto& memory = *devices_[device_id(0, c_.writeback_layer / (c_.layers / c_.pp), 0)].residency;
    const auto layer = static_cast<std::uint32_t>(base_object(c_.writeback_layer));
    const Time destination = kv_flash_ ? kv_flash_->allocate(aligned(object_bytes(layer), 4096)) : hbf_seed_bytes_;
    if (c_.hbf_capacity && destination + aligned(object_bytes(layer), 4096) > c_.hbf_capacity)
      throw std::runtime_error("HBF capacity exhausted by explicit weight writeback");
    auto admission = memory.try_acquire(layer, [this, &memory, layer, destination](const auto& ready) {
      if (!ready.resident) throw std::runtime_error("writeback source unavailable");
      auto write = memory.try_access(layer, hbmsim::HbmOp::Write,
        [this, &memory, layer, destination](const auto&) {
          memory.release(layer);
          if (!memory.writeback(layer, destination, [this](const auto& completion) {
                if (!completion.success) throw std::runtime_error("HBF writeback failed");
                writeback_done_ = true;
                writeback_bytes_ = completion.bytes_written;
              })) throw std::runtime_error("HBF writeback rejected");
        });
      if (!write.accepted()) throw std::runtime_error("HBM write before writeback rejected");
    });
    if (admission == hbfsim::ResidencyAdmission::Backpressure)
      throw std::runtime_error("writeback source admission blocked");
  }
  void report_extended() const {
    std::cout << ",\"moe\":{\"token_layer_visits\":" << moe_token_visits_
              << ",\"routed_assignments\":" << routed_tokens_
              << ",\"dispatch_payload_bytes\":" << dispatch_bytes_
              << ",\"combine_payload_bytes\":" << combine_bytes_
              << ",\"straggler_ps_sum\":" << moe_straggler_ps_
              << ",\"expert_tokens\":[";
    for (std::size_t i = 0; i < expert_tokens_.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << expert_tokens_[i];
    }
    std::cout << "],\"device_assignments\":[";
    for (std::size_t i = 0; i < expert_device_tokens_.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << expert_device_tokens_[i];
    }
    std::cout << "]},\"network_tiers\":{";
    bool first = true;
    for (const auto& [tier, stats] : tiers_) {
      if (!first) std::cout << ',';
      first = false;
      std::cout << '\"' << tier << "\":{\"bytes\":" << stats.bytes
                << ",\"busy_ps\":" << stats.busy << ",\"queue_wait_ps\":" << stats.wait << '}';
    }
    std::cout << "},\"onchip\":{\"dma_bytes\":" << dma_bytes_
              << ",\"dma_tiles\":" << dma_tiles_ << ",\"dma_busy_ps\":" << dma_busy_ps_
              << ",\"peak_sram_bytes\":" << sram_peak_ << "},\"device_compute_ps\":[";
    for (std::size_t i = 0; i < device_compute_ps_.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << device_compute_ps_[i];
    }
    std::cout << ']';
  }
  void report_storage(Time weights_fill, Time hbm_kv_peak) const {
    TieredKV::Stats total;
    Time live = 0, resident_peak = 0, logical_peak = 0;
    for (const auto& device : devices_) if (device.kv) {
      const auto& s = device.kv->stats;
      total.hits += s.hits; total.misses += s.misses; total.evictions += s.evictions;
      total.hot_protections += s.hot_protections;
      total.write_aware_protections += s.write_aware_protections;
      total.sealed_writebacks += s.sealed_writebacks; total.partial_writebacks += s.partial_writebacks;
      total.rewrite_writebacks += s.rewrite_writebacks;
      total.dirty_evictions += s.dirty_evictions; total.reloads += s.reloads;
      total.append_bytes += s.append_bytes; total.read_bytes += s.read_bytes;
      total.fill_bytes += s.fill_bytes; total.writeback_bytes += s.writeback_bytes;
      total.dirty_bytes_written += s.dirty_bytes_written; total.discarded_dirty_bytes += s.discarded_dirty_bytes;
      total.wait_ps += s.wait_ps; live += device.kv->live_bytes();
      resident_peak = std::max(resident_peak, s.resident_peak);
      logical_peak = std::max(logical_peak, s.logical_peak);
    }
    std::cout << ",\"storage_metrics\":{\"weights\":{\"hbm_resident_bytes_per_device\":" << resident_weights_
      << ",\"hbm_cache_reserved_bytes_per_device\":" << (c_.hbf ? c_.cache_bytes : 0)
      << ",\"hbf_backing_bytes\":" << hbf_seed_bytes_
      << ",\"hbm_read_logical_bytes\":" << weights_hbm_read_bytes_
      << ",\"hbf_fill_bytes\":" << weights_fill
      << ",\"access_service_ps_sum\":" << weights_wait_ps_ << "},\"kv\":{"
      << "\"hbm_read_logical_bytes\":" << kv_hbm_read_logical_ + total.read_bytes
      << ",\"hbm_append_logical_bytes\":" << kv_hbm_append_logical_ + total.append_bytes
      << ",\"offloaded_append_bytes\":" << total.append_bytes
      << ",\"hbf_fill_bytes\":" << total.fill_bytes
      << ",\"hbf_writeback_bytes\":" << total.writeback_bytes
      << ",\"dirty_logical_bytes_written\":" << total.dirty_bytes_written
      << ",\"discarded_dirty_bytes_at_completion\":" << total.discarded_dirty_bytes
      << ",\"hot_protections\":" << total.hot_protections
      << ",\"write_aware_protections\":" << total.write_aware_protections
      << ",\"sealed_block_writebacks\":" << total.sealed_writebacks
      << ",\"partial_block_writebacks\":" << total.partial_writebacks
      << ",\"rewrite_writebacks\":" << total.rewrite_writebacks
      << ",\"write_amplification\":";
    if (total.dirty_bytes_written)
      std::cout << static_cast<long double>(total.writeback_bytes) / total.dirty_bytes_written;
    else std::cout << "null";
    std::cout
      << ",\"cache_hits\":" << total.hits << ",\"cache_misses\":" << total.misses
      << ",\"evictions\":" << total.evictions << ",\"dirty_evictions\":" << total.dirty_evictions
      << ",\"reloads\":" << total.reloads << ",\"migration_wait_ps_sum\":" << total.wait_ps
      << ",\"hbm_fixed_layers_peak_bytes_per_device\":" << hbm_kv_peak
      << ",\"hbm_cache_reserved_bytes_per_device\":" << (c_.kv_offload ? c_.kv_cache_bytes : 0)
      << ",\"hbm_cache_peak_bytes_per_device\":" << resident_peak
      << ",\"logical_peak_bytes_per_device\":" << logical_peak
      << ",\"live_bytes_at_end\":" << live << ",\"page_bytes\":" << kv_page_bytes_
      << "},\"hbf_log\":{\"capacity_bytes\":" << c_.hbf_capacity
      << ",\"allocated_bytes\":" << (kv_flash_ ? kv_flash_->allocated() : hbf_seed_bytes_ + writeback_bytes_)
      << ",\"kv_read_bytes\":" << (kv_flash_ ? kv_flash_->read_bytes : 0)
      << ",\"kv_write_bytes\":" << (kv_flash_ ? kv_flash_->write_bytes : 0)
      << ",\"kv_write_dlu_ops\":" << (kv_flash_ ? kv_flash_->write_dlu_ops : 0)
      << ",\"kv_queue_wait_ps_sum\":" << (kv_flash_ ? kv_flash_->queue_wait_ps : 0)
      << ",\"reclamation\":\"append_only_no_gc\"}}";
  }
  void report() const {
    Time max_time = 0, total_ttft = 0, total_latency = 0, total_gap = 0;
    std::uint64_t total_gaps = 0, tokens = 0, batches = 0, peak_kv = 0;
    for (const auto& r : c_.requests) {
      max_time = std::max(max_time, r.done);
      total_ttft += r.first_token - r.arrival;
      total_latency += r.done - r.arrival;
      total_gap += r.gap_sum; total_gaps += r.gaps; tokens += r.output;
    }
    for (const auto& replica : replicas_) {
      batches += replica.batches;
      peak_kv = std::max(peak_kv, replica.peak_kv);
    }
    std::uint64_t hbm_read = 0, hbm_write = 0, hbf_fill = 0;
    Time hbm_bus_busy = 0, hbm_queue_wait = 0;
    std::uint64_t hbf_prefetches = 0, hbf_prefetch_hits = 0, hbf_evictions = 0;
    for (const auto& device : devices_) {
      if (device.hbm) {
        hbm_read += device.hbm->stats().read_bytes;
        hbm_write += device.hbm->stats().write_bytes;
      } else {
        hbm_read += device.residency->hbm_stats().read_bytes;
        hbm_write += device.residency->hbm_stats().write_bytes;
        hbf_fill += device.residency->stats().fill_bytes;
        hbf_prefetches += device.residency->stats().prefetches;
        hbf_prefetch_hits += device.residency->stats().prefetch_hits;
        hbf_evictions += device.residency->stats().evictions;
      }
      const auto& stats = device.hbm ? device.hbm->stats() : device.residency->hbm_stats();
      for (const auto& channel : stats.channels) {
        hbm_bus_busy += channel.data_bus_busy_time;
        hbm_queue_wait += channel.queue.queue_wait_time;
      }
    }
    const auto service_ps = max_time - epoch_ps_;
    std::cout << "{\"mode\":\"detailed\",\"requests\":" << c_.requests.size()
              << ",\"output_tokens\":" << tokens << ",\"batches\":" << batches
              << ",\"makespan_ps\":" << max_time - epoch_ps_
              << ",\"mean_ttft_ps\":" << total_ttft / c_.requests.size()
              << ",\"mean_request_latency_ps\":" << total_latency / c_.requests.size()
              << ",\"mean_tpot_ps\":" << (total_gaps ? total_gap / total_gaps : 0)
              << ",\"tokens_per_second\":" <<
                 (static_cast<long double>(tokens) * 1e12 / (max_time - epoch_ps_))
              << ",\"compute_flops\":" << compute_flops_
              << ",\"compute_service_ps_sum\":" << compute_ps_sum_
              << ",\"achieved_tflops_total\":" <<
                 (static_cast<long double>(compute_flops_) / service_ps)
              << ",\"peak_kv_bytes_per_replica\":" << peak_kv
              << ",\"peak_hbm_bytes_per_device\":" << memory_reserved_ + peak_kv
              << ",\"direct_hbm_requested_bytes\":" << hbm_requested_bytes_
              << ",\"hbm_read_bytes\":" << hbm_read
              << ",\"hbm_write_bytes\":" << hbm_write
              << ",\"service_hbm_read_bytes\":" << service_hbm_read_bytes_
              << ",\"service_hbm_write_bytes\":" << service_hbm_write_bytes_
              << ",\"service_hbm_effective_GBps\":" <<
                 (static_cast<long double>(service_hbm_read_bytes_ + service_hbm_write_bytes_)
                  * 1000 / service_ps)
              << ",\"hbm_data_bus_busy_ps_sum\":" << hbm_bus_busy
              << ",\"hbm_queue_wait_ps_sum\":" << hbm_queue_wait
              << ",\"hbm_backpressure\":" << hbm_backpressure_
              << ",\"hbf_fill_bytes\":" << hbf_fill
              << ",\"service_hbf_fill_bytes\":" << service_hbf_fill_bytes_
              << ",\"hbf_seed_bytes\":" << hbf_seed_bytes_
              << ",\"hbf_writeback_bytes\":" << writeback_bytes_
              << ",\"hbf_writeback_duration_ps\":" <<
                 (writeback_done_ ? now() - writeback_begin_ps_ : 0)
              << ",\"hbf_prefetches\":" << hbf_prefetches
              << ",\"hbf_prefetch_hits\":" << hbf_prefetch_hits
              << ",\"hbf_evictions\":" << hbf_evictions
              << ",\"fabric_bytes\":" << fabric_bytes_
              << ",\"fabric_busy_ps\":" << fabric_busy_ps_
              << ",\"fabric_queue_wait_ps_sum\":" << fabric_queue_wait_ps_
              << ",\"service_fabric_effective_GBps\":" <<
                 (static_cast<long double>(fabric_bytes_) * 1000 / service_ps)
              << ",\"fabric_topology\":\"" << c_.fabric_topology << "\""
              << ",\"hardware_kind\":\"" << c_.hardware_kind << "\""
              << ",\"requests_detail\":[";
    for (std::size_t i = 0; i < c_.requests.size(); ++i) {
      const auto& r = c_.requests[i];
      if (i) std::cout << ',';
      std::cout << "{\"ttft_ps\":" << r.first_token - r.arrival
                << ",\"latency_ps\":" << r.done - r.arrival
                << ",\"output_tokens\":" << r.output
                << ",\"queue_wait_ps\":" << r.queue_wait_ps
                << ",\"completion_node\":" << r.trace_node << ",\"token_gaps_ps\":[";
      for (std::size_t j = 0; j < r.token_gaps.size(); ++j) {
        if (j) std::cout << ',';
        std::cout << r.token_gaps[j];
      }
      std::cout << "]}";
    }
    std::cout << ']';
    report_extended();
    report_storage(hbf_fill, peak_kv);
    trace_.report();
    std::cout << "}\n";
  }

  std::unique_ptr<KvFlashLog> kv_flash_;
  std::vector<std::map<std::uint32_t, Time>> weight_backing_;
  Time max_hbf_object_bytes_ = 0;
  Time resident_weights_ = 0, kv_page_bytes_ = 0, kv_hbm_layers_ = 0, kv_hbf_layers_ = 0;
  Time weights_hbm_read_bytes_ = 0, weights_wait_ps_ = 0;
  Time kv_hbm_read_logical_ = 0, kv_hbm_append_logical_ = 0;
  Config c_;
  hbmsim::EventQueue events_;
  std::unique_ptr<hbfsim::Simulator> hbf_;
  std::unique_ptr<hbfsim::HbmSimSchedulerAdapter> hbf_scheduler_;
  hbmsim::ISimScheduler* scheduler_ = nullptr;
  std::vector<Device> devices_;
  std::vector<Replica> replicas_;
  std::unordered_map<hbmsim::TransactionId, std::function<void()>> memory_callbacks_;
  Trace trace_;
  Time attention_params_ = 0, expert_params_ = 0, objects_per_layer_ = 1;
  std::vector<std::size_t> dma_ready_, compute_ready_;
  std::vector<Time> device_compute_ps_, expert_tokens_, expert_device_tokens_;
  Time routed_tokens_ = 0, moe_token_visits_ = 0, dispatch_bytes_ = 0, combine_bytes_ = 0;
  Time dma_bytes_ = 0, dma_tiles_ = 0, dma_busy_ps_ = 0, sram_peak_ = 0;
  Time moe_straggler_ps_ = 0;
  std::map<std::string, std::size_t> link_ready_;
  struct TierStats { Time bytes = 0, busy = 0, wait = 0; };
  std::map<std::string, TierStats> tiers_;
  Time params_layer_ = 0, weight_bytes_layer_ = 0, kv_bytes_token_layer_ = 0;
  Time weight_bytes_device_ = 0, scratch_bytes_device_ = 0;
  Time memory_reserved_ = 0, kv_capacity_ = 0, next_memory_id_ = 1;
  Time compute_flops_ = 0, compute_ps_sum_ = 0;
  Time hbm_requested_bytes_ = 0, hbm_backpressure_ = 0;
  Time service_hbm_read_bytes_ = 0, service_hbm_write_bytes_ = 0;
  Time service_hbf_fill_bytes_ = 0, writeback_begin_ps_ = 0;
  Time epoch_ps_ = 0, hbf_seed_bytes_ = 0, writeback_bytes_ = 0;
  bool writeback_done_ = false;
  Time fabric_bytes_ = 0, fabric_busy_ps_ = 0, fabric_queue_wait_ps_ = 0;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: infer_sim <validated scenario.txt>");
    Engine engine(parse(argv[1]));
    engine.run();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
