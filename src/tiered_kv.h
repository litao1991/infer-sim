#pragma once
#include <optional>
#include <tuple>

// One append-only HBF log shared by KV caches. Fresh destinations avoid NAND
// overwrite/auto-erase. Capacity exhaustion is explicit; no implicit free GC.
class KvFlashLog {
 public:
  using Time = std::uint64_t;
  using Done = std::function<void(std::size_t)>;
  KvFlashLog(hbfsim::Simulator& sim, Trace& trace, Time first, Time capacity)
      : sim_(sim), trace_(trace), next_(first), capacity_(capacity) {
    if (first > capacity) throw std::invalid_argument("HBF capacity cannot hold weight backing");
  }
  Time allocate(Time bytes) {
    if (bytes > capacity_ - next_) throw std::runtime_error("HBF append log capacity exhausted; increase hbf_capacity_bytes");
    const auto address = next_; next_ += bytes; return address;
  }
  Time allocated() const { return next_; }
  Time queue_wait_ps = 0, read_bytes = 0, write_bytes = 0;
  void transfer(bool write, Time address, Time bytes, int device, std::size_t origin, Done done) {
    jobs_.push_back({write, address, bytes, 0, device, origin, sim_.now_ps(), std::move(done)});
    if (jobs_.size() == 1) start();
  }
 private:
  struct Job { bool write; Time address, bytes, offset; int device; std::size_t origin; Time queued; Done done; };
  void start() {
    auto& job = jobs_.front();
    queue_wait_ps += sim_.now_ps() - job.queued;
    job.origin = trace_.add("hbf_kv_queue", job.device, job.queued, sim_.now_ps(), {job.origin});
    chunk();
  }
  void chunk() {
    auto& job = jobs_.front();
    const auto bytes = std::min<Time>(4096, job.bytes - job.offset);
    const auto begin = sim_.now_ps();
    auto accepted = sim_.submit({sim_.next_submission_time_ns(), job.write ? hbfsim::OpType::Write : hbfsim::OpType::Read,
                                job.address + job.offset, bytes, 0},
      [this, begin, bytes](const auto& response) {
        if (!response.ok() || response.bytes_completed != bytes)
          throw std::runtime_error("KV HBF transfer failed");
        auto& current = jobs_.front();
        (current.write ? write_bytes : read_bytes) += bytes;
        current.origin = trace_.add(current.write ? "hbf_kv_writeback" : "hbf_kv_fill", current.device,
                                   begin, sim_.now_ps(), {current.origin});
        current.offset += bytes;
        if (current.offset < current.bytes) { chunk(); return; }
        auto done = std::move(current.done); const auto node = current.origin;
        jobs_.pop_front();
        // Start queued transfers before completion can enqueue another job.
        if (!jobs_.empty()) start();
        done(node);
      });
    if (!accepted) throw std::runtime_error("KV HBF admission rejected");
  }
  hbfsim::Simulator& sim_; Trace& trace_;
  Time next_, capacity_;
  std::deque<Job> jobs_;
};

class TieredKV {
 public:
  using Time = std::uint64_t;
  using Done = std::function<void(std::size_t)>;
  using Memory = std::function<void(bool, Time, Time, const std::string&, std::size_t, Done)>;
  using Key = std::tuple<std::size_t, std::uint32_t, Time>; // request, layer, block
  struct Stats {
    Time hits = 0, misses = 0, evictions = 0, dirty_evictions = 0, reloads = 0;
    Time append_bytes = 0, read_bytes = 0, fill_bytes = 0, writeback_bytes = 0, dirty_bytes_written = 0;
    Time discarded_dirty_bytes = 0, logical_peak = 0, resident_peak = 0, wait_ps = 0;
    Time hot_protections = 0;
  } stats;
  TieredKV(KvFlashLog& flash, Time page, Time capacity, Time token_bytes, Time tp,
           Time block_tokens, bool hot, Time hot_blocks, int device, Memory memory,
           std::function<Time()> now)
      : flash_(flash), page_(page), token_bytes_(token_bytes), tp_(tp), block_tokens_(block_tokens),
        hot_(hot), hot_blocks_(hot_blocks), device_(device), memory_(std::move(memory)), now_(std::move(now)),
        slots_(capacity / page) {
    if (slots_.empty()) throw std::invalid_argument("KV cache needs at least one block slot");
  }
  void access(Key key, Time before, Time after, Time positions, Time latest,
              std::size_t origin, Done done) {
    if (busy_) throw std::logic_error("concurrent KV cache operation on one device");
    busy_ = true;
    latest_[{std::get<0>(key), std::get<1>(key)}] = latest;
    auto& page = pages_[key];
    if (page.valid != before || after < before || after > block_tokens_)
      throw std::logic_error("KV append/read version mismatch");
    const auto begin = now_();
    acquire(key, origin, [this, key, after, positions, begin, done](std::size_t ready) {
      auto& page = pages_.at(key);
      const auto delta = after - page.valid;
      auto read = [this, key, after, positions, done](std::size_t written) {
        auto& page = pages_.at(key);
        page.valid = after;
        const auto bytes = shard_bytes(positions);
        stats.read_bytes += bytes;
        memory_(false, bytes, address(*page.slot), "hbm_kv_read", written,
          [this, key, done](std::size_t read_node) {
            pages_.at(key).last_use = ++use_;
            busy_ = false;
            auto releases = std::move(pending_releases_);
            pending_releases_.clear();
            for (auto request : releases) release_request(request);
            done(read_node);
          });
      };
      stats.wait_ps += now_() - begin;
      if (!delta) { read(ready); return; }
      const auto bytes = shard_bytes(delta);
      stats.append_bytes += bytes;
      const auto offset = shard_bytes(page.valid);
      memory_(true, bytes, address(*page.slot) + offset, "hbm_kv_append", ready, std::move(read));
    });
    stats.logical_peak = std::max<Time>(stats.logical_peak, pages_.size() * page_);
  }
  void release_request(std::size_t request) {
    // A downstream PP stage can finish the request while an upstream cache
    // is still writing its victim on behalf of another request. Keep the page
    // alive until that operation commits its backing/version update.
    if (busy_) { pending_releases_.insert(request); return; }
    for (auto it = pages_.begin(); it != pages_.end();) {
      if (std::get<0>(it->first) != request) { ++it; continue; }
      auto& page = it->second;
      if (page.slot) slots_[*page.slot].reset();
      stats.discarded_dirty_bytes += shard_bytes(page.valid - page.backed_valid);
      it = pages_.erase(it);
    }
    for (auto it = latest_.begin(); it != latest_.end();)
      if (it->first.first == request) it = latest_.erase(it); else ++it;
  }
  Time live_bytes() const { return pages_.size() * page_; }
  Time resident_bytes() const {
    return std::count_if(slots_.begin(), slots_.end(), [](const auto& s) { return s.has_value(); }) * page_;
  }
 private:
  struct Page { Time valid = 0, backed_valid = 0, last_use = 0; std::optional<Time> backing; std::optional<std::size_t> slot; };
  Time shard_bytes(Time tokens) const { return (tokens * token_bytes_ + tp_ - 1) / tp_; }
  Time address(std::size_t slot) const { return 0x40000000ULL + slot * page_; }
  bool is_hot(const Key& key) const {
    const auto latest = latest_.at({std::get<0>(key), std::get<1>(key)});
    return hot_ && hot_blocks_ && std::get<2>(key) + hot_blocks_ > latest;
  }
  void acquire(Key key, std::size_t origin, Done done) {
    auto& page = pages_.at(key);
    if (page.slot) { ++stats.hits; done(origin); return; }
    ++stats.misses;
    for (std::size_t slot = 0; slot < slots_.size(); ++slot)
      if (!slots_[slot]) { load(key, slot, origin, std::move(done)); return; }
    std::size_t victim = 0;
    for (std::size_t slot = 1; slot < slots_.size(); ++slot) {
      const auto a = *slots_[slot], b = *slots_[victim];
      if (std::make_pair(is_hot(a), pages_.at(a).last_use) <
          std::make_pair(is_hot(b), pages_.at(b).last_use)) victim = slot;
    }
    const auto old = *slots_[victim];
    if (hot_ && !is_hot(old))
      for (const auto& candidate : slots_)
        if (is_hot(*candidate) && pages_.at(*candidate).last_use < pages_.at(old).last_use) {
          ++stats.hot_protections; break;
        }
    auto evicted = [this, key, old, victim, done](std::size_t ready) {
      ++stats.evictions;
      pages_.at(old).slot.reset(); slots_[victim].reset();
      load(key, victim, ready, done);
    };
    auto& previous = pages_.at(old);
    if (previous.valid == previous.backed_valid) { evicted(origin); return; }
    ++stats.dirty_evictions;
    memory_(false, page_, address(victim), "hbm_kv_evict", origin,
      [this, old, evicted](std::size_t read) {
        const auto destination = flash_.allocate(page_);
        flash_.transfer(true, destination, page_, device_, read,
          [this, old, destination, evicted](std::size_t written) {
            auto& previous = pages_.at(old);
            stats.dirty_bytes_written += shard_bytes(previous.valid - previous.backed_valid);
            previous.backing = destination; previous.backed_valid = previous.valid;
            stats.writeback_bytes += page_;
            evicted(written);
          });
      });
  }
  void load(Key key, std::size_t slot, std::size_t origin, Done done) {
    auto& page = pages_.at(key);
    page.slot = slot; slots_[slot] = key;
    stats.resident_peak = std::max(stats.resident_peak, resident_bytes());
    if (!page.backing) {
      if (page.valid) throw std::logic_error("KV lost its backing copy");
      done(origin); return;
    }
    if (page.valid != page.backed_valid) throw std::logic_error("dirty KV evicted before writeback");
    ++stats.reloads;
    flash_.transfer(false, *page.backing, page_, device_, origin,
      [this, slot, done](std::size_t read) {
        stats.fill_bytes += page_;
        memory_(true, page_, address(slot), "hbm_kv_fill", read, done);
      });
  }
  KvFlashLog& flash_;
  Time page_, token_bytes_, tp_, block_tokens_; bool hot_; Time hot_blocks_; int device_;
  Memory memory_; std::function<Time()> now_;
  std::vector<std::optional<Key>> slots_;
  std::map<Key, Page> pages_;
  std::map<std::pair<std::size_t, std::uint32_t>, Time> latest_;
  std::set<std::size_t> pending_releases_;
  Time use_ = 0; bool busy_ = false;
};
