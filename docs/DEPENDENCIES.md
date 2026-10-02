# 可重现的依赖构建

本仓库的推理核心依赖同一作者的两个仿真器：

| 依赖 | 固定提交 |
| --- | --- |
| [HBMSim](https://github.com/litao1991/hbmsim) | `f147ab2bfde0671cd3d7ea4437925f5bf386b743` |
| [HBFSim](https://github.com/litao1991/hbfsim) | `28b1123ae314f50d2568387091bd20db4e630356` |

`dependencies.lock.json` 是版本入口。运行 `python3 tools/bootstrap_deps.py` 获取
两个固定版本，并将 HBFSim 补丁应用到项目的 `.deps/` 下，不依赖开发者原来的本地目录。
初始化脚本会验证补丁 SHA-256；重复执行会识别已应用的补丁。已有 checkout 的提交
不符合锁定版本时会失败，不会重置或清除用户修改。

## HBFSim 补丁

`patches/hbfsim-shared-hbm.patch` 包含必要的 33 行接口、实现及测试变更：

- 增加 `ResidencyManager::try_submit_hbm`。
- KV 和激活流量通过已有 HBM 控制器提交，保留真实争用。
- 拒绝与权重驻留槽重叠的地址范围，并检查边界溢出。
- 包含接口完成回调和槽位冲突的回归检查。

该补丁由本推理仿真器开发过程产生，尚未依赖 HBFSim 上游合入；将补丁作为源码
保存，使本仓库在全新环境中可构建。两个后端的完整 Git 仓库、构建产物和 Python
环境不提交到本仓库。

## 使用自有依赖

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DHBFSIM_SOURCE_DIR=/path/to/patched/hbfsim \
  -DHBMSIM_SOURCE_DIR=/path/to/hbmsim \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
```

提交中的 `reports/` 是历史验证快照；其中的源码指纹、依赖 dirty 状态、编译器和
本地路径对应当时的运行。应重新运行测试获得当前 checkout 的结果。应用受记录的
补丁后 HBFSim 显示 dirty 是预期状态，补丁内容和指纹可从锁文件核对。
