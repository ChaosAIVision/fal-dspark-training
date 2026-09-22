# Source provenance

The bootstrap script pins
[LightSeek TorchSpec](https://github.com/lightseekorg/TorchSpec) and the public
[DSpark SGLang branch](https://github.com/Dogacel/sglang/tree/dspark-support) by
full commit hash in `upstream.lock`.

The local TorchSpec patch adds bounded dataset preprocessing, offline runner
cleanup and continual checkpoint behavior required by the shard-at-a-time
workflow. It is applied after cloning and remains reviewable as a normal diff.
Two SGLang import/list expressions are reformatted to avoid a GitHub push
protection false positive; no identifier or runtime behavior is changed.

The DSpark architecture was contributed publicly through
[TorchSpec pull request 129](https://github.com/lightseekorg/TorchSpec/pull/129).
No private upstream source, training data, model weights, operational logs or
benchmark results are included in this repository.
