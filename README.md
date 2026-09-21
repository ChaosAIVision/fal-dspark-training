# DSpark training trên một GPU 24 GB

Pipeline có thể chạy bền trong `tmux`, dành riêng một GPU vật lý cho training,
chia dữ liệu thành shard, lưu checkpoint sau từng shard và tiếp tục từ marker
nếu SSH hoặc máy cá nhân bị ngắt. Repo không chứa dataset, model, checkpoint,
token, địa chỉ máy hay log vận hành.

> Đây là recipe kỹ thuật cộng đồng dựa trên TorchSpec/DSpark. Hãy benchmark trên
> workload thật trước khi dùng production; tốc độ phụ thuộc mạnh vào acceptance.

## Luồng chạy

```text
dataset(s) -> validate + deterministic shards -> materialize target tensors
           -> train one shard -> atomic checkpoint promotion -> repeat
           -> export HF -> launch SGLang DSpark -> health + A/B benchmark
```

## 1. Chuẩn bị máy

Yêu cầu: Ubuntu, NVIDIA driver hoạt động, Python 3.12, `git`, `tmux`, khoảng
80 GiB disk trống và một GPU 24 GB trở lên.

```bash
git clone https://github.com/ChaosAIVision/fal-dspark-training.git
cd fal-dspark-training
cp .env.example .env
```

Điền `.env` bằng đường dẫn/repo của riêng bạn. `EXPECTED_*_GPU_UUID` là tùy
chọn nhưng nên đặt để script từ chối chạy nếu thứ tự GPU thay đổi:

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv
```

Sau đó bootstrap dependency đã pin:

```bash
./scripts/bootstrap.sh
```

Mặc định bootstrap tạo riêng `.venv` cho training và `.venv-sglang` cho
serving. Nếu máy chỉ dùng để train, đặt `INSTALL_SGLANG=0` để bỏ qua môi trường
serving.

## 2. Chuẩn dữ liệu

Mỗi dòng nguồn cần đúng năm trường sau. `output` phải là chuỗi JSON chứa một
array; Markdown fence hoặc thẻ `final_answer` được chuẩn hóa tự động.

```json
{"system_prompt":"...","input":"...","thinking":"...","output":"[{\"id\":1}]"}
```

Nguồn có thể là Hugging Face datasets, phân cách bằng dấu phẩy trong
`DATASET_REPOS`, hoặc JSONL cục bộ:

```bash
source .venv/bin/activate
python scripts/prepare_dataset.py --local data/example.jsonl \
  --output-dir data/shards --shard-size 128
```

Script dừng ngay khi thiếu `system_prompt`, `input`, `thinking`, hoặc khi
`output` không phải JSON array hợp lệ.

## 3. Chạy training bền trong tmux

Đầu tiên khởi tạo checkpoint DSpark theo hướng dẫn upstream cho kiến trúc đích,
đặt nó ở `outputs/single_gpu/dspark/checkpoints`, rồi chạy:

```bash
tmux new-session -d -s dspark-train \
  "cd '$PWD' && exec ./scripts/run_sharded_training.sh"
```

Theo dõi mà không attach:

```bash
./scripts/status.sh
tail -f logs/sharded-training.log
```

Nếu tiến trình bị ngắt, chạy lại đúng lệnh tmux. Các shard có marker `.done`
được bỏ qua; checkpoint chỉ được thay thế sau khi checkpoint mới vượt kiểm tra
tính toàn vẹn. Không xóa `state/` hoặc checkpoint hiện tại khi resume.

## 4. Inference

Sau export, server chỉ nhìn thấy GPU được chỉ định bởi `INFER_GPU_INDEX`:

```bash
tmux new-session -d -s dspark-infer \
  "cd '$PWD' && exec ./scripts/start_inference.sh"
curl -f http://127.0.0.1:30000/health
```

Dừng đúng server này bằng `./scripts/stop_inference.sh`; script không kill tiến
trình GPU khác.

## 5. Benchmark acceptance và throughput

Chạy baseline và DSpark server lần lượt trên hai port, rồi dùng cùng prompt,
sampling và concurrency:

```bash
python scripts/benchmark_ab.py \
  --baseline-url http://127.0.0.1:8000/v1 \
  --dspark-url http://127.0.0.1:30000/v1 \
  --input data/example.jsonl --requests 20
```

Đọc cả output tok/s, latency p50/p95 và metric accepted tokens của SGLang. DSpark
chỉ tăng tốc rõ khi draft khớp target; thêm epoch không đảm bảo acceptance tăng.

## An toàn vận hành

- Workload CUDA luôn đi qua `scripts/gpu_lock.sh`.
- Không có UUID hard-code; có thể khóa UUID trong `.env`.
- Token chỉ nằm trong `.env`, vốn bị Git ignore.
- `outputs/`, `data/`, `logs/`, `state/`, cache và upstream clone không được commit.
- `scripts/preflight.sh` kiểm tra cấu hình, disk, GPU và secret trước khi chạy.

## Nguồn upstream

Commit được pin trong `upstream.lock`. `bootstrap.sh` clone đúng commit rồi áp
dụng `patches/torchspec.patch`, giúp môi trường lần sau tái tạo được thay vì phụ
thuộc snapshot cục bộ.
