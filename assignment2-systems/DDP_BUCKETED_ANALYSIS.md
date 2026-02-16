# Bucketed DDP: Theoretical Analysis and Results

## Theoretical Analysis of DDP Overhead

### Assumption
We assume that the time to compute gradients for a bucket equals the time to communicate that bucket's gradients.

### DDP Overhead Model

**Variables:**
- `s` = Total size of model parameters (bytes)
- `w` = All-reduce bandwidth (bytes/second) - computed as data size / all-reduce time
- `o` = Overhead per communication call (seconds)
- `n_b` = Number of buckets

**DDP Overhead Equation:**

When computation and communication times are equal for each bucket, the DDP overhead (time spent waiting after backward completes) can be modeled as:

```
DDP_overhead = o * n_b
```

**Explanation:**

Under the assumption that computation time equals communication time for each bucket:
- Each bucket's all-reduce can fully overlap with the computation of subsequent buckets' gradients
- The only overhead is the fixed latency cost `o` per communication call
- With `n_b` buckets, we pay the latency overhead `n_b` times

More detailed model including partial overlap:

If we denote:
- `t_compute` = total time to compute all gradients
- `t_comm_per_bucket` = s / (n_b * w) = time to communicate one bucket
- Assuming perfect overlap except for the first bucket:

```
DDP_overhead = o * n_b + max(0, (s / (n_b * w)) - (t_compute / n_b))
```

For the special case where t_compute/n_b = s/(n_b * w) (computation = communication per bucket):

```
DDP_overhead = o * n_b
```

### Optimal Bucket Size

To minimize DDP overhead, we want to minimize `o * n_b` where:
- `n_b = s / bucket_size` (number of buckets given total parameter size `s`)

Substituting:

```
DDP_overhead = o * (s / bucket_size)
```

To minimize this, we want to **maximize bucket_size**.

However, there are practical constraints:
1. **Memory constraints**: Larger buckets require more temporary buffer memory
2. **Overlap constraints**: Buckets must fit within the backward computation time to maintain overlap

**Optimal bucket size equation:**

Given memory constraint `M` (maximum buffer size) and the assumption that we want full overlap:

```
optimal_bucket_size = min(M, s)
```

In practice, PyTorch DDP uses ~25MB buckets as a balance between:
- Maximizing bucket size (reducing `n_b` and thus `o * n_b`)
- Maintaining reasonable memory overhead
- Ensuring buckets complete within backward computation time for overlap

For the pure latency-limited case with no memory constraints:

```
optimal_bucket_size = s  (single bucket)
```

But this sacrifices all overlap! So the real optimization involves balancing:

**Refined optimal bucket size:**

```
optimal_bucket_size = min(M, t_compute_per_layer * w)
```

Where `t_compute_per_layer` is the time to compute gradients for one layer. This ensures each bucket can be communicated within the time it takes to compute the next bucket's gradients.

## Implementation Details

### Bucketed DDP (`cs336_systems/ddp_bucketed.py`)

**Key Design Choices:**

1. **Reverse Order Bucketing**: Parameters are bucketed in reverse order of `model.parameters()` because gradients become ready in that order during backward pass

2. **Post-Accumulate Gradient Hooks**: Uses `register_post_accumulate_grad_hook()` to detect when each parameter's gradient is ready

3. **Bucket Triggering**: When all parameters in a bucket have gradients ready, the bucket is all-reduced asynchronously

4. **Gradient Flattening**: Uses `torch._utils._flatten_dense_tensors()` to efficiently flatten bucket gradients into a single tensor

**Bucket Creation Algorithm:**
```python
buckets = []
current_bucket = []
current_size = 0
bucket_size_bytes = bucket_size_mb * 1024 * 1024

for param in reversed(list(model.parameters())):
    if not param.requires_grad:
        continue

    param_size = param.numel() * param.element_size()

    if current_size + param_size > bucket_size_bytes and current_bucket:
        buckets.append(current_bucket)
        current_bucket = []
        current_size = 0

    current_bucket.append(param)
    current_size += param_size

if current_bucket:
    buckets.append(current_bucket)
```

**Gradient Synchronization:**
```python
def _all_reduce_bucket(self, bucket_idx):
    bucket = self.buckets[bucket_idx]
    grads = [p.grad.data for p in bucket['params'] if p.grad is not None]

    # Flatten gradients
    flat_grads = torch._utils._flatten_dense_tensors(grads)

    # Async all-reduce
    work = dist.all_reduce(flat_grads, op=ReduceOp.SUM, async_op=True)

    # Store for later synchronization
    self._async_work_handles.append({
        'work': work,
        'flat_grads': flat_grads,
        'grads': grads
    })
```

## Test Results

All tests pass reliably (5/5 runs):
```bash
$ uv run pytest tests/test_ddp.py -v
============================== 6 passed in 14.13s ===============================
```

Tests verify correctness with different bucket sizes:
- ✓ 0.0016 MB (very small buckets)
- ✓ 0.0001 MB (tiny buckets)
- ✓ 0.01 MB (small buckets)
- ✓ Works with tied weights
- ✓ Produces identical results to single-process training

## Performance Expectations

### Expected Results

Based on the theoretical model, we expect:

1. **Very small buckets (1MB)**: High overhead due to many communication calls (`o * n_b` is large)

2. **Medium buckets (10-25MB)**: Optimal balance between:
   - Reduced communication calls (lower `o * n_b`)
   - Good overlap with computation
   - Reasonable memory overhead

3. **Large buckets (100MB)**: Should approach flattened DDP performance:
   - Fewer communication calls
   - But reduced overlap (buckets take longer to accumulate)

4. **Very large buckets (1000MB)**: Should match flattened DDP:
   - Essentially a single bucket
   - No overlap benefits
   - Minimal latency overhead

5. **Comparison with baselines**:
   - Should beat overlap DDP (fewer communication calls)
   - Should beat flattened DDP for small-medium buckets (maintains overlap)

### Potential Mismatches

Real results may not match expectations due to:

1. **CPU-bound computation**: On CPU with Gloo, computation may be much slower than communication, making overlap less beneficial

2. **Small model size**: If model is small, latency overhead `o` dominates, making bucket size less impactful

3. **Synchronous hook execution**: If hooks execute synchronously, we lose overlap benefits

4. **Memory allocation overhead**: Frequent flatten/unflatten operations add CPU overhead

5. **GIL contention**: Python's GIL may prevent true parallelism between compute and communication

## Experimental Setup for Better Results

To see results that better match theoretical expectations:

1. **Use GPU with NCCL**: GPU asynchrony enables better overlap
   - Kernel launches are async
   - Communication can truly overlap with computation

2. **Larger models**: Use models >100MB parameters
   - Makes communication time more significant
   - Latency overhead becomes smaller relative to transfer time

3. **High-latency networks**: Test across nodes (not localhost)
   - Makes `o` (latency overhead) more significant
   - Clearer benefit of reducing `n_b`

4. **Profile with PyTorch profiler**: Measure actual overlap
   - Identify where communication blocks computation
   - Verify async operations are truly async

## Files

- [`cs336_systems/ddp_bucketed.py`](cs336_systems/ddp_bucketed.py) - Bucketed DDP implementation
- [`benchmark_bucketed_ddp.py`](benchmark_bucketed_ddp.py) - Performance benchmark script
- [`tests/test_ddp.py`](tests/test_ddp.py) - Test suite

## Usage

```bash
# Run tests
uv run pytest tests/test_ddp.py -v

# Benchmark different bucket sizes
python benchmark_bucketed_ddp.py --backend gloo --device cpu --world-size 2 \
    --bucket-sizes 1 10 25 100 1000

# Compare with baseline implementations
python benchmark_bucketed_ddp.py --backend gloo --device cpu --world-size 2 \
    --bucket-sizes 10 25 100 --compare-baseline

# GPU benchmark (if available)
python benchmark_bucketed_ddp.py --backend nccl --device cuda --world-size 2 \
    --bucket-sizes 1 10 25 100 1000 --compare-baseline
```
