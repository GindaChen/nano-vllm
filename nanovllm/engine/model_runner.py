import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        self._prev_decode_bs = 0
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # Interleaved KV layout [L, B, 2, S, H, D] for FlashInfer NHD
        # kv_cache[l] shape [B, 2, S, H, D]: kv_cache[l, :, 0] = K, kv_cache[l, :, 1] = V
        self.kv_cache = torch.empty(hf_config.num_hidden_layers, config.num_kvcache_blocks, 2, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache = self.kv_cache[layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(range(seq.num_cached_tokens, seqlen))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(range(start, end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        fi_kv_indptr = [0]
        fi_kv_indices = []
        fi_kv_last_page_len = []
        temperatures = [] if self.rank == 0 else None
        bs = self.block_size
        for seq in seqs:
            n = seq.num_tokens
            bt = seq.block_table
            input_ids.append(seq.last_token)
            positions.append(n - 1)
            slot_mapping.append(bt[-1] * bs + (n - 1) % bs)
            fi_kv_indptr.append(fi_kv_indptr[-1] + len(bt))
            fi_kv_indices.extend(bt)
            fi_kv_last_page_len.append((n - 1) % bs + 1)
            if temperatures is not None:
                temperatures.append(seq.temperature)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        fi_total_pages = fi_kv_indptr[-1]
        fi_kv_indptr = torch.tensor(fi_kv_indptr, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        fi_kv_indices = torch.tensor(fi_kv_indices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        fi_kv_last_page_len = torch.tensor(fi_kv_last_page_len, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping=slot_mapping,
                    fi_kv_indptr=fi_kv_indptr, fi_kv_indices=fi_kv_indices,
                    fi_kv_last_page_len=fi_kv_last_page_len, fi_total_pages=fi_total_pages)
        if temperatures is not None:
            temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return input_ids, positions, temperatures

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill:
            return self.model.compute_logits(self.model(input_ids, positions))
        if self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions)).argmax(-1)
        else:
            bs = input_ids.size(0)
            context = get_context()
            bucket = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[bucket]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            prev_bs = self._prev_decode_bs
            if prev_bs > bs:
                graph_vars["slot_mapping"][bs:prev_bs].fill_(-1)
            elif prev_bs == 0:
                graph_vars["slot_mapping"].fill_(-1)
            self._prev_decode_bs = bs
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            # Update FlashInfer static buffers for this bucket's wrapper and re-plan
            fi_total = context.fi_total_pages
            fi_buf = self.fi_bufs[bucket]
            fi_buf["indptr"][:bs + 1].copy_(context.fi_kv_indptr, non_blocking=True)
            fi_buf["indices"][:fi_total].copy_(context.fi_kv_indices, non_blocking=True)
            fi_buf["last_page_len"][:bs].copy_(context.fi_kv_last_page_len, non_blocking=True)
            # Extra slots beyond bs: zero pages (bucket - bs extra dummy seqs in the graph)
            if bs < bucket:
                fi_buf["indptr"][bs + 1:bucket + 1].fill_(fi_total)
                fi_buf["last_page_len"][bs:bucket].fill_(1)
            self.fi_wrappers[bucket].begin_forward(
                fi_buf["indptr"],
                fi_buf["indices"],
                fi_buf["last_page_len"],
                self._fi_num_qo_heads,
                self._fi_num_kv_heads,
                self._fi_head_dim,
                self.block_size,
                self.config.hf_config.torch_dtype,
            )
            graph.replay()
            return graph_vars["token_ids_out"][:bs]

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill)
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        else:
            input_ids, positions, temperatures = self.prepare_decode(seqs)
            token_ids_tensor = self.run_model(input_ids, positions, is_prefill)
            token_ids = token_ids_tensor.tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        import flashinfer
        config = self.config
        hf_config = config.hf_config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        max_total_pages = max_bs * max_num_blocks
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        num_qo_heads = hf_config.num_attention_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        self._fi_num_qo_heads = num_qo_heads
        self._fi_num_kv_heads = num_kv_heads
        self._fi_head_dim = head_dim

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 8))

        # Shared workspace for all FlashInfer wrappers (one active at a time)
        fi_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8)

        # Per-bucket FlashInfer wrappers + static buffers
        self.fi_wrappers = {}
        self.fi_bufs = {}
        for bs in self.graph_bs:
            fi_indptr = torch.zeros(bs + 1, dtype=torch.int32)
            fi_indices = torch.zeros(max_total_pages, dtype=torch.int32)
            fi_last_page_len = torch.ones(bs, dtype=torch.int32)
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                fi_workspace, "NHD",
                use_cuda_graph=True,
                paged_kv_indptr_buffer=fi_indptr,
                paged_kv_indices_buffer=fi_indices,
                paged_kv_last_page_len_buffer=fi_last_page_len,
            )
            self.fi_wrappers[bs] = wrapper
            self.fi_bufs[bs] = dict(indptr=fi_indptr, indices=fi_indices, last_page_len=fi_last_page_len)

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.full((max_bs,), -1, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        token_ids_out = torch.zeros(max_bs, dtype=torch.int64)
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            wrapper = self.fi_wrappers[bs]
            fi_buf = self.fi_bufs[bs]

            # Dummy FlashInfer inputs: each seq has max_num_blocks pages
            dummy_pages = bs * max_num_blocks
            fi_buf["indptr"][:] = torch.arange(0, bs + 1, dtype=torch.int32) * max_num_blocks
            fi_buf["indices"][:dummy_pages] = torch.arange(dummy_pages, dtype=torch.int32)
            fi_buf["last_page_len"][:] = self.block_size

            # Set this bucket's wrapper on all attention modules, then warmup
            for module in self.model.modules():
                if isinstance(module, Attention):
                    module.decode_wrapper = wrapper

            wrapper.begin_forward(
                fi_buf["indptr"], fi_buf["indices"][:dummy_pages], fi_buf["last_page_len"],
                num_qo_heads, num_kv_heads, head_dim, self.block_size, hf_config.torch_dtype,
            )
            set_context(False, slot_mapping=slot_mapping[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            token_ids_out[:bs] = self.model.compute_logits(outputs[:bs]).argmax(-1)

            # Re-plan before capture (same dummy inputs)
            wrapper.begin_forward(
                fi_buf["indptr"], fi_buf["indices"][:dummy_pages], fi_buf["last_page_len"],
                num_qo_heads, num_kv_heads, head_dim, self.block_size, hf_config.torch_dtype,
            )
            set_context(False, slot_mapping=slot_mapping[:bs])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
                token_ids_out[:bs] = self.model.compute_logits(outputs[:bs]).argmax(-1)
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            outputs=outputs,
            token_ids_out=token_ids_out,
        )
