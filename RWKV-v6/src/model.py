### ---
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
### ---

global RWKV_JIT_ON, RWKV_TORCH_COMPILE, RWKV_NO_CUDA

from .module.CoreDependencies import *
from .module.ChannelMix import RWKV_ChannelMix6_0
from .module.TimeMix import RWKV_TimeMix6_0

from . import metrics
# ---
# Isolating out known operations that **does not work** with torch.compile
# and wrapping them within a torch._dynamo.disable, this is required to get
# the baseline torc.compile to work
# ---

# In the latest version of deepspeed + torch compile,
# deepspeed.checkpointing now works ? - this is inconsistent, so i am disabling for now
@TCompileDisable
def deepspeed_checkpoint(*args, **kwargs):
    return deepspeed.checkpointing.checkpoint(*args, **kwargs)

### ---
# RWKV: State Blocks
### ---

class BlockState:

    def __init__(self, time_mix_state: tuple[torch.Tensor,torch.Tensor],
                 channel_mix_state: torch.Tensor):
        self.time_mix_state = time_mix_state
        self.channel_mix_state = channel_mix_state


class BlockStateList:

    def __init__(self, shift_states, wkv_states):
        self.wkv_states = wkv_states
        self.shift_states = shift_states

    # @ TCompileMax (no difference)
    @staticmethod
    def create(N, B, C, n_head, head_size, device, dtype):
        result = BlockStateList.empty(N, B, C, n_head, head_size, device, dtype)
        result.wkv_states[:] = 0
        # result.wkv_states[:, :, :, -1] = -1e38
        result.shift_states[:] = 0
        return result

    # @ TCompileMax (no difference)
    @staticmethod
    def empty(N, B, C, n_head, head_size, device, dtype):
        # @TODO: confirm if dtype can be changed from .flaot to dtype=dtype (when bf16)
        wkv_states = torch.empty((N, B, n_head, head_size, head_size),
        # wkv_states = torch.empty((N, B, 1, n_head, head_size, head_size),
                                 device=device,
                                #  dtype=dtype)
                                 dtype=torch.float)
        shift_states = torch.empty((N, 2, B, C), device=device, dtype=dtype)
        return BlockStateList(shift_states, wkv_states)

    def __getitem__(self, layer: int):
        return BlockState(
            (self.shift_states[layer, 0], self.wkv_states[layer]),
            (self.shift_states[layer, 1]))

    def __setitem__(self, layer: int, state: BlockState):
        self.shift_states[layer, 0] = state.time_mix_state[0]
        self.wkv_states[layer] = state.time_mix_state[1]
        self.shift_states[layer, 1] = state.channel_mix_state

### ---
# The RWKV Model blocks
### ---

class Block(JITModClass):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dropout, dim_att, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
        else:
            self.ln0 = nn.Identity()

        self.att = RWKV_TimeMix6_0(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.ffn = RWKV_ChannelMix6_0(layer_id, n_layer, n_embd, dim_ffn)

        # Setup droupout at block level
        self.dropout = dropout
        if dropout > 0:            
            self.drop0 = nn.Dropout(p = dropout)
            self.drop1 = nn.Dropout(p = dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    @JITModMethod
    @TCompileBaseline
    def forward(self, x, last_state: BlockState):
        x = self.ln0(x)

        att_out, att_state = self.att(
            self.ln1(x),
            last_state.time_mix_state,
        )
        x = self.drop0(x + att_out)
        
        ffn_out, ffn_state = self.ffn(
            self.ln2(x),
            last_state.channel_mix_state,
        )
        x = self.drop1(x + ffn_out)
        
        return x, BlockState(att_state, ffn_state)


class L2Wrap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, loss, y, factor, currentMask):
        # Currently (8th July 2023), save_for_backward, causes an issue with
        # pytorch.compile (see: https://github.com/pytorch/pytorch/blob/e600505e3209eaf539e8bc99870ea55236cefbf5/torch/_dynamo/variables/higher_order_ops.py#L735)
        # 
        # Due to L2Wrap being a major hotspot, we should monitor this for future support.
        # so that once its resolved, we can include the L2Wrap step in the torch.compile path
        #
        # See also:
        # - checkpointed_step
        ctx.save_for_backward(y, factor, currentMask)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y, factor, currentMask = ctx.saved_tensors

        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)

        # We ensure the mask is reshaped accordingly, and apply it against gy
        gy = gy * currentMask.reshape(gy.shape[0],gy.shape[1],1) # currentMask[:, None][None, :]
        return (grad_output, gy, None, None)

### ---
# Static optimized functions
### ---

# @ TCompileMax (no speed improvement)
# def F_cross_entropy_reduction_none_optimized(logits, targets):
#     return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")

### ---
# Core RWKV module
### ---
class RWKV(L.LightningModule):

    def __init__(self,
                 # Model file path to load from
                 load_model: str,
                 # Model size settings, which we either
                 # "auto detect", or use the user specified settings
                 n_embd: int = -1,
                 n_layer: int = -1,
                 vocab_size: int = -1,
                 # Context length size for the model
                 ctx_len: int = 2048,
                 # Context length schedule
                 ctx_len_cutoffs: List[int] = [],
                 ctx_len_warmup_steps: List[int] = [],
                 # Learning rate schedule
                 # use only target_lr_init / lr_init
                 # to configure a constant learning rate
                 lr_init: float = -1.0,
                 lr_final: float = -1.0,
                 lr_period: int = -1,
                 lr_period_type: str = 'epoch',
                 # Use either "cosine" or "linear"
                 lr_type: str = 'cosine',

                 # Dropout rate
                 dropout: float = 0.0,
                 # Adam optimizer settings
                 beta1: float = 0.9,
                 beta2: float = 0.99,
                 adam_eps: float = 1.0e-08,
                 weight_decay: float = 0.001,
                 warmup_steps: int = -1,

                 # loss bias start
                 position_loss_bias: float = 1.0,
                 position_loss_bias_in_validation: bool = False,
                 
                 # Selective loss settings
                 token_loss_threshold: float = 0.0,
                 token_dropout_rate: float = 0.0, # Dropout rate should be between 0-1

                 # Backprop settings
                 grad_cp: bool = True,
                 bptt_learning: bool = True,
                 bptt_learning_range: int = -1,
                 bptt_truncated_learning: bool = True,
                 layerwise_lr: bool = True,
                 dim_att: Optional[int] = None,
                 dim_ffn: Optional[int] = None,
                 substep_cuda_cache_clear: bool = False,
                 substep_logging: bool = False,
                 torch_set_float32_matmul_precision:str = 'high',

                 # Consolidated loss, and total token count tracking
                 # this introduces a slight perf penalty if enabled
                 consolidated_logging: bool = True
                 ):

        # Lets save everything in one shot
        # (this is used for wandb logging)
        self.setup_args = locals()
        del self.setup_args["self"]
        del self.setup_args["__class__"]

        # Setup the parent class
        super().__init__()

        # Load the model, unless its the special ".//<#|=@%!$init_model$!%@=|#>//." path
        # which is reserved to be used with the `init_model.py`
        #
        # We intentionally used several filesystem illegal characters, to ensure it
        # is not accidentally used by the user for a real file
        model_weights = None
        model_keys = None
        if load_model != ".//<#|=@%!$init_model$!%@=|#>//.":
            # Check if the load_model path exists, and is a file
            if not os.path.isfile(load_model):
                raise ValueError(f"load_model file '{load_model}' does not exist")

            # Load the model weights
            if IS_TORCH_2_1_COMPATIBLE:
                model_weights = torch.load(load_model, map_location='cpu', weights_only=True, mmap=True)
            else:
                model_weights = torch.load(load_model, map_location='cpu')

            # Get the model keys
            model_keys = list(model_weights.keys())

        # Lets compute the model various sizes, if they are not provided
        if n_layer < 0:
            max_block_id = 0
            for x in model_keys:
                if 'blocks.' in x:
                    block_id = int(x.split('.')[1])
                    max_block_id = max(max_block_id, block_id)
            n_layer = max_block_id + 1

        if n_embd < 0:
            n_embd = model_weights['head.weight'].shape[1]
        
        if vocab_size < 0:
            vocab_size = model_weights['head.weight'].shape[0]

        # Save the various other params for later
        self.ctx_len = ctx_len
        self.ctx_len_cutoffs = ctx_len_cutoffs
        self.ctx_len_warmup_steps = ctx_len_warmup_steps
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.layerwise_lr = layerwise_lr
        self.grad_cp = grad_cp
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_period = lr_period
        self.lr_period_type = lr_period_type
        self.lr_type = lr_type
        self.dropout = dropout
        self.warmup_steps = warmup_steps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.bptt_learning = bptt_learning
        self.bptt_learning_range = bptt_learning_range
        self.bptt_truncated_learning = bptt_truncated_learning
        self.substep_cuda_cache_clear = substep_cuda_cache_clear
        self.substep_logging = substep_logging
        self.consolidated_logging = consolidated_logging

        # # Add warning that bptt_truncated_learning is forced to be true
        # # due to incomplete implementation of CUDA kernel for bptt_learning
        # #
        # # @TODO : remove this warning once the CUDA kernel, with state gradient, is implemented
        # if self.bptt_truncated_learning == False:
        #     print("====================================================================")
        #     print("[WARNING]: bptt_truncated_learning is set as true (was configured as false), due to incomplete implementation of CUDA kernel for bptt_learning")
        #     print("====================================================================")
        #     self.bptt_truncated_learning = True

        # Save the position loss params, and selective loss settings
        self.position_loss_bias = position_loss_bias
        self.position_loss_bias_in_validation = position_loss_bias_in_validation
        self.token_loss_threshold = token_loss_threshold
        self.token_dropout_rate = token_dropout_rate

        # Previous step timestamp, for tracking step by step tok/s perf
        self._prev_step_endin_timestamp = 0

        dim_att = dim_att or n_embd
        dim_ffn = dim_ffn or int((n_embd * 3.5) // 32 * 32)
        self.dim_att = dim_att
        self.dim_ffn = dim_ffn

        # Compute the RWKV-v5 n_head / headsize
        head_size = 64
        self.head_size = head_size
        self.head_size_divisor = 8

        n_head = dim_att // head_size
        self.n_head = n_head
        assert dim_att % n_head == 0 ,  f"dim_att must be divisible by head_size ({self.head_size})"

        # Validate various sizes
        assert n_embd  % 32 == 0, f"n_embd must be divisible by 32"
        assert dim_att % 32 == 0, f"dim_att must be divisible by 32"
        assert dim_ffn % 32 == 0, f"dim_ffn must be divisible by 32"
        
        # Matmu precision check
        if torch_set_float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(torch_set_float32_matmul_precision)
        self.emb = nn.Embedding(vocab_size, n_embd)

        # load(name=f"wkv_{self.ctx_len}_bf16",
        #      sources=[
        #         os.path.join(CUDA_DIR, "wkv_op_bf16.cpp"),
        #         os.path.join(CUDA_DIR, "wkv_cuda_bf16.cu")
        #     ],
        #      verbose=True,
        #      extra_cflags=["-std=c++17", "-O3", f"-DTmax={self.ctx_len}"],
        #      extra_cuda_cflags=[
        #          "-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60",
        #          "--use_fast_math", "-O3", "-Xptxas -O3",
        #          "--extra-device-vectorization", f"-DTmax={self.ctx_len}"
        #      ],
        #      is_python_module=False)

        self.blocks = nn.ModuleList([
            Block(i, n_layer, n_embd, n_head, head_size, dropout, dim_att, dim_ffn) for i in range(n_layer)
        ])

        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Dropout handling
        if dropout > 0:
            self.drop0 = nn.Dropout(p = dropout)

        # load the state, and GC the original cpu copy
        if model_weights != None:
            self.load_state_dict(model_weights)
            del model_weights
            gc.collect()

        # Training based timings to track, and initialize
        self._counting_tokens = 0.0
        self._counting_time_start = 0

        self.metrics = dict(loss=metrics.Loss())


    def configure_optimizers(self):
        if self.bptt_learning == False:
            if self.deepspeed_stage >= 2 or self.deepspeed_offload:
                print(f"[WARNING]: it is highly recommended to enable bptt_learning when used to deepspeed 2/3/offloading, otherwise an exception will occur when training with dataset records, larger then the configured context length ({self.ctx_len})")
        else:
            if self.trainer.num_devices > 1:
                if self.bptt_learning_range <= 0:
                    print("[WARNING]: unlimited bptt_learning_range across multiple GPU's has a performance penalty with datasets of mixed sizes due to its constant need to keep all GPU's in sync (consider using bptt_learning_range=1 instead)")
        
        # Get the learning rate used for the optimizer
        lr_init = self.lr_init
        lr_final = self.lr_final
        # If the final learning rate is not specified, use the initial learning rate
        if lr_final < 0:
            lr_final = self.lr_init

        # Log the learning rate, and various other parameters
        if self.trainer.local_rank == 0:

            # Add the important notes, for informing users of common gotchas
            print((
                "#\n"
                "# RWKV lighting_trainer.py important notes \n"
                "# https://github.com/RWKV/RWKV-infctx-trainer \n"
                "#\n"
                "# - Ensure your host is not running cuda 12.0 (use either 11.8, or >=12.1), as this is known to have freeze issues\n"
                "# - The terms used in wandb / the progress bar can be confusing, see the github README.md for beter clarifications\n"
                "# - When resuming from checkpoint, the estimated time is inaccurate\n"
                "#"
            ))

            lr_init_e = "{:.3e}".format(lr_init)
            lr_final_e = "{:.3e}".format(lr_final)
            print(f"\n[RWKV.model] Configuring optimizer with\n"+
                  f"    - lr_init:  {lr_init_e} ({lr_init})\n"+
                  f"    - lr_final: {lr_final_e} ({lr_final})\n")

            # Get the setup args
            model_args = dict(self.setup_args)
            model_args["__lr_init"] = lr_init
            model_args["__lr_final"] = lr_final

            # Update WANDB
            if wandb.run is not None:
                wandb.config.update({ "model": model_args })

        # Setup layerwise learning rate
        if self.layerwise_lr:
            lr_decay = set()
            lr_1x = set()
            lr_2x = set()
            lr_3x = set()
            for n, p in self.named_parameters():
                if ("_w1" in n) or ("_w2" in n):
                    lr_1x.add(n)
                elif ("time_mix" in n) or ("time_maa" in n):
                    lr_1x.add(n)
                elif ("time_decay" in n) or ("time_daaaa" in n):
                    lr_2x.add(n)
                # V5-R2 changes
                elif "time_faaaa" in n:
                    lr_1x.add(n)
                elif "time_first" in n:
                    lr_3x.add(n)
                elif (len(p.squeeze().shape) >= 2) and (self.weight_decay > 0):
                    lr_decay.add(n)
                else:
                    lr_1x.add(n)
            lr_decay = sorted(list(lr_decay))
            lr_1x = sorted(list(lr_1x))
            lr_2x = sorted(list(lr_2x))
            lr_3x = sorted(list(lr_3x))
            # print('1x', lr_1x)
            # print('2x', lr_2x)
            # print('3x', lr_3x)
            param_dict = {n: p for n, p in self.named_parameters()}
            optim_groups = [
                {
                    "params": [param_dict[n] for n in lr_1x],
                    "weight_decay": 0.0,
                    "lr": 1.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_2x],
                    "weight_decay": 0.0,
                    "lr": 2.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_3x],
                    "weight_decay": 0.0,
                    "lr": 3.0 * lr_init
                },
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": self.weight_decay,
                    "lr": 1.0 * lr_init
                },
            ]
        else:
            optim_groups = [
                {
                    "params": [p for n, p in self.named_parameters()],
                    "weight_decay": self.weight_decay
                },
            ]

        # Setup the adam optimizers
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups,
                                         lr=lr_init,
                                         betas=(self.beta1, self.beta2),
                                         eps=self.adam_eps,
                                         bias_correction=True,
                                         adamw_mode= (self.weight_decay > 0),
                                         amsgrad=False)
        else:
            optimizer = FusedAdam(optim_groups,
                                  lr=lr_init,
                                  betas=(self.beta1, self.beta2),
                                  eps=self.adam_eps,
                                  bias_correction=True,
                                  adam_w_mode= (self.weight_decay > 0),
                                  amsgrad=False)
            
        # Throw if wramup_steps and lr_period are both set (not supported)
        if self.warmup_steps > 0 and self.lr_period > 0:
            raise ValueError(
                "Use either warmup_steps or lr_period, not both.")

        if self.warmup_steps > 0:
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor = 0.2,
                end_factor = 1.0,
                total_iters = self.warmup_steps
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        else:
            # Skip the lr_scheduler process if lr_init and lr_final are the same
            if lr_init == lr_final:
                return optimizer

            # The total number of steps to perform training rate decay with
            lr_total_step = 0

            # Handle lr_period -1 default behaviour of using the max_step / max_epoch
            if self.lr_period == -1:
                # Get trainer max_step / max_epoch
                trainer_max_step = self.trainer.max_steps
                trainer_max_epoch = self.trainer.max_epochs
                if trainer_max_step > 0:
                    lr_total_step = trainer_max_step
                elif trainer_max_epoch > 0:
                    lr_total_step = trainer_max_epoch * self.num_step_per_epoch()
                else :
                    print("Warning: max_step/max_epoch not set, we would be performing lr_init to lr_final shift assuming 10 epoch")
                    lr_total_step = 10 * self.num_step_per_epoch()
            else:
                # Calculate lr_total_step based on lr_period
                if self.lr_period_type == "step":
                    lr_total_step = self.lr_period
                elif self.lr_period_type == "epoch":
                    lr_total_step = self.lr_period * self.num_step_per_epoch() # * self.trainer.microbatch_size
                else:
                    raise ValueError(f"lr_period_type {self.lr_period_type} not supported.")

            # Lets initialize the lr_scheduler
            if self.lr_type == "cosine":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=lr_total_step,
                    eta_min=lr_final
                )
            elif self.lr_type == "linear":
                lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor= lr_final / lr_init,
                    total_iters=lr_total_step
                )
            else:  
                raise ValueError(f"lr_type {self.lr_type} not supported.")

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
                
    
    # We have to compute the number of steps per epoch ourselves
    # as this value is not provided directly by pytorch lightning
    # https://github.com/Lightning-AI/lightning/issues/5449#issuecomment-1501597319
    def num_step_per_epoch(self) -> int:
        # Estimated number of steps in total, added as the following
        # https://github.com/Lightning-AI/lightning/pull/11599
        #
        # This MUST be called before len(self.trainer.train_loader)
        # otherwise there is a bug in which the train_dataloader is not
        # fully initialized, which seems to be resolved by computing the 
        # self.trainer.estimated_stepping_batches
        estimated_stepping_batches = self.trainer.estimated_stepping_batches

        # Get the train_dataloader
        train_dataloader = self.trainer.train_dataloader
        if train_dataloader is None:
            train_dataloader = self.trainer.fit_loop._data_source.dataloader()

        # Update the dataloader - to include a reference to the model "self"
        #
        # This is an extreamly hacky work around, to ensure we can get the completed step
        # from the dataloader iteration process - to ensure we properly offset the data
        # on a checkpoint resumption
        #
        # Basically workaround hack for: 
        # https://discuss.pytorch.org/t/resume-iterating-dataloader-from-checkpoint-batch-idx/60683/14 
        #
        # See: data.py -> CheckPointResumeSafeDataLoader
        train_dataloader._set_model_self(self)
        
        # Get the number of epochs, 
        # use estimated_stepping_batches if max_epochs is set
        max_epochs = self.trainer.max_epochs
        if max_epochs > 0:
            return estimated_stepping_batches // max_epochs

        # Max epoch is not set, use the train_dataloader
        dataset_size = len(train_dataloader)

        num_devices = max(1, self.trainer.num_devices)
        num_nodes = max(1, self.trainer.num_nodes)
        num_steps = dataset_size // (self.trainer.microbatch_size * num_devices * num_nodes)
        return num_steps
    
    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "offload_optimizer" in cfg or "offload_parameters" in cfg
        return False
    
    @property
    def deepspeed_stage(self) -> int:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return "stage" in cfg
        return -1

    # @TCompileBaseline
    def forward(self, idx: torch.Tensor, last_shift_states: torch.Tensor = None,
                last_wkv_states: torch.Tensor = None):
        B, T = idx.size()
        assert T <= self.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        # Handle dropout (input)
        if self.dropout > 0.0:
            x = self.drop0(x)

        new_states = BlockStateList.empty(self.n_layer, B, self.n_embd, 
                                          self.n_head, self.head_size,
                                          x.device, x.dtype)
        
        # last_shift_states can be None, when we are performing direct inference
        if last_shift_states is None:
            cur_bs_list = BlockStateList.create(
                self.n_layer, B, self.n_embd, 
                self.n_head, self.head_size,
                x.device, x.dtype
            )
        else:
            cur_bs_list = BlockStateList(last_shift_states, last_wkv_states)

        ## The output X token
        output_x = x

        ########
        ### Non forking block loop
        #######

        # Avoid using the zip operation, as torch.compile throws an exception on it
        # with `zip not reconized as a valid function`
        # ---
        # for i, (block, last_state) in enumerate(
        #         zip(self.blocks,
        #             BlockStateList(last_shift_states, last_wkv_states))):
        # ---
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            last_state = cur_bs_list[i]
            if self.grad_cp:
                output_x, new_state = deepspeed_checkpoint(
                    block, output_x, last_state)
            else:
                output_x, new_state = block(output_x, last_state)
            new_states[i] = new_state

        ########
        ### Forking block loop (its slower sadly)
        #######

        # # Configuring the chunk sizes
        # first_round_chunk_size = 256
        
        # # Next round chunk sizes forumlation
        # def nextRoundChunkSize(t):
        #     return first_round_chunk_size

        # # First round, first block
        # def firstRound_firstBlock_subProcess(
        #         block:Block, last_state:BlockState, 
        #         in_x:torch.tensor, grad_cp):
        #     if grad_cp:
        #         out_x, new_state = deepspeed_checkpoint(
        #             block, in_x, last_state)
        #     else:
        #         out_x, new_state = block(in_x, last_state)
        #     return out_x, new_state
            
        # # First round, next block
        # def firstRound_nextBlock_subProcess(
        #         block:Block, last_state:BlockState, 
        #         in_x_promise: torch.jit.Future[torch.Tensor], 
        #         grad_cp):
        #     in_x, prv_layer_state = torch.jit.wait(in_x_promise)
        #     return firstRound_firstBlock_subProcess(block, last_state, in_x, grad_cp)
        
        # # Next round, sub process
        # def nextRound_firstBlock_subProcess(
        #     block:Block, last_state_promise: torch.jit.Future[BlockState],
        #     in_x:torch.Tensor, grad_cp):
        #     last_x, last_state = torch.jit.wait(last_state_promise)
        #     return firstRound_firstBlock_subProcess(block, last_state, in_x, grad_cp)
    
        # # Next round, next block
        # def nextRound_nextBlock_subProcess(
        #     block:Block, last_state_promise: torch.jit.Future[BlockState],
        #     in_x_promise: torch.jit.Future[torch.Tensor], 
        #     grad_cp):
        #     last_x, last_state = torch.jit.wait(last_state_promise)
        #     in_x, prv_layer_state = torch.jit.wait(in_x_promise)
        #     return firstRound_firstBlock_subProcess(block, last_state, in_x, grad_cp)

        # # Final x value futures
        # output_x_futures = []
        
        # # Highly experimental first round token pass with JIT fork
        # first_round_futures = []
        # for i in range(len(self.blocks)):
        #     if i == 0:
        #         future = torch.jit.fork(
        #             firstRound_firstBlock_subProcess, self.blocks[i], 
        #             cur_bs_list[i], x[:,:first_round_chunk_size], self.grad_cp
        #         )
        #     else:
        #         future = torch.jit.fork(
        #             firstRound_nextBlock_subProcess, self.blocks[i], 
        #             cur_bs_list[i], first_round_futures[i-1], self.grad_cp
        #         )
        #     first_round_futures.append(future)
        # output_x_futures.append(first_round_futures[-1])

        # # Lets start doing the next round iterations
        # next_round_futures = first_round_futures

        # # Lets start the next round iterations
        # idx = first_round_chunk_size
        # while idx < T:
        #     increment = nextRoundChunkSize(idx)
        #     for i in range(len(self.blocks)):
        #         if i == 0:
        #             future = torch.jit.fork(
        #                 nextRound_firstBlock_subProcess, self.blocks[i], 
        #                 next_round_futures[i], x[:,idx:idx+increment], self.grad_cp
        #             )
        #         else:
        #             future = torch.jit.fork(
        #                 nextRound_nextBlock_subProcess, self.blocks[i], 
        #                 next_round_futures[i], next_round_futures[i-1], self.grad_cp
        #             )
        #         next_round_futures[i] = future
        #     output_x_futures.append(next_round_futures[-1])
        #     idx += increment

        # # Lets get the new states from the final round futures
        # for i in range(len(self.blocks)):
        #     tmp_x, new_state = torch.jit.wait(next_round_futures[i])
        #     new_states[i] = new_state
        
        # # Lets process the final output_x_futures
        # output_x, tmp_state = torch.jit.wait(output_x_futures[0])
        # for i in range(1, len(output_x_futures)):
        #     tmp_x, tmp_state = torch.jit.wait(output_x_futures[i])
        #     output_x = torch.cat((output_x, tmp_x), dim=1)
        # output_x = output_x[:, :T]

        # Final layernorm and head output
        output_x = self.ln_out(output_x)
        output_x = self.head(output_x)

        return output_x, new_states.shift_states, new_states.wkv_states

    #
    # Custom overwrite of manual_backwards operation, to skip the "manual_backwards"
    # safety check, so we can perform manual backward operation step, while using
    # the default trainer loop. This is modified from the original code found here:
    # https://github.com/Lightning-AI/lightning/blob/37c244f94be365496def82870b22c2faf0ab889e/src/lightning/pytorch/core/module.py#L999
    #
    # ---
    # 
    # This allow us to avoid disabling the "automatic_optimization" flag
    #
    # Which would have been required to do "segmented learning", or "Backpropagation Through Time"
    # where we would need to implement manual optimization as per
    # https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    #
    # Otherwise an error will be thrown if we call `self.manual_backward`
    #
    # However this would mean that we would need to do a full reimplementation
    # of several features that were handled by the automatic optimization.
    # - accumulate_grad_batches
    # - gradient_clip_val
    # - logging behaviour
    # - distributed training co-ordination
    # - (And probably other features that I am not aware of)
    #
    # So this is a hacky work around, to avoid reimplementing all of the above.
    # 
    # From the current code implementatiion, it seem like this is blocked only by 
    # automatic_optimization flag - and has no adverse side effect otherwise
    # https://lightning.ai/docs/pytorch/stable/_modules/lightning/pytorch/core/module.html#LightningModule.manual_backward
    #
    # If anyone have a better idea, let me know
    # (have experimented with, reimplementing the above, but it is not trivial, unfortunately)
    #
    def manual_backward(self, loss: torch.Tensor, *args, **kwargs):
        if self._fabric:
            self._fabric.backward(loss, *args, **kwargs)
        else:
            # self._verify_is_manual_optimization("manual_backward")
            self.trainer.strategy.backward(loss, None, *args, **kwargs)

    #
    # Main compute_loss function, this is called by the trainer loop
    #
    # @TCompileBaseline
    def compute_loss(self, batch, batch_idx, is_training_run: bool = False, is_validation_run: bool = False):

        # Start time for the step
        step_start_time = time.time()

        # Used for token/second performance tracking
        if self._counting_tokens is None:
            self._counting_tokens = 0
        if self._counting_time_start is None or self._counting_time_start == 0:
            self._counting_time_start = step_start_time
        
        # Get the input sequence, and attention mask
        seq = batch['input_ids']
        assert isinstance(seq, torch.Tensor) and seq.ndim == 2
        ori_seq_mask = batch['attention_mask']

        # # Get the dataset index
        # dataset_index = 0
        # dataset_name = "dataset_0"
        # if "dataset_index" in batch:
        #     dataset_index = batch["dataset_index"]
        #     dataset_name = f"dataset_{dataset_index}"
        # if "dataset_name" in batch and dataset_name is not None:
        #     dataset_name = batch["dataset_name"]

        # Check if attent mask is set, if not initialize it
        if ori_seq_mask is None or ori_seq_mask.ndim != 2:
            ori_seq_mask = torch.ones_like(seq[:, 1:])

        # Initialize the total_mask_sum (but not compute it)
        total_mask_sum = 0

        # Number of GPUs used in training, note that if it is > 1
        # it is requried that all operations here are in sync with
        # all other GPUs, as such "quick return" on this function
        # should not be allowed
        num_devices = self.trainer.num_devices

        # ### ---
        # ### Positional loss bias handling
        # ### ---
        
        # # Get the starting and ending loss bias
        # loss_bias_start = self.position_loss_bias
        # loss_bias_end   = 2.0 - loss_bias_start

        # # Skip loss bias calculation, if loss_bias_start is 1.0
        # if loss_bias_start == 1.0 or (is_training_run == False and self.position_loss_bias_in_validation == False):
        #     seq_mask = ori_seq_mask
        # else:
        #     # Lets get the torch mask sum
        #     total_mask_sum = torch.sum(ori_seq_mask)

        #     # Lets get a linear multiplier for the loss bias
        #     # seq_mask_sum = torch.sum(ori_seq_mask)
        #     bias_mask = torch.linspace(loss_bias_start, loss_bias_end, int(total_mask_sum.item()), device=ori_seq_mask.device)

        #     # Boolean flag of seq_mask > 0
        #     seq_mask_index = ori_seq_mask[0] > 0

        #     # Apply the bias mask only to positive seq_mask values
        #     final_mask = torch.zeros(ori_seq_mask.shape[1], device=ori_seq_mask.device)
        #     final_mask[seq_mask_index] = ori_seq_mask[0][seq_mask_index] * bias_mask

        #     # And save it as seq_mask
        #     seq_mask = final_mask.unsqueeze(0)

        # Since we are no longer doing positional loss above, use seq_mask directly
        seq_mask = ori_seq_mask

        ### ---
        ### Training cutoff logic handling 
        ### ---
        
        # Perform cutoff for training run
        if is_training_run:
            prev_step = 0

            # Avoid using the zip operation, as torch.compile throws an exception on it
            # with `zip not reconized as a valid function`
            # 
            # This skip if ctx_len_warmup_steps/ctx_len_cutoffs is not set
            # ---
            # for step, len_cut in zip(self.ctx_len_warmup_steps,
            #                          self.ctx_len_cutoffs):
            # ---
            for i in range(min(len(self.ctx_len_warmup_steps), len(self.ctx_len_cutoffs))):
                step = self.ctx_len_warmup_steps[i]
                len_cut = self.ctx_len_cutoffs[i]

                if prev_step <= self.global_step < step and len_cut < seq.shape[
                        1] - 1:
                    pos = randint(0, seq.shape[1] - len_cut - 1)

                    # Original
                    # seq = seq[:, pos:pos + len_cut + 1]

                    # Changed to use masking for prefix cutoff (i do not know if this makes sense)
                    seq = seq[:, :pos + len_cut + 1]
                    seq_mask = seq_mask[:, :pos + len_cut + 1]
                    # Set the attention mask to 0 for the skipped tokens
                    seq_mask[:, :pos] = 0
                    break
                prev_step = step
        
        ### ---
        ### Various size checking, and implementing the core checkpoint_step
        ### ---
        
        # BPTT, and training steps, and various size fetching
        do_bptt_learning = self.bptt_learning and is_training_run
        idx, targets = seq[:, :-1], seq[:, 1:]
        B, T = idx.shape
        C = self.n_embd

        # If total_mask_sum, we skip, as there is no tokens of value to learn from anyway
        total_mask_sum = torch.sum(seq_mask)
        avg_mask_sum = ( total_mask_sum / B )

        # # Do a quick return, if there is no tokens of value to learn from due to full masking
        # # DO NOT DO THIS : This causes multi node / multi GPU to go out of sync
        # if num_devices <= 1 and total_mask_sum == 0:
        #     return 0
        
        # Checkpoint steps
        def checkpointed_step(idx, targets, mask, last_shift_states,
                              last_wkv_states):
            # # Skip if there is no tokens of value to learn from
            # if idx.shape[1] == 0:
            #     # Prepare dummy loss
            #     train_loss = torch.tensor(0, dtype=self.emb.weight.dtype).requires_grad_()
            #     sample_loss = train_loss.clone().detach().requires_grad_(False)

            #     # Return the checkpoint values
            #     return sample_loss, train_loss, last_shift_states, last_wkv_states, 0

            # Get the logits, and the new states
            logits, new_shift_states, new_wkv_states = self(
                idx, last_shift_states, last_wkv_states)
            
            # Ensure logits, targets, and mask are contiguous
            # this is required to avoid view is not compatible with size and stride error
            logits = logits.contiguous()
            targets = targets.contiguous()
            mask = mask.contiguous()

            # Compute the token loss
            token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                    targets.view(-1),
                                    reduction="none")
            submask = mask.view(-1)[:token_loss.shape[0]]

            # to encourage the logits to be close to 0
            # factor_divisor is typically the total token count
            L2Wrap_factor = 1e-4 / avg_mask_sum

            # Submask count
            submask_count = torch.sum(submask)
            
            # Selective token loss logic
            if submask_count <= 0.0:
                train_loss = torch.tensor(0, dtype=self.emb.weight.dtype).requires_grad_()
                sample_loss = train_loss.clone().detach().requires_grad_(False)
                train_token_count = 0
                train_mask = submask

            elif self.token_loss_threshold > 0.0 or self.token_dropout_rate > 0.0:

                # Sample loss, without backprop 
                with torch.no_grad():
                    sample_loss = (torch.sum(token_loss * submask) / total_mask_sum).clone().detach().requires_grad_(False)

                # Building the training mask
                train_mask = submask

                # Selective loss gating
                if self.token_loss_threshold > 0.0:
                    above_threshold = token_loss > self.token_loss_threshold
                    train_mask = train_mask * above_threshold

                # Dropout logic
                if self.token_dropout_rate > 0.0:
                    dropout_mask = torch.rand(train_mask.shape, device=train_mask.device) > self.token_dropout_rate
                    train_mask = train_mask * dropout_mask
                
                # The training loss to use
                train_loss = torch.sum(token_loss * train_mask) / total_mask_sum  
                train_token_count = torch.sum(train_mask)

                # Adjust the factor accordingly
                # L2Wrap_factor = L2Wrap_factor * (submask_count / train_token_count)

            else:
                train_loss = torch.sum(token_loss * submask) / total_mask_sum
                sample_loss = train_loss.clone().detach().requires_grad_(False)
                train_token_count = submask_count
                train_mask = submask

            if train_loss <= 0.0:
                segment_train_loss = torch.tensor(0, dtype=self.emb.weight.dtype).requires_grad_()
            else:
                # L2Wrap for the backprop process
                segment_train_loss = L2Wrap.apply(train_loss, logits, L2Wrap_factor, train_mask)

            # Return the checkpoint values
            return sample_loss, segment_train_loss, new_shift_states, new_wkv_states, train_token_count

        # Initialize the states, and compute the segment count
        states = BlockStateList.create(self.n_layer, B, C, 
                                       self.n_head, self.head_size,
                                       seq.device, self.emb.weight.dtype)
        segment_count = math.ceil(T / self.ctx_len)

        # Initialize the training loss, and the token count
        training_loss = torch.tensor(0, dtype=self.emb.weight.dtype).requires_grad_()
        training_tokens = 0

        # Raw sample loss (before selective token training)
        sampling_loss = 0

        ### ---
        ### Learning process logic (BPTT or not)
        ### ---
        
        #
        # BPTT learning, we split the sequence into segments
        # and perform a backward pass for each segment, on its own.
        #
        # Allowing us to perform backpropagation across context sizes much larger
        # then what is supported by the current GPU memory.
        #
        # This reduces the need for the checkpointing process, and mitigate
        # a known error where multiple backwards pass throws an exception.
        #
        # While not mathematically equivalent to full context size learning,
        # it makes "infctx" size training possible with deepspeed 2/3
        #
        # ---
        # 
        # See the following, for more details on "Gradient computed twice" error:
        # https://github.com/microsoft/DeepSpeed/issues/988#issuecomment-1549417269
        #
        # Other possibly related issues on the topic:
        # https://github.com/microsoft/DeepSpeed/pull/677
        # https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-766366413
        #
        if do_bptt_learning:

            gradient_accumulation_steps = max(1, self.trainer.accumulate_grad_batches)
            optimizer = self.optimizers()
            cur_device = self.device
            
            # # We use the average segment size, instead of ctx length size.
            # # this helps ensure that the segment cutoffs do not make the last segment too small.
            # # (eg, the last chunk having only 1 token)
            # #
            # # it also helps ensure the segment cutoff points are more varied, across mixed dataset sizes
            # # and avoid potentially undesired training behaviour at fixed cutoff points
            # # (this only applies for segmented learning)
            # segment_size = min(math.ceil(T / segment_count)+2, self.ctx_len)

            # We now enforce segment_size, to be self.ctx_len, to improve overall torch.compile training performance
            segment_size = self.ctx_len

            # Dummy 2D tensor of shape [B,0], are used to do "dummy checkpoint/forward/backprop" to keep everything in sync
            dummy_empty_zero = torch.zeros(B,0, dtype=torch.long, device=cur_device)

            # Get the max segment count across all GPUs, in the current substep, which is used to keep all devices are in sync
            # Once a thread has completed all its segments, it will do dummy checkpoint/forward/backprop with one token,
            # and stay in sync with the thread that are still working on their segments
            #
            # This is used to work around an undocumented behaviour for either lightning / deepspeed loss.backward multi-gpu handling
            # where the `self.manual_backward()` / `loss.backward()` call will block / freeze / hang when being too "out of sync"
            #
            # This can be viewed as a form of `fabric.barrier()` which is invoked implicitly by each `self.manual_backward()` call
            # except that it isn't exactly a `fabric.barrier()` - because it does not block immediately and instead blocks in 
            # the next `self.manual_backward()` call if the previous ones are too far out of sync. 
            # (its confusing, but makes sense for efficency)
            #
            # Additionally because the "code line position" and params actually matter for the 'barrier' code effect,
            # we cant work around this issue by doing dummy `self.manual_backward()` calls, in another if/else branch or loop
            #
            # Combined, this makes this issue very hard to trace and debug, as it will manifest itself as randomly "freezing"
            # when out of sync during `self.manual_backward()` calls. Without the current work around put in place
            #
            # We only do this, if we are doing bptt learning on all segments (-1), and gpu count > 1
            # otherwise we just use the segment count as it is
            if self.trainer.num_devices > 1:
                if self.bptt_learning_range <= 0:
                    # We perform forward/backward on the shared max segment count across all GPUs
                    # ---
                    # we map it to be a tensor, instead of the int directly, as this is more reliable across certain versions of torch/lightning
                    # https://discord.com/channels/992359628979568762/1148755392638234697/1148821863749931008
                    
                    if self.device.type == "cuda":
                        forward_segment_count = self.trainer.strategy.reduce(
                            torch.cuda.IntTensor([segment_count], device=self.device), 
                            reduce_op="max"
                        )
                    else:
                        forward_segment_count = self.trainer.strategy.reduce(
                            torch.Tensor([segment_count], dtype=torch.int),
                            reduce_op="max"
                        )

                    # Convert to int, if its a torch tensor
                    if isinstance(forward_segment_count, torch.Tensor):
                        forward_segment_count = forward_segment_count.item()
                    # We perform as many backward pass as we need to be equal or more then bptt_learning_range
                    backward_segment_count = forward_segment_count
                else:
                    # We perform as many forward pass as we need to be equal or more then bptt_learning_range
                    # and perform an equal amount of backward pass
                    forward_segment_count  = max(segment_count, self.bptt_learning_range)
                    backward_segment_count = self.bptt_learning_range
            else:
                if self.bptt_learning_range <= 0:
                    # Since we do not need to sync GPUs here, we perform as much forward as we exactly need
                    forward_segment_count  = segment_count
                    backward_segment_count = forward_segment_count
                else:
                    # We clamp the backward segment count to the forward count, and bptt_learning_range
                    forward_segment_count  = segment_count
                    backward_segment_count = min(self.bptt_learning_range, segment_count)

            # We compute when we start the segmented learning process
            if forward_segment_count != backward_segment_count:
                start_learning_segment = max(segment_count - self.bptt_learning_range, 0)
            else:
                start_learning_segment = 0

            # # Segment loss array to track (and reduce later)
            # # of size equal to forward_segment_count
            # segment_loss_arr = [0] * forward_segment_count

            # Lets go through and forward all the segments 
            # (including dummy ones)
            for i in range(forward_segment_count):
                # Apply state truncation, if truncated learning is enabled
                # this limits the backprop process, reduces loss learning rate, 
                # but save vram across extreamly large backpropagation steps
                if self.bptt_truncated_learning:
                    prv_shift_states = states.shift_states.clone().detach().requires_grad_(False)
                    prv_wkv_states = states.wkv_states.clone().detach().requires_grad_(False)
                else:
                    prv_shift_states = states.shift_states
                    prv_wkv_states = states.wkv_states
                
                # We use a dummy masked token 0, to do additional dummy checkpoint/forward/backprop when needed
                # for each additional call after the current "segment_count" max
                if i <= segment_count - 1:
                    cur_idx = idx[:, i * segment_size:(i + 1) * segment_size]
                    cur_tar = targets[:, i * segment_size:(i + 1) * segment_size]
                    cur_msk = seq_mask[:, i * segment_size:(i + 1) * segment_size]
                else:
                    cur_idx = dummy_empty_zero
                    cur_tar = dummy_empty_zero
                    cur_msk = dummy_empty_zero

                # Segmented learning, applies the forward/pass over each chunk seperately
                segment_sample_loss, segment_train_loss, new_shift_states, new_wkv_states, segment_train_tokens = checkpointed_step(
                    cur_idx,
                    cur_tar,
                    cur_msk,
                    prv_shift_states,
                    prv_wkv_states
                )
                states = BlockStateList(new_shift_states, new_wkv_states)

                # # Keep the segment loss (for backpassing in reverse)
                # segment_loss_arr[i] = segment_loss

                # Perform the backward pass accordingly, for valid segments (besides the last segment)
                # In this version, we do backward passes together with the forward passes in the main segment loop
                # Instead of after all segment losses are computed
                #
                # In the past, we have implemented to do all forward, and all backwards. But this was found to be "slow"
                if i >= start_learning_segment and i < start_learning_segment + backward_segment_count:
                    # The learning loss, should be normalized against the accumulation steps
                    # as we are bypassing the pytorch lightning normalization
                    # https://lightning.ai/docs/pytorch/2.0.4/common/lightning_module.html#backward
                    learning_loss = segment_train_loss / gradient_accumulation_steps

                    # Perform the backward pass accordingly, for valid segments (besides the last segment)
                    if i == start_learning_segment + backward_segment_count - 1:
                        # This is the last backward pass, we let the default pytorch lightning handle the backward pass
                        # and return the segment loss as part of the total loss
                        training_loss = training_loss + segment_train_loss
                    else:
                        # Undocumented multiple backward pass support
                        # https://github.com/Lightning-AI/lightning/blob/678f642808c54e4c490caee4df5d357301c976bb/tests/trainer/optimization/test_manual_optimization.py#L251
                        self.manual_backward(learning_loss, optimizer, retain_graph=True)

                        # Accumulate without gradient, as we already did the backward pass
                        training_loss = training_loss + segment_train_loss.clone().detach().requires_grad_(False)
                else:
                    # Even if its not the segments we use for backward pass, we still need to accumulate the loss
                    training_loss = training_loss + segment_train_loss.clone().detach().requires_grad_(False)
                
                # Add token count and raw sampling loss
                training_tokens = training_tokens + segment_train_tokens
                sampling_loss = sampling_loss + segment_sample_loss

                # GC collect unused memory
                # gc.collect()
                # torch.cuda.empty_cache()
        else:

            #
            # Normal operations without BPTT
            #
            segment_size = self.ctx_len
            for i in range(segment_count):
                if i < segment_count-1 and is_training_run:
                    segment_sample_loss, segment_train_loss, new_shift_states, new_wkv_states, segment_train_tokens = deepspeed_checkpoint(
                        checkpointed_step,
                        idx[:, i * segment_size:(i + 1) * segment_size],
                        targets[:, i * segment_size:(i + 1) * segment_size],
                        seq_mask[:, i * segment_size:(i + 1) * segment_size],
                        states.shift_states,
                        states.wkv_states
                    )
                else:
                    segment_sample_loss, segment_train_loss, new_shift_states, new_wkv_states, segment_train_tokens = checkpointed_step(
                        idx[:, i * segment_size:(i + 1) * segment_size],
                        targets[:, i * segment_size:(i + 1) * segment_size],
                        seq_mask[:, i * segment_size:(i + 1) * segment_size],
                        states.shift_states,
                        states.wkv_states
                    )
                
                # Add them up
                training_loss = training_loss + segment_train_loss
                training_tokens = training_tokens + segment_train_tokens
                sampling_loss = sampling_loss + segment_sample_loss

                # Update the states
                states = BlockStateList(new_shift_states, new_wkv_states)
                gc.collect()
                # torch.cuda.empty_cache()

        global_rank = self.global_rank
        global_device_count = self.trainer.num_devices * self.trainer.num_nodes

        # Wandb logging only, if an active run exists (only applies for training)
        if is_training_run:
            microbatch_size = self.trainer.microbatch_size

            # Get the total dataset context length
            batch_ctx_len = 0
            if "data_ctx_len" in batch:
                batch_ctx_len = torch.sum(batch["data_ctx_len"]).item()
            else:
                batch_ctx_len = T * microbatch_size

            # Increment the counting tokens, and log it accordingly
            batch_ktokens = batch_ctx_len / 1000.0
            self._counting_tokens += batch_ktokens

            # Calculate various log values
            ctx_len = batch_ctx_len / microbatch_size
            training_tokens_avg = training_tokens / microbatch_size

            avg_ctx_len = ctx_len
            avg_data_loss = sampling_loss
            avg_learn_tokens = training_tokens_avg
            avg_learn_loss = training_loss

            # Ending time for the step
            step_endin_time = time.time()

            # Get the previous step endin time
            step_prev_endin_time = self._prev_step_endin_timestamp
            if step_prev_endin_time == 0:
                step_prev_endin_time = self._counting_time_start
            self._prev_step_endin_timestamp = step_endin_time

            # Compute the kT values, per GPU
            kT_per_sec = self._counting_tokens / max(step_endin_time - self._counting_time_start, 1e-8)
            kT_per_sec_step = (batch_ktokens) / max(step_endin_time - step_prev_endin_time, 1e-8)
            
            # The log object to use
            logobj = {
                # The original loss and ctx_len (averaged by batch size)
                'train/data_ctxlen': ctx_len, 
                'train/data_loss': sampling_loss,
                # The selective training tokens, and loss
                'train/learn_tokens': training_tokens_avg,
                'train/learn_loss': training_loss,

                # # Dataset based tracking (not working)
                # f'dataset/train/{dataset_index}.loss': training_loss,
                # f'dataset/train/{dataset_index}.data_loss': sampling_loss,
                # f'dataset/train/{dataset_index}.tokens': tokens,
                # f'dataset/train/{dataset_index}.ctx_len': ctx_len,
                # f'dataset/train/{dataset_index}.name': dataset_name,

                # Perf tracking
                f'perf/kTokens_per_sec.gpu.{global_rank}': kT_per_sec,
                f'perf/kTokens_per_sec_step.gpu.{global_rank}': kT_per_sec_step,
                f'perf/kTokens_total.gpu.{global_rank}': self._counting_tokens,

                # Perf tracking
                f'perf/cluster/kTokens_per_sec': kT_per_sec * global_device_count,
                f'perf/cluster/kTokens_per_sec_step': kT_per_sec_step * global_device_count,
                f'perf/cluster/kTokens_total': self._counting_tokens * global_device_count,

                # Step and trainer tracking
                'global_rank': global_rank, 
                'substep': (batch_idx * global_device_count + global_rank),
                'trainer/global_step':self.global_step,
                'trainer/learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'],
                'batchidx': batch_idx
            }

            # Consolidated logging
            if self.consolidated_logging:
                # Prepare the floating tensor values, containing the total kTokens, batch kTokens, data_loss, and learn_loss
                sync_tensor = torch.tensor([
                    self._counting_tokens, 
                    batch_ktokens, 
                    sampling_loss, ctx_len,
                    training_loss, training_tokens_avg
                ], device=self.device, dtype=torch.float32)

                # Get the sum across all devices, for each values
                sync_tensor = self.trainer.strategy.reduce(sync_tensor, reduce_op="sum")

                # Get the sum values
                sum_total_kTokens = sync_tensor[0].item()
                sum_batch_kTokens = sync_tensor[1].item()
                avg_data_loss = sync_tensor[2].item() / global_device_count
                avg_ctx_len = sync_tensor[3].item() / global_device_count
                avg_learn_loss = sync_tensor[4].item() / global_device_count
                avg_learn_tokens = sync_tensor[5].item() / global_device_count

                # Update the log object, merge with the previous log object
                logobj = {
                    **logobj,
                        
                    # Perf tracking
                    f'perf/cluster/kTokens_per_sec': sum_total_kTokens / max(step_endin_time - self._counting_time_start, 1e-8),
                    f'perf/cluster/kTokens_per_sec_step': (sum_batch_kTokens) / max(step_endin_time - step_prev_endin_time, 1e-8),
                    f'perf/cluster/kTokens_total': sum_total_kTokens,

                    # Update the consolidate loss and lengths
                    'train/data_ctxlen': avg_ctx_len, 
                    'train/data_loss': avg_data_loss,
                    'train/learn_tokens': avg_learn_tokens,
                    'train/learn_loss': avg_learn_loss,

                    # # The original loss and ctx_len (averaged by batch size)
                    # f'train/data_ctxlen.gpu.{global_rank}': ctx_len, 
                    # f'train/data_loss.gpu.{global_rank}': sampling_loss,
                    # # The selective training tokens, and loss
                    # f'train/learn_tokens.gpu.{global_rank}': training_tokens_avg,
                    # f'train/learn_loss.gpu.{global_rank}': training_loss,
                }

            # Actual logging into wandb
            if wandb.run is not None:
                wandb.log(logobj)
            
            margs = metrics.MetricArgs(idx, None, None, targets, torch.tensor(avg_learn_loss,device=sync_tensor.device))
            for metric in self.metrics.values():
                metric.update(margs)
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0 and (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
                B, T = idx.shape
                self.log('train/tok', int(batch_idx * global_device_count * B * T), on_step=True, prog_bar=True, rank_zero_only=True)
                self.log('global_step', self.global_step, rank_zero_only=True)
                for name, metric in self.metrics.items():
                    metric_value = metric.compute()
                    metric.clear()
                    self.log('train/'+name, metric_value, on_step=True, prog_bar=True, rank_zero_only=True)

        if is_validation_run:
            # Log the line values
            logobj = {
                # The original loss and ctx_len (averaged by batch size)
                'validation/data_ctxlen': T, 
                'validation/data_loss': sampling_loss,
                # "validation/dataset_index": dataset_index,

                # # The selective training tokens, and loss
                # 'validation/learn_tokens': training_tokens,
                # 'validation/learn_loss': training_loss,

                # # Dataset based tracking (not working)
                # f'dataset/validation/{dataset_index}.loss': training_loss,
                # f'dataset/validation/{dataset_index}.data_loss': sampling_loss,
                # f'dataset/validation/{dataset_index}.ctx_len': T,
                # f'dataset/validation/{dataset_index}.name': dataset_name,

                # Step and trainer tracking
                'global_rank': global_rank, 
                'trainer/global_step':self.global_step,
                'batchidx': batch_idx
            }

            # Consolidated logging
            if self.consolidated_logging:
                # Prepare the floating tensor values, containing the validation values
                sync_tensor = torch.tensor([
                    sampling_loss, T, 
                    training_loss, training_tokens
                ], device=self.device, dtype=torch.float32)

                # Get the sum across all devices, for each values
                sync_tensor = self.trainer.strategy.reduce(sync_tensor, reduce_op="sum")

                # Get the sum values
                avg_data_loss = sync_tensor[0].item() / global_device_count
                avg_ctx_len = sync_tensor[1].item() / global_device_count
                avg_learn_loss = sync_tensor[2].item() / global_device_count
                avg_learn_tokens = sync_tensor[3].item() / global_device_count

                # Update the log object, merge with the previous log object
                logobj = {
                    **logobj,
                        
                    # Update the consolidate loss and lengths
                    'validation/data_ctxlen': avg_ctx_len, 
                    'validation/data_loss': avg_data_loss,
                    # 'validation/learn_tokens': avg_learn_tokens,
                    # 'validation/learn_loss': avg_learn_loss,
                }

            # Actual logging into wandb
            if wandb.run is not None:
                wandb.log(logobj)

        # Throw if total loss is NaN
        assert not torch.isnan(training_loss), "training_loss is NaN"
        return sampling_loss, training_loss

    #
    # Training and validation steps
    #
    def training_step(self, batch, batch_idx):

        # Update the dataloader skip steps (fix dataset offset issues)
        # train_dataloader._set_skip_offset(self.global_step * self.trainer.accumulate_grad_batches)

        # print("=== BATCH ID SHAPE ===", batch["input_ids"].shape)
        # print("=== BATCH AM SHAPE ===", batch["attention_mask"].shape)

        sampling_loss, training_loss = self.compute_loss(batch, batch_idx, True, False)

        #self.log('train/loss', training_loss, prog_bar=True)
                
        # If set - forces the above train/loss log line to always be on a new line
        if self.substep_logging:
            print("")
        
        if self.substep_cuda_cache_clear:
            gc.collect()
            torch.cuda.empty_cache()

        # if loss not a number return None
        if torch.isnan(training_loss):
            return None

        return training_loss

    # @TCompileBaseline
    def validation_step(self, batch, batch_idx):
        sampling_loss, training_loss = self.compute_loss(batch, batch_idx, False, True)
        self.log('validation/loss', sampling_loss, prog_bar=True, sync_dist=True)

        # Reset the token tracking accordingly
        # self._counting_tokens = 0
        # self._counting_time_start = time.time()

        return sampling_loss

### ---
# SimpleRWKV, a wrapper for RWKV that allows for simple usage of the model
### ---

# SimpleRWKV specific imports
from transformers import PreTrainedTokenizerFast

# Current script dir
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCRIPT_PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../'))

# SimpleRWKV is a wrapper for RWKV that allows for simple usage of the model
#
# it is not meant to be highly performant, but rather a simple minimal way to run the RWKV trainer module
# in inference mode, and can be used to validate the model trainer code / its changes
class SimpleRWKV():

    def __init__(
            self,
            model_path: str,
            ctx_len:int = 1024,
            device:str = "cuda",
            dtype:str = "fp32"
        ):

        # Log the mismatch dtype
        if dtype != "fp32":
            print("[SimpleRWKV] Warning: dtype mismatch, only fp32 is supported (for now)")

        # Prepare the model config with the model path, and custom torch load
        model_config = {}
        model_config["load_model"] = model_path
        model_config["ctx_len"] = ctx_len

        # This feature depends on deepspeed
        model_config["grad_cp"] = False
        # model_config["_torch_load_state"] = loaded_state

        # Save the config settings
        self.ctx_len = ctx_len
        self.device = device

        # Lets actually load the model
        self.model = RWKV(**model_config)

        # Lets map it over to the respective device type
        # and set it to run as eval/inference mode
        self.model.to(device)
        self.model.eval()

        # Get the model detected vocab size
        vocab_size = self.model.vocab_size

        # The tokenizer object values
        self.fastTokenizer = None
        self.worldTokenizer = None

        # Setup the tokenizer
        if vocab_size == 50277:
            # Use the neox tokenizer
            tokenizer_file = os.path.join(SCRIPT_DIR,"./dataflow/20B_tokenizer.json")
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
            self.fastTokenizer = tokenizer
        elif vocab_size == 65536:
            # Use the world tokenizer
            from .dataflow.trie_tokenizer import MT_TRIE_TOKENIZER
            world_tokenizer = MT_TRIE_TOKENIZER(os.path.join(SCRIPT_DIR, "./dataflow/rwkv_vocab_v20230424.txt"))
            self.worldTokenizer = world_tokenizer
        else:
            raise NotImplementedError(f"Unsupported vocab size ({vocab_size}) - custom tokenizer not supported")

    # Encoding strings
    def encode(self, text: str):
        if self.worldTokenizer != None:
            return self.worldTokenizer.encode(text)
        return self.fastTokenizer.encode(text)

    # Decoding strings
    def decode(self, tokens: list):
        if self.worldTokenizer != None:
            return self.worldTokenizer.decode(tokens)
        return self.fastTokenizer.decode(tokens)

    # Forwarding logic, withoout torch._no_grad() context
    def _forward(
            self, tokens, 
            stateObj = None,
            all_logits = False
        ):

        logits_arr = None
        token_len = len(tokens)

        # Get the shift/wkv state
        if stateObj is None:
            shift_states = None
            wkv_states = None
        else:
            shift_states = stateObj["shift_states"]
            wkv_states = stateObj["wkv_states"]
        
        # The all_logits array, if requested
        all_logits_arr = None

        # For each token, process the state, in batches up to ctx_len
        for i in range(0, token_len, self.ctx_len):
            # Token set
            token_set = tokens[i:i+self.ctx_len]

            # Check if tokens are already tensors
            batch_tokens = torch.tensor(
                token_set, 
                dtype=torch.long, device=self.device
            ).unsqueeze(0)
            
            # Compute the logits and state
            logits_arr, shift_states, wkv_states = self.model.forward(
                batch_tokens, shift_states, wkv_states
            )

            # Build the all_logits array
            if all_logits:
                if all_logits_arr is None:
                    all_logits_arr = logits_arr[0]
                else:
                    all_logits_arr = torch.cat([all_logits_arr, logits_arr[0]], dim=0)

        # Return the logits and state
        if all_logits:
            return all_logits_arr, { "shift_states": shift_states, "wkv_states": wkv_states }
        else:
            return logits_arr[0][-1], { "shift_states": shift_states, "wkv_states": wkv_states }
    
    # Forwarding logic, with torch._no_grad() context
    def forward(
            self, tokens:list, 
            stateObj = None,
            all_logits = False
        ):
        with torch.no_grad():
            return self._forward(tokens, stateObj, all_logits)

    # Sampling logits
    def sample_logits(
            self, logits, 
            prv_tokens=[0], 
            temperature=1.0, top_p=0.9,
            token_ban: list = []
            ):
        # Copy to CPU first
        logits = logits.cpu()

        # Max negative float
        max_neg = -torch.finfo(torch.float).max

        # Apply token ban
        for x in token_ban:
            logits[x] = max_neg
        
        # Remove NaNs from logits
        for x in range(len(logits)):
            if torch.isnan(logits[x]):
                logits[x] = max_neg

        # Handle sampling with temperature
        if temperature > 0.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs = torch.sort(probs, descending=True)[0]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(cumulative_probs > top_p)])
            probs[probs < cutoff] = 0
            if temperature != 1.0:
                probs = probs.pow(1.0 / temperature)
            out = torch.multinomial(probs, num_samples=1)[0]
            return out
        else: 
            # Since the tokenizer sample does not support temp==0
            # we handle this case ourself, by fining the top token
            return torch.argmax(logits, dim=-1).item()

    # Completion API
    def completion(self, 
            prompt, 
            max_tokens: int = 32,
            temperature: float = 1.0,
            top_p: float = 0.9,
            token_ban: list = [],
            start_state = None,
            stream_to_stdout: bool = False,
        ):
        # Encode the context, if its a string
        if isinstance(prompt, str):
            enc = self.encode(prompt)
        # Check if the prompt is a list of tokens
        elif isinstance(prompt, list):
            enc = prompt
        else:
            raise ValueError("Prompt must be a string or a list of tokens")

        # Keep track of the logits and state
        logits = None
        stateObj = start_state

        # For each token, process the state
        logits, stateObj = self.forward(enc, stateObj)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Generate each token
        out_tokens = []
        for i in range(max_tokens):
            ttt = self.sample_logits(
                logits, 
                # prv_tokens=full_tokens,
                temperature=temperature, top_p=top_p,
                token_ban=token_ban
            )
            
            # Append the token
            out_tokens.append(ttt)
            # full_tokens.append(ttt)
            if stream_to_stdout:
                print(self.decode([ttt]), end="", flush=True)

            # Perform the forward pass
            logits, stateObj = self.forward([ttt], stateObj)

        # Decode the tokens
        out_str = self.decode(out_tokens)

        # # Garbage collect
        # gc.collect()
        # torch.cuda.empty_cache()

        # Return the output string, and state
        return out_str, stateObj
