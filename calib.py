import torch
from tqdm import tqdm
from transformers.utils import logging
import os
import pickle

logger = logging.get_logger(__name__)


def _relative_ridge_cholesky(data):
    """Cholesky after upstream's absolute 7e-6 shift has already failed.

    Scales the ridge to the matrix instead of using a fixed constant, escalating
    until the factor is finite. Returns the same upper-triangular convention as
    the call site (`cholesky(...).T`).
    """
    scale = torch.diagonal(data).abs().max().clamp(min=1.0)
    eye = torch.eye(data.shape[0], dtype=data.dtype, device=data.device)
    for rel in (1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        try:
            L = torch.linalg.cholesky(data + (rel * scale) * eye)
        except Exception:
            continue
        if torch.isfinite(L).all():
            print(f"Info: relative ridge {rel:g}*{scale:.3e} restored definiteness")
            return L.T
    raise RuntimeError("scaling_diag_matrix is not salvageable by a relative ridge")


class Hook:
    def __init__(self, module, backward=False):
        self.calib = None
        if not backward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        with torch.no_grad():
            inp = input[0].detach().float()
            inp = inp.flatten(start_dim=0, end_dim=-2)
            if self.calib is None:
                self.calib = (inp.T @ inp).cpu()
            else:
                self.calib += (inp.T @ inp).cpu()

    def close(self):
        self.hook.remove()
        del self.calib


class Calib:
    @staticmethod
    def save(path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_calib_data(group, name, save_path=None):
        assert save_path is not None
        # calibration data for up and gate is the same
        if name == 'mlp.gate_proj':
            name = 'mlp.up_proj'
        if name == "self_attn.q_proj" or name == "self_attn.v_proj":
            name = "self_attn.k_proj"
        data = None
        file_path = os.path.join(save_path, name)
        for item in group:
            file_name = os.path.join(file_path, "{}.pkl".format(item))
            if os.path.exists(file_name):
                tmp_data = Calib.load(file_name)
                if data is None:
                    data = tmp_data
                else:
                    data += tmp_data
            else:
                raise FileNotFoundError(
                    "{} not found. You should run build_calibration_dataset first!".format(file_name))
        return data

    @staticmethod
    def get_s_inv_s(group, name, model_type, calib_path=None):
        data = Calib.get_calib_data(group, name, calib_path).double()
        # The following code is from https://github.com/AIoT-MLSys-Lab/SVD-LLM
        try:
            scaling_diag_matrix = torch.linalg.cholesky(data).T
        except Exception as e:
            print("Warning: eigen scaling_diag_matrix is not positive!")
            eigenvalues = torch.linalg.eigvalsh(data)
            data += (- eigenvalues[0] + 7e-6) * torch.eye(data.shape[0]).to(data.device)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(data).T
            except Exception:
                # Upstream leaves this second cholesky unguarded. The 7e-6 shift is
                # ABSOLUTE, but these Gram matrices are sums over 256*2048 tokens
                # with norms many orders larger, so on the bigger models it is far
                # below the fp32 accumulation noise it has to dominate and the
                # retry raises again. (SVD-LLM ships the same code with 1e-6 and
                # died exactly this way on opt-6.7b.) Escalate a RELATIVE ridge.
                # opt-125m enters the branch above 7 times and never gets here, so
                # the validated 76.4902/436.3088/165.9375 cannot move.
                scaling_diag_matrix = _relative_ridge_cholesky(data)
            eigenvalues = None
            del eigenvalues
        try:
            invs = torch.linalg.inv(scaling_diag_matrix)
        except Exception:
            # Upstream does not guard this at all.
            _scale = torch.diagonal(scaling_diag_matrix).abs().max().clamp(min=1.0)
            _eye = torch.eye(scaling_diag_matrix.shape[0], dtype=scaling_diag_matrix.dtype,
                             device=scaling_diag_matrix.device)
            invs = None
            for _rel in (1e-10, 1e-8, 1e-6, 1e-4):
                _cand = scaling_diag_matrix + (_rel * _scale) * _eye
                try:
                    _inv = torch.linalg.inv(_cand)
                except Exception:
                    continue
                if torch.isfinite(_inv).all():
                    print(f"Info: relative ridge {_rel:g}*{_scale:.3e} restored invertibility")
                    scaling_diag_matrix, invs = _cand, _inv
                    break
            if invs is None:
                print("Info: falling back to pinv")
                invs = torch.linalg.pinv(scaling_diag_matrix)
        return scaling_diag_matrix, invs

    @staticmethod
    def build_calibration_dataset(model, dataloader, names, model_type, save_path):
        print("Start building calibration data.")

        if model_type == "gpt2":
            tmp_model = model.transformer.h
        elif model_type == "llama2":
            tmp_model = model.model.layers
        elif model_type == "opt":
            tmp_model = model.model.decoder.layers
        elif model_type == "mistral":
            tmp_model = model.model.layers
        else:
            raise NotImplementedError

        # Every layer's hooks are live at once and each holds a float32
        # d_in x d_in Gram in HOST memory until the whole pass ends:
        #   opt-6.7b  1.22 GB/layer x 32 =  39 GiB (measured on disk)
        #   opt-13b   1.86 GB/layer x 40 =  74 GiB
        #   opt-30b   3.90 GB/layer x 48 = 175 GiB  <- 93% of the 187.5 GiB cgroup
        # Trillium caps a 1-GPU job at 24 cores x 7.8125 GiB, so the cgroup cannot
        # be raised. Hooking a SLICE of the layers per pass bounds the resident set
        # at the cost of one extra forward pass per group.
        #
        # This is numerically exact: each Linear's Gram depends only on its own
        # inputs, and those come from the forward pass, which is identical whether
        # or not a hook is attached elsewhere. BS_CALIB_GROUPS defaults to 1 --
        # one pass over every layer, i.e. upstream's exact behaviour -- so the
        # validated opt-125m/6.7b/13b and Llama runs are untouched.
        n_groups = max(1, int(os.environ.get("BS_CALIB_GROUPS", "1")))
        n_layers = len(tmp_model)
        per = (n_layers + n_groups - 1) // n_groups
        assert save_path is not None
        model.config.use_cache = False
        model.eval()

        for g0 in range(0, n_layers, per):
            g1 = min(g0 + per, n_layers)
            if n_groups > 1:
                print(f"Calibration pass for layers [{g0}, {g1}) "
                      f"of {n_layers}", flush=True)
            hooks = {}
            for name in names:
                hooks[name] = []
                for layer in tmp_model[g0:g1]:
                    target = layer.get_submodule(name)
                    hooks[name].append(Hook(target, backward=False))

            for i, batch in tqdm(enumerate(dataloader)):
                with torch.no_grad():
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    out = model(**batch)

            for name in names:
                tmp_save_path = os.path.join(save_path, name)
                if not os.path.exists(tmp_save_path):
                    os.makedirs(tmp_save_path)
                for i, hook in enumerate(hooks[name]):
                    data = hook.calib.cpu()
                    # index within the full model, not within the group
                    tmp_name = str(g0 + i) + ".pkl"
                    Calib.save(os.path.join(tmp_save_path, tmp_name), data)
                    hook.close()
            del hooks

    @staticmethod
    def build_update_dataset(model, dataloader, names, model_type, save_path):
        print("Start building update dataset.")
        if model_type == "gpt2":
            tmp_model = model.transformer
            num_layers = len(tmp_model.h)
        elif model_type == "llama2" or model_type == "mistral":
            tmp_model = model.model
            num_layers = len(tmp_model.layers)
        elif model_type == "opt":
            tmp_model = model.model.decoder
            num_layers = len(tmp_model.layers)
        else:
            raise NotImplementedError

        hooks = {}
        for name in names:
            hooks[name] = []
            for i in range(num_layers):
                target = tmp_model.get_submodule(name)[str(i)]
                hooks[name].append(Hook(target, backward=False))

        model.config.use_cache = False
        model.eval()
        for i, batch in tqdm(enumerate(dataloader)):
            with torch.no_grad():
                batch = {k: v.to(model.device) for k, v in batch.items()}
                out = model(**batch)

        assert save_path is not None
        for name in names:
            tmp_save_path = os.path.join(save_path, name)
            if not os.path.exists(tmp_save_path):
                os.makedirs(tmp_save_path)
            for i, hook in enumerate(hooks[name]):
                data = hook.calib.cpu()
                tmp_name = str(i) + ".pkl"
                Calib.save(os.path.join(tmp_save_path, tmp_name), data)
                hook.close()
