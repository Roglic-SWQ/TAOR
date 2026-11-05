"""Fix vLLM 0.11's upstream-FA vision metadata placement in the TAOR-only venv.

Never change shared environments. The CPU cu_seqlens metadata must be on CUDA
when Qwen2-VL falls back to the upstream flash_attn kernel (A100, head size 80).
"""
from pathlib import Path
import hashlib
import json

ROOT = Path('/medisco/linjiashi/venvs/taor-qwen7b/lib/python3.11/site-packages')
TARGET = ROOT / 'vllm/model_executor/models/qwen2_vl.py'
MARKER = '# TAOR: upstream FlashAttention requires CUDA sequence metadata.'


def main():
    if TARGET.is_symlink() or not TARGET.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Refusing to modify a shared/symlinked vLLM installation")
    source = TARGET.read_text()
    before = hashlib.sha256(source.encode()).hexdigest()
    needle = '            output = flash_attn_varlen_func(q,\n'
    if MARKER not in source:
        if source.count(needle) != 1:
            raise RuntimeError('Unexpected vLLM Qwen2-VL implementation; inspect before patching')
        source = source.replace(needle,
            '            ' + MARKER + '\n'
            '            if self.use_upstream_fa:\n'
            '                cu_seqlens = cu_seqlens.to(device=q.device)\n\n' + needle)
        TARGET.write_text(source)
    print(json.dumps({'path': str(TARGET), 'before_sha256': before,
                      'after_sha256': hashlib.sha256(source.encode()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
