"""Verify response ownership and real vLLM concurrency before a large run."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from urllib import request
import uuid

from scripts.run_wn18rr_completion_batch import save


def check(root, endpoint='http://127.0.0.1:12345', image_path=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    barrier = threading.Barrier(10)
    finished = threading.Event()
    samples = []

    def metrics():
        while not finished.is_set():
            try:
                text = request.urlopen(endpoint + '/metrics', timeout=2).read().decode()
                running = sum(float(value) for value in re.findall(
                    r'^vllm:num_requests_running(?:\{[^\n]*\})?\s+([0-9.eE+\-]+)', text, re.M))
                samples.append({'seconds': time.monotonic(), 'running': running})
            except Exception:
                pass
            finished.wait(0.05)

    def submit(index, parallel=True):
        key = f'taor-batch-{index:02d}'
        prompt = ('Return only a JSON object with probe_id exactly ' + key +
                  ', number exactly ' + str(index) + ', and an explanation of around 80 words '
                  'describing a water channel. Do not use thinking or a code fence.')
        body = {'model': 'Qwen/Qwen3.5-9B', 'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0, 'max_tokens': 256,
                'chat_template_kwargs': {'enable_thinking': False}}
        destination = root / ('parallel' if parallel else 'serial')
        save(destination / f'{key}-request.json', body)
        if parallel:
            barrier.wait(timeout=15)
        started = time.monotonic()
        req = request.Request(endpoint + '/v1/chat/completions', data=json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json', 'X-Request-Id': uuid.uuid4().hex})
        with request.urlopen(req, timeout=120) as response:
            payload = json.load(response)
        save(destination / f'{key}-response.json', payload)
        assert payload['model'] == body['model']
        choice = payload['choices'][0]
        assert choice['finish_reason'] == 'stop', 'Empty/truncated output is not a JSON parse failure'
        content = choice['message']['content']
        assert content and '<think>' not in content
        decoded = json.loads(content)
        assert decoded['probe_id'] == key and decoded['number'] == index, 'Batch responses crossed requests'
        return {'probe_id': key, 'seconds': time.monotonic()-started,
                'completion_tokens': payload['usage']['completion_tokens']}

    monitor = threading.Thread(target=metrics, daemon=True)
    monitor.start()
    try:
        start = time.monotonic()
        serial = [submit(index, False) for index in range(10)]
        serial_seconds = time.monotonic()-start
        print(json.dumps({'event': 'qwen_serial_probe_done', 'seconds': serial_seconds}), flush=True)
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(submit, range(10)))
        parallel_seconds = time.monotonic()-started
    finally:
        finished.set()
        monitor.join(timeout=3)
        save(root / 'scheduler-metrics.json', samples)
    peak = max((sample['running'] for sample in samples), default=0)
    report = {'requests': len(results), 'all_response_ids_matched': True,
              'peak_engine_running_requests': peak, 'wall_seconds': parallel_seconds,
              'serial_seconds': serial_seconds, 'serial_results': serial,
              'serial_tokens_per_second': sum(row['completion_tokens'] for row in serial)/serial_seconds,
              'parallel_tokens_per_second': sum(row['completion_tokens'] for row in results)/parallel_seconds,
              'results': results}
    save(root / 'summary.json', report)
    assert peak >= 2, 'HTTP concurrency alone does not prove batched model inference'
    tokenize_request = request.Request(endpoint + '/tokenize', data=json.dumps({
        'model': 'Qwen/Qwen3.5-9B', 'messages': [{'role': 'user', 'content': 'Hello'}]}).encode(),
        headers={'Content-Type': 'application/json'})
    with request.urlopen(tokenize_request, timeout=30) as response:
        tokenizer = json.load(response)
    assert tokenizer.get('count', 0) > 0
    save(root / 'tokenizer.json', tokenizer)
    if image_path is not None:
        image_path = Path(image_path)
        image = image_path.read_bytes()
        body = {'model': 'Qwen/Qwen3.5-9B', 'temperature': 0, 'max_tokens': 256,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Describe the visible shape of the main object in one sentence.'},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'+base64.b64encode(image).decode()}}]}]}
        req = request.Request(endpoint+'/v1/chat/completions', data=json.dumps(body).encode(),
                              headers={'Content-Type': 'application/json'})
        with request.urlopen(req, timeout=120) as response:
            vision = json.load(response)
        save(root / 'vision-response.json', vision)
        assert vision['choices'][0]['finish_reason'] == 'stop'
        assert vision['choices'][0]['message']['content'].strip()
        report['vision'] = {'source_path': str(image_path), 'sha256': hashlib.sha256(image).hexdigest(),
                            'content': vision['choices'][0]['message']['content']}
    save(root / 'passed.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    check(args.root, image_path=args.image)
