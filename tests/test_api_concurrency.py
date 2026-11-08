from concurrent.futures import ThreadPoolExecutor
import asyncio
import io
import threading
import time
from urllib import request

import pytest
from scripts import run_wn18rr_api_preflight as api


def test_api_permits_cover_response_body_and_release_after_errors(monkeypatch):
    active, peak, lock = [0], [0], threading.Lock()
    class Response(io.BytesIO):
        def read(self, *args, **kwargs):
            time.sleep(.002)
            return super().read(*args, **kwargs)
        def close(self):
            if not self.closed:
                with lock:
                    active[0] -= 1
            super().close()
    def opened(url, **kwargs):
        if str(url).endswith('/fail'):
            raise ValueError('transport failed')
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        return Response(b'{}')
    monkeypatch.setattr(request, 'urlopen', opened)
    restore = api.pace_api_requests('api.test', 0, 4)
    try:
        def read(index):
            if index % 11 == 0:
                with pytest.raises(ValueError):
                    request.urlopen('https://api.test/fail', timeout=2)
            else:
                with request.urlopen('https://api.test/data', timeout=2) as response:
                    assert response.read(1) == b'{'
                    assert response.read() == b'}'
        with ThreadPoolExecutor(max_workers=128) as pool:
            list(pool.map(read, range(256)))
        assert peak[0] <= 4 and active[0] == 0
    finally:
        restore()
    assert request.urlopen is opened


def test_admission_timeout_never_sends_a_late_request_and_localhost_bypasses(monkeypatch):
    sent = []
    def opened(url, **kwargs):
        sent.append(url)
        return io.BytesIO(b'{}')
    monkeypatch.setattr(request, 'urlopen', opened)
    restore = api.pace_api_requests('api.test', 0, 1)
    try:
        first = request.urlopen('https://api.test/first', timeout=1)
        with request.urlopen('http://127.0.0.1/v1', timeout=.01) as local:
            assert local.read() == b'{}'
        with pytest.raises(TimeoutError):
            request.urlopen('https://api.test/must-not-send', timeout=.01)
        first.close()
        with request.urlopen('https://api.test/next', timeout=1) as response:
            assert response.read() == b'{}'
        assert 'https://api.test/must-not-send' not in sent
    finally:
        restore()


def test_async_and_sync_transports_share_one_limit(monkeypatch):
    active, peak = [0], [0]
    async def opened(client, method, url, **kwargs):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        try:
            await asyncio.sleep(.002)
            if url.endswith('/fail'):
                raise ValueError('failed')
            return 'ok'
        finally:
            active[0] -= 1
    monkeypatch.setattr(api.httpx.AsyncClient, 'request', opened)
    restore = api.pace_api_requests('api.test', 0, 3)
    async def check():
        async with api.httpx.AsyncClient() as client:
            results = await asyncio.gather(*[client.request('GET','https://api.test/'+('fail' if i%10==0 else 'ok'))
                                           for i in range(128)], return_exceptions=True)
            assert sum(isinstance(x, ValueError) for x in results) == 13
    try:
        asyncio.run(check())
        assert peak[0] <= 3 and active[0] == 0
    finally:
        restore()
