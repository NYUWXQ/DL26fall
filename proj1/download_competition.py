"""Download the competition manifest and images with the authenticated Kaggle API."""
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
from requests import HTTPError

ROOT = Path(__file__).resolve().parent / 'data' / 'competition'
COMPETITION = 'cifar-10-res-net-architecture-challenge-dl'
LOCAL = threading.local()
REQUEST_LOCK = threading.Lock()
NEXT_REQUEST = 0.


def request_slot():
    global NEXT_REQUEST
    while True:
        with REQUEST_LOCK:
            delay = NEXT_REQUEST - time.monotonic()
            if delay <= 0:
                NEXT_REQUEST = time.monotonic() + 1.
                return
        time.sleep(delay)


def api():
    if not hasattr(LOCAL, 'client'):
        LOCAL.client = KaggleApi()
        LOCAL.client.authenticate()
    return LOCAL.client


def download(name):
    global NEXT_REQUEST
    path = ROOT / name
    if path.exists():
        if path.suffix != '.png':
            return
        try:
            with Image.open(path) as image:
                image.verify()
            return
        except (OSError, SyntaxError):
            pass
    for attempt in range(20):
        try:
            request_slot()
            api().competition_download_file(COMPETITION, name, path=str(ROOT), force=True, quiet=True)
            if path.suffix == '.png':
                with Image.open(path) as image:
                    assert image.size == (32, 32) and image.mode == 'RGB'
                    image.verify()
            return
        except HTTPError as error:
            if error.response.status_code not in (429, 500, 502, 503, 504) or attempt == 19:
                raise
            delay = max(float(error.response.headers.get('Retry-After', 0)), min(300, 30 * 2**attempt))
            with REQUEST_LOCK:
                NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic() + delay)
            print(f'HTTP {error.response.status_code}; all downloads backing off for {delay:g}s', flush=True)
    raise RuntimeError(f'Unable to download {name}')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ('test.csv', 'sample_submission.csv'):
        download(name)
    with (ROOT/'test.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (ROOT/'sample_submission.csv').open() as f:
        sample = list(csv.DictReader(f))
    assert len(rows) == len(sample) and len({r['id'] for r in rows}) == len(rows)
    assert [r['id'] for r in rows] == [r['id'] for r in sample]
    assert all(Path(r['filename']).name == r['filename'] for r in rows)
    names = ['test_images/' + r['filename'] for r in rows]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, _ in enumerate(pool.map(download, names), 1):
            if count % 100 == 0:
                print(f'{count}/{len(names)} images verified', flush=True)
    print('Competition data ready', flush=True)


if __name__ == '__main__':
    main()
