"""Run in a private Kaggle CPU notebook to retrieve competition data in one file."""
import csv
import hashlib
from pathlib import Path
import zipfile


def main():
    manifests = list(Path('/kaggle/input').rglob('test.csv'))
    assert len(manifests) == 1, 'Attach only the Project 1 competition'
    root = manifests[0].parent
    with manifests[0].open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2000 and len({r['id'] for r in rows}) == 2000
    output = Path('/kaggle/working/competition_test.zip')
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('test.csv', 'sample_submission.csv'):
            archive.write(root/name, name)
        for row in rows:
            name = row['filename']
            assert Path(name).name == name
            path = root/'test_images'/name
            assert path.is_file()
            archive.write(path, 'test_images/'+name)
    with zipfile.ZipFile(output) as archive:
        assert len(archive.namelist()) == 2002 and archive.testzip() is None
    print('Archived 2000 test images and two manifests')
    print('SHA256', hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
