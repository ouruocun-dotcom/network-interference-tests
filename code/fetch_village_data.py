"""Download the author's public mirror, verifying the Dataverse file checksum.

Dataverse doi:10.7910/DVN/U3BIHX, released version 9.4, CC0 1.0.
File datav4.0.zip, id 2460959, 16,037,457 bytes. The public author mirror
is used independently of Dataverse's visitor-registration interface.
No registration form, account information or participant records are submitted.
"""
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,time

URL='https://web.stanford.edu/~jacksonm/diffusion-science-data.zip'
SIZE=16037457
MD5='a61cc1f89000677b46dcfbc492e81b64'

def fetch(destination):
    destination=Path(destination)
    if destination.exists() and hashlib.md5(destination.read_bytes()).hexdigest()==MD5:return destination
    size=(SIZE+3)//4
    def segment(index):
        start=index*size;end=min(SIZE,(index+1)*size)-1
        request=Request(URL,headers={'Range':f'bytes={start}-{end}'})
        for attempt in range(3):
            try:
                with urlopen(request,timeout=240) as response:
                    data=response.read()
                    assert response.status==206 and len(data)==end-start+1
                return index,data
            except Exception:
                if attempt==2:raise
                time.sleep(1)
    parts={}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(segment,i) for i in range(4)]):
            index,data=future.result();parts[index]=data
            print('Downloaded segment',index+1,'of 4',flush=True)
    data=b''.join(parts[i] for i in range(4))
    if len(data)!=SIZE or hashlib.md5(data).hexdigest()!=MD5:raise ValueError('Dataset checksum mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(data)
    print('Verified original Dataverse MD5 and size.',flush=True)
    return destination

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('inputs/verified_network_data.zip'))
    fetch(p.parse_args().output)
