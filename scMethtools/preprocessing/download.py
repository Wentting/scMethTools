import os
import time
import requests
def data_downloader(url,path,title):
    r"""datasets downloader
    
    Arguments
    ---------
    - url: `str`
        the download url of datasets
    - path: `str`
        the save path of datasets
    - title: `str`
        the name of datasets
    
    Returns
    -------
    - path: `str`
        the save path of datasets
    """
    if os.path.isfile(path):
        print("......Loading dataset from {}".format(path))
        return path
    else:
        print("......Downloading dataset save to {}".format(path))
        
    dirname, _ = os.path.split(path)
    try:
        if not os.path.isdir(dirname):
            print("......Creating directory {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
    except OSError as e:
        print("......Unable to create directory {}. Reason {}".format(dirname,e))
    
    start = time.time()
    size = 0
    res = requests.get(url, stream=True)

    chunk_size = 1024000
    content_size = int(res.headers["content-length"]) 
    if res.status_code == 200:
        print('......[%s Size of file]: %0.2f MB' % (title, content_size/chunk_size/10.24))
        with open(path, 'wb') as f:
            for data in res.iter_content(chunk_size=chunk_size):
                f.write(data)
                size += len(data) 
                print('\r'+ '......[Downloader]: %s%.2f%%' % ('>'*int(size*50/content_size), float(size/content_size*100)), end='')
        end = time.time()
        print('\n' + ".......Finish！%s.2f s" % (end - start))
    
    return path