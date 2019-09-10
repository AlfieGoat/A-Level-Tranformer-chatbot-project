import zstandard as zstd
import time

dctx = zstd.ZstdDecompressor()

#dirs = ["RC_2019-01", "RC_2019-02", "RC_2019-03", "RC_2019-04", "RC_2019-05"]
dirs = ["RS_2018-11", "RS_2018-12", "RS_2019-01","RS_2019-02", "RS_2019-03", "RS_2019-04", "RS_2019-05"]

for i in dirs:
    
    time1 = time.time()
    print(i)
    with open(f"Z:/Code/Neural_Network_stuff/Datasets/reddit_comments/{i}.zst", 'rb') as ifh, open(f"Z:/Code/Neural_Network_stuff/Datasets/reddit_comments/unzipped/{i}", 'wb') as ofh:
        dctx.copy_stream(ifh, ofh, write_size=65536)
        
    print(time.time()-time1)
