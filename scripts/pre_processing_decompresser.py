import time

import zstandard as zstd

# Just a script which de-compresses the zst files.

dctx = zstd.ZstdDecompressor()

dirs = ["Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-05",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-05",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-04",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-03",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-02",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2019-01",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-12",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-11",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RC_2018-10",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-10",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-11",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2018-12",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-01",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-02",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-03",
        "Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/RS_2019-04"]


for i in dirs:
    start_time = time.time()
    print(i)
    with open(f"Z:/Code/A-Level-Tranformer-chatbot-project/data/compressed/{i}.zst", 'rb') as ifh, \
            open(f"Z:/Code/A-Level-Tranformer-chatbot-project/data/uncompressed/{i}", 'wb') as ofh:
        dctx.copy_stream(ifh, ofh, write_size=65536)
    print(time.time()-start_time)
