import multiprocessing
from training_skeleton import DataLoader
import pickle
import time

def get_batch_processisng(n):
    DL = DataLoader(45000)
    batch = DL.get_batch()
    pickle.dump(batch, open(f"batch_{n}", "wb"))
    print(n)
    while True:
        try:
            time.sleep(0.05)
            if pickle.load(open(f"batch_{n}", "rb"))[0] is None:
                batch = DL.get_batch()
                pickle.dump(batch, open(f"batch_{n}", "wb"))
                print(n)
        except Exception:
            pass


def lol():
    processes = []
    for n in range(10):
        p = multiprocessing.Process(target=get_batch_processisng, args=(n, ))
        processes.append(p)
        p.start()



    # wait until process 1 is finished

    # both processe =s finished


if __name__ == "__main__":

    lol()






