'''
multiprocessing
the ability to run multiple processes at the same time

global interpreter lock(GIL)
    1)allows only one thread to run at a time
    2)when 2 threads try to access same data at a time or try to write onto same memory
    it will lead to memory leakage
    3)GIL is used to avoid deadlocks and memory leakages

differences b/w multithreading and multiprocessing

multithreading                                      multiprocessing
1)used for input/output bound tasks         1)used for CPU bound tasks
Ex:image downloading,network connections     ex:tasks where extensive processing involves
2)cpu has to switch b/w multiple thraeds    2)cpu has to switch b/w multiple processes
so that it appears they are running simulta  so that it appears they are running simulta
3)creation of thread economical in time      3)creation of process is slow and resource
and resource                                   specific
4)threads belongs to same process            4)multiple process have different memory
and share same memory and resource             and resource for each process
that of process
5)takes moderate time for processing         5)takes low time for processing
6)only one thread can run at a time          6)overcomes GIL and multiple process
due to GIL                                        at a time

picking/serialization
1)process of serializing objects into binary streams,also known as serialization
2)picking is useful when we want to save the state of objects to any file  and
reuse them at another time without loosing any data

pickle.dump(object,filename)

unpickling/marshalling/flattening.
1)its inverse of picking,retreiving back the original python objects from a saved binary
stream file

pickle.load(file)

pickle is the module used

'''
import multiprocessing
import time

start=time.perf_counter()
def do_something(seconds):
    print(f'sleeping for {seconds} seconds')
    time.sleep(seconds)
    print(f' Done sleeping for {seconds} seconds')
if __name__ == '__main__':
    secs=[5,4,3,2,1]
    process=[]
    for sec in secs:
        p=multiprocessing.Process(target=do_something,args=[sec])
        p.start()
        process.append(p)

    for p in process:
        p.join()

    start=time.perf_counter()