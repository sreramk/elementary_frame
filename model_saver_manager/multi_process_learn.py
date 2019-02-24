from multiprocessing import Pool, Process, Pipe, Queue

def f2(name):
    print('hello', name)

def f(x):
    return x*x



def f3(conn):
    send_obj = [42, None, 'hello']
    conn.send(send_obj)
    conn.close()
    # del send_obj

def f4(q):
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    child_conn, parent_conn = Pipe()
    p = Process(target=f3, args=(child_conn,))
    p.start()
    print(parent_conn.recv())   # prints "[42, None, 'hello']"
    p.join()

    q = Queue()
    p = Process(target=f4, args=(q,))
    p.start()
    print(q.get())  # prints "[42, None, 'hello']"
    p.join()

    pool = Pool()
    with pool as p:
        print(p.map(f, [1, 2, 3,3,3,3,3,3,3,3,3,3]))



    p = Process(target=f2, args=('bob',))
    p.start()
    p.join()