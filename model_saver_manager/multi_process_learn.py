import time
from multiprocessing import Pool, Process, Pipe, Queue


def f2(name):
    print('hello', name)


def f(x):
    return x * x


def f3(conn):
    send_obj = [42, None, 'hello']
    conn.send(send_obj)
    conn.close()
    # del send_obj


def f4(q, cur_id):
    if cur_id == 0:
        q.put([42, None, 'hello'])
        q_send = Queue()
        q_receive = Queue()
        p = Process(target=f4, args=((q_send, q_receive), 1,))
        q.put([p, q_send, q_receive])
        #q.put([p, q_send, q_receive])
        #q.put([p, q_send, q_receive])
        #q.put([p, q_send, q_receive])
        p.start()
        while True:
            pass

    else:
        # q.put(["awesome!"])
        q_send, q_receive = q
        # q.put([p,q_send, q_receive])
        q_send.put(["awesome"])
        while True:
            value = q_receive.get()
            value[0] = value[0] + 1
            q_send.put(value)


if __name__ == '__main__':
    def main_fn():
        child_conn, parent_conn = Pipe()
        p = Process(target=f3, args=(child_conn,))
        p.start()
        print(parent_conn.recv())  # prints "[42, None, 'hello']"
        p.join()

        cur_id = 0
        q = Queue()
        p = Process(target=f4, args=(q, cur_id,))
        p.start()
        print(q.get())  # prints "[42, None, 'hello']"
        time.sleep(5)
        print(q.qsize())
        ls = q.get()
        print(ls)
        new_process = ls[0]
        new_queue_receive = ls[1]
        new_queue_send = ls[2]
        print(new_queue_receive.get())
        # p.join()
        print("First process joined...")
        print(new_queue_receive.get())
        new_queue_send.put([0])
        val = new_queue_receive.get()
        print(val)
        new_queue_send.put(val)
        val = new_queue_receive.get()
        print(val)
        new_queue_send.put(val)
        val = new_queue_receive.get()
        print(val)
        new_queue_send.put(val)
        val = new_queue_receive.get()
        print(val)
        new_queue_send.put(val)
        val = new_queue_receive.get()
        print(val)
        new_queue_send.put(val)
        val = new_queue_receive.get()
        print(val)
        new_queue_send.put(val)
        val = new_queue_receive.get()
        print(val)
        new_process.terminate()
        new_process.join()

        pool = Pool()
        with pool as p:
            print(p.map(f, [1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]))

        p = Process(target=f2, args=('bob',))
        p.start()
        p.join()


    main_fn()
