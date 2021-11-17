/******************************************************************************/
//
//	Copyright (C) 2018, Florian Hirschberger <florian.hirschberger@uol.de>
//
//	LICENSE: THE SOFTWARE IS PROVIDED "AS IS" UNDER THE
//	ACADEMIC FREE LICENSE (AFL) v3.0.
//
/******************************************************************************/

#ifndef UTILITY_THREADS_HPP
#define UTILITY_THREADS_HPP

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>

template <bool flag>
class tp;

template <>
class tp<true> {
    std::vector<std::thread> threads;
    std::size_t nthreads;
    std::size_t cap;
    std::size_t up;
    std::size_t ret;
    std::mutex m;
    std::mutex dyn;
    std::condition_variable c;
    std::condition_variable w;

    void (tp::*call)(std::size_t t);
    const void *args;

    template <class L>
    void wrap(std::size_t t);

    void loop(void);
    void wait();

public:
    tp(std::size_t _nthreads);
    ~tp();

    template <class L>
    void parallel(std::size_t tasks, const L& immu);

    std::size_t size(void);
};

tp<true>::tp(std::size_t _nthreads)
    : threads()
    , nthreads(_nthreads)
    , cap(0)
    , ret(0)
{
    for (std::size_t t = 0; t < nthreads; t++) {
        threads.push_back(std::thread(&tp::loop, this));
    }
    wait();
}

tp<true>::~tp()
{
    call = nullptr;
    args = nullptr;

    c.notify_all();
    for (std::size_t t = 0; t < nthreads; t++) {
        threads[t].join();
    }
}

void
tp<true>::loop(void)
{
    std::size_t t;
    {
        std::lock_guard<std::mutex> lock(m);
        t = cap++;
    }

    while (true)
    {
        {
            std::unique_lock<std::mutex> lock(m);
            ret++;
            if (ret == nthreads) {
                w.notify_all();
            }
            c.wait(lock);
        }

        if (this->call) {
            (this->*call)(t);
        } else {
            break;
        }
    }
}

void
tp<true>::wait()
{
    std::unique_lock<std::mutex> lock(m);
    while (ret < nthreads) {
        w.wait(lock);
    }
}

template <class L>
void
tp<true>::wrap(std::size_t t)
{
    std::size_t delta = (cap + nthreads - 1) / nthreads;
    std::size_t from  = t * delta;
    std::size_t to    = (t + 1) * delta;
    std::size_t min_to = std::min(cap, to);

    for (std::size_t it = from; it < min_to; it++) {
        (*reinterpret_cast<const L*>(args))(it, t);
    }
}

template <class L>
void
tp<true>::parallel(std::size_t tasks, const L& immu)
{
    call = &tp::wrap<L>;
    args = &immu;

    cap = tasks;
    ret = 0;
    up  = 0;

    c.notify_all();
    wait();
}

std::size_t
tp<true>::size(void)
{
    return nthreads;
}

template <>
class tp<false>
{
public:
    std::size_t size(void);

    template <class L>
    void parallel(std::size_t, const L&);
};

template <class L>
void
tp<false>::parallel(std::size_t tasks, const L& immu)
{
    for (std::size_t i(0); i < tasks; i++)
    {
        immu(i, 0);
    }
}


std::size_t
tp<false>::size(void)
{
    return 1;
}


#endif
