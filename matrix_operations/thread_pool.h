#pragma once

#include <thread>
#include <mutex>
#include <condition_variable>
#include <latch>
#include <memory>
#include <iostream>
#include <boost/lockfree/spsc_queue.hpp>
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <chrono>
#include <queue>

namespace thread_pool
{
    struct Task
    {
        explicit Task(std::function<void()> &&func) : func_(std::move(func)) {}
        Task(const Task &) = delete;
        Task(Task &&) noexcept = default;

        void operator()() const
        {
            func_();
        }

        std::function<void()> func_;
    };

    class WorkerBlocking
    {
    public:
        using TaskPtr = std::shared_ptr<Task>;
        using TaskQueue = std::queue<TaskPtr>;

        WorkerBlocking() = default;
        WorkerBlocking(const WorkerBlocking &) = delete;
        WorkerBlocking(WorkerBlocking &&) noexcept = default;

        /* thread function */
        void run(std::stop_token token)
        {
            while (!token.stop_requested())
            {
                std::unique_lock guard(mtx_);
                if (!queue_.empty())
                {
                    const auto task = queue_.front();
                    queue_.pop();
                    guard.unlock();
                    (*task.get())();
                }
            }
        }

        bool enqueue(const TaskPtr &task)
        {
            std::lock_guard guard(mtx_);
            queue_.push(task);
            return true;
        }

    private:
        std::mutex mtx_{};
        TaskQueue queue_{};
    };

    class WorkerBlockingSpin
    {
    public:
        using TaskPtr = std::shared_ptr<Task>;
        using TaskQueue = std::queue<TaskPtr>;

        WorkerBlockingSpin() = default;
        WorkerBlockingSpin(const WorkerBlockingSpin &) = delete;
        WorkerBlockingSpin(WorkerBlockingSpin &&) = default;

        /* thread function */
        void run(std::stop_token token)
        {
            while (!token.stop_requested())
            {
                boost::unique_lock guard(mtx_);
                if (!queue_.empty())
                {
                    const auto task = queue_.front();
                    queue_.pop();
                    guard.unlock();
                    (*task.get())();
                }
            }
        }

        bool enqueue(const TaskPtr &task)
        {
            boost::lock_guard guard(mtx_);
            queue_.push(task);
            return true;
        }

    private:
        boost::detail::spinlock mtx_{};
        TaskQueue queue_{};
    };

    class WorkerBlockingLazy
    {
    public:
        using TaskPtr = std::shared_ptr<Task>;
        using TaskQueue = std::queue<TaskPtr>;

        WorkerBlockingLazy() = default;
        WorkerBlockingLazy(const WorkerBlockingLazy &) = delete;
        WorkerBlockingLazy(WorkerBlockingLazy &&) = default;

        /* thread function */
        void run(std::stop_token token)
        {
            while (!token.stop_requested())
            {
                std::unique_lock guard(mtx_);
                cv_.wait(guard, [this]()
                         { return !queue_.empty(); });

                const auto task = queue_.front();
                queue_.pop();
                guard.unlock();
                (*task.get())();
            }
        }

        bool enqueue(const TaskPtr &task)
        {
            boost::lock_guard guard(mtx_);
            queue_.push(task);
            cv_.notify_one(); /* doesn't need the lock to notify */
            return true;
        }

    private:
        std::mutex mtx_{};
        std::condition_variable cv_{};
        TaskQueue queue_{};
    };

    class WorkerLockFree
    {
    public:
        using TaskPtr = std::shared_ptr<Task>;
        using TaskQueue = boost::lockfree::spsc_queue<TaskPtr>;

        WorkerLockFree() = default;
        WorkerLockFree(const WorkerLockFree &) = delete;
        WorkerLockFree(WorkerLockFree &&) = default;

        void run(std::stop_token token)
        {
            while (!token.stop_requested())
            {
                if (queue_.read_available() > 0)
                {
                    const auto task = queue_.front();
                    queue_.pop();
                    (*task.get())();
                }
            }
        }

        bool enqueue(const TaskPtr &task)
        {
            if (queue_.write_available())
            {
                queue_.push(task);
                return true;
            }
            return false;
        }

    private:
        TaskQueue queue_{5};
    };

    template <typename Worker>
    class ThreadPoolImpl
    {
    public:
        ThreadPoolImpl() = default;
        ThreadPoolImpl(const ThreadPoolImpl &) = delete;
        ThreadPoolImpl(ThreadPoolImpl &&) noexcept = default;

        void init(std::size_t number_of_threads)
        {
            for (std::size_t i = 0; i < number_of_threads; i++)
            {
                workers_.emplace_back(std::make_shared<Worker>());
            }
            for (const auto &worker : workers_)
            {
                threads_.emplace_back(std::bind_front(&Worker::run, worker.get()));
            }
        }

        void join()
        {
            for (auto &thread : threads_)
            {
                thread.join();
            }
            std::cout << "joined" << std::endl;
        }

        using Workers = std::vector<std::shared_ptr<Worker>>;
        Workers workers_{};
        using Threads = std::vector<std::jthread>;
        Threads threads_{};
    };

    
    // using Worker = WorkerBlocking;
    using Worker = WorkerBlockingSpin;
    //using Worker = WorkerLockFree;
    
    using ThreadPool = ThreadPoolImpl<Worker>;

    /* only for benchmark. need a static object to avoid creating a new pool everytime */
    class ThreadPoolInstance
    {
    public:
        ThreadPoolInstance()
        {
            tp.init(8);
        }
        ThreadPoolInstance(const ThreadPoolInstance&) = delete;
        ThreadPoolInstance(ThreadPoolInstance&&) = delete;

        static ThreadPool &get_instance()
        {
            return tp;
        }

    private:

        inline static ThreadPool tp{};
    };
}
