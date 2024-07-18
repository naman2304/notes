## The trouble with distributed systems

### Faults and partial failures
* A program on a single computer either works or it doesn't. There is no reason why software should be flaky (non deterministic). An individual computer with good software is either fully functional or entirely broken, but not something in between.
* In a distributed systems we have no choice but to confront the messy reality of the physical world. There will be parts that are broken in an unpredictable way, while others work. Partial failures are _nondeterministic_. Things will unpredictably fail.

### Cloud Computing and Supercomputing
**Supercomputers**
* High-performing computing (HPC), also known as supercomputers are used for computationally intensive scientific computing tasks, like weather forecasting or molecular dynamics.
* In supercomputers, job typically checkpoints state of its computation to durable storage from time to time. If one node fails, common solution is to stop the entire cluster workload. After faulty node is repaired, computation is restarted from last checkpoint. So, supercomputer is more like single node computer (it deals with partial failure by letting it escalate into total failure)
* Built using specialized hardware, nodes communicate through shared memory and remote direct memory access (RDMA).
* Uses specialized network topologies

**Cloud computing**
* Cloud computing, uses multi-tenant datacenters, commodity computers.
* Here, making the service unavailable (stopping the cluster for repair) is not possible
* Commodity hardware, shared-nothing architecture
* Commodity network toplogies

We need to accept the possibility of partial failure and build fault-tolerant mechanism into the software. **We need to build a reliable system from unreliable components.**
* IP (Layer 3, Network Layer) is unreliable: delay, drop, duplicate and reorder packets.
* TCP (Layer 4, Transport Layer) which is built on top of IP, ensures missing packets are transmitted (not drop), duplicates are eliminated, and packets are reassembled into the order in which they are sent (TCP can't do anything about "delay" though).

### Unreliable networks

Focusing on _shared-nothing systems_ the network is the only way machines communicate.

The internet and most internal networks are _asynchronous packet networks_. A message is sent and the network gives no guarantees as to when it will arrive, or whether it will arrive at all. Things that could go wrong (all 6 of these are indistinguashable from perspective of client). If you send a request to another node and don't receive a response, it is _impossible_ to tell why.
1. Request lost
2. Request waiting in a queue to be delivered later
3. Remote node may have failed
4. Remote node may have temporarily stopped responding
5. Response has been lost on the network
6. The response has been delayed and will be delivered later

* **The usual way of handling this issue is a _timeout_**: after some time you give up waiting and assume that the response is not going to arrive.
* EC2 (Elastic Compute Cloud) is notorious for having frequent network glitches.
* Nobody is immune to network problems. You do need to know how your software reacts to network problems to ensure that the system can recover from them. It may make sense to deliberately trigger network problems and test the system's response.
* **Detecting Faults**
  * Uncertainity about network makes it difficult to tell whether a node is working or not. Although you may get feedback sometimes
    * If you can reach the node (but the process crashed), OS will helpfully close or refuse TCP connections by sending a RST or FIN packet in reply.
    * If node crashes, but OS is still running, it can notify other nodes about the crash
    * If you have access to management interface of network switches of datacenter, you can query to detect link failures at hardware level.
   * But all of these don't work all the time. If you want to be sure that a request was successful, you need a positive response from the application itself.
   * If something has gone wrong, you have to assume that you will get no response at all.
* **Timeouts and unbounded delays**
   * A long timeout means a long wait until a node is declared dead.
   * A short timeout detects faults faster, but carries a higher risk of incorrectly declaring a node dead (when it could be a slowdown). Premature declaring a node is problematic, if the node is actually alive the action may end up being performed twice.
   * When a node is declared dead, its responsibilities need to be transferred to other nodes, which places additional load on other nodes and the network.
* **Network congestion and queueing**
   * Variability of packet delays is most often due to queuing:
      * Different nodes try to send packets simultaneously to the same destination, the network switch must queue them and feed them to the destination one by one. The switch will discard packets when filled up, so dropped packets would be required to resent.
      * If CPU cores are busy, the request is queued by the operative system, until applications are ready to handle it.
      * TCP performs _flow control_ (also known as congestion avoidance or backpressure), in which a node limits its own rate of sending in order to avoid overloading a network link or the receiving node. This means additional queuing at the sender.
   * You can choose timeouts experimentally by measuring the distribution of network round-trip times over an extended period. TCP considers packet to be lost if it is not acknowledged within some timeout, and hence lost packets are automatically retransmitted. Systems can continually measure response times and their variability (_jitter_), and automatically adjust timeouts according to the observed response time distribution.
   * **TCP vs UDP**: UDP does not perform _flow control_ and retransmission of packets.
* **Synchronous vs ashynchronous networks**
   * A telephone network estabilishes a _circuit_, a fixed guaranteed amount of bandwidth is allocated for the call, we say is _synchronous_ even as the data passes through several routers as it does not suffer from queuing. The maximum end-to-end latency of the network is fixed (_bounded delay_).
   * A circuit is a fixed amount of reserved bandwidth which nobody else can use while the circuit is established, whereas packets of a TCP connection opportunistically use whatever network bandwidth is available. While TCP connection is idle, it doesn't use any bandwidth.
   * Internet networks are optimized for bursty traffic.
   * **Using circuits for bursty data transfers wastes network capacity and makes transfer unnecessary slow. By contrast, TCP dynamically adapts the rate of data transfer to the available network capacity.**
   * We have to assume that network congestion, queueing, and unbounded delays will happen. Consequently, there's no "correct" value for timeouts, they need to be determined experimentally.

### Unreliable clocks
* The time when a message is received is always later than the time when it is sent, we don't know how much later due to network delays. This makes difficult to determine the order of which things happened when multiple machines are involved.
* Each machine on the network has its own clock (actual hardware device usually a quartz crystal oscillator), slightly faster or slower than the other machines. It is possible to synchronise clocks with Network Time Protocol (NTP).
* Modern computers have 2 different types of clocks:
   * **Time-of-day clocks**
     * Return the current date and time according to some calendar (_wall-clock time_).
     * `clock_gettime(CLOCK_REALTIME)` on Linux and `System.currentTimeMillis()` in Java.
     * If the local clock is too far ahead of the NTP server, it may be forcibly reset and appear to jump back to a previous point in time. Also, then ignore leap seconds. **This makes it is unsuitable for measuring elapsed time.**
   * **Monotonic clocks**
     * The _absolute_ value of the clock is meaningless (as it might be number of nanoseconds since computer was started, or something similarly arbitrary)
     * `clock_gettime(CLOCK_MONOTONIC)` on Linux and `System.nanoTime()` in Java.
     * They are guaranteed to always move forward. The difference between clock reads can tell you how much time elapsed beween two checks.
     * NTP allows the clock rate to be speeded up or slowed down by up to 0.05%, but **NTP cannot cause the monotonic clock to jump forward or backward**. **In a distributed system, using a monotonic clock for measuring elapsed time (eg: timeouts), is usually fine**.
* If some piece of software is relying on an accurately synchronised clock, the result is more likely to be silent and subtle data loss than a dramatic crash.
* **Timestamps for ordering events**
   * **It is tempting, but dangerous to rely on clocks for ordering of events across multiple nodes.** This usually imply that _last write wins_ (LWW), often used in both multi-leader replication and leaderless databases like Cassandra and Riak, and data-loss may happen.
   * _Logical clocks_, based on counters instead of oscillating quartz crystal, are safer alternative for ordering events. Logical clocks do not measure time of the day or elapsed time, only relative ordering of events. This contrasts with time-of-the-day and monotonic clocks (also known as _physical clocks_).
* **Clock readings have a confidence interval**
   * It doesn't make sense to think of a clock reading as a point in time, it is more like a range of times, within a confidence interval: for example, 95% confident that the time now is between 10.3 and 10.5. 
   * Spanner
     * Google's TrueTime API in Spanner when asked for time, explicitly gives confidence interval [earliest, latest], which are the earliest possible and the latest possible timestamp.
     * The most common implementation of snapshot isolation requires a monotonically increasing transaction ID. ON single node DB, simple counter is sufficient for generating transaction IDs. On distributed DBs, a globally monotonically increasing transaction ID is difficult to generate, because it requires coordination
     * Spanner implements snapshot isolation across datacenters by using clock's confidence interval. If you have two confidence internvals where `A = [A earliest, A latest]` and `B = [B earliest, B latest]` and those two intervals do not overlap (`A earliest` < `A latest` < `B earliest` < `B latest`), then B definitely happened after A.
     * For solving for overlapping case, Spanner deliberately waits for the length of the confidence interval before commiting a read-write transaction. By doing so, it ensures that any transaction that may read the data is at a sufficiently later time, so their confidence intervals do not overlap.
     * Spanner needs to keep the clock uncertainty as small as possible, that's why Google deploys a GPS receiver or atomic clock in each datacenter, allowing clocks to be synchronized to within 7 ms as opposed to ~100 ms for a normal computer.

### Process pauses

How does a node know that it is still leader? One option is for the leader to obtain a _lease_ from other nodes (similar to a lock with a timeout). It will be the leader until the lease expires; to remain leader, the node must periodically renew the lease. If the node fails, another node can takeover when it expires.

```java
while(true) {
   request = getIncomingRequest();

   // Ensure that the lease always has at least 10 seconds remaining
   if (lease.expiryTimeMillis - System.currentTimeMillis < 10000) {
      lease = lease.renew();
   }

   if (lease.isValid()) {
      process(request);
   }
}
```
Problems in above code:
1. Relying on synchronized clocks (expiry time on lease is setup by different machine but it is being compared to local system clock)
2. Let's say we use monotonic clocks (and then problem#1) is not there anymore, even then, the code assumes that very little time passes between the point it checks the time and the time when it processes the request. Normally, 10 second buffer is more than enough so that lease doesn't expire in middle of processing a request, however unexpected pauses may occur
   * Stop the world garbage collector
   * Virtual machine can be suspended and resumed. This pause can be of arbitrary length of time.
   * Operating system context-switches
   * Swapping to disk (paging)

* **You cannot assume anything about timing**
* **Response time guarantees**
   * There are systems that require software to respond before a specific _deadline_ (_real-time operating system, or RTOS_).
   * For example, if your car's onboard sensors detect that you are currently experiencing a crash, you wouldn't want to release the airbag to be delayed due to inopportune GC pause.
   * Library functions must document their worst-case execution times; dynamic memory allocation may be restricted or disallowed and enormous amount of testing and measurement must be done. RTOS are extremely expensive.
   * Garbage collection could be treated like brief planned outages. If the runtime can warn the application that a node soon requires a GC pause, the application can stop sending new requests to that node, wait for it to finish processing outstanding requests, and then perform GC while no requests are in progress. Latency sensitive firms like HFTs uses this approach.

### Knowledge, truth and lies

A node cannot necessarily trust its own judgement of a situation. Many distributed systems rely on a _quorum_ (voting among the nodes).

Commonly, the quorum is an absolute majority of more than half of the nodes.

#### Fencing tokens

Assume every time the lock server grant sa lock or a lease, it also returns a _fencing token_, which is a number that increases every time a lock is granted (incremented by the lock service). Then we can require every time a client sends a write request to the storage service, it must include its current fencing token.

The storage server remembers that it has already processed a write with a higher token number, so it rejects the request with the last token.

If ZooKeeper is used as lock service, the transaciton ID `zcid` or the node version `cversion` can be used as a fencing token.

#### Byzantine faults

Fencing tokens can detect and block a node that is _inadvertently_ acting in error.

Distributed systems become much harder if there is a risk that nodes may "lie" (_byzantine fault_).

A system is _Byzantine fault-tolerant_ if it continues to operate correctly even if some of the nodes are malfunctioning.
* Aerospace environments
* Multiple participating organisations, some participants may attempt ot cheat or defraud others
