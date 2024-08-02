## Caching

### Distributed Caching

Caching
* Pros
  * faster reads (and writes?)
  * reduced load on database
* Cons
  * cache misses are expensive -- say our cache server is diff, and our application server makes a network call to just find out that our value is not there in cache. Then have to make disk seek.
  * data consistency is complex, depends on how much you care

What do we cache
* database results
* computation done by application servers
* popular static content (CDN)

Server-local cache (seeing at low level)
* L1 is smallest (fastest) cache. Different for different CPU core
* L2 is bigger (slower than L1) cache. Can be shared or different for different CPU core
* L3 is biggest (slowest than L2) cache. Shared across different CPU core.
* First we check L1, if not found, L2, if not found L3, if not found RAM, if not found disk seek.
* ![Single computer cache](/metadata/single_computer_cache.png)

Server-local cache (seeing at high level) 
* cache on application server, database, message broker
* say if we want to see number of likes on our post, application server can cache that result. If we make call again, first we need to reach the same application server (consistent hashing on load balancer), and then we can fetch the result from cache on application server and not make network call to database.
* Pros: fewer network calls, very fast
* Cons: cache size is proportional to number of servers

Global caching layer
* dedicated cache server(s)
* hit application server, which makes network call to caching server, if not present, make network call to database server
* Pros: scale independently (replication + partitioning) from application server
* Cons: extra network call

Writing to database strategies
1. Write Around Cache
   * We just write to database
   * What about cache consistency? Two ways:
     * Don't do anything -- stale reads for sometime (till entry expires due to TTL), then fresh reads
     * Invalidation: forcefully invalidate the cache entry. When another request comes, cache miss happens and we seek from disk and update the cache
2. Write Through Cache
   * We write to cache, and to database
   * What about cache consistency? Two ways:
     * Don't do anything -- can be troubling
     * Two phase commit: slow
3. Write Back Cache
   * We write to cache, which asynchronously write to database (can do batching)
   * What about cache consistency? Two ways:
     * Don't do anything -- durability
     * Distributed locking + replication

Cache Eviction Policies
1. FIFO
   * implemented using queue
   * easy to understand, but has huge downside
2. Least Recently Used (LRU)
   * implemented using hashmap + doubly linked list
3. Least Frequently Used (LFU)

Redis vs Memcached
| Memcached | Redis |
| --------- | ----- |
| Bare bones in memory | Feature rich: hashmaps, sorted sets, geo indexes, etc |
| Volatile; do not write to disk | Durable; writes to disk using WAL |
| Partioned using consistent hashing | Paritioned using fixed number of partitions via gossip protocol |
| Multithreaded | Singlethreaded (allows transaction; actual serial execution) |
| LRU eviction | |
| | Single leader replication |
