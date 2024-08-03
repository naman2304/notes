This is copied, modified and appended from [here](https://github.com/keyvanakbary/learning-notes/blob/master/books/designing-data-intensive-applications.md)

# [Designing Data-Intensive Applications](https://www.goodreads.com/book/show/23463279-designing-data-intensive-applications)

- [Reliable, scalable, and maintainable applications](#reliable-scalable-and-maintainable-applications)
  - [Reliability](#reliability)
  - [Scalability](#scalability)
  - [Maintainability](#maintainability)
- [Data models and query language](#data-models-and-query-language)
  - [Relational model vs document model vs network model](#relational-model-vs-document-model-vs-network-model)
  - [Query languages for data](#query-languages-for-data)
  - [Graph-like data models](#graph-like-data-models)
- [Storage and retrieval](#storage-and-retrieval)
  - [Transaction processing or analytics?](#transaction-processing-or-analytics)
  - [Column-oriented storage](#column-oriented-storage)
- [Encoding and evolution](#encoding-and-evolution)
  - [Formats for encoding data](#formats-for-encoding-data)
  - [Modes of dataflow](#modes-of-dataflow)
- [Replication](#replication)
  - [Implementation of replication logs](#implementation-of-replication-logs)
  - [Problems with replication lag](#problems-with-replication-lag)
  - [Multi-leader replication](#multi-leader-replication)
  - [Leaderless replication](#leaderless-replication)
- [Partitioning](#partitioning)
  - [Partitioning and replication](#partitioning-and-replication)
  - [Partition of key-value data](#partition-of-key-value-data)
  - [Partitioning and secondary indexes](#partitioning-and-secondary-indexes)
  - [Rebalancing partitions](#rebalancing-partitions)
  - [Request routing](#request-routing)
- [Transactions](#transactions)
  - [The slippery concept of a transaction](#the-slippery-concept-of-a-transaction)
  - [Weak isolation levels](#weak-isolation-levels)
  - [Serializability](#serializability)
- [The trouble with distributed systems](#the-trouble-with-distributed-systems)
  - [Faults and partial failures](#faults-and-partial-failures)
  - [Unreliable networks](#unreliable-networks)
  - [Unreliable clocks](#unreliable-clocks)
  - [Knowledge, truth and lies](#knowledge-truth-and-lies)
- [Consistency and consensus](#consistency-and-consensus)
  - [Consistency guarantees](#consistency-guarantees)
  - [Linearizability](#linearizability)
  - [Ordering guarantees](#ordering-guarantees)
  - [Distributed transactions and consensus](#distributed-transactions-and-consensus)
- [Batch processing](#batch-processing)
  - [Batch processing with Unix tools](#batch-processing-with-unix-tools)
  - [MapReduce](#mapreduce)
  - [Beyond MapReduce](#beyond-mapreduce)
- [Stream processing](#stream-processing)
  - [Transmitting event streams](#transmitting-event-streams)
  - [Databases and streams](#databases-and-streams)
  - [Processing Streams](#processing-streams)
- [The future of data systems](#the-future-of-data-systems)
  - [Data integration](#data-integration)
  - [Unbundling databases](#unbundling-databases)
  - [Aiming for correctness](#aiming-for-correctness)
  - [Doing the right thing](#doing-the-right-thing)
- [Essential Technologies](#essential-technologies)
  - [Caching](#caching)
  - [Search Indexes](#search-indexes)

| Database        | Database Model       | Storage Engine                                     | Replication log     | Replication type | Partitioning Strategy | Secondary index partition | Rebalancing strategy | ACID isolation |
| -------------   | -------------------- | -------------------------------------------------- | ------------------- | ---------------- | --------------------- | ------------------------- | ------- | ------- |
| MySQL           | Relational           | BTree (InnoDB)                                     | Logical (row-based). Was statement based before version 5.1 | Single + Multi (Tungsten Replicator)   | | | | serializable (2PL)
| PostgreSQL      | Relational           | BTree                                              | WAL based           | Single + Multi (BDR - Bi Directional Replication) | | | | serializable (SSI)
| Oracle          | Relational           | BTree                                              | WAL based           | Single + Multi (GoldenGate)  |
| SQL Server      | Relational           | BTree                                              |                     | Single + Multi   |
| VoltDB          | Relational           | Hash (in memory database - but maintains durability by WAL)                          | Statement based     |                  |                      | Local Index  | | serializable (actual serial execution)
| Redis           | Key-Value            | Hash (in memory one; disk one is custom format)  | | | | | Fixed number of partitions via gossip protocol | serializable (actual serial execution as its single threaded)
| Memcached       | Key-Value            | Hash (in memory one; no data is flushed to disk) | | | | | Consistent Hashing
| Riak            | Key-Value            | Hash (Bitcask), LSM (LevelDB)                      |                     | Leaderless       | Hash                 | Local Index  | Fixed number of partitions |
| RocksDB         | Key-Value            | LSM                                                |
| Voldemort       | Key-Value            |                                                    |                     | Leaderless       | Hash                 |              | Fixed number of partitions |
| Amazon DynamoDB | Key-Value            |                                                    |                     | Single           | Hash+Key combo       | Global Index |
| MongoDB         | Document             |                                                    |                     |                  | Key Range (<2.4)     | Local Index  |
| CouchDB         | Document             |                                                    |                     | Multi            | Key Range            |
| RethinkDB       | Document             |                                                    |                     |                  |                      |              | Dynamic partitioning |
| Espresso        | Document             |
| ElasticSearch   | Document             |                                                    |                     |                  |                      | Local Index  | Fixed number of partitions |
| Solr            | Document             |                                                    |                     |                  |                      | Local Index  |
| Neo4J           | Graph (property)     |
| Titan           | Graph (property)     |
| InfiniteGraph   | Graph (property)     |
| Datomic         | Graph (triple store) |
| AllegroGraph    | Graph (triple store) |
| Cassandra       | Wide Column          | LSM                                                |                     | Leaderless       | Hash+Key combo       | Local Index  |
| HBase           | Wide Column          | LSM                                                |                     |                  | Key Range            |              | Dynamic partitioning |
| BigTable        | Wide Column          | LSM                                                |                     |                  | Key Range
| Amazon Redshift | Wide Column          | BTree

## Reliable, scalable, and maintainable applications

Applications are today not compute intensive, rather data intensive. Bigger problem is usually the amount of data, complexity of the data and the speed at which it is changing. A data-intensive application is typically built from standard building blocks. They usually need to:
* Store data (_databases_)
* Speed up reads (_caches_) (Memcache)
* Search data (_search indexes_) (Elasticsearch or Solr)
* Send a message to another process asynchronously (_stream processing_)
* Periodically crunch data (_batch processing_)

Things are becoming fuzzy too, and boundaries are not obvious. For instance, some datastores are also used as message queues (Redis), and there are message queues which have database like durability (Apache Kafka).

If you have an application managed caching layer (using Memcached or similar), or a full text search server (such as ElasticSearch or Solr) separate from the main database, it is normally the application's responsibility to keep those caches and indexes in sync with the main database.

* **Reliability**. To work _correctly_ even in the face of _adversity_.
* **Scalability**. Reasonable ways of dealing with growth (data volume or traffic volume)
* **Maintainability**. Be able to work on it _productively_.

### Reliability

Typical expectations:
* Application performs the function the user expected
* Tolerate the user making mistakes
* Its performance is good (under expected load)
* The system prevents unauthorised access and abuse

Systems that anticipate faults and can cope with them are called _fault-tolerant_ or _resilient_.

**A fault is usually defined as one component of the system deviating from its spec**, whereas _failure_ is when the system as a whole stops providing the required service to the user.

You should generally **prefer tolerating faults over preventing failures** (there are cases where prevention is better than cure, because there is no cure -- for instance, a security failure which compromises data of users). Netflix Chaos Monkey.

* **Hardware faults**. Until recently redundancy of hardware components was sufficient for most applications. As data volumes increase, more applications use a larger number of machines, proportionally increasing the rate of hardware faults (MTTF of hard disk is 10-50 years, a datacentre with 10k disks on average have 1 disk per day failure). **There is a move towards systems that tolerate the loss of entire machines**. A system that tolerates machine failure can be patched one node at a time, without downtime of the entire system (_rolling upgrade_).
    - Disks: RAID (Redundant Array of Independent Disks) config in disks
    - Servers: dual power supplies or hot swappable CPUs in servers
    - Datacentres: battery generator in datacenters)
* **Software errors**. It is unlikely that a large number of hardware components will fail at the same time. Software errors are a systematic error within the system, they tend to cause many more system failures than uncorrelated hardware faults (leap second bug in 2012 that caused many applications to hang simultaneously)
* **Human errors**. Humans are known to be unreliable. Configuration errors by operators are a leading cause of outages. You can make systems more reliable:
    - Minimising the opportunities for error, example: with admin interfaces that make easy to do the "right thing" and discourage the "wrong thing".
    - Provide fully featured non-production _sandbox_ environments where people can explore and experiment safely.
    - Automated testing (unit and integration).
    - Quick and easy recovery from human error, fast to rollback configuration changes, roll out new code gradually and tools to recompute data.
    - Set up detailed and clear monitoring, such as performance metrics and error rates (_telemetry_).
    - Implement good management practices and training.

### Scalability

This is how do we cope with increased load. We need to succinctly describe the current load on the system (for instance, requests per second, ratio of reads to writes in DB, cache hit rate on cache, # of active people in chat room, etc); only then we can discuss growth questions.

---

#### Describing Load - Twitter example

Twitter main operations
- Post tweet: a user can publish a new message to their followers (4.6k req/sec, over 12k req/sec peak)
- Home timeline: a user can view tweets posted by the people they follow (300k req/sec)

Two ways of implementing those operations:
1. Posting a tweet simply inserts the new tweet into a global collection of tweets. When a user requests their home timeline, look up all the people they follow, find all the tweets for those users, and merge them (sorted by time). This could be done with a SQL `JOIN`.
2. Maintain a cache for each user's home timeline. When a user _posts a tweet_, look up all the people who follow that user, and insert the new tweet into each of their home timeline caches.

Approach 1, systems struggle to keep up with the load of home timeline queries. So the company switched to approach 2. The average rate of published tweets is almost two orders of magnitude lower than the rate of home timeline reads.

Downside of approach 2 is that posting a tweet now requires a lot of extra work. Some users have over 30 million followers. A single tweet may result in over 30 million writes to home timelines.

Twitter moved to an hybrid of both approaches. Tweets continue to be fanned out to home timelines but a small number of users with a very large number of followers are fetched separately and merged with that user's home timeline when it is read, like in approach 1.

---

#### Describing performance

What happens when the load increases:
* How is the performance affected?
* How much do you need to increase your resources?

In a batch processing system such as Hadoop, we usually care about _throughput_, or the number of records we can process per second.

> ##### Latency and response time
> The response time is what the client sees. Latency is the duration that a request is waiting to be handled.

Even in scenarios where we think that all requests will take same amount of time, there is variation due to
* Network delays (retransmission of packets on TCP)
* Node delays
  * Garbage Collection Pause
  * Context Switch
  * Page Fault forcing to read from disk
  * Mechanical vibrations in server rack
 
It's common to see the _average_ response time of a service reported. However, the mean is not very good metric if you want to know your "typical" response time, it does not tell you how many users actually experienced that delay.

**Better to use percentiles.**
* _Median_ (_50th percentile_ or _p50_). Half of user requests are served in less than the median response time, and the other half take longer than the median
* Percentiles _95th_, _99th_ and _99.9th_ (_p95_, _p99_ and _p999_) are good to figure out how bad your outliners are.

Amazon describes response time requirements for internal services in terms of the 99.9th percentile because the customers with the slowest requests are often those who have the most data. The most valuable customers.

On the other hand, optimising for the 99.99th percentile would be too expensive.

_Service level objectives_ (SLOs) and _service level agreements_ (SLAs) are contracts that define the expected performance and availability of a service.
An SLA may state the median response time to be less than 200ms and a 99th percentile under 1s. **These metrics set expectations for clients of the service and allow customers to demand a refund if the SLA is not met.**

Queueing delays often account for large part of the response times at high percentiles. **It is important to measure times on the client side.**

When generating load artificially, the client needs to keep sending requests independently of the response time.

> ##### Percentiles in practice
> Calls in parallel, the end-user request still needs to wait for the slowest of the parallel calls to complete.
> The chance of getting a slow call increases if an end-user request requires multiple backend calls. Tail latency amplification.

#### Approaches for coping with load

* _Scaling up_ or _vertical scaling_: Moving to a more powerful machine
* _Scaling out_ or _horizontal scaling_: Distributing the load across multiple smaller machines (_shared nothing architecture_)
* _Elastic_ systems: Automatically add computing resources when detected load increase. Quite useful if load is unpredictable.

Distributing stateless services across multiple machines is fairly straightforward. Taking stateful data systems from a single node to a distributed setup can introduce a lot of complexity. Until recently it was common wisdom to keep your database on a single node.

### Maintainability

The majority of the cost of software is in its ongoing maintenance. There are three design principles for software systems:
* **Operability**. Make it easy for operation teams to keep the system running.
* **Simplicity**. Easy for new engineers to understand the system by removing as much complexity as possible.
* **Evolvability**. Make it easy for engineers to make changes to the system in the future.

#### Operability: making life easy for operations

A good operations team is responsible for
* Monitoring and quickly restoring service if it goes into bad state
* Tracking down the cause of problems
* Deployment and keeping software and platforms up to date
* Perform complex maintenance tasks, like platform migration
* Maintaining the security of the system
* Keeping tabs on how different systems affect each other
* Anticipating future problems (capacity planning)
* Establishing good practices and tools for development
* Defining processes that make operations predictable
* Documentation to preserve organisation's knowledge

**Good operability means making routine tasks easy.**

#### Simplicity: managing complexity

When complexity makes maintenance hard, budget and schedules are often overrun. There is a greater risk of introducing bugs.

Making a system simpler means removing _accidental_ complexity, as non inherent in the problem that the software solves (as seen by users).

One of the best tools we have for removing accidental complexity is _abstraction_ that hides the implementation details behind clean and simple to understand APIs and facades.

#### Evolvability: making change easy

_Agile_ working patterns provide a framework for adapting to change. Test Driven Development (TDD).

* _Functional requirements_: what the application should do
* _Nonfunctional requirements_: general properties like reliability, scalability, maintainability, security and compliance.

---
---

## Data models and query language

Most applications are built by layering one data model on top of another. Each layer hides the complexity of the layers below by providing a clean data model. These abstractions allow different groups of people to work effectively.
* Real life objects &#8594; Document, Graph, Relational DB &#8594; LSM, BTree &#8594; bytes in terms of electrical currents.

_Not Only SQL_ has a few driving forces:
* Greater scalability
* preference for free and open source software
* Specialised query optimisations
* Desire for a more dynamic and expressive data model
  
### Relational model vs document model vs network model

**Relational model**
* Way to lay out all the data in the open -- a relation (table) is simply a collection of tuples (rows).
* The roots of relational databases lie in _business data processing_ in 1960s and 70s (_transaction processing_ and _batch processing_). The goal was to hide the implementation details behind a cleaner interface.
* Use B-Trees mostly
* Use 2 phase locking for transactions (or serializable snapshot isolation)
* Scaling
  * Vertical: easier (specialized hardware => mainframe)
  * Horizontal: sharding (generalized hardware). Might have to do distributed transaction (super tough!):
    * Involves multiple shards with one value incrementing on one, decrementing on another (like money transfer)
    * Joins involving multiple shards
* **Normalization** -- data is mostly normalized using foreign keys.
* **Impedance Mismatch** - With a SQL model, if data is stored in a relational tables, an awkward translation layer is translated, this is called _impedance mismatch_.
* **Joins** - In relational databases, it's normal to refer to rows in other tables by ID, because joins are easy. Great for many-to-many relationships.
* **Schema Flexibility** - schema-on-write, like static (compile time) type checking (C++, Java) where schema is handled at DB layer
* **Locality** - not so great locality as tables are stored physically at different place on disk. Bad data locality can lead to distributed transactions like two-phase commit or do joins across datacenters (which is slow). Some databases do it differently -- Google Spanner offers great locality properties by allowing schema to declare that a table's rows should be interleaved (nested) within a parent table.
* **Example**
  * traditional: MySQL, PostgreSQL, Oracle, SQL Server
  * NewSQL: VoltDB, Spanner

**Document model**
* The most popular database for business data processing in the 1970s was the IBM's _Information Management System_ (IMS).
* IMS used a _hierarchical model_ and like document databases worked well for one-to-many relationships, but it made many-to-any relationships difficult, and it didn't support joins.
* **Normalization** -- data is mostly denormalized
* **Impedance Mismatch** - JSON model reduces the impedance mismatch and the lack of schema is often cited as an advantage.
* **Joins** - In document databases, joins are not needed for one-to-many tree structures, and support for joins is poor. If the database itself does not support joins, you have to emulate a join in application code by making multiple queries (poor performance) (foreign key i.e. document reference concept is there, but have limitation that you cannot refer directly to a nested item within a document)
* **Schema Flexibility** - schema-on-read, like dynamic (runtime) type checking (Python) where schema is handled at application layer while reading (client have no guarantees what data they will get; called schemaless but that's misleading).
* **Locality** - JSON representation has better _locality_ than the multi-table SQL schema. All the relevant information is in one place, and one query is sufficient. The database typically needs to load the entire document, even if you access only a small portion of it. On updates, the entire document usually needs to be rewritten, it is recommended that you keep documents fairly small.
* **Example** - MongoDB, CouchDB, RethinkDB, Espresso

**PostgreSQL does support storing JSON documents in a single cell. RethinkDB supports relational-like joins in its query language. Relational and document databases are becoming more similar over time**

**The network model**
* Standardised by a committee called the Conference on Data Systems Languages (CODASYL) model was a generalisation of the hierarchical model. In the tree structure of the hierarchical model, every record has exactly one parent, while in the network model, a record could have multiple parents.
* The links between records are like pointers in a programming language. The only way of accessing a record was to follow a path from a root record called _access path_ (very bad)
* A query in CODASYL was performed by moving a cursor through the database by iterating over a list of records
  * Several different paths can lead to same record and hence programmer had to keep track of these different access paths.
  * If a record had multiple parents, application code had to keep track of all the various relationships.
* Moved away from access path &#8594; query optimiser automatically decides which parts of the query to execute in which order, and which indexes to use (the access path) [for all 3: relational, document and network model)

### Query languages for data
* SQL is a _declarative_ query language. In an _imperative language_ (like IMS and CODASYL) you tell the computer to perform certain operations in order.
* In a declarative query language you just specify the pattern of the data you want, but not _how_ to achieve that goal.
* A declarative query language hides implementation details of the database engine, making it possible for the database system to introduce performance improvements without requiring any changes to queries.
* Declarative languages often lend themselves to parallel execution while imperative code is very hard to parallelise across multiple cores because it specifies instructions that must be performed in a particular order. Declarative languages specify only the pattern of the results, not the algorithm that is used to determine results.
* **Declarative queries on the web** - In a web browser, using declarative CSS styling is much better than manipulating styles imperatively in JavaScript. Declarative languages like SQL turned out to be much better than imperative query APIs.

#### MapReduce querying

_MapReduce_ is a "programming model" (not some software) for processing large amounts of data in bulk across many machines, popularised by Google. Apache Hadoop leverages it. Also, MongoDB and CouchDB offers a MapReduce solution. MapReduce is neither imperative nor declarative, but something in between.

```js
db.observations.mapReduce(
    function map() { // 2 
        var year  = this.observationTimestamp.getFullYear();
        var month = this.observationTimestamp.getMonth() + 1;
        emit(year + "-" + month, this.numAnimals); // 3
    },
    function reduce(key, values) { // 4
        return Array.sum(values); // 5
    },
    {
        query: { family: "Sharks" }, // 1
        out: "monthlySharkReport" // 6
    }
);
```

The `map` and `reduce` functions must be _pure_ functions, they cannot perform additional database queries and they must not have any side effects. These restrictions allow the database to run the functions anywhere, in any order, and rerun them on failure.

A usability problem with MapReduce is that you have to write two carefully coordinated functions. A declarative language offers more opportunities for a query optimiser to improve the performance of a query. For there reasons, MongoDB 2.2 added support for a declarative query language called _aggregation pipeline_

```js
db.observations.aggregate([
    { $match: { family: "Sharks" } },
    { $group: {
        _id: {
            year:  { $year:  "$observationTimestamp" },
            month: { $month: "$observationTimestamp" }
        },
        totalAnimals: { $sum: "$numAnimals" }
    } }
]);
```

### Graph-like data models

* If many-to-many relationships are very common in your application, it becomes more natural to start modelling your data as a graph.
* A graph consists of _vertices_ (_nodes_ or _entities_) and _edges_ (_relationships_ or _arcs_).
* Well-known algorithms can operate on these graphs, like the shortest path between two points, or popularity of a web page.
* There are several ways of structuring and querying the data.
    * Property graph model
        * Neo4j &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&#8594; Cypher &nbsp; &nbsp; &nbsp;(declarative)
        * Titan &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &#8594; Gremlin &nbsp; &nbsp; (imperative)
        * Infinite Graph  &#8594; Gremlin &nbsp; &nbsp; (imperative)
    * Triple-store_ model
        * Datomic &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &#8594; Datalog &nbsp; &nbsp; (declarative)
        * AllegroGraph &nbsp;&#8594; SPARQL &nbsp; &nbsp; (declarative)

#### Property graphs

Each vertex consists of:
* Unique identifier
* Collection of properties (key-value pairs)
* Outgoing edges (optional)
* Incoming edges (optional)

Each edge consists of:
* Unique identifier
* Vertex at which the edge starts (_tail vertex_)
* Vertex at which the edge ends (_head vertex_)
* Label to describe the kind of relationship between the two vertices
* A collection of properties (key-value pairs)

Graphs provide a great deal of flexibility for data modelling. Graphs are good for evolvability.

```sql
CREATE TABLE vertices {
  vertex_id    integer  PRIMARY KEY,
  properties   json
};

CREATE TABLE edes {
  edge_id      integer  PRIMARY KEY,
  tail_vertex  integer  REFERENCES vertices {vertex_id},
  head_vertex  integer  REFERENCES vertices {vertex_id},
  label        text
  properties   json
};

CREATE INDEX edge_tails ON edges {tail_vertex};
CREATE INDEX head_tails ON edges {head_vertex};
```

**Cypher**
* _Cypher_ is a declarative language for property graphs created by Neo4j
* Graph queries in SQL. In a relational database, you usually know in advance which joins you need in your query. In a graph query, the number if joins is not fixed in advance. In Cypher `:WITHIN*0..` expresses "follow a `WITHIN` edge, zero or more times" (like the `*` operator in a regular expression). This idea of variable-length traversal paths in a query can be expressed using something called _recursive common table expressions_ (the `WITH RECURSIVE` syntax).

```sql
MATCH
  (person) -[:BORN_IN]-> () -[WITHIN*0..]-> (us:Location {name:"United States"}),
  (person) -[:LIVES_IN]-> () -[WITHIN*0..]-> (eu:Location {name:"Europe"})
RETURN person.name
```

#### Triple-stores

* In a triple-store, all information is stored in the form of very simple three-part statements: _subject_, _predicate_, _object_ (eg: _Jim_, _likes_, _bananas_).
* Subject is equivalent to vertex in graph
* Object is one of two
    * value in a primitive datatype (string or number). In that case, predicate and object of triple are equivalent to the key and value of a property on the subject. For example `{lucy, age, 33}`
    * another vertex in graph. In that case, predicate is edge in the graph and subject is tail vertex and object is head vertex. For example `{lucy, marriedTo, john}`
* _SPARQL_ is a query language for triple-stores using the Resource Description Framework (RDF) data model. RDF was intended as a mechanism for different websites to publish data in a consistent format, allowing data from different websites to be automatically combined into a web of data -- a kind of internet wide "database of everything"

#### The foundation: Datalog
* _Datalog_ provides the foundation that later query languages build upon. Its model is similar to the triple-store model, generalised a bit. Instead of writing a triple (_subject_, _predicate_, _object_), we write as _predicate(subject, object)_.
* Query language for Datomic
* Cascalog is the Datalog implementation for querying large datasets in Hadoop. For Hadoop, SQL like query languages are HiveQL (Apache Hive), Pig Latin (Apache Pig), Spark SQL (Apache Spark), Impala (Apache Impala) and HBase QL (Apache HBase). Cascalog is a non-SQL language (but still declarative).
    * Hive is relational data warehouse
    * Pig is data flow platform for writing scripts for data processing
    * Spark is cluster computing system
    * Impala is query engine
    * HBase is columnar data warehouse
    * Hadoop is distributed file system and batch processing framework
* Datalog is a subset of Prolog
* We define _rules_ that tell the database about new predicates and rules can refer to other rules, just like functions can call other functions or recursively call themselves.
* Rules can be combined and reused in different queries. It's less convenient for simple one-off queries, but it can cope better if your data is complex.

All three models are widely used today. One model can be emulated in terms of another model -- for example, graph data can be represented in a relational database, but the result is often awkward. That's why we have different systems for different purposes, not a single one-size-fits-all solution.

---
---

## Storage and retrieval

> A _log_ is an append-only sequence of records
* Databases need to do two things: store the data and give the data back to you.
* Many databases use a _log_, which is append-only data file. Real databases have more issues to deal with though (concurrency control, reclaiming disk space so the log doesn't grow forever and handling errors and partially written records).
* In order to efficiently find the value for a particular key, we need a different data structure: an _index_. An index is an _additional_ structure that is derived from the primary data.
* Well-chosen indexes speed up read queries but every index slows down writes. That's why databases don't index everything by default, but require you to choose indexes manually using your knowledge on typical query patterns.

### Hash indexes

* Key-value stores are quite similar to the _dictionary_ type (hash map or hash table).
* Let's say our storage consists only of appending to a file. The simplest indexing strategy is to keep an in-memory hash map (storing hash map on disk is super expensive due to writes at random places in hashmap) where "every key" is mapped to a byte offset in the data file. Whenever you append a new key-value pair to the file, you also update the hash map to reflect the offset of the data you just wrote.
* As we only ever append to a file, so how do we avoid eventually running out of disk space?
    * **A good solution is to break the log into segments of certain size by closing the segment file when it reaches a certain size, and making subsequent writes to a new segment file. We can then perform _compaction_ on these segments.**
    * Compaction means throwing away duplicate keys in the log, and keeping only the most recent update for each key.
    * We can also merge several segments together at the same time as performing the compaction.
    * Segments are never modified after they have been written, so the merged segment is written to a new file.
    * Merging and compaction of frozen segments can be done in a background thread. After the merging process is complete, we switch read requests to use the new merged segment instead of the old segments, and the old segment files can simply be deleted.
* "Each segment" now has its own in-memory hash table, mapping keys to file offsets.
* In order to find a value for a key, we first check the most recent segment hash map; if the key is not present we check the second-most recent segment and so on. The merging process keeps the number of segments small, so lookups don't need to check many hash maps.

Example:
* **Bitcask (the default storage engine in Riak)** does it like that. The only requirement it has is that all the keys fit in the available RAM. Values can use more space than there is available in memory, since they can be loaded from disk. A storage engine like Bitcask is well suited to situations where the value for each key is updated frequently. There are a lot of writes, but there are not too many distinct keys, you have a large number of writes per key, but it's feasible to keep all keys in memory.
* Redis, Memcached


Some issues that are important in a real implementation:
* File format. It is simpler to use binary format.
* Deleting records. Append special deletion record to the data file (_tombstone_) that tells the merging process to discard previous values.
* Crash recovery. If restarted, the in-memory hash maps are lost. You can recover from reading each segment but that would take long time. Bitcask speeds up recovery by storing a snapshot of each segment hash map on disk.
* Partially written records. The database may crash at any time. Bitcask includes checksums allowing corrupted parts of the log to be detected and ignored.
* Concurrency control. As writes are appended to the log in a strictly sequential order, a common implementation is to have a single writer thread. Segments are immutable, so they can be read concurrently by multiple threads.

Append-only design turns out to be good for several reasons:
* Appending and segment merging are sequential write operations, much faster than random writes, especially on magnetic spinning-disks.
* Concurrency and crash recovery are much simpler.
* Merging old segments avoids files getting fragmented over time.

Hash table has its limitations too:
* The hash table must fit in memory. It is difficult to make an on-disk hash map perform well.
* Range queries are not efficient.

### SSTables and LSM-Trees

We introduce a new requirement to segment files
* we require that the sequence of key-value pairs is _sorted by key_.
* we require that each key only appears once within each merged segment file

We call this _Sorted String Table_, or _SSTable_. SSTables have few big advantages over log segments with hash indexes
1. **Merging segments is simple and efficient** (we can use algorithms like _mergesort_). When multiple segments contain the same key, we can keep the value from the most recent segment and discard the values in older segments.
2. **You no longer need to keep an index of all the keys in memory.**
   * For a key like `handiwork`, when you know the offsets for the keys `handback` and `handsome`, you know `handiwork` must appear between those two. You can jump to the offset for `handback` and scan from there until you find `handiwork`, if not, the key is not present.
   * You still need an in-memory index to tell you the offsets for some of the keys. One key for every "few kilobytes" of segment file is sufficient.
4. Since read requests need to scan over several key-value pairs in the requested range anyway, **it is possible to group those records into a block and compress it** before writing it to disk.

How do we get the data sorted in the first place? With red-black trees or AVL trees, you can insert keys in any order and read them back in sorted order.
* When a write comes in, add it to an in-memory balanced tree structure (_memtable_).
* When the memtable gets bigger than some threshold (megabytes), write it out to disk as an SSTable file. Writes can continue to a new memtable instance.
* On a read request, try to find the key in the memtable, then in the most recent on-disk segment, then in the next-older segment, etc (sparse index + bloom filter)
    * LSM-tree algorithm can be slow when looking up keys that don't exist in the database.
    * To optimise this, storage engines often use additional _Bloom filters_ (a memory-efficient probabilistic data structure for approximating the contents of a set).
        * Key present? No &#8594; definitely not present
        * Key present? Yes &#8594; May or may not present
* From time to time, run merging and compaction in the background to discard overwritten and deleted values.

##### Bloom filter algorithm
* start with array of m bits initialized to 0s
* put key to bloom filter
  * for a key, hash it using k different hash functions (all spitting results from 0 to m-1)
  * change all the bits corresponding to the results of previous equation to 1 in the bloom filter. Example: if we used 3 hash functions and they spit values 11, 23 and 28, then we will switch on these 3 bits in our m sized array.
* check if key is present in bloom filter
  * compute k hashes for key using k hash functions
  * if any of k positions in the m bit array is 0 --> return false (definitely not present)
  * if all k positions in the m bit array are 1 --> return true (may or may not be present)

Other points
* If the database crashes, the most recent writes are lost. We can keep a separate log on disk to which every write is immediately appended. That log is not in sorted order, but that doesn't matter, because its only purpose is to restore the memtable after crash. Every time the memtable is written out to an SSTable, the log can be discarded.
* Storage engines that are based on this principle of merging and compacting sorted files are often called LSM structure engines (Log Structure Merge-Tree).
* Great for range queries
* Great for write heavy applications because disk writes are always sequential (WAL is sequential + dumping all values of memtable to disk is sequential)
* There are also different strategies to determine the order and timing of how SSTables are compacted and merged. Mainly two _leveled_ and _size-tiered_ compaction.
    * In leveled compaction, the key range is split up into smaller SSTables and older data is moved into separate "levels", which allows the compaction to use less disk space. LevelDB, RocksDB and BigTable use leveled compaction
    * In size-tiered compaction, newer and smaller SSTables are successively merged into older and larger SSTables. HBase use size-tiered
    * and Cassandra supports both.  

Examples
* LevelDB (can be used in Riak in place of Bitcask). RocksDB (Meta). BigTable (Google). HBase. Cassandra.
* Lucene, an indexing engine for full-text search used by Elasticsearch and Solr, uses a similar method for storing its _term dictionary_.

### B-trees

* This is the most widely used indexing structure. B-tress keep key-value pairs sorted by key, which allows efficient key-value lookups and range queries.
* Hash/LSM break the database down into variable-size _segments_ typically several megabytes or more. B-trees break the database down into fixed-size _blocks_ or _pages_, traditionally 4KB (aligns closely to underlying hardware)
* Tree Structure:
    * One page is designated as the _root_ and you start from there. The page contains several keys and references to child pages.
    * If you want to update the value for an existing key in a B-tree, you search for the leaf page containing that key -- now leaf page either contains the value itself or reference to the page with the value -- we change the value in this "whatever" page, and write the page back to disk.
    * If you want to add new key, find the page and add it to the page. If there isn't enough free space in the page to accommodate the new key, it is split in two half-full pages, and the parent page is updated to account for the new subdivision of key ranges.
    * Trees remain _balanced_. A B-tree with _n_ keys always has a depth of _O_(log _n_). Number of references to child pages in one page of the B-tree is called the branching factor. Four level tree of 4 KB pages with branching factor of 500 can store up to 256 TB.
* The basic underlying write operation of a B-tree is to overwrite a page on disk with new data. It is assumed that the overwrite does not change the location of the page, all references to that page remain intact. This is a big contrast to log-structured indexes such as LSM-trees, which only append to files.
* Crash Recovery:
    * Some operations require several different pages to be overwritten. When you split a page, you need to write the two pages that were split, and also overwrite their parent. If the database crashes after only some of the pages have been written, you end up with a corrupted index.
    * It is common to include an additional data structure on disk: a _write-ahead log_ (WAL, also know as the _redo log_) for crash recovery + ensuring the write (which can be multiple page writes) atomic.
* Careful concurrency control is required if multiple threads are going to access, typically done protecting the tree internal data structures with _latches_ (lightweight locks).

Optimizations
* During writes, instead of overwriting pages -- do copy-on-write i.e. modified page is written to a different location and new version of parent pages in the tree are created. Great for concurrency control and indexes for snapshot isolation / repeatable read. LMDB database does this.
* Abbreviate keys (this optimization is called B+ trees)
* Try to put leaf nodes sequentially on disk for more efficient range queries
* References to siblings can be kept

### B-trees vs LSM-trees

LSM-trees are typically faster for writes, whereas B-trees are thought to be faster for reads. Reads are typically slower on LSM-tress as they have to check several different data structures and SSTables at different stages of compaction.

| LSM                                      | BTree                                                          |
| -----------                              | -----------                                                    |
| High write throughput as sequential writes                 | Lower write throughput as writes involve several pages and overwriting                                         |
| Compressed better (small disk space)     | Fragmentation leaves disk space empty in multiple pages        |
| Lower read throughput                    | Higher read throughput                                         |
| Compaction happens which can sometimes interfere with the performance of ongoing reads and writes (especially at higher percentiles) | No compaction, so no such problem |
| At high write throughput, disk resource of writing is shared with compaction. If compaction not able to keep up, OOM. | No compaction, so no such problem |
| Same key may have multiple copies on different segments     | Each key exists in exactly one place in the index. This offers strong transactional semantics. Transaction isolation is implemented using locks on ranges of keys, and in a B-tree index, those locks can be directly attached to the tree.     |

### Other indexing structures
* We've only discussed key-value indexes, which are like _primary key_ index (one which uniquely identifies a row, or a document or a vertex). There are also _secondary indexes_.
* A secondary index can be easily constructed from a key-value index. The main difference is that in a secondary index, the indexed values are not necessarily unique. There are two ways of doing this: making each value in the index a list of matching row identifiers or by making a each entry unique by appending a row identifier to it.
* Both LSM and BTrees can be used.
* Clustered index: store the indexed row directly within the index (for example, primary key in InnoDB storage engine for MySQL is always clustered index)
* Compromise between clustered and nonclustered (storing only references to the data within the index) is known as covered index or index with included columns which stores some of the table's columns within the index.
* Say want to get restaurants b/w 2 latitudes and 2 longitudes
    * Standard LSM or B-tree index is not able to answer that kind of query efficiently -- it can give you either all the restaurants in a range of latitudes (but at any longitude) or all the restuarants in a range of longitudes (but at any latitude), but not both simultaneously.
    * Specialized index trees like R trees are used.
    * If say want to search all the ovservation during the year 2013 where temperature was between 25 and 30 degree celsius, can create an index on (day, temperature) and then use R trees -- this technique is called HyperDex.

### Keeping everything in memory
* Disks have two significant advantages: they are durable, and they have lower cost per gigabyte than RAM.
* It's quite feasible to keep them entirely in memory, this has lead to _in-memory_ databases.
* When an in-memory database is restarted, it needs to reload its state, either from disk or over the network from a replica. The disk is merely used as an append-only log for durability, and reads are served entirely from memory.
* Key-value stores, such as Memcached are intended for cache only, it's acceptable for data to be lost if the machine is restarted. Memcached is purely memory based (no disk at all).
* Other in-memory databases aim for durability, with special hardware, writing a log of changes to disk, writing periodic snapshots to disk or by replicating in-memory sate to other machines. Redis provide weak durability by writing every operation to append only log to disk + periodic snapshot to disk. 
* In-memory databases can be faster because they can avoid the overheads of encoding in-memory data structures in a form that can be written to disk.
* Another interesting area is that in-memory databases may provide data models that are difficult to implement with disk-based indexes (sets, priority queues, etc)

### Transaction processing or analytics?

* A _transaction_ is a group of reads and writes that form a logical unit, this pattern became known as _online transaction processing_ (OLTP).
* _Data analytics_ has very different access patterns. A query would need to scan over a huge number of records, only reading a few columns per record, and calculates aggregate statistics. These queries are often written by business analysts, and fed into reports. This pattern became known for _online analytics processing_ (OLAP).


### Data warehousing
* Also called analytics database. A _data warehouse_ is a separate database that analysts can query to their heart's content without affecting OLTP operations. It contains read-only copy of the data in all various OLTP systems in the company. Data is extracted out of OLTP databases (through periodic data dump or a continuous stream of update), transformed into an analysis-friendly schema, cleaned up, and then loaded into the data warehouse (process _Extract-Transform-Load_ or ETL).
* A data warehouse is most commonly relational, but the internals of the systems can look quite different because of different access patterns.
* Amazon RedShift is hosted version of ParAccel.
* Data warehouses are used in fairly formulaic style known as a _star schema_.
    * Central table is "fact table" in which each row is a fact -- facts are captured as individual events, because this allows maximum flexibility of analysis later. The fact table can become extremely large.
    * Some of the columns in the fact are attributes, others are foreign key references to other tables called "dimension tables". Dimensions represent the _who_, _what_, _where_, _when_, _how_ and _why_ of the event.
    * The name "star schema" comes from the fact than when the table relationships are visualised, the fact table is in the middle, surrounded by its dimension tables, like the rays of a star.
    * Fact tables often have over 100 columns, sometimes several hundred. Dimension tables can also be very wide.
    * Variation of this template is _snowflake_ schema where dimensions are further broken down into subdimensions.

### Column-oriented storage
* Data warehouses are usually stored using column oriented storage which usually are implemented using LSM. Here we talk about the fact table only as it is the biggest.
* In a row-oriented storage engine, when you do a query that filters on a specific field, the engine will load all those rows with all their fields into memory, parse them and filter out the ones that don't meet the requirement. This can take a long time.
* _Column-oriented storage_ is simple: don't store all the values from one row together, but store all values from each _column_ together instead. If each column is stored in a separate file, a query only needs to read and parse those columns that are used in a query, which can save a lot of work. Relies on each column file containing the rows in the same order.
* Column-oriented storage often lends itself very well to compression as the sequences of values for each column look quite repetitive, which is a good sign for compression.
    * A technique that is particularly effective in data warehouses is _bitmap encoding_ (or bitmap indexing)
    * If say for a column, there are `n` distinct possible values only, then for each of these `n` value create a bit array of `r` length (where `r` is the number of rows) -- each bit is 0 or 1, where 1 denotes that this row index has the value
    * We can further do _run length encoding_ to compress these bitmaps (111100 can be written as 42)

```sql
// This is called vectorized processing.
WHERE product_sk IN (30, 68, 69) # load the three bitmaps for 30, 68, 69 from product_sk column file and take bitwise OR.
WHERE product_sk = 30 AND store_sk = 3 # load bitmap for product_sk=30 and store_sk=3, and calculate bitwise AND.
```

> Cassandra and HBase have a concept of _column families_, which they inherited from Bigtable.

* **Column-oriented storage, compression, and sorting helps to make read queries faster and make sense in data warehouses, where most of the load consist on large read-only queries run by analysts. The downside is that writes are more difficult.**
* An update-in-place approach, like B-tree use, is not possible with compressed columns. If you insert a row in the middle of a sorted table, you would most likely have to rewrite all column files.
* Clever extension for sorting the data was adopted by **Vertica**. Different queries benefit from different sort orderes, so why not store the same data sorted in several different ways? Data needs to be replicated to multiple machines anyway, so that you don't lose data inf one machine fails. You might as well store that redundant data sorted in different ways so that when you're processing a query, you can use the version that best fits the query pattern.
* **Materialised views**
    * It's worth mentioning _materialised aggregates_ as some cache of the counts and the sums that queries use most often.
    * A way of creating such a cache is with a _materialised view_, on a relational model this is usually called a _virtual view_: a table-like object whose contents are the results of some query. A materialised view is an actual copy of the query results, written in disk, whereas a virtual view is just a shortcut for writing queries.
    * When the underlying data changes, a materialised view needs to be updated, because it is denormalised copy of the data. Database can do it automatically, but writes would become more expensive.
    * A common special case of a materialised view is know as a _data cube_ or _OLAP cube_, a grid of aggregates grouped by different dimensions.

---
---

## Encoding and evolution

* Change to an application's features also requires a change to data it stores.
* Relational databases conforms to one schema although that schema can be changed, there is one schema in force at any point in time. **Schema-on-read (or schemaless) contain a mixture of older and newer data formats.**
* In large applications changes don't happen instantaneously. You want to perform a _rolling upgrade_ and deploy a new version to a few nodes at a time, gradually working your way through all the nodes without service downtime.

Old and new versions of the code, and old and new data formats, may potentially all coexist. We need to maintain compatibility in both directions
* Backward compatibility, newer code can read data that was written by older code.
* Forward compatibility, older code can read data that was written by newer code.

### Formats for encoding data

Two different representations:
* In memory
* When you want to write data to a file or send it over the network, you have to encode it

Thus, you need a translation between the two representations. In-memory representation to byte sequence is called _encoding_ (_serialisation_ or _marshalling_), and the reverse is called _decoding_ (_parsing_, _deserialisation_ or _unmarshalling_). Note: encoding has nothing to do with encryption.

Programming languages come with built-in support for encoding in-memory objects into byte sequences, but is usually a bad idea to use them. Precisely because of a few problems.
* Often tied to a particular programming language. Java has `java.io.Serializable`, Ruby has `Marshal`, Python has `pickle`.
* The decoding process needs to be able to instantiate arbitrary classes and this is frequently a security hole.
* Versioning (forward and backward compatability is not thought out)
* Efficiency is bad (example: Java's in built serialization is very bad performant)

Standardised encodings can be written and read by many programming languages.

JSON, XML, and CSV are human-readable (**textual**) and popular specially as data interchange formats, but they have some subtle problems:
* Ambiguity around the encoding of numbers (numbers and strings are indistinguishable in XML and CSV), dealing with large numbers and precision problems with floating numbers (in JSON)
* Support of Unicode character strings, but no support for binary strings. People get around this by encoding binary data as Base64, which increases the data size by 33%.
* There is "optional" schema support for both JSON and XML
* CSV does not have any schema, and is quite a vague format (what happens if a value contains a comma or a newline character?)
* Not compact!

#### Binary encoding

JSON is less verbose than XML, but both still use a lot of space compared to binary formats. There are binary encodings for JSON (MessagePack, BSON, BJSON, UBJSON, BISON and Smile), similar thing for XML (WBXML and Fast Infoset).

```sql
# As there is no schema for JSON and XML, we have to store "userName", "favouriteNumber" and "interests" in encoding
# (even if we use binary encoding formats like MessagePack)
{
   "userName": "Naman",
   "favouriteNumber": 11,
   "interests": ["sleeping", "eating"]
}
```

**Apache Thrift (Meta) and Protocol Buffers (protobuf) (Google) are binary encoding libraries.** Both made open source in 2007-08.

Thrift offers two different protocols:
* **BinaryProtocol**, there are no field names like `userName`, `favouriteNumber`. Instead the data contains _field tags_, which are numbers (`1`, `2`)
* **CompactProtocol**, which is equivalent to BinaryProtocol but it packs the same information in less space. It packs the field type and the tag number into the same byte and using variable length integers.

Protocol Buffers are very similar to Thrift's CompactProtocol, bit packing is a bit different and that might allow smaller compression.

Schemas inevitable need to change over time (_schema evolution_), how do Thrift and Protocol Buffers handle schema changes while keeping backward and forward compatibility changes?

* **Forward compatible support**. As with new fields you add new tag numbers, old code trying to read new code, it can simply ignore not recognised tags.
* **Backwards compatible support**. As long as each field has a unique tag number, new code can always read old data. Every field you add after initial deployment of schema must be optional or have a default value.

Removing fields is just like adding a field with backward and forward concerns reversed. You can only remove a field that is optional, and you can never use the same tag again.

What about changing the data type of a field? There is a risk that values will lose precision or get truncated.

##### Avro

* Apache Avro is another binary format that has two schema languages, one intended for human editing (Avro IDL), and one (based on JSON) that is more easily machine-readable. There are no tag numbers.

```sql
record Person {
   string               userName;
   union [null, long]   favouriteNumber = null;
   array<strings>       interests;
}
```

* You go go through the fields in the order they appear in the schema and use the schema to tell you the datatype of each field. Any mismatch in the schema between the reader and the writer would mean incorrectly decoded data.
* What about schema evolution?
     * When an application wants to encode some data, it encodes the data using whatever version of the schema it knows (_writer's schema_).
     * When an application wants to decode some data, it is expecting the data to be in some schema (_reader's schema_).
* In Avro the writer's schema and the reader's schema _don't have to be the same_. They just have to be compatible. The Avro library resolves the differences by looking at the writer's schema and the reader's schema.
* If the code reading the data encounters a field that appears in the writer's schema but not in the reader's schema, it is ignored. If the code reading the data expects some field, but the writer's schema does not contain a field of that name, it is filled in with a default value declared in the reader's schema.
* Forward compatibility means you can have a new version of the schema as writer and an old version of the schema as reader. Conversely, backward compatibility means that you can have a new version of the schema as reader and an old version as writer.
* To maintain compatibility, you may only add or remove a field that has a default value.
* If you were to add a field that has no default value, new readers wouldn't be able to read data written by old writers.
* Changing the datatype of a field is possible, provided that Avro can convert the type. Changing the name of a field is tricky (backward compatible but not forward compatible).
* The schema is identified encoded in the data.
     * In a large file with lots of records, the writer of the file can just include the schema (full schema and not just version number) at the beginning of the file (like sending a whole bunch of records in Hadoop)
     * On a database with individually written records, you cannot assume all the records will have the same schema, so you have to include a version number at the beginning of every encoded record.
     * While sending records over the network, you can negotiate the schema version on connection setup.
* Avro is friendlier to _dynamically generated schemas_ (dumping into a file the database). You can fairly easily generate an Avro schema in JSON. If the database schema changes, you can just generate a new Avro schema for the updated database schema and export data in the new Avro schema. By contrast with Thrift and Protocol Buffers, every time the database schema changes, you would have to manually update the mappings from database column names to field tags
  * Very useful for ETL processes. If we want to export all rows from a database to HDFS, we want to fully automate this process. With Thrift / Protocol Buffers, we would have somebody manually convert database schema to a serialization schema to follow rules of only increasing tag numbers (database is not aware of this). 

---

Although textual formats such as JSON, XML and CSV are widespread, binary encodings based on schemas are also a viable option. As they have nice properties:
* Can be much more compact, since they can omit field names from the encoded data.
* Schema is a valuable form of documentation, required for decoding, you can be sure it is up to date.
* Database of schemas allows you to check forward and backward compatibility changes.
* Generate code from the schema is useful, since it enables type checking at compile time.

### Modes of dataflow

Different process on how data flows between processes

#### Via databases

* The process that writes to the database encodes the data, and the process that reads from the database decodes it.
* A value in the database may be written by a _newer_ version of the code, and subsequently read by an _older_ version of the code that is still running.
* When a new version of your application is deployed, you may entirely replace the old version with the new version within a few minutes. The same is not true in databases, the five-year-old data will still be there, in the original encoding, unless you have explicitly rewritten it. _Data outlives code_.
* Rewriting (_migrating_) data into a new schema is certainly possible, but it is expensive, most relational databases allow simple schema changes, such as adding a new column with a `null` default value without rewriting existing data. When an old row is read, the database fills in `null`s for any columns that are missing.
* When you create backup of DB, or create data warehouses, then perhaps you can copy the data with the latest schema even though the main DB is a mixture of various schema versions.
* LinkedIn's document database Espresso uses Avro for storage.

#### Via service calls

* You have processes that need to communicate over a network of _clients_ and _servers_. The API exposed by the server is called _service_.
* Services are similar to databases, each service should be owned by one team and that team should be able to release versions of the service frequently, without having to coordinate with other teams. We should expect old and new versions of servers and clients to be running at the same time, and hence data encodings used by client and server must be compatible across versions of the service API.

**Hyper Text Transfer Protocol (HTTP)**
* HTTP is
     * apps or browsers making call to services.
     * intra-org communications (software that supports this kind of use case is sometimes called middleware)
     * inter-org communications
* Common HTTP methods include GET (retreive data), POST (submit data to be processed), PUT (update data), DELETE (remove data).

| REST      | SOAP |
| ----------- | ----------- |
| Representational State Transfer      | Simple Object Access Protocol       |
| not a protocol, but rather a design philosophy   | a protocol        |
| loose guidelines     | pre defined rules     |
| stateless i.e all requests are independent    | stateless and stateful both types available     |
| JSON, XML, YAML   | XML only        |
| HTTP only      | HTTP, SMTP, etc       |
| Best for web APIs and services needing a lightweight, scalable, and easily maintainable solution | Ideal for enterprise-level services needing robust security, ACID-compliance, and reliable transactions. |

Here are some other differences between gRPC and REST:

* Protocol: gRPC uses HTTP/2 for transport, while REST typically uses HTTP/1.1.
* Data format: gRPC employs Protocol Buffers for serialization, while REST usually leverages JSON or XML.
* API design: gRPC is based on the RPC (Remote Procedure Call) paradigm, while REST follows the architectural constraints of the Representational State Transfer model.
* Streaming: gRPC supports bidirectional streaming, whereas REST is limited to request-response communication patterns.

##### Microservices essentials

Docker (quick detour)
* many things to deploy, each having unique requirements (one in Java, another in Python, etc)
* containers allow us to package our dev environments into an easy to run package
* applications are packaged into containers (each container has all the dependencies of application, versions, environment, etc)
* when deploying to cloud, you typically preconfigure the capacity of virtual machine
  * if we want microservice running on each virtual machine, we would have to figure out capacity of each microservice needed in advanced -- would likely result in a lot of bad utilization
  * Docker allows running multiple containers on just one virtual machine, giving each container the ability to take resources as they need (one service can be heavily used, while other has little traffic)
 
Kubernetes
* like sigma in Google
* manages various applications on diff hosts (orchestrator)
 

**Remote procedure calls (RPC)**

RPC tries to make a request to a remote network service look the same as calling a function or method in your programming language, it seems convenient at first but the approach is flawed:
* A network request is unpredictable (request drop)
* A function call will either return result or raise exception or won't return result (due to infinite loop or crash), but network request have another problem: it may return without a result, due a _timeout_
* Retrying will cause the action to be performed multiple times, unless you build a mechanism for deduplication (_idempotence_).
* A network request is much slower than a function call, and its latency is wildly variable.
* Parameters need to be encoded into a sequence of bytes that can be sent over the network and becomes problematic with larger objects.
* The RPC framework must translate datatypes from one language to another, not all languages have the same types.

**There is no point trying to make a remote service look too much like a local object in your programming language, because it's a fundamentally different thing.**

* gRPC uses Protocol Buffers, Finagle uses Thrift, Rest.li uses JSON
* New generation of RPC frameworks are more explicit about the fact that a remote request is different from a local function call. Finagle and Rest.li use _features_ (_promises_) to encapsulate asynchronous actions.
* RESTful API has some significant advantages like being good for experimentation and debugging.
* REST seems to be the predominant style for public APIs. The main focus of RPC frameworks is on requests between services owned by the same organisation, typically within the same datacenter.

#### Via asynchronous message passing

In an _asynchronous message-passing_ systems, a client's request (usually called a _message_) is delivered to another process with low latency. The message goes via an intermediary called a _message broker_ (_message queue_ or _message-oriented middleware_) which stores the message temporarily. This has several advantages compared to direct RPC:
* It can act as a buffer if the recipient is unavailable or overloaded
* It can automatically redeliver messages to a process that has crashed and prevent messages from being lost
* It avoids the sender needing to know the IP address and port number of the recipient (useful in a cloud environment)
* It allows one message to be sent to several recipients
* **Decouples the sender from the recipient**

* The communication happens only in one direction. The sender doesn't wait for the message to be delivered, but simply sends it and then forgets about it (_asynchronous_).
* Open source implementations for message brokers are RabbitMQ, ActiveMQ, HornetQ, NATS, and Apache Kafka.
* One process sends a message to a named _queue_ or _topic_ and the broker ensures that the message is delivered to one or more _consumers_ or _subscribers_ to that queue or topic.
* Message brokers typically don't enforce a particular data model, you can use any encoding format.

##### Distributed actor frameworks
An _actor model_ is a programming model for concurrency in a single process. Rather than dealing with threads (and their complications), logic is encapsulated in _actors_. Each actor typically represent one client or entity, it may have some local state (not share with other actors), and it communicates with other actors by sending and receiving asynchronous messages. Message deliver is not guaranteed. Since each actor processes only one message at a time, it doesn't need to worry about threads.

In _distributed actor frameworks_, this programming model is used to scale an application across multiple nodes. It is better than say RPC which gives the false impression that it is making a local function call. But here, even in _actor model_ the message delivery is not guaranteed, which is consistent with what we deal with distributed systems (_distributed actor frameworks_). It basically integrates a message broker and the actor model into a single framework.

Examples of _distributed actor frameworks_:
* _Akka_ uses Java's built-in serialisation by default, which does not provide forward or backward compatibility. You can replace it with something like Protocol Buffers and the ability to do rolling upgrades.
* _Orleans_ by default uses custom data encoding format that does not support rolling upgrade deployments -- so you have to deploy new version to new cluster, move traffic from old to new cluster and shutdown the old one.
* In _Erlang OTP_ it is surprisingly hard to make changes to record schemas.

---
---

## Replication

Reasons why you might want to replicate data:
* To keep data geographically close to your users (reduce latency)
* Increase availability
* Increase read throughput

The difficulty in replication lies in handling _changes_ to replicated data. Popular algorithms for replicating changes between nodes: _single-leader_, _multi-leader_, and _leaderless_ replication.

##### Retrying State Updates
* say on social media website, you want to "like" a post made by some user. The request may be lost or the acknowledgement may be lost. If we are not careful, the retry could lead to the request being processed multiple times, leading to an incorrect state in the database.
* ![Retrying state updates](/metadata/retrying_state_updates.png)
* or say we want to deduct 100 rupees, a retry shouldn't cause it to be deducted twice.
* Solution
  * **Deduplicate requests**:  However, in a crash-recovery system model, this requires storing requests (or some metadata about requests, such as a vector clock) _in stable storage_, so that duplicates can be accurately detected even after a crash
  * **Idempotency**
    * Make requests idempotent
    * Incrementing a counter is not idempotent, but adding an element to a set is
    * Therefore, if a counter is required (as in the number of likes), it might be better to actually maintain the set of elements in the database, and to derive the counter value from the set by computing its cardinality.
    * allows an update to have "exactly-once semantics" i.e. the update may actually be applied multiple times, but the effect is the same as if it had been applied exactly once
    * idempotence has a limitation that becomes apparent when there are multiple updates in progress
    * Example 1
      * ![Idempotency limitation 1](/metadata/idempotency_limitation_1.png)
      * Client 1 adds a user ID to the set of likes to the post, but ack is lost. Client 2 reads the set of likes from DB (including the user ID added by client 1), and then makes a request to remove the user ID. Meanwhile, client 1 makes request again (unaware of client 2), which adds user ID again. Retry therefore has the effect of adding user ID -- this is unexpected because client 2 causally saw client 1's change, so the removal happened after the addition of the set element, and therefore we expect that in final state, user ID should NOT be present.
      * In this case, the fact that adding an element to a set is idempotent is not sufficient to make the retry safe.
    * Example 2
      * ![Idempotency limitation 2](/metadata/idempotency_limitation_2.png)
      * In both scenarios the outcome is the same: x is present on replica B, and absent from replica A. Yet the intended effect is different: in the first scenario, the client wanted x to be removed from both replicas, whereas in the second scenario, the client wanted x to be present on both replicas. When the two replicas reconcile their inconsistent states, we want them to both end up in the state that the client intended. However, this is not possible if the replicas cannot distinguish between these two scenarios.
    * To solve, we can do two things
      * First, we attach logical timestamp to every update operation, and store the timestamp in DB as part of the data written by the update.
      * Second, when asked to remove a record from the database, we don’t actually remove it, but rather write a special type of update (called a tombstone) marking it as deleted.
      * ![Timestamps and tombstones](/metadata/timestamps_and_tombstones.png)
      * In many replicated systems, replicas run a protocol to detect and reconcile any differences (this is called anti-entropy) so that the replicas eventually hold consistent copies of the same data. Using timestamps and tombstones, we can differentiate b/w two scenarios given in idempotency example 2.
      * Using timestamps and tombstones also helps in handling concurrent updates
        * ![Concurrent writes](/metadata/concurrent_writes.png)
        * Last Write Wins [use logical timestamps like lamport clocks with **total order** -- data loss as two concurrent updatates are ordered arbitrarily]
        * Version Vectors [**partial order** -- v2 replaces v1 if v2>v1, preserves both v1 and v2 if v1 || v2]
          * On-read (siblings)
          * On-write (CRDTs)

State machine replication (SMR)
* FIFO-total order broadcast every update to all replicas
* Replica delivers update messages: apply to its own state
* Applying an update is deterministic
* Replica is a state machine: starts in fixed initial state, goes through same sequence of state transitions in the same order => all replicas end in the same state
* Closely related idea
  * Serializable transactions (execute in delivery order)
  * Blockchains, distributed ledgers, smart contracts (the “chain of blocks” in a blockchain is nothing other than the sequence of messages delivered by a total order broadcast protocol)
* Limitations
  * If update happens on node A, then A cannot update state immediately, have to wait for delivery through broadcast (by coordinating with other nodes)
  * Need fault-tolerant total order broadcast (will see in future)

```python
on request to perform update u do
 send u via FIFO-total order broadcast
end on

on delivering u through FIFO-total order broadcast do
 update state using arbitrary deterministic logic!
end on
```

* Can we use other weak broadcast protocols for replication? YES with care
  * Total Order Broadcast: deterministic SMR
  * Causal Broadcast: deterministic, concurrent updates are commutative (f(g(x)) = (g(f(x))
  * Reliable Broadcast: deterministic, all updates commutative
  * Best effort Broadcast: deterministic, commutative, idempotent, tolerates message loss

#### Implementation of replication logs

##### Statement-based replication

The leader logs every _statement_ and sends it to its followers (every `INSERT`, `UPDATE` or `DELETE`).

This type of replication has some problems:
* Non-deterministic functions such as `NOW()` or `RAND()` will generate different values on replicas.
* Statements that depend on existing data (`UPDATE ... WHERE <some condition>`), or if statement uses auto-incrementing columns, must be executed in the same order in each replica. This can be limiting when there are multiple concurrently executing transactions.
* Statements with side effects (triggers, stored procedures, user definted functions) may result on different results on each replica, unless the side effects are absolutely deterministic.

A solution to this is to replace any nondeterministic function with a fixed return value in the leader. VoltDB does this and uses statement based replication. MySQL before version 5.1 used statement based replication.

##### Write-ahead log (WAL) shipping

* The log is an append-only sequence of bytes containing all writes to the database. The leader can send it to its followers. This way of replication is used in PostgreSQL and Oracle.
* The main disadvantage is that the log describes the data at a very low level (like which bytes were changed in which disk blocks), coupling it to the storage engine. If the database changes its storage format from one version to another, it is typically not possible to run different versions of database software on the leader and the followers. This can have a big operational impact, like making it impossible to have a zero-downtime upgrade of the database.

##### Logical (row-based) log replication

* Decouples replication from storage engine as both are different.
* Basically a sequence of records describing writes to database tables at the granularity of a row:
    * For an inserted row, the new values of all columns.
    * For a deleted row, the information that uniquely identifies that column (primary key)
    * For an updated row, the information to uniquely identify that row and all the new values of the columns.
* A transaction that modifies several rows, generates several of such logs (this can be cited as disadvantage as it takes lots of storage), followed by a record indicating that the transaction was committed. MySQL binlog uses this approach.
* Since logical log is decoupled from the storage engine internals, it's easier to make it backwards compatible (allowing leader and follower to run different version of database software or even different storage engines).
* Logical logs are also easier for external applications to parse, useful for data warehouses, custom indexes and caches (this is called _change data capture_).

##### Trigger-based replication

* There are some situations were you may need to move replication up to the application layer (if you want to only replicate a subset of the dat, or want to replicate from one kind of database to another, or if you need conflict resolution logic).
* A trigger lets you register custom application code that is automatically executed when a data change occurs. This is a good opportunity to log this change into a separate table, from which it can be read by an external process.
* Main disadvantages is that this approach has greater overheads, is more prone to bugs but it may be useful due to its flexibility.

### Problems with replication lag

* Node failures is just one reason for wanting replication. Other reasons are scalability and latency.
* In a _read-scaling_ architecture, you can increase the capacity for serving read-only requests simply by adding more followers. However, this only realistically works on asynchronous replication. The more nodes you have, the likelier is that one will be down, so a fully synchronous configuration would be unreliable.
* With an asynchronous approach, a follower may fall behind, leading to inconsistencies in the database (_eventual consistency_).
* The _replication lag_ could be a fraction of a second or several seconds or even minutes.

The problems that may arise in asynchronous replication (eventual consistency) and how to solve them are as follows

#### Reading your own writes

![Read your writes](/metadata/read_your_writes.png)

_Read-after-write consistency_, also known as _read-your-writes consistency_ is a guarantee that if the user reloads the page, they will always see any updates they submitted themselves. IT makes no promises about other users: other users' updates may not be visible until some later time..

How to implement it:
* **When reading something that the user may have modified, read it from the leader.** For example, user profile information on a social network is normally only editable by the owner. A simple rule is always read the user's "own" profile from the leader, and any other users' profiles from a follower.
* If most things are potentially editable by the user, above approach won't work as everything will go to leader only, negating the benefit of read scaling. You could track the time of the latest update and, for one minute till after the last update, make all reads from the leader (i.e. pin the user to master for one minute after they made the update so that all (or relevant APIs) reads goes to master)
* The client can remember the timestamp of the most recent write, then the system can ensure that the replica serving any reads for that user reflects updates at least until that timestamp. Timestamp can be logical timestamp (something that indicates ordreing of writes such as log sequence number) or actual system clock (unreliable)
* If your replicas are distributed across multiple datacenters, then any request needs to be routed to the datacenter that contains the leader.


Another complication is that the same user is accessing your service from multiple devices, you may want to provide _cross-device_ read-after-write consistency: if user enters some information on one device and then views it on another device, they should see the information they just entered. Some additional issues to consider:
* Remembering the timestamp of the user's last update becomes more difficult, because the code running on one device doesn't know what updates have happened on other device. The metadata will need to be centralised.
* If replicas are distributed across datacenters, there is no guarantee that connections from different devices will be routed to the same datacenter. You may need to route requests from all of a user's devices to the same datacenter.

#### Monotonic reads

![Monotonic Reads](/metadata/monotonic_reads.png)

* Because of followers falling behind, it's possible for a user to see things _moving backward in time_.
* Lesser guarantee than strong consistency, but a stronger guarantee than eventual consistency.
* When you read data, you may see an old value; monotonic reads only means that if one user makes several reads in sequence, they will not see time go backward.
* Solution: Make sure that each user always makes their reads from the same replica. The replica can be chosen based on a hash of the user ID. However, if the replica fails, the user's queries will need to be rerouted to another replica.

#### Consistent prefix reads

![Consistent prefix reads](/metadata/consistent_prefix_reads.png)

* If a sequence of writes happens in a certain order, then anyone reading those writes will see them appear in the same order.
* This is a particular problem in partitioned (sharded) databases as there is no global ordering of writes.
* A solution is to make sure any writes casually related to each other are written to the same partition.

#### Solutions for replication lag

_Transactions_ exist so there is a way for a database to provide stronger guarantees so that the application can be simpler.

### Single leader replication

Each node that stores a copy of the database is called a _replica_.

Every write to the database needs to be processed by every replica. The most common solution for this is called _leader-based replication_ (_active/passive_ or _master-slave replication_).
1. One of the replicas is designated the _leader_ (_master_ or _primary_). Writes to the database must send requests to the leader.
2. Other replicas are known as _followers_ (_read replicas_, _slaves_, _secondaries_ or _hot standbys_). The leader sends the data change to all of its followers as part of a _replication log_ or _change stream_.
3. Reads can be query either the leader or any of the followers, while writes are only accepted on the leader.

MySQL, PostgreSQL, Oracle Data Guard, SQL Server's AlwaysOn Availability Groups, MongoDB, RethinkDB, Espresso, Kafka and RabbitMQ are examples of these kind of databases.

#### Synchronous vs asynchronous

* **The advantage of synchronous replication is that the follower is guaranteed to have an up-to-date copy of the data that is consistent with the leader. The disadvantage is that it the synchronous follower doesn't respond, the write cannot be processed.**. Synchronous replication => Strong Consistency.
    * _Chain replication_ is a variant of synchronous replication that has been successfully implemented in a few systems such as Microsoft Azure Storage.
    * There is a strong connection between consistency of replication and consensus (getting several nodes to agree on a value).
* It's impractical for all followers to be synchronous. If you enable synchronous replication on a database, it usually means that _one_ of the followers is synchronous, and the others are asynchronous. This guarantees up-to-date copy of the data on at least two nodes (this is sometimes called _semi-synchronous_).
* Often, leader-based replication is asynchronous. Writes are not guaranteed to be durable, the main advantage of this approach is that the leader can continue processing writes. Disadvantage is weakened durability. Asynchronous replication => Eventual Consistency.

#### Setting up new followers

Copying data files from one node to another is typically not sufficient.

Setting up a follower can usually be done without downtime. The process looks like:
1. Take a snapshot of the leader's database, without taking lock on entire database. Most databases have this feature, as it is also required for backups. Snapshot Isolation. Sometimes third party tools are used, like _innobackupex_ for MySQL.
2. Copy the snapshot to the follower node
3. Follower requests data changes that have happened since the snapshot was taken. This requires that snapshot is associated with exact position in leader's replication log. PostgreSQL calls it log sequence number, MySQL calls it binlog coordinates.
4. Once follower processed the backlog of data changes since snapshot, it has _caught up_.

#### Handling node outages

How does high availability works with leader-based replication?

##### Follower failure: catchup recovery

Follower can connect to the leader and request all the data changes that occurred during the time when the follower was disconnected using it's log.

##### Leader failure: failover

One of the followers needs to be promoted to be the new leader, clients need to be reconfigured to send their writes to the new leader and followers need to start consuming data changes from the new leader.

Automatic failover consists:
1. Determining that the leader has failed. There is no full proof system for detecting what has gone wrong (node has crashed, or are there network issues, etc). So most systems simply use a timeout: nodes frequently bounce messages back and forth between each other and if a node doesn't respond for some period of time (say 30s), it is assumed dead. Or, all the nodes ping an orchestrator. There are many failure detection systems (example: phi accrual failure detection algorithm that instead of expecting a heartbeat at regular intervals estimates the probability and sets a dynamic threshold).
2. Choosing a new leader. The best candidate for leadership is usually the replica with the most up-to-date changes from the old leader. Either new leader is appointed by election or a previously elected _controller node_ chooses the leader. Consensus problem. 
3. Reconfiguring the system to use the new leader. The system needs to ensure that the old leader becomes a follower and recognises the new leader.

Things that could go wrong:
* If asynchronous replication is used, the new leader may have received conflicting writes in the meantime. Most common solution is for old leader's unreplicated writes to simply be discarded, which may violate clients' durability expectations. We can use semi-synchronous and promote the synchronous "passive master" to being main master, hence this loss of durability won't occur.
* Discarding writes is especially dangerous if other storage systems outside of the database need to be coordinated with the database contents. Example: an out-of-date MySQL follower was promoted to leader. Database used autoincrementing counter to assign primary keys to new rows, but because the new leader's counter lagged behind old leader's it reused some primary keys that were previously assigned by the old leader. These primary keys were also used in a Redis store, so the reuse of primary keys resulted in inconsistency between Redis and MySQL, which caused some private data to be disclosed to the wrong users.
* It could happen that two nodes both believe that they are the leader (_split brain_). Data is likely to be lost or corrupted. Like multi-leader replication => Conflicts!
* What is the right time before the leader is declared dead?

For these reasons, some operation teams prefer to perform failovers manually, even if the software supports automatic failover.

### Multi-leader replication

* Leader-based replication has one major downside: there is only one leader, and all writes must go through it.
* A natural extension is to allow more than one node to accept writes (_multi-leader_, _master-master_ or _active/active_ replication) where each leader simultaneously acts as a follower to the other leaders.

#### Use cases for multi-leader replication

It rarely makes sense to use multi-leader setup within a single datacenter.

##### Multi-datacenter operation

![Multi Datacenter](/metadata/multi_datacenter.png)

You can have a leader in _each_ datacenter. Within each datacenter, regular leader-follower replication is used. Between datacenters, each datacenter leader replicates its changes to the leaders in other datacenters.

Compared to a single-leader replication model deployed in multi-datacenters
* **Performance.** With single-leader, every write must go across the internet to wherever the leader is, adding significant latency. In multi-leader every write is processed in the local datacenter and replicated asynchronously to other datacenters. The network delay is hidden from users and perceived performance may be better.
* **Tolerance of datacenter outages.** In single-leader if the datacenter with the leader fails, failover can promote a follower in another datacenter. In multi-leader, each datacenter can continue operating independently from others.
* **Tolerance of network problems.** Single-leader is very sensitive to problems in this inter-datacenter link as writes are made synchronously over this link. Multi-leader with asynchronous replication can tolerate network problems better.

Multi-leader replication is implemented with Tungsten Replicator for MySQL (doesn't even try to detect conflicts), BDR for PostgreSQL (does not provide casual ordering of writes) or GoldenGate for Oracle.

It's common to fall on subtle configuration pitfalls. Autoincrementing keys, triggers and integrity constraints can be problematic. Multi-leader replication is often considered dangerous territory and avoided if possible.

##### Clients with offline operation

If you have an application that needs to continue to work while it is disconnected from the internet, every device that has a local database can act as a leader, and there will be some asynchronous multi-leader replication process (imagine, a Calendar application).

CouchDB is designed for this mode of operation.

#### Collaborative editing

* _Real-time collaborative editing_ applications allow several people to edit a document simultaneously. Like Etherpad or Google Docs. "Operational transformation" is the conflict resolution algorithm behind both of these.
* The user edits a document, the changes are instantly applied to their local replica and asynchronously replicated to the server and any other user.
* If you want to avoid editing conflicts, you must the lock the document before a user can edit it. But this is equivalent to single leader replication as any user will have to wait for first user to complete it's writes.
* For faster collaboration, you may want to make the unit of change very small (like a keystroke) and avoid locking.

#### Handling write conflicts

* The biggest problem with multi-leader replication is when conflict resolution is required. This problem does not happen in a single-leader database.
* In single-leader the second writer can be blocked and wait the first one to complete, forcing the user to retry the write. On multi-leader if both writes are successful, the conflict is only detected asynchronously later in time.
* If you want synchronous conflict detection, you might as well use single-leader replication.

##### Conflict avoidance

* The simplest strategy for dealing with conflicts is to avoid them. If all writes for a particular record go through the same leader, then conflicts cannot occur. Like partitioning.
* On an application where a user can edit their own data, you can ensure that requests from a particular user are always routed to the same datacenter and use the leader in that datacenter for reading and writing.
* However sometimes you may want to change the home datacenter for a user due to
     * home datacenter failed
     * user moved to different location

##### Converging toward a consistent state

* On single-leader, the last write determines the final value of the field.
* In multi-leader, it's not clear what the final value should be.
* The database must resolve the conflict in a _convergent_ way, all replicas must arrive a the same final value when all changes have been replicated.

Different ways of achieving convergent conflict resolution.
* Give each write a unique ID (timestamp, long random number, UUID, or a hash of the key and value), pick the write with the highest ID as the _winner_ and throw away the other writes. This is known as _last write wins_ (LWW) and it is dangerously prone to data loss.
* Give each replica a unique ID, writes that originated at a higher-numbered replica always take precedence. This approach also implies data loss.
* Somehow merge the values together. Say if value is string, then concatenate the string?

##### Custom conflict resolution

Multi-leader replication tools let you write conflict resolution logic using application code. Version Vectors can be used here for both.

* **On write.** As soon as the database system detects a conflict in the log of replicated changes, it calls the conflict handler. CRDTs used here.
* **On read.** All the conflicting writes are stored. On read, multiple versions of the data are returned to the application. The application may prompt the user or automatically resolve the conflict. CouchDB works this way. 

**Note that conflict resolution usually applies at the level of an indivudual row or document, and not at transaction level. Thus, if you have a transaction that atomically makes several different writes, each write is still considered separately for the purposes of conflict resolution.**

#### Multi-leader replication topologies

* A _replication topology_ describes the communication paths along which writes are propagated from one leader to another.
* The most general topology is _all-to-all_ in which every leader sends its writes to every other leader. MySQL uses _circular topology_, where each nodes receives writes from one node and forwards those writes to another node. Another popular topology has the shape of a _star_, one designated node forwards writes to all of the other nodes.
* In circular and star topologies a write might need to pass through multiple nodes before they reach all replicas. To prevent infinite replication loops each node is given a unique identifier and the replication log tags each write with the identifiers of the nodes it has passed through.
* In all-to-all topology fault tolerance is better as messages can travel along different paths avoiding a single point of failure. It has some issues too, some network links may be faster than others and some replication messages may "overtake" others. Consistent Prefix Reads problem. To order events correctly. there is a technique called _version vectors_.

### Leaderless replication

* Simply put, any replica can directly accept writes from clients. Databases like look like Amazon's in-house _Dynamo_ datastore. _Riak_, _Cassandra_ and _Voldemort_ follow the _Dynamo style_.
* Note: Amazon DynamoDB is based on principles of Dynamo paper, but DynamoDB is single leader replication. The Dynamo paper was describing leaderless replication as a concept.
* In a leaderless configuration, failover does not exist. Clients send the write to all replicas in parallel.
* _Read requests are also sent to several nodes in parallel_. The client may get different responses. Version numbers are used to determine which value is newer.
* In reality, the client does not send reads and writes to these n nodes, rather client talk to a coordinator node, which in turn talks to these n nodes (has the logic for read repair, etc) and then sends back the final result to the client. Cassandra does this.

![Read repair](/metadata/read_repair.png)

Eventually, all the data is copied to every replica. After a unavailable node come back online, it has two different mechanisms to catch up:
* **Read repair.** When a client detect any stale responses, write the newer value back to that replica.
* **Anti-entropy process.** There is a background process that constantly looks for differences in data between replicas and copies any missing data from one replica to he other. **It does not copy writes in any particular order**. Voldemort does not have anti entropy process. Without it, values that are rarely read may be missing from some replicas and thus have reduced durability.

##### Merkle Tree
* Useful for efficiently calculating the difference between two sets of data or files, and determine the difference in logarithmic time
* Algorithm
  * we have n values in set, calculate hashes of them, take pairs of them and calculate the sum, take hash and repeat
  * to find difference b/w two merkle trees, start from the root and traverse depth first finding differences in hashes
* the single unit (leaf value) is usually the whole file (say a.txt), for which we find the hash (hashing all the contents of the file)
* use case
  * git revision system
    * when there is a change to say file foo.cc, then we compute the hash, we get a different valye, create another leaf node in the existing tree, keep creating nodes till the root in the same tree. So similar to copy-on-write we can do in BTree to implement snapshot isolation. This way we keep history of changes. Note: here we are only tracking if a file has been changed, and not the diff in contents of the changed file
  * anti entropy process in leaderless databases
    * single unit of comparison is the whole SSTable segment in case of Cassandra.
  * blockchain
* ![Merkle Tree](/metadata/merkle_tree.jpeg)

#### Quorums for reading and writing

* If there are _n_ replicas, every write must be confirmed by _w_ nodes to be considered successful, and we must query at least _r_ nodes for each read. As long as _w_ + _r_ > _n_, we expect to get an up-to-date value when reading. _r_ and _w_ values are called _quorum_ reads and writes. Are the minimum number of votes required for the read or write to be valid.
* A common choice is to make _n_ and odd number (typically 3 or 5) and to set _w_ = _r_ = (_n_ + 1)/2 (rounded up).
* Quoram does not mean strong consistency.

```sql
# 3 clients writing to all 3 DBs, but due to network delays reach in weird state
L1 L2 L3
W1 W1 W2 v1
W2 W3 W3 v2
W3 W2 W1 v3
# State is above line eventually at the end. Two clients reading from say these will see different results. Hence, not strongly consistent.
```

Limitations:
* Sloppy quorum, the _w_ writes may end up on different nodes than the _r_ reads, so there is no longer a guaranteed overlap.
* If two writes occur concurrently, and is not clear which one happened first, the only safe solution is to merge them (above code snippet). Writes can be lost due to clock skew.
* If a write happens concurrently with a read, the write may be reflected on only some of the replicas.
* If a write succeeded on some replicas but failed on others, it is not rolled back on the replicas where it succeeded. Reads may or may not return the value from that write.
* If a node carrying a new value fails, and its data is restored from a replica carrying an old value, the number of replicas storing the new value may break the quorum condition.

**Dynamo-style databases are generally optimised for use cases that can tolerate eventual consistency.**

* It is important to monitor whether databases are running with up-to-date values.
* For leader-based, we can calculate replication lag (L2 log sequence number minus L1 log sequence number)
* But in leaderless, there is no fixed order in which writes are applied, which makes monitoring more difficult.

#### Sloppy quorums and hinted handoff

* Leaderless replication may be appealing for use cases that require high availability and low latency, and that can tolerate occasional stale reads.
* It's likely that the client won't be able to connect to _some_ database nodes during a network interruption.
* Is it better to return errors to all requests for which we cannot reach quorum of _w_ or _r_ nodes?
* Or should we accept writes anyway, and write them to some nodes that are reachable but aren't among the _n_ nodes on which the value usually lives?
* The latter is known as _sloppy quorum_: writes and reads still require _w_ and _r_ successful responses, but those may include nodes that are not among the designated _n_ "home" nodes for a value.
* Once the network interruption is fixed, any writes are sent to the appropriate "home" nodes (_hinted handoff_).
* Sloppy quorums are useful for increasing write availability: as long as any _w_ nodes are available, the database can accept writes. This also means that you cannot be sure to read the latest value for a key, because it may have been temporarily written to some nodes outside of _n_.

##### Multi-datacenter operation

Each write from a client is sent to all replicas, regardless of datacenter, but the client usually only waits for acknowledgement from a quorum of nodes within its local datacenter so that it is unaffected by delays and interruptions on cross-datacenter link.

#### Detecting concurrent writes
* Applicable for multi-leader setup + leaderless setup.
* In Dynamo style database (leaderless), conflicts can also arise during read repair or hinted handoff.
* In order to become eventually consistent, the replicas should converge toward the "same" value. If you want to avoid losing data, you application developer, need to know a lot about the internals of your database's conflict handling.

* **Last write wins (discarding concurrent writes).** Even though the writes don't have a natural ordering, we can force an arbitrary order on them. We can attach a timestamp to each write and pick the most recent. There are some situations such as caching on which lost writes are acceptable. If losing data is not acceptable, LWW is a poor choice for conflict resolution. Cassandra only supports this.
* **Ordering of messages: "happens-before" relationship and concurrency.** Whether one operation happens before another operation is the key to defining what concurrency means.
     * An operation A happens before another operation B if B knows about A, or depends on A, or builds upon A in some way (consistent prefix reads).
       * An event is something happening at one node (local execution step, or sending a message, or receiving a message)
       * We say that `a` happens before event `b` (writtern `a` &#8594; `b`) iff:
         * `a` and `b` occurred at same node (assume single threaded), and `a` occurred before `b` in that node's local execution order (there is a _strict total order_ on the events that occur at the same node); or
         * event `a` is the sending of message m, and event `b` is the receipt of that same message m (assuming sent messages are unique, we can make messages unique say using UUID OR sender node ID + sequence number for each message); or
         * there exists an event `c` such that `a` &#8594; `c` and `c` &#8594; `b`
       * Happens-before is a **partial order**
     * **We can simply say that to operations are _concurrent_ if neither happens before the other.** Concurrent does not mean literally at the exact same time; we simply say two operations are concurrent, if they are both unaware of each other (or they are independent)
       * It is possible that neither `a` &#8594; `b` and `b` &#8594; `a`, in that case we say that `a` and `b` are concurrent (`a` || `b`)
     * Either A happened before B, or B happened before A, or A and B are concurrent.
     * Example:
       * ![Happens-before-concurrent](/metadata/happens-before-concurrent.png)
     * We can't just use time-of-day clock for ordering the messages as timestamp across different nodes are not synchronized perfectly. Also, monotonic clocks can't be compared across nodes, so that can't be used either.
     * If one operation happened before another, later operation should overwrite the earlier operation, but if the operations are concurrent, we have a conflict that needs to be resolved.

##### Capturing the happens-before relationship

![Happen before](/metadata/happen_before.png)
1. Client 1 adds milk to cart.
   * Server: [Version: 1 {1: milk}]
2. Client 2 adds eggs to cart (not knowing that client 1 concurrently added milk). Server assigns version 2 to this write, and stores eggs and milk as two separate values. Returns both values to client with version number 2
   * Server: [Version: 2 {1: milk, 2: eggs}]
3. Client 1, oblivious to client 2's write, adds flour to cart. Sends [milk, flour] with version 1, server sees [milk, flour] supersedes [milk] but is concurrent with [eggs]
   * Server: [Version: 3 {3: [milk, flour], 2:eggs}]
4. Client 2, oblivious to client 1's write, adds ham to cart. Sends [milk, eggs, ham] with version number 2
   * Server: [Version: 4 {3: [milk, flour], 4: [milk, eggs, ham]}]
5. Client 1, adds bacon to cart. Sends [milk, flour, eggs, bacon] with version number 3
   * Server: [Version: 5 {5: [milk, flour, eggs, bacon], 4:[milk, eggs, ham]}]

The server can determine whether two operations are concurrent by looking at the version numbers.
* The server maintains a version number for every key, increments the version number every time that key is written, and stores the new version number along the value written.
* Client reads a key, the server returns all values that have not been overwrite, as well as the latest version number. A client must read a key before writing.
* Client writes a key, it must include the version number from the prior read, and it must merge together all values that it received in the prior read.
* Server receives a write with a particular version number, it can overwrite all values with that version number or **"below"** (since it knows that they have been merged into the new value), but it must keep all values with a higher version number.

##### Merging concurrently written values

* No data is silently dropped. It requires clients do some extra work, they have to clean up afterward by merging the concurrently written values. Riak calls these concurrent values _siblings_.
* Merging sibling values is the same problem as conflict resolution in multi-leader replication. A simple approach is to just pick one of the values on a version number or timestamp (last write wins). You may need to do something more intelligent in application code to avoid losing data.
* If you want to allow people to _remove_ things, union of siblings may not yield the right result. An item cannot simply be deleted from the database when it is removed, the system must leave a marker with an appropriate version number to indicate that the item has been removed when merging siblings (_tombstone_).
* Merging siblings in application code is complex and error-prone, there are efforts to design data structures that can perform this merging automatically (CRDTs).

#### Version vectors

* We need a version number _per replica_ as well as per key. Each replica increments its own version number when processing a write, and also keeps track of the version numbers it has seen from each of the other replicas.
* The collection of version numbers from all the replicas is called a _version vector_.
* Version vector are sent from the database replicas to clients when values are read, and need to be sent back to the database when a value is subsequently written. Riak calls this _casual context_. Version vectors allow the database to distinguish between overwrites and concurrent writes.

![Version vector](/metadata/version_vector.jpeg)

Algorithm
* every time a client reads from DB, DB gives it's version vector of key
* every time a client writes to DB, it passes DB its most recently read version vector for key and DB supplies it with new one.
* two values have "happen before" relationship if one version vector is strictly greater than other (for all i, v1[i] >= v2[i])
* all other version vectors are concurrent
     * corresponding values can be either
          *  merged ("on write"). CRDTs.
          *  kept as sibling ("on read")
     * merging version vectors means taking max of both version vectors at every index
 
#### CRDT (Conflict-free Replication Data Types)

* Where used? Riak.
* Types
  * **Operational**
    * Example: increment(x). Tells database which index of list to increment. Less data to send over network.
    * Downsides: say DB1 got two updates: add("ham"), and then remove("ham"). And now say these 2 updates get replicated to DB2, but add("ham") get dropped. These were 2 causually dependent messages. So this won't work.
  * **State based**
    * Example: merge([5,4], [4,6]) = [5,6]
    * merge function must be
      * commutative: f(a,b) = f(b,a)
      * associative: f(f(a,b),c) = f(a, f(b,c))
      * idempotent: f(a,b) = f(f(a,b),b)
    * since order doesn't matter, we can communicate via **gossip protocol** (nodes randomly keep sending message to few neighbour nodes)
    * How to implement set? Keep two lists: one **adds**, another **removes** [so if adds={1, 2, 3} and removes={3}, then actual set is {1, 2}
      * merge (adds1, adds2, removes1, removes2) = union(adds1, adds2) difference union(removes1, removes2)
      * problem: if one element goes to **removes**, we can never add it again. Solution: give each add a unique tag
        * adds = {1:t1, 2:t2, 1:t3}, removes = {1:t1}, then set is {2:t2, 1:t3} which is {2, 1}
  * **Sequence CRDTs**
    * buld eventually consistent ordered list (Etherpad and Google Docs) 
      
---
---

## Partitioning

* In horizontal scaling, if we want to scale reads we do replication, if we want to scale writes we do partitioning.
* Replication, for very large datasets or very high query throughput is not sufficient, we need to break the data up into _partitions_ (_sharding_).
* Two types of partitioning
  * Horizontal - by rows (relational and non-relational DBs)
  * Vertical - by columns (data warehouses)
* MongoDB, ElasticSearch and SolrCloud (shard); HBase (region); Bigtable (tablet); Cassandra and Riak (vnode);
* Basically, each partition is a small database of its own.
* The main reason for wanting to partition data is _scalability_, query load can be distributed across many processors. Throughput can be scaled by adding more nodes.

### Partitioning and replication

* Each record belongs to exactly one partition, it may still be stored on several nodes for fault tolerance.
* A node may store more than one partition.
* Choice of partitioning schem is mostly independent of the choice of replication scheme.

### Partition of key-value data

* Our goal with partitioning is to spread the **data and the query load** evenly across nodes.
* If partition is unfair, we call it _skewed_. It makes partitioning much less effective. A partition with disproportionately high load is called a _hot spot_.
* The simplest approach is to assign records to nodes randomly. The main disadvantage is that if you are trying to read a particular item, you have no way of knowing which node it is on, so you have to query all nodes in parallel.

#### Partition by key range

* Assign a continuous range of keys, like the volumes of a paper encyclopaedia. Boundaries might be chose manually by an administrator, or the database can choose them automatically. On each partition, keys are in sorted order so scans are easy. A classic use-case where range-based partitioning fails is when we range-partition the time-series data on timestamp because load comes on the latest partition as writes are done on that only. Hence, for time series data paritioning by time is not good.
* Advantage: Range queries efficient.
* Disadvantage: Ranges of keys are not necessarily evenly spaced, because your data may not be evenly distributed. Hence certain access patterns can lead to hot spots.
* Rebalancing typically done using "Dynamic partitioning"

#### Partitioning by hash of key

* A good hash function takes skewed data and makes it uniformly distributed. There is no need to be cryptographically strong (MongoDB uses MD5 and Cassandra uses Murmur3). You can assign each partition a range of hashes. The boundaries can be evenly spaced or they can be chosen pseudorandomly (_consistent hashing_).
* Advantage: less hot spots
* Disadvantage: we lose the ability to do efficient range queries. Keys that were once adjacent are now scattered across all the partitions. Any range query has to be sent to all partitions.
* Rebalancing typically done using "Fixed number of partitions". Although "Dynamic partitioning" can also be used.

Cassandra (and Amazon DynamoDB) achieves a compromise between the two partition strategies. Table in Cassandra can be declared with a _compound primary key_ consisting of several columns. Only first part of key is hashed to determine the partition, but the other columns are used as concatenated index for sorting the data in SSTables. Great then for say queries (user_id, timestamp) => hash the user_id and get the partition on which this user data is present, then scan it's data in the timestamp as data is sorted by timestamp.

#### Skewed workloads and relieving hot spots

* You can't avoid hot spots entirely. For example, you may end up with large volume of writes to the same key (celebrity user with millions of followers)
* It's the responsibility of the application to reduce the skew. A simple technique is to add a random number to the beginning or end of the key.
* Splitting writes across different keys, makes reads now to do some extra work and combine them.

### Partitioning and secondary indexes

The situation gets more complicated if secondary indexes are involved. A secondary index usually doesn't identify the record uniquely. They don't map neatly to partitions.

#### Partitioning secondary indexes by document

* Each partition maintains its secondary indexes, covering only the documents in that partition (_local index_).
* Advantage: Writes are efficient as only have to go to one partition.
* Disadvantage: You need to send the query to _all_ partitions, and combine all the results you get back (_scatter/gather_). Hence, reads are inefficient.
* Widely used in MongoDB, Riak, Cassandra, Elasticsearch, SolrCloud and VoltDB.

#### Partitioning secondary indexes by term

* We construct a _global index_ that covers data in all partitions. The global index must also be partitioned so it doesn't become the bottleneck.
* It is called the _term-partitioned_ because the term we're looking for determines the partition of the index.
* It is a kind of re-partitioning of data on a different partition key.
* Partitioning by term can be useful for range scans, whereas partitioning on a hash of the term gives a more even distribution load.
* Advantage: Reads more efficient; rather than doing scatter/gather over all partitions, a client only needs to make a request to the partition containing the term that it wants.
* Disadvantage: writes are slower and complicated.

In practice, updates to global secondary indexes are often asynchronous.

### Rebalancing partitions
The process of moving load from one node in the cluster to another. Have to add more nodes because
* Query throughput increases
* Dataset size increases
* Machine fails

Requirements
* Data and query load should be evenly distributed
* No downtime, while rebalancing is happening, the database should continue accepting reads and writes.
* Least possible data needs to be moved to minimize network and disk I/O load.

Strategies for rebalancing:
* **How not to do it: Hash mod n.**
  * The problem with _mod N_ is that if the number of nodes _N_ changes, most of the keys will need to be moved from one node to another.
* **Fixed number of partitions.**
  * Create many more partitions than there are nodes and assign several partitions to each node.
  * If a node is added to the cluster, we can _steal_ a few partitions from every existing node until partitions are fairly distributed once again. The number of partitions does not change, nor does the assignment of keys to partitions. The only thing that change is the assignment of partitions to nodes.
  * This is used in Riak, Elasticsearch, Couchbase, and Voldemort.
  * **You need to choose a high enough number of partitions to accommodate future growth.** Neither too big or too small. Choose too less? Maximum number of nodes is bottlenecked by number of partitions; also rebalancing such a big partition is expensive. Choose too high? More overhead to store this metadata in Zookeeper. 
* **Dynamic partitioning.**
  * The number of partitions adapts to the total data volume.
  * An empty database starts with an empty partition. While the dataset is small, all writes have to processed by a single node while the others nodes sit idle. HBase and MongoDB allow an initial set of partitions to be configured (_pre-splitting_).
* **Partitioning proportionally to nodes.**
  * Fixed number of partitions: size of dataset increases => size of partition increases. # of partitions is independent of # of nodes.
  * Dynamic partition: size of dataset increases => # of partition increases. # of partitions is independent of # of nodes.
  * Cassandra and Ketama make the number of partitions proportional to the number of nodes. Have a fixed number of partitions _per node_.
  * When a new node joins the cluster, it randomly chooses a fixed number of existing partitions to split, and then takes ownership of one half of each of those split partitions while leaving the other half of each partition in place.
  * This approach also keeps the size of each partition fairly stable.
* **Consistent Hashing**
  * distributes keys evenly, minimal data sent over networks on rebalance
  * ring is the range of hash function (say 0 to 2^32 - 1), and then we put nodes randomly on the ring (can put a node at **multiples places** (this number is same for all nodes say K) on the ring at the same time) -- we can think each node having fixed number of partitions (K)
  * when we want to remove or add a node, just put it randomly on the ring
  * this is equivalent to **partition proportionally to nodes**
  * apart from rebalancing partitions, consistent hashing is also used in load balancers (to create requests sticky such that request from user goes to one application server only, say to leverage caching benefits etc) (ring is hash of user IDs, and then application server is put on the ring) 

#### Automatic versus manual rebalancing

* Fully automated rebalancing may seem convenient but the process can overload the network or the nodes and harm the performance of other requests while the rebalancing is in progress.
* It can be good to have a human in the loop for rebalancing. You may avoid operational surprises.

### Request routing

![Routing](/metadata/routing.png)

This problem is also called _service discovery_. There are different approaches:
1. Allow clients to contact any node (round robin) and make them handle the request directly, or forward the request to the appropriate node. Gossip protocol between nodes then.
2. Send all requests from clients to a routing tier first that acts as a partition-aware load balancer.
3. Make clients aware of the partitioning and the assignment of partitions to nodes.

In many cases the problem is: how does the component making the routing decision learn about changes in the assignment of partitions to nodes?

![ZooKeeper](/metadata/zookeeper.png)
* Many distributed data systems rely on a separate coordination service such as ZooKeeper to keep track of this cluster metadata. Each node registers itself in ZooKeeper, and ZooKeeper maintains the authoritative mapping of partitions to nodes. The routing tier or the partitioning-aware client, can subscribe to this information in ZooKeeper. Whenever a partition changes ownership, or a node is added or removed, ZooKeeper notifies the routing tier so that it can keep its routing information up to date.
* HBase, SolrCloud and Kafka use ZooKeeper to track partition assignment. MongoDB relies on its own _config server_. Cassandra and Riak take a different approach: they use a _gossip protocol_.

#### Parallel query execution

_Massively parallel processing_ (MPP) relational database products are much more sophisticated in the types of queries they support.

---
---
## Transactions

In data systems, things can go wrong
* DB software or hardware can fail
* Application server can fail
* Network problems (interruptions b/w server and DB OR DB and DB
* concurrent writes/reads from several clients

Implementing fault-tolerant mechanisms is a lot of work. So, we have transactions.

### The slippery concept of a transaction

* _Transactions_ have been the mechanism of choice for simplifying these issues. Conceptually, all the reads and writes in a transaction are executed as one operation: either the entire transaction succeeds (_commit_) or it fails (_abort_, _rollback_).
* The application is free to ignore certain potential error scenarios and concurrency issues, because DB takes care of them (_safety guarantees_). For example, if a transaction fails, the application can safely retry and don't have to worry about partial failures
* Not all applications actually need transaction, because it comes at a cost.
* Wrong belief: transactions means reduced scalability, so any large application has to abandon transaction to achieve performance and high availability. Not entirely true!

#### ACID

ACID is the "safety guarantee" provided by transaction. There is another term called BASE (Basically Available, Soft State, Eventually Consistent)
* **Atomicity**
  * refers to something that cannot be broken into parts. In multithreaded programming, one thread executes an atomic operation, meaning another thread could not see half-finished result of the operation. System can only in the staet it was before the operation or after the operation, not something in between.
  * But here, atomicity is _not_ about concurrency (that is "isolation"). It is what happens if a client wants to make several writes, but a fault occurs after some of the writes have been processed. Then the transaction is aborted, and DB must undo any writes it has made so far. So application can safely retry if transaction was aborted, because there is no half state. _Abortability_ would have been a better term than _atomicity_.
* **Consistency**
  * _Invariants_ on your data must always be true. The idea of consistency depends on the application's notion of invariants. DB checks some kind of invariants (uniqueness, foreign key) -- however, in general, the application defines what data is valid or invalid -- the database just stores it.
  * Atomicity, isolation, and durability are properties of the database, whereas consistency (in an ACID sense) is a property of the application. Application may rely on DB's atomicity and isolation properties in order to achieve consistency, but it's not up to the database alone. 
* **Isolation**
  * Concurrently executing transactions are isolated from each other. It's also called _serializability_, each transaction can pretend that it is the only transaction running on the entire database, and the result is the same as if they had run _serially_ (one after the other). Serializability comes at performance penalty.
* **Durability**
  * Once a transaction has committed successfully, any data it has written will not be forgotten, even if there is a hardware fault or the database crashes.
  * In a single-node database this means the data has been written to nonvolatile storage. In a replicated database it means the data has been successfully copied to some number of nodes.

#### Single Object and Multi Object Operations
* Both atomicity and isolation assumes that there are multiple operations we are doing in a transaction (modifying multiple objects i.e. rows, documents or records)
* DBs first ensures that single object writes are atomic (using WAL) and isolated (using locks) -- so that such thing doesn't happen like a row is updated partially and then node goes down.
* Then we group many such single object operations to transactions (aka multi object transaction) using `BEGIN TRANSACTION` and `COMMIT` statement
* Need for multi object transaction
  * Foreign key references needs to be maintained
  * Denormalized data needs to be updated
  * Secondary index needs to be updated

* Note: many non-relational DBs provide multi-object API (like taking multiple key-value pairs to overwrite), but they don't follow transaction semantics; the command may fail for some keys and succeed for others.

#### Handling errors and aborts

* A key feature of a transaction is that it can be aborted and safely retried if an error occurred (atomicity)
* In datastores with leaderless replication is the application's responsibility to recover from errors.
* The whole point of aborts is to enable safe retries. However, sometimes retying an aborted transaction is troublesome
  * Transaction actually succeeded, but network failed when served tried to sent ack of successful commit to client. So client sends again -- need to have application level deduplication
  * Transaction aborted because system was overloaded -- limit the number of retries using exponential backoff
  * Transaction aborted was not due to transient error (deadlock, isolation violation) but permanent error (invariant violation)

### Weak isolation levels

* Concurrency issues (race conditions) come into play when one transaction reads data that is concurrently modified by another transaction, or when two transactions try to simultaneously modify the same data.
* Databases have long tried to hide concurrency issues by providing _transaction isolation_.
* In practice, is not that simple. Serializable isolation has a performance cost. It's common for systems to use weaker levels of isolation, which protect against _some_ concurrency issues, but not all.

#### Read committed (no dirty read and writes)

##### Dirty Writes
* Definition: Overwrite data that has been uncommitted

```sql
# say two values k1 and k2 have to be in consistent state
    BEGIN T1
                                         BEGIN T2
    write k1 = x         
                                         write k1 = y
                                         write k2 = b
    write k2 = a
    COMMIT
                                         COMMIT

# now k1 is y and k2 is a ==> inconsistent state
```

* Solution
  * when writing, take lock and hold it until transaction is committed or aborted
  * deadlock may happen
 
```sql
    BEGIN T1
                                         BEGIN T2
    write k1 = x # get lock on k1      
                                         write k2 = b # get lock on k2
    write k2 = a # waiting before this operation for k2 to be released
                                         write k1 = y # waiting before this operation for k1 to be released 
    COMMIT
                                         COMMIT
```

##### Dirty Read
* Definition: Read data that has been uncommitted
* Solution
  * in addition to above, hold the lock before reading a value and release it again immediately after reading (is some other uncommitted transaction wrote the value, lock will be held by that, and we won't be able to acquire the lock for read)
  * However, a long running write will force read-only transactions to wait until long running transaction has completed. Hence for every object that is written, the database remembers both the old committed value and the new value set by the transaction that currently holds the write lock. While the transaction is ongoing, any other transactions that read the object are simply given the old value.

#### Snapshot isolation and repeatable read

* There are still plenty of ways in which you can have concurrency bugs when using this isolation level.
* _Nonrepeatable read_ or _read skew_, when during the course of a transaction, a row is retrieved twice and the values within the row differ between reads.

```sql
    # say before k value was 100
    BEGIN T1
    read k # returns 100
                                         BEGIN T2  
                                         write k = 200
                                         COMMIT
    read k # returns 200. Non repeatable read.
    COMMIT
```

There are some situations that cannot tolerate such temporal inconsistencies (essentially long running read operations)
* **Backups.** During the time that the backup process is running, writes will continue to be made to the database. If you need to restore from such a backup, inconsistencies can become permanent.
* **Analytic queries and integrity checks.** You may get nonsensical results if they observe parts of the database at different points in time.

* Solution
  * _Snapshot isolation_. Each transaction reads from a _consistent snapshot_ of the database.
  * The implementation of snapshots typically use write locks to prevent dirty writes (writes blocks writes). However, reads does not require locks (thus, readers don't block writers, and writes don't block readers)
  * The database must potentially keep several different committed versions of an object (_multi-version concurrency control_ or MVCC).
  * Read committed uses a separate snapshot for each query, while snapshot isolation uses the same snapshot for an entire transaction.
  * When a transaction is started, it is given always increasing transaction ID (**txid**)
* Implementation
  * ![Snapshot Isolation Implementation](/metadata/snapshot_isolation_implementation.png)
  * Each row in a table has a created_by field, containing the ID of the transaction that inserted this row into the table. Moreover, each row has a deleted_by field, which is initially empty.
  * If a transaction deletes a row, the row isn’t actually deleted from the database, but it is marked for deletion by setting the deleted_by field to the ID of the transaction that requested the deletion. At some later time, when it is certain that no transaction can any longer access the deleted data, a garbage collection process in the database removes any rows marked for deletion and frees their space.
  * An update is internally translated into a delete and a create.
  * MVCC don't take much space, because it is sparse.
* Algorithm
  * **at the start of the transaction**, create list of transactions ongoing
  * consider data only till those transactions whose ID is less than ongoing transaction's ID. Also remove those transaction which are in above list. 
* Indexes and snapshot isolation
  * How do indexes work in a multi-version database? One option is to have the index simply point to all versions of an object and require an index query to filter out any object versions that are not visible to the current transaction. When garbage collection removes old object versions that are no longer visible to any transaction, the corresponding index entries can also be removed.
  * Another approach: **copy on write** variant. With append-only B-trees, every write transaction (or batch of transactions) creates a new B-tree root, and a particular root is a consistent snapshot of the database at the point in time when it was created. There is no need to filter out objects based on transaction IDs because subsequent writes cannot modify an existing B-tree; they can only create new tree roots. However, this approach also requires a background process for compaction and garbage collection

#### Preventing lost updates

```sql
    # say before k value was 100
    BEGIN T1
                                         BEGIN T2 
    read k # returns 100
                                         read k # returns 100
    write k = k + 1 // k is 101
    COMMIT
                                         write k = k + 1 // k is 101
                                         COMMIT
```

* Till now, except dirty writes, we have only discussed how concurrent read and writes affect each other. Now will talk about concurrent writes affecting each other.
* Definition
  * if an application reads some value from the database, modifies it, and writes it back (_read-modify-write_ cycle)
  * If two transactions do this concurrently, one of the modifications can be lost (later write _clobbers_ the earlier write).
  * Example: incrementing a counter, making a local change in JSON document.
* Solutions
  * Atomic write operations
    * Avoid the need to implement read-modify-write cycles and provide atomic operations
    * MongoDB provides atomic operations for making local modifications, and Redis provides atomic operations for modifying data structures such as priority queues
    * Implemented by taking exclusive lock on object when it is read so that no other transaction can read it, until the update has been committed. Called _cursor stability_.

      ```sql
      UPDATE counters SET value = value + 1 WHERE key = 'foo';
      ```
  * Explicit locking
    * The application explicitly lock objects that are going to be updated.
    * The **FOR UPDATE** clause indicates that the database should take a lock on all rows returned by this query.

      ```sql
      BEGIN TRANSACTION;
      
      SELECT * FROM figures WHERE name = 'robot' AND game_id = 222 FOR UPDATE;
      
      -- Check whether move is valid, then update the position
      -- of the piece that was returned by the previous SELECT.
      UPDATE figures SET position = 'c4' WHERE id = 1234;
      
      COMMIT;
      ```

  * Automatically detecting lost updates
    * Allow them to execute in parallel, if the transaction manager detects a lost update, abort the transaction and force it to retry its read-modify-write cycle.
    * An advantage of this approach is that databases can perform this check efficiently in conjunction with snapshot isolation. When writing to a value, we can check if original value (value at beginning of transaction) is same as pre-overwritten value
  * Compare-and-set
    * If the current value does not match with what you previously read, the update has no effect.

    ```SQL
    UPDATE wiki_pages SET content = 'new content'
      WHERE id = 1234 AND content = 'old content';
    ```

**With multi-leader or leaderless replication, compare-and-set and explict lock do not apply. A common approach in replicated databases is to allow concurrent writes to create several conflicting versions of a value (also know as _siblings_), and to use application code or special data structures to resolve and merge these versions after the fact.**

#### Write skew and phantoms

* Write skew can occur if two transactions read the same objects, and then update some of those objects (different transactions may update different objects). In the special case where different transactions update the same object, you get a dirty write or lost update anomaly (depending on the timing).
* Nature
  * SELECT query checks whether some requirement is satisfied
  * Depending on result, application code decides how to continue
  * If application decides to go ahead, it makes a write (INSERT, UPDATE, DELETE). Effect of write is that it changes result of the SELECT query above
* Example
  * Imagine Alice and Bob are two on-call doctors for a particular shift. Imagine both the request to leave because they are feeling unwell. Unfortunately they happen to click the button to go off call at approximately the same time.
  * Since database is using snapshot isolation, both checks return 2. Both transactions commit, and now no doctor is on call. The requirement of having at least one doctor has been violated.

    ```sql
        ALICE                                   BOB
    
        ┌─ BEGIN TRANSACTION                    ┌─ BEGIN TRANSACTION
        │                                       │
        ├─ currently_on_call = (                ├─ currently_on_call = (
        │   select count(*) from doctors        │    select count(*) from doctors
        │   where on_call = true                │    where on_call = true
        │   and shift_id = 1234                 │    and shift_id = 1234
        │  )                                    │  )
        │  // now currently_on_call = 2         │  // now currently_on_call = 2
        │                                       │
        ├─ if (currently_on_call >= 2) {        │
        │    update doctors                     │
        │    set on_call = false                │
        │    where name = 'Alice'               │
        │    and shift_id = 1234                ├─ if (currently_on_call >= 2) {
        │  }                                    │    update doctors
        │                                       │    set on_call = false
        └─ COMMIT TRANSACTION                   │    where name = 'Bob'  
                                                │    and shift_id = 1234
                                                │  }
                                                │
                                                └─ COMMIT TRANSACTION
    ```

* Solution
  * Atomic operations don't help as things involve more objects.
  * Automatically prevent write skew requires true serializable isolation.
  * The second-best option in this case is probably to explicitly lock the rows that the transaction depends on.
    ```sql
    BEGIN TRANSACTION;
  
    SELECT * FROM doctors
    WHERE on_call = true
    AND shift_id = 1234 FOR UPDATE;
  
    UPDATE doctors
    SET on_call = false
    WHERE name = 'Alice'
    AND shift_id = 1234;
  
    COMMIT;
    ```

**Phantoms**
* Nature -- in SELECT query as above in write skew, we are checking for **absence of rows**. Thus, we can't event lock the SELECT query using FOR UPDATE.
* Examples
  * want to enforce that there cannot be two bookings for the same room at the same time.
  * no two users can get same username
* Solution (**materializing conflicts**)
  * artificially introduce lock object into the database.
  * For example, in the meeting room booking case you could imagine creating a table of time slots and rooms. Each row in this table corresponds to a particular room for a particular time period (say, 15 minutes). You create rows for all possible combinations of rooms and time periods ahead of time, e.g. for the next six months

### Serializability

This is the strongest isolation level. It guarantees that even though transactions may execute in parallel, the end result is the same as if they had executed one at a time, _serially_, without concurrency. Basically, the database prevents _all_ possible race conditions. There are three techniques for achieving this

#### Actual serial execution

The simplest way of removing concurrency problems is to remove concurrency entirely and execute only one transaction at a time, in serial order, on a single thread. This approach is implemented by VoltDB/H-Store, Redis and Datomic. Have to do 2 things to make this perfomant:
1. For disk latency: keep data in memory
   * Put entire dataset in memory
2. For network latency: stored procedure
   * OLTP transactions are usually short and only make a small number of reads and writes. Long-running analytic queries are typically readonly, so they can be run on a consistent snapshot (using snapshot isolation) outside of the serial execution loop
   * With interactive style of transaction, a lot of time is spent in network communication between the application and the database.
   * For this reason, systems with single-threaded serial transaction processing don't allow interactive multi-statement transactions. The application must submit the entire transaction code to the database ahead of time, as a _stored procedure_, so all the data required by the transaction is in memory and the procedure can execute very fast.

There are a few cons for stored procedures:
* Each database vendor has its own language for stored procedures. They usually look quite ugly and archaic from today's point of view, and they lack the ecosystem of libraries.
* It's harder to debug, more awkward to keep in version control and deploy, trickier to test, and difficult to integrate with monitoring.

Modern implementations of stored procedures include general-purpose programming languages instead: VoltDB uses Java or Groovy, Datomic uses Java or Clojure, and Redis uses Lua. VoltDB also uses stored procedure for replication: instead of copying a transaction's writes from one node to another, it executes the same stored procedure on each replica, ensuring that stored procedures are deterministic.

##### Partitioning
* Executing all transactions serially limits the transaction throughput to the speed of a single CPU.
* In order to scale to multiple CPU cores you can potentially partition your data and each partition can have its own transaction processing thread. You can give each CPU core its own partition.
* For any transaction that needs to access multiple partitions, the database must coordinate the transaction across all the partitions (distributed transactions using say 2 phase commit). They will be vastly slower than single-partition transactions.

#### Two-phase locking (2PL)

* Two-phase locking (2PL) sounds similar to two-phase _commit_ (2PC) but be aware that they are completely different things.
* Several transactions are allowed to concurrently read the same object as long as nobody is writing it. When somebody wants to write (modify or delete) an object, exclusive access is required.
* Writers don't just block other writers; they also block readers and vice versa. It protects against all the race conditions discussed earlier.

Blocking readers and writers is implemented by a having lock on each object in the database. The lock is used as follows:
* if a transaction wants to read an object, it must first acquire a lock in shared mode. Several txns are allowed to hold the lock in shared mode simultaneously, but if another txn already has an exclusive lock on the object, these txns must wait
* If a transaction wants to write to an object, it must first acquire the lock in exclusive mode. No other txn may hold the lock at the same time; so if there is any existing lock (shared or exclusive), txn must wait.
* If a transaction first reads and then writes an object, it may upgrade its shared lock to an exclusive lock. Applicable if only this txn has the shared lock on the object.
* After a transaction has acquired the lock, it must continue to hold the lock until the end of the transaction (commit or abort). **First phase is when the locks are acquired, second phase is when all the locks are released.**

Properties
* It can happen that transaction A is stuck waiting for transaction B to release its lock, and vice versa (_deadlock_). DB automatically detects deadlocks and aborts one of them so that others can progress. Aborted txn has to be retried by the application.
* The performance for transaction throughput and response time of queries are significantly worse under two-phase locking than under weak isolation.
  * due to overhead of acquiring and releasing locks
  * reduced concurrency
  * too many deadlocks, hence txn has to be aborted and retried again
* A transaction may have to wait for several others to complete before it can do anything.
* Databases running 2PL can have unstable latencies, and they can be very slow at high percentiles. One slow transaction, or one transaction that accesses a lot of data and acquires many locks can cause the rest of the system to halt.

##### Predicate locks

* With _phantoms_, one transaction may change the results of another transaction's search query.
* In order to prevent phantoms, we need a _predicate lock_. Rather than a lock belonging to a particular object, it belongs to all objects that match some search condition.
* Predicate locks applies even to objects that do not yet exist in the database, but which might be added in the future (phantoms).
* Algorithm
  * If txn A wants to read objects matching some conditions, it must acquire shared mode predicate lock on the conditions of the query. If another txn B currently has an exclusive lock on any object matching those conditions, A must wait until B releases its lock before it is allowed to make the query
  * If txn A wants to write (insert, update, delete) any object, it must first check whether either the old or new value matches any existing predicate lock. If there is a matching predicate lock held by txn B, then A must wait until B has committed or aborted before it can continue. 

##### Index-range locks

* Predicate locks do not perform well. Checking for matching locks becomes time-consuming and for that reason most databases implement _index-range locking_.
* It's safe to simplify a predicate by making it match a greater set of objects.
* Example: if predicate lock is to lock rooms from 12pm to 1pm for room 123, then either I can lock room 123 for entire day, or lock all rooms from 12pm to 1pm
* These locks are not as precise as predicate locks would be, but since they have much lower overheads, they are a good compromise.

#### Serializable snapshot isolation (SSI)

* It provides full serializability and has a small performance penalty compared to snapshot isolation. SSI is fairly new and might become the new default in the future.
* Pessimistic versus optimistic concurrency control
  * Two-phase locking is called _pessimistic_ concurrency control because if anything might possibly go wrong, it's better to wait.
  * Serial execution is also _pessimistic_ as is equivalent to each transaction having an exclusive lock on the entire database.
  * Serializable snapshot isolation is _optimistic_ concurrency control (OCC) technique. Instead of blocking if something potentially dangerous happens, transactions continue anyway, in the hope that everything will turn out all right. When a transaction wants to commit, database checks whether anything bad happened. If so, the transaction is aborted and has to be retried.
* If contention between transactions is not too high, optimistic concurrency control techniques tend to perform better than pessimistic ones. For cases of high contention (say creating counter), SSI is bad.
* SSI is based on snapshot isolation, reads within a transaction are made from a consistent snapshot of the database. On top of snapshot isolation, SSI adds an algorithm for detecting serialization conflicts among writes and determining which transactions to abort.

The database needs to assume that any change in the query result (the premise) means thata writes in that transaction may be invalid. In other words there is a causal dependency between the queries and the writes in the transaction:
* **Detecting reads of a stale MVCC object version.**
  * Another txn modifies data, but is not yet committed, which is read by another txn
  * The database needs to track when a transaction ignores another transaction's writes due to MVCC visibility rules. When a transaction wants to commit, the database checks whether any of the ignored writes have now been committed. If so, the transaction must be aborted.

    ```sql
        # say before k value was 100
        BEGIN T1
                                             BEGIN T2 
        SELECT * // query
        UPDATE some val
                                             SELECT * // query -- here I will see that a previous uncommitted txn has changed the data.
                                             UPDATE some val
        COMMIT
                                             COMMIT // database checks whether any of the ignored writes have now been committed. If so, txn must be aborted
    ```

* **Detecting writes that affect prior reads.**
  * Another txn modifies data after it is read by another txn
  * As with two-phase locking, SSI uses index-range locks except that it does not block other transactions. When a transaction writes to the database, it must look in the indexes for any other transactions that have recently read the affected data. It simply notifies the transactions that the data they read may no longer be up to date.

    ```sql
        # say before k value was 100
        BEGIN T1
                                             BEGIN T2 
        SELECT * // query
                                             SELECT * // query
        UPDATE some val // we see that this previous val was seen by T2, so it notifies T2 that T1 is going to change the val 
                                             UPDATE some val // // we see that this previous val was seen by T1, so it notifies T1 that T2 is going to change the val
        COMMIT
                                             COMMIT // database checks the notifications it got, and if those are committed, abort our txn 
    ```
##### Performance of serializable snapshot isolation

* Compared to two-phase locking, the big advantage of SSI is that one transaction doesn't need to block waiting for locks held by another transaction. Writers don't block readers, and vice versa.
* Compared to serial execution, SSI is not limited to the throughput of a single CPU core. Transactions can read and write data in multiple partitions while ensuring serializable isolation.
* The rate of aborts significantly affects the overall performance of SSI. SSI requires that read-write transactions be fairly short (long-running read-only transactions may be okay).

**MySQL uses 2PL, and PostgreSQL uses SSI. MySQL is ACID compliant with few storage engines like InnoDB which implements 2PL, but PostgreSQL is ACID compliant from ground up**

---
---

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

**TCP** (Transmission Control Protocol)
* retransmission of packets, deduplication, ordered (via sequence numbers)
* flow control and congestion control
* error checked (via checksums)
* Properties
  * **3-way handshake**: used to create a connection b/w two nodes over TCP
    * process
      * client sends SYN to server with some random sequence number x
      * server responds with SYN-ACK with sequence number y and acknowledgement value x+1
      * client sends ACK with acknowledgement value y+1
    * sequence number is for ordering of message at server's end
    * acknowledgement value is for telling the client that they have received message and now expect this ack value to be sent as sequence number by client
  * **reliable transmission**
    * after the handshake, we can start sending packets
    * duplicate ack retransmission (getting ack value, but not what we intended)
      * sender sends packet with sequence number 99, gets ack value 100 from receiver
      * sender sends packet with sequence number 100, gets ack value 100 from receiver.
      * Trouble: should have received 101 but got 100 as ack value. Thus we know that sequence number 100 has to be resend
    * timeout based estimates (**exponential backoff**) (not getting ack value back)
      * sender will resend packet if it doesn't receive SYN-ACK within some timeout
      * timeout is based on estimate of round trip time
      * on expiration double the round trip time
      * helps protect against DDOS attacks
  * **flow control**
    * it is possible to overwhelm the "receiver node"
    * receiving device specifies how much space it has in bytes to buffer new messages, and sending device can only send that much more until new acknowledgement comes in
    * if receiving device has no room left, sender waits for some time and then sends a small packet to see how much space is left now
  * **congestion control**
    * it is possible to overwhelm "network links"
    * mitigation:
      * congestion window keeps track of
        * max number of packets that can be sent over the network without being acked.
        * number of packets currently being sent over the network (ones that have not been acknowledged) and limits them.
      * size of window builds up over time using additive increase multiplicative decrease (AIMD) strategy (window increase in size linearly with successful acks, decrease in size exponentially if network overloaded)
  * **connection termination**
    * four way handshake: really just each half of the connection termination independently. Cannot terminate both connections at the same time, due to two generals problem. FIN from client to sender. ACK from sender to client. FIN from sender to client. ACK from client to sender.

UDP (User Datagram Protocol) does not perform reliable transmission or flow control or congestion control or retransmission of packets. Only does checksum. Good for video calls, video games, stock prices. Also we can do multicast (send to bunch of nodes using UDP)

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
* **Synchronous vs asynchronous networks**
   * A telephone network estabilishes a _circuit_, a fixed guaranteed amount of bandwidth is allocated for the call, we say is _synchronous_ even as the data passes through several routers as it does not suffer from queuing. The maximum end-to-end latency of the network is fixed (_bounded delay_).
   * A circuit is a fixed amount of reserved bandwidth which nobody else can use while the circuit is established, whereas packets of a TCP connection opportunistically use whatever network bandwidth is available. While TCP connection is idle, it doesn't use any bandwidth.
   * Internet networks are optimized for bursty traffic.
   * **Using circuits for bursty data transfers wastes network capacity and makes transfer unnecessary slow. By contrast, TCP dynamically adapts the rate of data transfer to the available network capacity.**
   * We have to assume that network congestion, queueing, and unbounded delays will happen. Consequently, there's no "correct" value for timeouts, they need to be determined experimentally.

### Unreliable clocks
* Distributed systems often need to measure time:
  * Determining order of events across several nodes (time when a message is received is always later than the time when it is sent, we don't know how much later due to network delays. This makes difficult to determine the order of which things happened when multiple machines are involved)
  * Cache time invalidation
  * Logging
  * Schedulers, timeouts, failure detector, retry timers
* **Physical Clock**
  * Each machine on the network has its own clock (actual hardware device usually a quartz crystal oscillator), but they are not accurate, due to following reasons:
    * **Clock Drift**
      * The rate by which it is fast or slow is called drift, measured in ppm (parts per million)
      * 1 ppm = 1 microsecond per second = 86 millisecond per day = 32 second per year
      * Most computers are correct within 50 ppm.
      * ![Temperature clock drift](/metadata/temp_clock_drift.png)
    * **Leap seconds**
      * UTC (Universal Coordinated Time) is how time is defined.
        * GMT (Greenwhich Mean Time) -- based on astronomy -- it's noon when the son is in south as seen from Greenwhich Meridian
        * TAI (Time Atomic International) -- based on caesium atomic clocks resonant frequency
        * Above 2 don't match up, because speed of Earth's rotation is not constant.
        * Hence, UTC is TAI with corrections to account for Earth's rotation -- these corrections are called leap seconds.
        * Timezones, etc are then defined on UTC.
      * every year on 30 June and 31 Dec, 1 second is either added, removed or unchanged (immediate jump)
      * poor handling of leap seconds caused outage on 30 June 2012
  * Clock Synchronisation
    * Computer tracks time, but due to clock drift + leap seconds, they are not perfectly synchronized.
    * Difference between two clocks at any point in time is called clock skew. We calculate using following:
      * ![NTP](/metadata/ntp.png)
      * Way of handling clock skew depends on its amount:
        * |θ| < 125 ms, **slew** the clock (slightly speed it up or down by upto 500 ppm -- brings back clock in sync within 5 minutes)
        * 125 ms ≤ |θ| < 1000 seconds, **step** the clock (sudden reset to correct time)
        * |θ| ≥ 1000 seconds, **panic** (do nothing, and let user do it manually -- hence clock skew needs to be carefully monitored)
    * Hence to synchronise clocks, computers do this via Network Time Protocol (NTP)
      * NTP is a stratum of clock servers
        * Stratum 0: atomic clocks or GPS receiver
        * Stratum 1: synced directly with stratum 0
        * Stratum 2: synced directly with stratum 1
* Modern computers have 2 different types of physical clocks
   * **Time-of-day clocks**
     * Return the current date and time according to some calendar (_wall-clock time_) (example UNIX timestamp gives number of seconds since epoch: 1 Jan 1970). Also, they ignore leap seconds
     * `clock_gettime(CLOCK_REALTIME)` on Linux and `System.currentTimeMillis()` in Java.
     * Due to **step**, it may be forcibly reset and appear to jump back to a previous point in time.
     * Unsuitable for measuring elapsed time on a single node
     * Suitable for measuring time across nodes.
   * **Monotonic clocks**
     * Return the current date and time since some arbitrary point. The _absolute_ value of the clock is meaningless (as it might be number of nanoseconds since computer was started, or something similarly arbitrary)
     * `clock_gettime(CLOCK_MONOTONIC)` on Linux and `System.nanoTime()` in Java.
     * Only **skew**, no **step**, hence are guaranteed to always move forward. The difference between clock reads can tell you how much time elapsed beween two checks.
     * Suitable for measuring elapsed time on a single node.
     * Unsuitable for measuring time across nodes.
    
```java
// BAD
long startTime = System.currentTimeMillis();
doSomething();
long endTime = System.currentTimeMillis();
long elapsedTime = endTime - startTime; // may be negative!

// GOOD
long startTime = System.nanoTime();
doSomething();
long endTime = System.nanoTime();
long elapsedTime = endTime - startTime; // always >= 0
```

* If some piece of software is relying on an accurately synchronised clock, the result is more likely to be silent and subtle data loss than a dramatic crash.
* **Timestamps for ordering events**
   * **It is tempting, but dangerous to rely on physical clocks for ordering of events across multiple nodes.** This usually imply that _last write wins_ (LWW), often used in both multi-leader replication and leaderless databases like Cassandra and Riak, and data-loss may happen.
   * **_Logical clocks_**
     * counts number of events occurred (rather than number of seconds elapsed in case of physical clocks)
     * based on counters instead of oscillating quartz crystal, are safer alternative for ordering events. Logical clocks do not measure time of the day or elapsed time, only relative ordering of events. This contrasts with time-of-the-day and monotonic clocks (also known as _physical clocks_).
     * designed to capture causal dependencies. e1 &#8594; e2 ⮕ T(e1) < T(e2)
     * Examples: lamport clocks and vector clocks.
    
##### Lamport Clocks
```java
// Lamport Clocks Algorithm.
// The algorithm assumes a crash-stop model (or a crash-recovery model if the timestamp is maintained in stable storage, i.e. on disk).

on initialisation do
  t := 0 // each node has its own local variable t
end on

on any event occurring at the local node do
  t := t + 1
end on

on request to send message m do
  t := t + 1;
  send (t, m) via the underlying network link
end on

on receiving (T, m) via the underlying network link do
  t := max(t, T) + 1
  deliver m to the application
end on
```
* each node maintains a counter `t`, incremented on every local event `e`
* Let `L(e)` be the value of `t` after the increment
* Properties of this schema
  * If `a` &#8594; `b`, then `L(a)` < `L(b)`
  * If `L(a)` < `L(b)`, does not imply `a` &#8594; `b` (it's definitely not `b` &#8594; `a`, but both possibilities are there i.e. `a` happened before `b` [`a` &#8594; `b`] OR `a` and `b` are concurrent [`a` || `b`]
  * Possible that for `a` != `b`, `L(a)` = `L(b)`
  * It is **partial order** here.
  * Forceful
    * If we need a unique timestamp for every event, each timestamp can be extended with the name or identifier of the node on which that event occurred (trying to define a total order)
    * Within the scope of a single node, each event is assigned a unique timestamp; thus, assuming each node has a unique name, the combination of timestamp and node name is globally unique (across all nodes).
    * Here, it is **total order**.
  * ![Lamport clock example](/metadata/lamport_example.png)
 
##### Vector Clocks
* Given the Lamport timestamps of two events, it is in general not possible to tell whether those events are concurrent or whether one happened before the other (if `L(a)` < `L(b)`, we can't tell whether `a` &#8594; `b` OR `a` || `b`).
* If we do want to detect when events are concurrent, we need a different type of logical time: a vector clock.

```java
// Vector Clocks Algorithm.

on initialisation at node Ni do
  // T[i] is the number of events observed by node Ni
  T := <0, 0, . . . , 0> // local variable at node Ni
end on

on any event occurring at node Ni do
  T[i] := T[i] + 1 
end on

on request to send message m at node Ni do
  T[i] := T[i] + 1;
  send (T, m) via network
end on

on receiving (K, m) at node Ni via the network do
  T[j] := max(T[j], K[j]) for every j ∈ {1, . . . , n}
  T[i] := T[i] + 1; deliver m to the application
end on
```

* ![Vector Clocks example](/metadata/vector_clock_example.png)
* This defines a **partial order** (happens-before relationship)
* Defining the order
  * T = K iff T[i] = K[i] for all i ∈ {1, . . . , n}
  * T ≤ K iff T[i] ≤ K[i] for all i ∈ {1, . . . , n}
  * T < K iff T ≤ K and T != K
  * T || K iff neither T ≤ K nor K ≤ T
* Properties
  * V(a) <  V(b) ⬌ (a &#8594; b)
  * V(a) =  V(b) ⬌ (a = b)
  * V(a) || V(b) ⬌ (a || b)


#### Clock readings have a confidence interval
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
* Unreliable networks with variable delays, unreliable clocks and process pauses makes distributed systems tough.
* Problems in network cannot reliably be distinguished from problems at a node.
* A node cannot necessarily trust its own judgement of a situation. Many distributed systems rely on a _quorum_ (voting among the nodes). That includes decisions about declaring nodes dead. IF a quorum of nodes declared another node dead, it must be considered dead, even if that node still very much feels alive. Individual node must abide by the quorum decision and step down. Commonly, the quorum is an absolute majority of more than half of the nodes.
* Unreliable networks and nodes make it very problematic when we need "one" of something -- one leader, one process holding a distributed lock, one account with username "xyz123!".

![Bad dist lock](/metadata/bad_dist_lock.png)

Client holding the lease is paused for too long, its lease expires. Another client can obtain a lease for the same file, and start writing to the file. When the paused client comes back, it believes (incorrectly) that it still has a valid lease and proceeds to also write to the file. As a result, the clients' writes clash and corrupt the file.

#### Fencing tokens

![Fencing tokens](/metadata/fencing_tokens.png)

* Assume every time the lock server grants a lock or a lease, it also returns a _fencing token_, which is a number that increases every time a lock is granted (incremented by the lock service). Then we can require every time a client sends a write request to the storage service, it must include its current fencing token.
* The storage server remembers that it has already processed a write with a higher token number, so it rejects the request with the last token.
* If ZooKeeper is used as lock service, the transaciton ID `zxid` or the node version `cversion` can be used as a fencing token.
* Checking a token on the server side may seem like a downside, but it is arguably a good thing: it is unwise for a service to assume that tis clients will always be well behaved, because the clients are often run by people whose priorities are very different from the people running the service.

#### Byzantine faults
* We assume that nodes are unreliable, but honest (may be slow or never respond, but we assume that if a node does respond, it is telling the "truth" to the best of its knowledge)
* Distributed systems become much harder if there is a risk that nodes may "lie" (_byzantine fault_), and problem of reaching consensus in this untrusting environment is known as Byzantine General Problem
* A system is _Byzantine fault-tolerant_ if it continues to operate correctly even if some of the nodes are malfunctioning or malicious
 * Aerospace environments where memory or CPU register could be corrupted due to radiation.
 * Multiple participating organisations, some participants may attempt to cheat or defraud others. Peer to peer networks like Bitcoin and other blockchains can be considered to be a way of getting mutually untrusting parties to agree whether a transaction happened or not, without relying on a central authority.
 * In our web applications with datacenter, maybe there is no radiation and nodes can be trusted. We typically don't use Byzantine fault tolerant protocols here, but put some guardrails
   * Make servers the authority of deciding what client behaviour is and isn't allowed.
   * Checksums in TCP and UDP to detect corrupted data.
   * Sanitize user inputs to prevent denial of service attacks.
  
 ### System Model and Reality
 * Timing assumptions:
   * Synchronous model: Bounded network delays, bounded process pauses and bounded clock error. All within a known upper bound. Non realistic model.
   * Partially synchronous model: System behaves like synchronous most of the time, but sometimes exceeds the bounds. Realistic model.
   * Asynchronous model: No bounds at all. **There is not even a clock, so no concept of timeout**
 * Nodes assumptions:
   * Crash-stop faults: Node suddenly stop responding at any moment, and then it never comes back.
   * Crash-recovery faults: Nodes suddently stop responding at any moment, and perhaps start responding again after some unknown time. Nodes are assumed to have stable storage (i.e. nonvolatile disk storage) that is preserved across crashes, whil in memory state is assumed to be lost. Realistic model.
   * Byzantine (arbitrary) faults: Nodes can do anything (be even malicious)
  
Partially synchronous model, with crash recovery faults is a realistic model.

* Correctness of an algorithm
  * To define what it means for an algorithm to be correct, we can describe its properties.
  * There are two types of properties:
    * Safety
      * **nothing bad happens**
      * If safety property is violated, the violation cannot be undone. Havoc!
      * It is required that safety guarantee **always** hold, even if all nodes crash, or entire network fails.
    * Liveness
      * **something good eventually happens**
      * It may not hold at some point in time, but there is always hope that it may be satisfied in the future. Usually include the word "eventually" in their definition (like eventual consistency).
      * Can make caveats that request will get a response if a majority of nodes have not crashed, and only if network evetnually recovers from an outage.
  * For example, if we are generating fencing tokens, we may require algorithm to have following properties:
    * Uniqueness: no two requests for a fencing token return the same value
    * Monotonic sequence: if request A return token x, and request B return token y, and A completed before B began, then x < y.
    * Availability: node that requests a fencing token and does not crash eventually receives a response.
  * The theoritical description of algorithm can declare that certain things are simply assumed not to happen. However, a real implementation may still have to include code to handle the case where something happens that was assumed to be impossible (even if handling is just printing logs or exiting process).

---
---

## Consistency and consensus

The simplest way of handling faults is to simply let the entire service fail. We need to find ways of _tolerating_ faults.

### Consistency guarantees

Write requests arrive on different nodes at different times.

Most replicated databases provide at least _eventual consistency_. The inconsistency is temporary, and eventually resolves itself (_convergence_).

With weak guarantees, you need to be constantly aware of its limitations. Systems with stronger guarantees may have worse performance or be less fault-tolerant than systems with weaker guarantees.

### Linearizability

Make a system appear as if there were only one copy of the data, and all operaitons on it are atomic.

* `read(x) => v` Read from register _x_, database returns value _v_.
* `write(x,v) => r` _r_ could be _ok_ or _error_.

If one client read returns the new value, all subsequent reads must also return the new value.

* `cas(x_old, v_old, v_new) => r` an atomic _compare-and-set_ operation. If the value of the register _x_ equals _v_old_, it is atomically set to _v_new_. If `x != v_old` the registers is unchanged and it returns an error.

**Serializability**: Transactions behave the same as if they had executed _some_ serial order.

**Linearizability**: Recency guarantee on reads and writes of a register (individual object).

#### Locking and leader election

To ensure that there is indeed only one leader, a lock is used. It must be linearizable: all nodes must agree which nodes owns the lock; otherwise is useless.

Apache ZooKeepr and etcd are often used for distributed locks and leader election.

#### Constraints and uniqueness guarantees

Unique constraints, like a username or an email address require a situation similiar to a lock.

A hard uniqueness constraint in relational databases requires linearizability.

#### Implementing linearizable systems

The simplest approach would be to have a single copy of the data, but this would not be able to tolerate faults.

* Single-leader repolication is potentially linearizable.
* Consensus algorithms is linearizable.
* Multi-leader replication is not linearizable.
* Leaderless replication is probably not linearizable.

Multi-leader replication is often a good choice for multi-datacenter replication. On a network interruption betwen data-centers will force a choice between linearizability and availability.

With multi-leader configuraiton, each data center can operate normally with interruptions.

With single-leader replication, the leader must be in one of the datacenters. If the application requires linearizable reads and writes, the network interruption causes the application to become unavailable.

* If your applicaiton _requires_ linearizability, and some replicas are disconnected from the other replicas due to a network problem, the some replicas cannot process request while they are disconnected (unavailable).

* If your application _does not require_, then it can be written in a way tha each replica can process requests independently, even if it is disconnected from other replicas (peg: multi-leader), becoming _available_.

**If an application does not require linearizability it can be more tolerant of network problems.**

#### The unhelpful CAP theorem

CAP is sometimes presented as _Consistency, Availability, Partition tolerance: pick 2 out of 3_. Or being said in another way _either Consistency or Available when Partitioned_.

CAP only considers one consistency model (linearizability) and one kind of fault (_network partitions_, or nodes that are alive but disconnected from each other). It doesn't say anything about network delays, dead nodes, or other trade-offs. CAP has been historically influential, but nowadays has little practical value for designing systems.

---

The main reason for dropping linearizability is _performance_, not fault tolerance. Linearizabilit is slow and this is true all the time, not on only during a network fault.

### Ordering guarantees

Cause comes before the effect. Causal order in the system is what happened before what (_causally consistent_).

_Total order_ allows any two elements to be compared. Peg, natural numbers are totally ordered.

Some cases one set is greater than another one.

Different consistency models:

* Linearizablity. _total order_ of operations: if the system behaves as if there is only a single copy of the data.
* Causality. Two events are ordered if they are causally related. Causality defines _a partial order_, not a total one (incomparable if they are concurrent).

Linearizability is not the only way of preserving causality. **Causal consistency is the strongest possible consistency model that does not slow down due to network delays, and remains available in the face of network failures.**

You need to know which operation _happened before_.

In order to determine the causal ordering, the database needs to know which version of the data was read by the application. **The version number from the prior operation is passed back to the database on a write.**

We can create sequence numbers in a total order that is _consistent with causality_.

With a single-leader replication, the leader can simply increment a counter for each operation, and thus assign a monotonically increasing sequence number to each operation in the replication log.

If there is not a single leader (multi-leader or leaderless database):
* Each node can generate its own independent set of sequence numbers. One node can generate only odd numbers and the other only even numbers.
* Attach a timestamp from a time-of-day clock.
* Preallocate blocks of sequence numbers.

The only problem is that the sequence numbers they generate are _not consistent with causality_. They do not correctly capture ordering of operations across different nodes.

There is simple method for generating sequence numbers that _is_ consistent with causality: _Lamport timestamps_.

Each node has a unique identifier, and each node keeps a counter of the number of operations it has processed. The lamport timestamp is then simply a pair of (_counter_, _node ID_). It provides total order, as if you have two timestamps one with a greater counter value is the greater timestamp. If the counter values are the same, the one with greater node ID is the greater timestamp.

Every node and every client keeps track of the _maximum_ counter value it has seen so far, and includes that maximum on every request. When a node receives a request of response with a maximum counter value greater than its own counter value, it inmediately increases its own counter to that maximum.

As long as the maximum counter value is carried along with every operation, this scheme  ensure that the ordering from the lamport timestamp is consistent with causality.

Total order of oepration only emerges after you have collected all of the operations.

Total order broadcast:
* Reliable delivery: If a message is delivered to one node, it is delivered to all nodes.
* Totally ordered delivery: Mesages are delivered to every node in the same order.

ZooKeeper and etcd implement total order broadcast.

If every message represents a write to the database, and every replica processes the same writes in the same order, then the replcias will remain consistent with each other (_state machine replication_).

A node is not allowed to retroactgively insert a message into an earlier position in the order if subsequent messages have already been dlivered.

Another way of looking at total order broadcast is that it is a way of creating a _log_. Delivering a message is like appending to the log.

If you have total order broadcast, you can build linearizable storage on top of it.

Because log entries are delivered to all nodes in the same order, if therer are several concurrent writes, all nodes will agree on which one came first. Choosing the first of the conflicting writes as the winner and aborting later ones ensures that all nodes agree on whether a write was commited or aborted.

This procedure ensures linearizable writes, it doesn't guarantee linearizable reads.

To make reads linearizable:
* You can sequence reads through the log by appending a message, reading the log, and performing the actual read when the message is delivered back to you (etcd works something like this).
* Fetch the position of the latest log message in a linearizable way, you can query that position to be delivered to you, and then perform the read (idea behind ZooKeeper's `sync()`).
* You can make your read from a replica that is synchronously updated on writes.

For every message you want to send through total order broadcast, you increment-and-get the linearizable integer and then attach the value you got from the register as a sequence number to the message. YOu can send the message to all nodes, and the recipients will deliver the message consecutively by sequence number.

### Distributed transactions and consensus

Basically _getting several nodes to agree on something_.

There are situations in which it is important for nodes to agree:
* Leader election: All nodes need to agree on which node is the leader.
* Atomic commit: Get all nodes to agree on the outcome of the transacction, either they all abort or roll back.

#### Atomic commit and two-phase commit (2PC)

A transaction either succesfully _commit_, or _abort_. Atomicity prevents half-finished results.

On a single node, transaction commitment depends on the _order_ in which data is writen to disk: first the data, then the commit record.

2PC uses a coordinartor (_transaction manager_). When the application is ready to commit, the coordinator begins phase 1: it sends a _prepare_ request to each of the nodes, asking them whether are able to commit.

* If all participants reply "yes", the coordinator sends out a _commit_ request in phase 2, and the commit takes place.
* If any of the participants replies "no", the coordinator sends an _abort_ request to all nodes in phase 2.

When a participant votes "yes", it promises that it will definitely be able to commit later; and once the coordiantor decides, that decision is irrevocable. Those promises ensure the atomicity of 2PC.

If one of the participants or the network fails during 2PC (prepare requests fail or time out), the coordinator aborts the transaction. If any of the commit or abort request fail, the coordinator retries them indefinitely.

If the coordinator fails before sending the prepare requests, a participant can safely abort the transaction.

The only way 2PC can complete is by waiting for the coordinator to revover in case of failure. This is why the coordinator must write its commit or abort decision to a transaction log on disk before sending commit or abort requests to participants.

#### Three-phase commit

2PC is also called a _blocking_ atomic commit protocol, as 2Pc can become stuck waiting for the coordinator to recover.

There is an alternative called _three-phase commit_ (3PC) that requires a _perfect failure detector_.

---

Distributed transactions carry a heavy performance penalty due the disk forcing in 2PC required for crash recovery and additional network round-trips.

XA (X/Open XA for eXtended Architecture) is a standard for implementing two-phase commit across heterogeneous technologies. Supported by many traditional relational databases (PostgreSQL, MySQL, DB2, SQL Server, and Oracle) and message brokers (ActiveMQ, HornetQ, MSQMQ, and IBM MQ).

The problem with _locking_ is that database transactions usually take a row-level exclusive lock on any rows they modify, to prevent dirty writes.

While those locks are held, no other transaction can modify those rows.

When a coordinator fails, _orphaned_ in-doubt transactions do ocurr, and the only way out is for an administrator to manually decide whether to commit or roll back the transaction.

#### Fault-tolerant consensus

One or more nodes may _propose_ values, and the consensus algorithm _decides_ on those values.

Consensus algorithm must satisfy the following properties:
* Uniform agreement: No two nodes decide differently.
* Integrity: No node decides twice.
* Validity: If a node decides the value _v_, then _v_ was proposed by some node.
* Termination: Every node that does not crash eventually decides some value.

If you don't care about fault tolerance, then satisfying the first three properties is easy: you can just hardcode one node to be the "dictator" and let that node make all of the decisions.

The termination property formalises the idea of fault tolerance. Even if some nodes fail, the other nodes must still reach a decision. Termination is a liveness property, whereas the other three are safety properties.

**The best-known fault-tolerant consensus algorithms are Viewstamped Replication (VSR), Paxos, Raft and Zab.**

Total order broadcast requires messages to be delivered exactly once, in the same order, to all nodes.

So total order broadcast is equivalent to repeated rounds of consensus:
* Due to agreement property, all nodes decide to deliver the same messages in the same order.
* Due to integrity, messages are not duplicated.
* Due to validity, messages are not corrupted.
* Due to termination, messages are not lost.

##### Single-leader replication and consensus

All of the consensus protocols dicussed so far internally use a leader, but they don't guarantee that the lader is unique. Protocols define an _epoch number_ (_ballot number_ in Paxos, _view number_ in Viewstamped Replication, and _term number_ in Raft). Within each epoch, the leader is unique.

Every time the current leader is thought to be dead, a vote is started among the nodes to elect a new leader. This election is given an incremented epoch number, and thus epoch numbers are totallly ordered and monotonically increasing. If there is a conflic, the leader with the higher epoch number prevails.

A node cannot trust its own judgement. It must collect votes from a _quorum_ of nodes. For every decision that a leader wants to make, it must send the proposed value to the other nodes and wait for a quorum of nodes to respond in favor of the proposal.

There are two rounds of voting, once to choose a leader, and second time to vote on a leader's proposal. The quorums for those two votes must overlap.

The biggest difference with 2PC, is that 2PC requires a "yes" vote for _every_ participant.

The benefits of consensus come at a cost. The process by which nodes vote on proposals before they are decided is kind of synchronous replication.

Consensus always require a strict majority to operate.

Most consensus algorithms assume a fixed set of nodes that participate in voting, which means that you can't just add or remove nodes in the cluster. _Dynamic membership_ extensions are much less well understood than static membership algorithms.

Consensus systems rely on timeouts to detect failed nodes. In geographically distributed systems, it often happens that a node falsely believes the leader to have failed due to a network issue. This implies frequest leader elecctions resulting in terrible performance, spending more time choosing a leader than doing any useful work.

#### Membership and coordination services

ZooKeeper or etcd are often described as "distributed key-value stores" or "coordination and configuration services".

They are designed to hold small amounts of data that can fit entirely in memory, you wouldn't want to store all of your application's data here. Data is replicated across all the nodes using a fault-tolerant total order broadcast algorithm.

ZooKeeper is modeled after Google's Chubby lock service and it provides some useful features:
* Linearizable atomic operations: Usuing an atomic compare-and-set operation, you can implement a lock.
* Total ordering of operations: When some resource is protected by a lock or lease, you need a _fencing token_ to prevent clients from conflicting with each other in the case of a process pause. The fencing token is some number that monotonically increases every time the lock is acquired.
* Failure detection: Clients maintain a long-lived session on ZooKeeper servers. When a ZooKeeper node fails, the session remains active. When ZooKeeper declares the session to be dead all locks held are automatically released.
* Change notifications: Not only can one client read locks and values, it can also watch them for changes.

ZooKeeper is super useful for distributed coordination.

ZooKeeper/Chubby model works well when you have several instances of a process or service, and one of them needs to be chosen as a leader or primary. If the leader fails, one of the other nodes should take over. This is useful for single-leader databases and for job schedulers and similar stateful systems.

ZooKeeper runs on a fixed number of nodes, and performs its majority votes among those nodes while supporting a potentially large number of clients.

The kind of data managed by ZooKeeper is quite slow-changing like "the node running on 10.1.1.23 is the leader for partition 7". It is not intended for storing the runtime state of the application. If application state needs to be replicated there are other tools (like Apache BookKeeper).

ZooKeeper, etcd, and Consul are also often used for _service discovery_, find out which IP address you need to connect to in order to reach a particular service. In cloud environments, it is common for virtual machines to continually come an go, you often don't know the IP addresses of your services ahead of time. Your services when they start up they register their network endpoints ina  service registry, where they can then be found by other services.

ZooKeeper and friends can be seen as part of a long history of research into _membership services_, determining which nodes are currently active and live members of a cluster.

## Batch processing

Three types of systems
* Service (online): waits for a request, sends a response back. Response time is usually the primary performance metric.
* Batch processing system (offline): takes a large amount of input data, runs a _job_ to process it, and produces some output. Throughput (time taken to crunch data of certain size) is usually the primary performance metric.
* Stream processing systems (near-real-time): somewhere b/w online and offline systems. A stream processor consumes input and produces outputs (rather than responding to requests). A stream job operates on events shortly after they happen, as compared to batch job which operates on a fixed set of input data.

### Batch processing with Unix tools

* We can build a simple log analysis job to get the five most popular pages on your site

  ```
  cat /var/log/nginx/access.log |      # read log file
    awk '{print $7}' |                 # give 7th element on the file (which is the page actually in this example.
    sort             |                 # sort the list
    uniq -c          |                 # group same elements and count them
    sort -r -n       |                 # sort by count (in reverse order)
    head -n 5        |                 # pick top 5
  ```

* You could write the same thing with a simple python program.

  ```
  for line in lines:
    page = line.split(7)
    count[page]++;
  ```

* The difference is that with Unix commands automatically handle larger-than-memory datasets (say `sort` spills dataset to disk) and automatically parallelizes sorting across multiple CPU cores. In the simple python program, `count` has to be kept in memory, and single threaded, and would have to manage everything ourselvse if have to do multithreaded.
* UNIX philosophy
  * Programs must have the same data format to pass information to one another. In Unix, that interface is a file (file descriptor), an ordered sequence of bytes.
  * By convention Unix programs treat this sequence of bytes as ASCII text.
  * Input file immutable.
  * For debugging
    * you can put `less` and look at it to see if it has the expected formk OR
    * write intermediate result to a file, and inspect that, if ok, pipe it to next stage.
* The UNIX approach works best if a program simply uses `stdin` and `stdout`. This allows a shell user to wire up the input and output in whatever way they want; the program doesn't know or care where the input is coming from and where the output is going to. Pipes let you attach the stdout of one process to the stdin of another process (with a small in-memory buffer, and without writing the entire intermediate data stream to disk).

Hadoop consists of 2 components: HDFS (storage) and MapReduce (processing)

### HDFS
* is based on the _shared-nothing_ principle, requiring no special hardware, only computers connected by a conventional datacenter network.
* consists of a daemon process running on each machine, exposing a network service that allows other nodes to access files stored on that machine. A central server called the _NameNode_ keeps track of which file blocks are stored on which machine (_DataNode_). Thus, HDFS creates one big filesystem that can use the space on the disks of all machines running the daemon.
* **Fault Tolerance**: In order to tolerate machine and disk failures, file blocks are replicated on multiple machines. Replication may mean simply several copies of the same data on multiple machines, or an _erasure coding_ scheme such as Reed-Solomon codes, which allow lost data to be recovered with low storage overhead than full replication.
* **Extensibility**: building block for MapReduce, and data flow engines (Spark, Tez, Flink), and other databases (OLTP like HBase, OLAP like Impala)

### MapReduce

* MapReduce is a bit like UNIX tools, but distributed across potentially thousands of machines (single MapReduce job is comparable to a single Unix process).
* Running a MapReduce job normally does not modify the input and does not have any side effects other than producing the output.
* While Unix tools use `stdin` and `stdout` as input and output, MapReduce jobs read and write files on a distributed filesystem. In Hadoop, that filesystem is called HDFS (Hadoop Distributed File System), an open source reimplementation of Google File System (GFS). Other distributed filesystems are GlusterFS, and other object storage services are Amazon S3 and Microsoft Azure Blob Storage.
* is a programming framework with which you can write code to process large datasets in a distributed filesystem like HDFS.
  1. Read a set of input files, and break it up into _records_. This is handled by input parser.
  2. Call the mapper function to extract a key and value from each input record.
  3. Sort all of the key-value pairs by key. This is implicit in MapReduce -- output of mapper is always sorted before it is given to reducer.
  4. Call the reducer function to iterate over the sorted key-value pairs.
* **Mapper**: Called once for every input record, and its job is to extract the key and value from the input record. For each record, can generate any number of k-v pairs (including none). Does not keep any state from one input to other -- all records treated independently.
* **Reducer**: Takes the key-value pairs produced by the mappers, collects all the values belonging to the same key, and calls the reducer with an interator over that collection of values. Can generate any number of records. Output records are written to a file on distributed filesystem
* **Properties**
  * MapReduce programming model has separated the physical network communication aspects of the computation (getting the data to the right machine) from the application logic (processing the data once you have it).
  * MapReduce can parallelise a computation across many machines, without you having to write code to explicitly handle the parallelism. The mapper and reducer only operate on one record at a time; they don't need to know where their input is coming from or their output is going to.
  * In Hadoop MapReduce, the mapper and reducer are each a Java class that implements a particular interface. In MongoDB and CouchDB, they are JavaScript functions.
  * The reduce side of the computation is also partitioned. While the number of map tasks is determined by the number of input file blocks, the number of reduce tasks is configured by the job author. To ensure that all key-value pairs with the same key end up in the same reducer, the framework uses a hash of the key to determine which reducer task should receive a particular key-value pair.
  * The key-value pair must be sorted, but the dataset is likely too large to be sorted with a conventional sorting algorithm on a single machine. Sorting is performed in stages. First, each map task partitions its output by reducer, based on hash of the key. Each of these partitions is written to a sorted file on mapper's local disk (like LSM). Whenever a mapper finishes reading its input file and writing its sorted output files, the MapReduce scheduler notifies the reducers that they can start fetching the output files from that mapper. The reducers connect to each of the mappers and download the files of sorted key-value pairs for their partition. Process of partitioning by reducer, sorting and copying data partitions from mappers to reducers is called **_shuffle_**. The reduce task takes the files from the mappers and merges them together, preserving the sort order.
* **Advantages**
  * The MapReduce scheduler tries to run each mapper on one of the machines that stores a replica of the input file, _putting the computation near the data_ -- saves copying the input file over the network, reducing network load and increasing locality.
  * Can run arbitrary code
  * Designed for fault tolerance

#### Joins
* It is common in datasets for one record to have an association with another record: a _foreign key_ in a relational model, a _document reference_ in a document model, or an _edge_ in graph model. If the query involves joins, it may require multiple index lookups. MapReduce has no concept of indexes.
* When a MapReduce job is given a set of files as input, it reads the entire content of all of those files, like a _full table scan_. In analytics it is common to want to calculate aggregates over a large number of records. Scanning the entire input might be quite reasonable.
* In order to achieve good throughput in a batch process, the computation must be local to one machine. Requests over the network are too slow. Queries to other database for example would be prohibitive. A better approach is to take a copy of the data (eg: the database) and put it in the same distributed filesystem. So, say we have 2 tables, and want to join on a common column -- put both the databases on HDFS.
* GROUP BY is achieved similarly. One common case of grouping is collating all the activity events for a particular user session, this is called sessionization.

##### Sort-merge join
* reducer-side join
* one set of mappers go over one DB (set of files), another set of mappers go over another DB (another set of files). Reducer takes data from both sets of mappers, merges them and do cross join.
* ![Sort merge join](/metadata/sort_merge_join.png)
* Advantages: can always be done
* Disadvantages
  * SLOW (data over the network + shuffle (repartition, sorting and merging))
  * Skew:
    * "Breaking all records with same key to **same place**" is the problem.
    * For example, in a batch process wanting to assess users, we record activity events of user and do join across multiple DBs (a fact table and multiple dimension tables) to get the data. For celebrities, their reducers will be hotspots and can lead to significant _skew_ i.e. they must process significantly more records than the others.
    * Solution
      * The _skewed join_ method in Pig first runs a sampling job to determine which keys are hot and then mapper sends them to not just one reducer, but to several reducers (choosen randomly). Other DB records (dimension tables) related to the hot key need to be replicated to _all_ reducers handling that key. Allows celebrities to be handled parallely across several reducers. In Crunch, hot keys have to be specified explicitly rather than sampling job. For aggegation (like GROUP) here we want to have one value per key, so we can do the total work in two stages, first like explained here so that multiple reducer performs the grouping on a subset of records for the hot key, second,  we can create another MapReduce job which takes data for a celebrity from multiple reducers from above method, and combine them to one value per key. 
      * Hive's skewed join optimisation requries hot keys to be specified explicitly and then it stores hot keys data in separate files, and for them uses **uses map-side join**.

If you _can_ make certain assumptions about your input data, it is possible to make joins faster. A MapReducer job with no reducers and no sorting, each mapper simply reads one input file and writes one output file. This is done using mapper-side joins

##### Broadcast hash join
* mapper-side join
* large dataset joined with small dataset (small enough to be loaded entirely in memory of each mapper). Small input is effectively "broadcast" to all partitions
* Mappers have partitioned fact table records. Dimension table records are small enough to be put on every mapper.

##### Partition hash join
* mapper-side join
* inputs on map-side join are partitioned in the same way, then the hash join approach can be applied to each partition independently.
* only works if both of join's inputs have the same number of partitions, with records assigned to partitions based on the same key and the same hash function

Output of reduce-side join is partitioned and sorted by the join key, whereas the output of a map-side join (broadcast or partition hash join) is partitioned and sorted in the same way as large input.

#### Batch Workflows
* MapReduce jobs can be chained together into _workflows_, the output of one job becomes the input to the next job. In Hadoop this chaining is done implicitly by directory name: the first job writes its output to a designated directory in HDFS, the second job reads that same directory name as its input. A batch job's output is only considered valid when the job has completed successfully (MapReduce discards the partial output of a failed job). Thus, one job in a workflow can only start when the prior jobs have completed successfully.
* Workflow schedulers for Hadoop: Oozie, Luigi
* Batch process is closer to analytics (OLAP) rather than transaction processing (OLTP).
* Google's original use of MapReduce was to build indexes for its search engine. Hadoop MapReduce remains a good way of building indexes for Lucene/Solr. If you need to perform a full-text search, a batch process is very effective way of building indexes: the mappers partition the set of documents as needed, each reducer builds the index for its partition, and the index files are written to the distributed filesystem. It pararellises very well.
* Machine learning systems such as classifiers (spam filters, anamoly detection) and recommendation systems are a common use for batch processing.
* The output of a batch process is often not a report, but some other kind of structure i.e. database. So, how does the output from the batch process get back into a database?
  * Writing from the batch job directly to the database server is a bad idea:
    * Making a network request for every single record is magnitude slower than the normal throughput of a batch task.
    * Mappers or reducers concurrently write to the same output database which can overwhelm the database.
    * You have to worry about the results from partially completed jobs being visible to database.
  * A much better solution is to build a brand-new database _inside_ the batch job and write it as files to the job's output directory, so it can be loaded in bulk into servers that handle read-only queries. Various key-value stores support building database files in MapReduce including Voldemort, Terrapin, ElephanDB and HBase bulk loading.
* By treating inputs as immutable and avoiding side effects (such as writing to external databases), batch jobs not only achieve good performance but also become much easier to maintain.

### Comparing Hadoop to Distributed Databases
* Design principles that worked well for Unix also seem to be working well for Hadoop.
* The MapReduce paper was not at all new. Things we discussed till now had been already implemented in so-called _massively parallel processing_ (MPP) databases.
* Difference
  * **Diversity of storage**
    * Databases require to structure data according to particular model, whereas files in distributed filesystem are just byte sequences, which can be written using any data model and encoding (text, images, videos, genome sequence, any data)
    * Shifts the burden of interpreting the data: instead of forcing the produce of a dataset to bring it into a standardized format, the interpretation of data becomes consumer's problem (schema on read approach)
    * Thus, Hadoop has often been used for implementing ETL processes: data from transaction processing is dumped into distributed filesystem in raw form, MapReduce jobs cleans up the data, transform into relational form, and import it into MPP databases for analytic purposes
  * **Diversity of processing models**
    * Not all processing makes sense with SQL (recommendation systems, ranking models, etc)
    * With time, even both Hadooop and SQL were insufficient, so other models were required.
    * With openness of Hadoop platform, it was possible to build other models on top of it.
      * SQL query execution engine build on top of it, and indeed this is what the Hive project did.
      * OLTP database HBase and OLAP database Impala both use HDFS as storage, but do not use MapReduce.
  * **Designing of frequent faults**
    * handling of faults + memory/disk usage
    * If a node crashes while a query is executing, most MPP databases abort the entire query. MPP databases also prefer to keep as much data as possible in memory.
    * MapReduce can tolerate the failure of a map or reduce task without it affecting the job. It is also very eager to write data to disk, partly for fault tolerance, and partly because the dataset might not fit in memory anyway. Thus, MapReduce is more appropriate for larger jobs.
    * At Google, they use mixed datacenters, online production services and offline batch jobs run on same machines for better resource usage. Online services were high priority and offline batch jobs are low priority. If there is spike in resource consumption in online, offline job can easily be dropped. MapReduce task that runs for an hour has an approximately 5% risk of being terminated to make space for higher-priority process. This is why MapReduce is designed to tolerate frequent unexpected task termination.

### Beyond MapReduce
* Depending on volume of data, structure of data, and type of processing done, other tools are developed.

#### Higher level programming models
* In response to the difficulty of using MapReduce directly, various higher-level programming models emerged on top of it: **Pig, Hive, Cascading, Crunch**. Essentially all abstracts out different components of MapReduce
  * Pig has Pig Latin scripting language to create complex programs easily
  * Hive uses HiveQL (similar semantics to SQL)
  * Cascading uses Java application framework
  * Crunch uses Java library.

#### Data flow engines
Pitfalls of MapReduce
* A MapReduce job can only start when all tasks in the preceding jobs have completed, whereas processes connected by a Unix pipe are started at the same time. Lots of waiting
* In complex tasks like recommendation engine, there are many independent MapReduce jobs taking output of one as input of another and so on. These **intermediate files** are still put on HDFS which requires replication and all. The process of writing out the intermediate state to files is called **_materialisation_**. In contrast, UNIX connects two programs by pipes which do not fully materialize the intermediate results, but streams the output to the input incrementally using in-memory buffer. MapReduce hence uses a ton of disk (store intermediate results + do replication)
* Expensive sorting operation between every mapper and reducer.
* Mappers are often redundant: they just read back the same file that was just written by a reducer; and do expensive partition and sorting thingy. Mapper code could easily be part of previous reducer

To fix these problems with MapReduce, new execution engines for distributed batch computations were developed: **Spark, Tez and Flink**. 
* They can handle an entire workflow as one job, rather than breaking it up into independent subjobs. These are called _dataflow engines_.
* Dataflow engines look much more like UNIX pipes, rather than MapReduce which writes outpout of each command to temporary file.
* They parallelize work by partitioning inputs, and they copy the output of one function over the network to become input to another function. These functions need not to take the strict roles of alternating map and reduce, they are assembled in flexible ways, in functions called **_operators_**.
  * One option is to repartition and sort records by key, like in the shuffle stage of MapReduce. This feature enables sort-merge joins and grouping in the same way as in MapReduce.
  * Another possibility is to take several inputs and to partition them in the same way, but skip the sorting. This saves effort on partitioned hash joins, where the partitioning of records is important but the order is irrelevant because building the hash table randomizes the order anyway.
  * For broadcast hash joins, the same output from one operator can be sent to all partitions of the join operator.
* **Optimizations**
  * operators can start executing as soon as their input is ready; there is no need to wait for the entire preceding stage to finish before the next one starts.
  * can keep intermediate results in memory
  * Expensive operation like sorting can only be performed in places where it is actually required, rather than default between every map and reduce stage
  * No unncessary map tasks
  * Because all joins and dependencies are explicitly declared, can do locality optimisations.
* **Interoperability**: Since operators are a generalization of map and reduce, the same processing code can run on either execution engine: workflows implemented in Pig, Hive, or Cascading can be switched from MapReduce to Tez or Spark with a simple configuration change, without modifying code
* **Fault tolerance**:
  * Spark uses the resilient distributed dataset (RDD)(these are in memory) abstraction for tracking the ancestry of data, while Flink checkpoints operator state, allowing it to resume running an operator that ran into a fault during its execution
  * Spark, Flink, and Tez avoid writing intermediate state to HDFS, so they take a different approach to tolerating faults: if a machine fails and the intermediate state on that machine is lost, it is recomputed from other data that is still available (prior intermediary stage if possible, or otherwise the original input data).
  * Better to make operators deterministic because if it is nondeterministic (O1), and it's output is say consumed by two other operators (O2 and O3), and O2 failed, and we have to recompute O1, then we would have to fail O3 too, because O1 is nondeterministic.
  * If intermediate data is small than original data, or was very CPU intensive to generate, it is better to actually materialize the intermediate result.

##### Apache Spark
Instead of materializing intermediate state, Spark uses RDD (Resilient Distributed Dataset):
* in memory data structure representing the contents of variable in Spark program (in reality the data is probably held across multiple nodes in a cluster
* can create RDDs from data on disk or by performing an operation on another RDD
* tracks lineage of each RDD
* Fault tolerance
  * Narrow dependency: each partition of the child RDD depends on a small number of partitions from the parent RDD. Typically, each partition in the child RDD can be computed using only one partition of the parent RDD. No shuffling of data is required across the network. If a task fails, only the partitions directly linked to that task need to be recomputed.
  * Wide dependency: each partition of the child RDD depends on multiple partitions of the parent RDD. This typically requires data from multiple partitions to be shuffled across the network to create the child RDD. Shuffling or movement of data b/w different nodes is required which is expensive. If a task fails, Spark may need to recompute multiple partitions from the parent RDD, involving a more complex and time-consuming process. Solution: output of such operators which involve inputs from multiple partition to produce result are written to disk, so that if the node fails, this expensive computation is there and we don't have to recompute.
 

#### Graphs and iterative processing
* Dataflow engines like Spark, Flink, and Tez typically arrange the operators in a job as a directed acyclic graph (DAG). This is not the same as graph databases: in dataflow engines, the flow of data from one operator to another is structured as a graph, while the data itself typically consists of relational-style tuples. In graph processing, the data itself has the form of a graph. We are talking here about putting graph data in distributed filesystem.
* It's interesting to look at graphs in batch processing context, where the goal is to perform some kind of offline processing or analysis on an entire graph. This need often arises in machine learning applications such as recommendation engines, or in ranking systems.
* "repeating until done" cannot be expressed in plain MapReduce as it runs in a single pass over the data and some extra trickery is necessary.
* MapReduce does not account for the iterative nature of the algorithm: it will always read the entire input dataset and produce a completely new output dataset, even if only a small part of the graph has changed compared to the last iteration
* An optimisation for batch processing graphs, the _bulk synchronous parallel_ (BSP) has become popular. It is implemented by Apache Giraph, Spark's GraphX API, and Flink's Gelly API. This is also called Pregel model, as Google Pregel paper popularised it.
* One vertex can "send a message" to another vertex, and typically those messages are sent along the edges in a graph. In each iteration, a function is called for each vertex, passing it all the messages that were sent to it—much like a call to the reducer. The difference from MapReduce is that in the Pregel model, a vertex remembers its state in memory from one iteration to the next, so the function only needs to process new incoming messages. If no messages are being sent in some part of the graph, no work needs to be done.
* The fact that vertices can only communicate by message passing helps improve the performance of Pregel jobs, since messages can be batched.
* Fault tolerance is achieved by periodically checkpointing the state of all vertices at the end of an interation.
* The framework may partition the graph in arbitrary ways.
* Graph algorithms often have a lot of cross-machine communication overhead, and the intermediate state is often bigger than the original graph.
* If your graph can fit into memory on a single computer, it's quite likely that a single-machine algorithm will outperform a distributed batch process. If the graph is too big to fit on a single machine, a distributed approach such as Pregel is unavoidable.

In summary, think of batch processing with 2 requirements
1. Partitioning
   * In MapReduce, mappers are partitioned according to input file blocks. The output of mappers is repartitioned, sorted, and merged into a configurable number of reducer partitions. The purpose of this process is to bring all the related data e.g., all the records with the same key—together in the same place.
   * Dataflow engines try to avoid sorting unless it is required, but they otherwise take a broadly similar approach to partitioning.
2. Fault Tolerance
   * MapReduce frequently writes to disk, which makes it easy to recover from an individual failed task without restarting the entire job but slows down execution in the failure-free case.
   * Dataflow engines perform less materialization of intermediate state and keep more in memory, which means that they need to recompute more data if a node fails. Deterministic operators reduce the amount of data that needs to be recomputed.
  
---
---

## Stream processing

We can run the processing continuously, abandoning the fixed time slices entirely and simply processing every event as it happens, that's the idea behind _stream processing_. Data that is incrementally made available over time.

### Transmitting event streams

A record is more commonly known as an _event_. Something that happened at some point in time, it usually contains a timestamp indicating when it happened acording to a time-of-day clock.

An event is generated once by a _producer_ (_publisher_ or _sender_), and then potentially processed by multiple _consumers_ (_subcribers_ or _recipients_). Related events are usually grouped together into a _topic_ or a _stream_.

A file or a database is sufficient to connect producers and consumers: a producer writes every event that it generates to the datastore, and each consumer periodically polls the datastore to check for events that have appeared since it last ran.

However, when moving toward continual processing, polling becomes expensive. It is better for consumers to be notified when new events appear.

Databases offer _triggers_ but they are limited, so specialised tools have been developed for the purpose of delivering event notifications.

#### Messaging systems

##### Direct messaging from producers to consumers

Within the _publish_/_subscribe_ model, we can differentiate the systems by asking two questions:
1. _What happens if the producers send messages faster than the consumers can process them?_ The system can drop messages, buffer the messages in a queue, or apply _backpressure_ (_flow control_, blocking the producer from sending more messages).
2. _What happens if nodes crash or temporarily go offline, are any messages lost?_ Durability may require some combination of writing to disk and/or replication.

A number of messaging systems use direct communication between producers and consumers without intermediary nodes:
* UDP multicast, where low latency is important, application-level protocols can recover lost packets.
* Brokerless messaging libraries such as ZeroMQ
* StatsD and Brubeck use unreliable UDP messaging for collecting metrics
* If the consumer expose a service on the network, producers can make a direct HTTP or RPC request to push messages to the consumer. This is the idea behind webhooks, a callback URL of one service is registered with another service, and makes a request to that URL whenever an event occurs

These direct messaging systems require the application code to be aware of the possibility of message loss. The faults they can tolerate are quite limited as they assume that producers and consumers are constantly online.

If a consumer if offline, it may miss messages. Some protocols allow the producer to retry failed message deliveries, but it may break down if the producer crashes losing the buffer or messages.

##### Message brokers

An alternative is to send messages via a _message broker_ (or _message queue_), which is a kind of database that is optimised for handling message streams. It runs as a server, with producers and consumers connecting to it as clients. Producers write messages to the broker, and consumers receive them by reading them from the broker.

By centralising the data, these systems can easily tolerate clients that come and go, and the question of durability is moved to the broker instead. Some brokers only keep messages in memory, while others write them down to disk so that they are not lost inc ase of a broker crash.

A consequence of queueing is that consuemrs are generally _asynchronous_: the producer only waits for the broker to confirm that it has buffered the message and does not wait for the message to be processed by consumers.

Some brokers can even participate in two-phase commit protocols using XA and JTA. This makes them similar to databases, aside some practical differences:
* Most message brokers automatically delete a message when it has been successfully delivered to its consumers. This makes them not suitable for long-term storage.
* Most message brokers assume that their working set is fairly small. If the broker needs to buffer a lot of messages, each individual message takes longer to process, and the overall throughput may degrade.
* Message brokers often support some way of subscribing to a subset of topics matching some pattern.
* Message brokers do not support arbitrary queries, but they do notify clients when data changes.

This is the traditional view of message brokers, encapsulated in standards like JMS and AMQP, and implemented in RabbitMQ, ActiveMQ, HornetQ, Qpid, TIBCO Enterprise Message Service, IBM MQ, Azure Service Bus, and Google Cloud Pub/Sub.

When multiple consumers read messages in the same topic, to main patterns are used:
* Load balancing: Each message is delivered to _one_ of the consumers. The broker may assign messages to consumers arbitrarily.
* Fan-out: Each message is delivered to _all_ of the consumers.

In order to ensure that the message is not lost, message brokers use _acknowledgements_: a client must explicitly tell the broker when it has finished processing a message so that the broker can remove it from the queue.

The combination of laod balancing with redelivery inevitably leads to messages being reordered. To avoid this issue, youc an use a separate queue per consumer (not use the load balancing feature).

##### Partitioned logs

A key feature of barch process is that you can run them repeatedly without the risk of damaging the input. This is not the case with AMQP/JMS-style messaging: receiving a message is destructive if the acknowledgement causes it to be deleted from the broker.

If you add a new consumer to a messaging system, any prior messages are already gone and cannot be recovered.

We can have a hybrid, combining the durable storage approach of databases with the low-latency notifications facilities of messaging, this is the idea behind _log-based message brokers_.

A log is simply an append-only sequence of records on disk. The same structure can be used to implement a message broker: a producer sends a message by appending it to the end of the log, and consumer receives messages by reading the log sequentially. If a consumer reaches the end of the log, it waits for a notification that a new message has been appended.

To scale to higher throughput than a single disk can offer, the log can be _partitioned_. Different partitions can then be hosted on different machines. A topic can then be defined as a group of partitions that all carry messages of the same type.

Within each partition, the broker assigns monotonically increasing sequence number, or _offset_, to every message.

Apache Kafka, Amazon Kinesis Streams, and Twitter's DistributedLog, are log-based message brokers that work like this.

The log-based approach trivially supports fan-out messaging, as several consumers can independently read the log reading without affecint each other. Reading a message does not delete it from the log. To eachieve load balancing the broker can assign entire partitions to nodes in the consumer group. Each client then consumes _all_ the messages in the partition it has been assigned. This approach has some downsides.
* The number of nodes sharing the work of consuming a topic can be at most the number of log partitions in that topic.
* If a single message is slow to process, it holds up the processing of subsequent messages in that partition.

In situations where messages may be expensive to process and you want to pararellise processing on a message-by-message basis, and where message ordering is not so important, the JMS/AMQP style of message broker is preferable. In situations with high message throughput, where each message is fast to process and where message ordering is important, the log-based approach works very well.

It is easy to tell which messages have been processed: al messages with an offset less than a consumer current offset have already been processed, and all messages with a greater offset have not yet been seen.

The offset is very similar to the _log sequence number_ that is commonly found in single-leader database replication. The message broker behaves like a leader database, and the consumer like a follower.

If a consumer node fails, another node in the consumer group starts consuming messages at the last recorded offset. If the consumer had processed subsequent messages but not yet recorded their offset, those messages will be processed a second time upon restart.

If you only ever append the log, you will eventually run out of disk space. From time to time old segments are deleted or moved to archive.

If a slow consumer cannot keep with the rate of messages, and it falls so far behind that its consumer offset poitns to a deleted segment, it will miss some of the messages.

The throughput of a log remains more or less constant, since every message is written to disk anyway. This is in contrast to messaging systems that keep messages in memory by default and only write them to disk if the queue grows too large: systems are fast when queues are short and become much slower when they start writing to disk, throughput depends on the amount of history retained.

If a consumer cannot keep up with producers, the consumer can drop messages, buffer them or applying backpressure.

You can monitor how far a consumer is behind the head of the log, and raise an alert if it falls behind significantly.

If a consumer does fall too far behind and start missing messages, only that consumer is affected.

With AMQP and JMS-style message brokers, processing and acknowledging messages is a destructive operation, since it causes the messages to be deleted on the broker. In a log-based message broker, consuming messages is more like reading from a file.

The offset is under the consumer's control, so you can easily be manipulated if necessary, like for replaying old messages.

### Databases and streams

A replciation log is a stream of a database write events, produced by the leader as it processes transactions. Followers apply that stream of writes to their own copy of the database and thus end up with an accurate copy of the same data.

If periodic full database dumps are too slow, an alternative that is sometimes used is _dual writes_. For example, writing to the database, then updating the search index, then invalidating the cache.

Dual writes have some serious problems, one of which is race conditions. If you have concurrent writes, one value will simply silently overwrite another value.

One of the writes may fail while the other succeeds and two systems will become inconsistent.

The problem with most databases replication logs is that they are considered an internal implementation detail, not a public API.

Recently there has been a growing interest in _change data capture_ (CDC), which is the process of observing all data changes written to a database and extracting them in a form in which they can be replicated to other systems.

For example, you can capture the changes in a database and continually apply the same changes to a search index.

We can call log consumers _derived data systems_: the data stored in the search index and the data warehouse is just another view. Change data capture is a mechanism for ensuring that all changes made to the system of record are also reflected in the derived data systems.

Change data capture makes one database the leader, and turns the others into followers.

Database triggers can be used to implement change data capture, but they tend to be fragile and have significant performance overheads. Parsing the replication log can be a more robust approach.

LinkedIn's Databus, Facebook's Wormhole, and Yahoo!'s Sherpa use this idea at large scale. Bottled Watter implements CDC for PostgreSQL decoding the write-ahead log, Maxwell and Debezium for something similar for MySQL by parsing the binlog, Mongoriver reads the MongoDB oplog, and GoldenGate provide similar facilities for Oracle.

Keeping all changes forever would require too much disk space, and replaying it would take too long, so the log needs to be truncated.

You can start with a consistent snapshot of the database, and it must correspond to a known position or offset in the change log.

The storage engine periodically looks for log records with the same key, throws away any duplicates, and keeps only the most recent update for each key.

An update with a special null value (a _tombstone_) indicates that a key was deleted.

The same idea works in the context of log-based mesage brokers and change data capture.

RethinkDB allows queries to subscribe to notifications, Firebase and CouchDB provide data synchronisation based on change feed.

Kafka Connect integrates change data capture tools for a wide range of database systems with Kafka.

#### Event sourcing

There are some parallels between the ideas we've discussed here and _event sourcing_.

Similarly to change data capture, event sourcing involves storing all changes to the application state as a log of change events. Event sourcing applyies the idea at a different level of abstraction.

Event sourcing makes it easier to evolve applications over time, helps with debugging by making it easier to understand after the fact why something happened, and guards against application bugs.

Specialised databases such as Event Store have been developed to support applications using event sourcing.

Applications that use event sourcing need to take the log of evetns and transform it into application state that is suitable for showing to a user.

Replying the event log allows you to reconstruct the current state of the system.

Applications that use event sourcing typically have some mechanism for storing snapshots.

Event sourcing philosophy is careful to distinguis between _events_ and _commands_. When a request from a user first arrives, it is initially a command: it may still fail (like some integrity condition is violated). If the validation is successful, it becomes an event, which is durable and immutable.

A consumer of the event stream is not allowed to reject an event: Any validation of a command needs to happen synchronously, before it becomes an event. For example, by using a serializable transaction that atomically validates the command and publishes the event.

Alternatively, the user request to serve a seat could be split into two events: first a tentative reservation, and then a separate confirmation event once the reservation has been validated. This split allows the validation to take place in an asynchronous process.

Whenever you have state changes, that state is the result of the events that mutated it over time.

Mutable state and an append-only log of immutable events do not contradict each other.

As an example, financial bookkeeping is recorded as an append-only _ledger_. It is a log of events describing money, good, or services that have changed hands. Profit and loss or the balance sheet are derived from the ledger by adding them up.

If a mistake is made, accountants don't erase or change the incorrect transaction, instead, they add another transaction that compensates for the mistake.

If buggy code writes bad data to a database, recovery is much harder if the code is able to destructively overwrite data.

Immutable events also capture more information than just the current state. If you persisted a cart into a regular database, deleting an item would effectively loose that event.

You can derive views from the same event log, Druid ingests directly from Kafka, Pistachio is a distributed key-value sotre that uses Kafka as a commit log, Kafka Connect sinks can export data from Kafka to various different databases and indexes.

Storing data is normally quite straightforward if you don't have to worry about how it is going to be queried and accessed. You gain a lot of flexibility by separating the form in which data is written from the form it is read, this idea is known as _command query responsibility segregation_ (CQRS).

There is this fallacy that data must be written in the same form as it will be queried.

The biggest downside of event sourcing and change data capture is that consumers of the event log are usually asynchronous, a user may make a write to the log, then read from a log derived view and find that their write has not yet been reflected.

The limitations on immutable event history depends on the amount of churn in the dataset. Some workloads mostly add data and rarely update or delete; they are wasy to make immutable. Other workloads have a high rate of updates and deletes on a comparaively small dataset; in these cases immutable history becomes an issue because of fragmentation, performance compaction and garbage collection.

There may also be circumstances in which you need data to be deleted for administrative reasons.

Sometimes you may want to rewrite history, Datomic calls this feature _excision_.

### Processing Streams

What you can do with the stream once you have it:
1. You can take the data in the events and write it to the database, cache, search index, or similar storage system, from where it can thenbe queried by other clients.
2. You can push the events to users in some way, for example by sending email alerts or push notifications, or to a real-time dashboard.
3. You can process one or more input streams to produce one or more output streams.

Processing streams to produce other, derived streams is what an _operator job_ does. The one crucial difference to batch jobs is that a stream never ends.

_Complex event processing_ (CEP) is an approach for analising event streams where you can specify rules to search for certain patterns of events in them.

When a match is found, the engine emits a _complex event_.

Queries are stored long-term, and events from the input streams continuously flow past them in search of a query that matches an event pattern.

Implementations of CEP include Esper, IBM InfoSphere Streams, Apama, TIBCO StreamBase, and SQLstream.

The boundary between CEP and stream analytics is blurry, analytics tends to be less interested in finding specific event sequences and is more oriented toward aggregations and statistical metrics.

Frameworks with analytics in mind are: Apache Storm, Spark Streaming, Flink, Concord, Samza, and Kafka Streams. Hosted services include Google Cloud Dataflow and Azure Stream Analytics.

Sometimes there is a need to search for individual events continually, such as full-text search queries over streams.

Message-passing ystems are also based on messages and events, we normally don't think of them as stream processors.

There is some crossover area between RPC-like systems and stream processing. Apache Storm has a feature called _distributed RPC_.

In a batch process, the time at which the process is run has nothing to do with the time at which the events actually occurred.

Many stream processing frameworks use the local system clock on the processing machine (_processing time_) to determine windowing. It is a simple approach that breaks down if there is any significant processing lag.

Confusing event time and processing time leads to bad data. Processing time may be unreliable as the stream processor may queue events, restart, etc. It's better to take into account the original event time to count rates.

You can never be sure when you have received all the events.

You can time out and declare a window ready after you have not seen any new events for a while, but it could still happen that some events are delayed due a network interruption. You need to be able to handle such _stranggler_ events that arrive after the window has already been declared complete.

1. You can ignore the stranggler events, tracking the number of dropped events as a metric.
2. Publish a _correction_, an updated value for the window with stranglers included. You may also need to retrat the previous output.

To adjust for incofrrect device clocks, one approach is to log three timestamps:
* The time at which the event occurred, according to the device clock
* The time at which the event was sent to the server, according to the device clock
* The time at which the event was received by the server, according to the server clock.

You can estimate the offset between the device clock and the server clock, then apply that offset to the event timestamp, and thus estimate the true time at which the event actually ocurred.

Several types of windows are in common use:
* Tumbling window: Fixed length. If you have a 1-minute tumbling window, all events between 10:03:00 and 10:03:59 will be grouped in one window, next window would be 10:04:00-10:04:59
* Hopping window: Fixed length, but allows windows to overlap in order to provide some smoothing. If you have a 5-minute window with a hop size of 1 minute, it would contain the events between 10:03:00 and 10:07:59, next window would cover 10:04:00-10:08:59
* Sliding window: Events that occur within some interval of each other. For example, a 5-minute sliding window would cover 10:03:39 and 10:08:12 because they are less than 4 minutes apart.
* Session window: No fixed duration. All events for the same user, the window ends when the user has been inactive for some time (30 minutes). Common in website analytics

The fact that new events can appear anytime on a stream makes joins on stream challenging.

#### Stream-stream joins

You want to detect recent trends in searched-for URLs. You log an event containing the query. Someone clicks one of the search results, you log another event recording the click. You need to bring together the events for the search action and the click action.

For this type of join, a stream processor needs to maintain _state_: All events that occurred in the last hour, indexed by session ID. Whenever a search event or click event occurs, it is added to the appropriate index, and the stream processor also checks the other index to see if another event for the same session ID has already arrived. If there is a matching event, you emit an event saying search result was clicked.

#### Stream-table joins

Sometimes know as _enriching_ the activity events with information from the database.

Imagine two datasets: a set of usr activity events, and a database of user profiles. Activity events include the user ID, and the the resulting stream should have the augmented profile information based upon the user ID.

The stream process needs to look at one activity event at a time, look up the event's user ID in the database, and add the profile information to the activity event. THe database lookup could be implemented by querying a remote database., however this would be slow and risk overloading the database.

Another approach is to load a copy of the database into the stream processor so that it can be queried locally without a network round-trip. The stream processor's local copy of the database needs to be kept up to date; this can be solved with change data capture.

#### Table-table join

The stream process needs to maintain a database containing the set of followers for each user so it knows which timelines need to be updated when a new tweet arrives.

#### Time-dependence join

The previous three types of join require the stream processor to maintain some state.

If state changes over time, and you join with some state, what point in time do you use for the join?

If the ordering of events across streams is undetermined, the join becomes nondeterministic.

This issue is known as _slowly changing dimension_ (SCD), often addressed by using a unique identifier for a particular version of the joined record. For example, we can turn the system deterministic if every time the tax rate changes, it is given a new identifier, and the invoice includes the identifier for the tax rate at the time of sale. But as a consequence makes log compation impossible.

#### Fault tolerance

Batch processing frameworks can tolerate faults fairly easy:if a task in a MapReduce job fails, it can simply be started again on another machine, input files are immutable and the output is written to a separate file.

Even though restarting tasks means records can be processed multiple times, the visible effect in the output is as if they had only been processed once (_exactly-once-semantics_ or _effectively-once_).

With stream processing waiting until a tasks if finished before making its ouput visible is not an option, stream is infinite.

One solution is to break the stream into small blocks, and treat each block like a minuature batch process (_micro-batching_). This technique is used in Spark Streaming, and the batch size is typically around one second.

An alternative approach, used in Apache Flint, is to periodically generate rolling checkpoints of state and write them to durable storage. If a stream operator crashes, it can restart from its most recent checkpoint.

Microbatching and chekpointing approaches provide the same exactly-once semantics as batch processing. However, as soon as output leaves the stream processor, the framework is no longer able to discard the output of a failed batch.

In order to give appearance of exactly-once processing, things either need to happen atomically or none of must happen. Things should not go out of sync of each other. Distributed transactions and two-phase commit can be used.

This approach is used in Google Cloud Dataflow and VoltDB, and there are plans to add similar features to Apache Kafka.

Our goal is to discard the partial output of failed tasks so that they can be safely retired without taking effect twice. Distributed transactions are one way of achieving that goal, but another way is to rely on _idempotence_.

An idempotent operation is one that you can perform multiple times, and it has the same effect as if you performed it only once.

Even if an operation is not naturally idempotent, it can often be made idempotent with a bit of extra metadata. You can tell wether an update has already been applied.

Idempotent operations can be an effective way of achieving exactly-once semantics with only a small overhead.

Any stream process that requires state must ensure tha this state can be recovered after a failure.

One option is to keep the state in a remote datastore and replicate it, but it is slow.

An alternative is to keep state local to the stream processor and replicate it periodically.

Flink periodically captures snapshots and writes them to durable storage such as HDFS; Samza and Kafka Streams replicate state changes by sending them to a dedicated Kafka topic with log compaction. VoltDB replicates state by redundantly processing each input message on several nodes.

## The future of data systems

### Data integration

Updating a derived data system based on an event log can often be made determinisitic and idempotent.

Distributed transactions decide on an ordering of writes by using locks for mutual exclusion, while CDC and event sourcing use a log for ordering. Distributed transactions use atomic commit to ensure exactly once semantics, while log-based systems are based on deterministic retry and idempotence.

Transaction systems provide linearizability, useful guarantees as reading your own writes. On the other hand, derived systems are often updated asynchronously, so they do not by default offer the same timing guarantees.

In the absence of widespread support for a good distributed transaction protocol, log-based derived data is the most promising approach for integrating different data systems.

However, as systems are scaled towards bigger and more coplex worloads, limitiations emerge:
* Constructing a totally ordered log requires all events to pass through a _single leader node_ that decides on the ordering.
* An undefined ordering of events that originate on multiple datacenters.
* When two events originate in different services, there is no defined order for those events.
* Some applications maintain client-side state. Clients and servers are very likely to see events in different orders.

Deciding on a total order of events is known as _total order broadcast_, which is equivalent to consensus. It is still an open research problem to design consensus algorithms that can scale beyond the throughput of a single node.

#### Batch and stream processing

The fundamental difference between batch processors and batch processes is that the stream processors operate on unbounded datasets whereas batch processes inputs are of a known finite size.

Spark performs stream processing on top of batch processing. Apache Flink performs batch processing in top of stream processing.

Batch processing has a quite strong functional flavour. The output depends only on the input, there are no side-effects. Stream processing is similar but it allows managed, fault-tolerant state.

Derived data systems could be maintained synchronously. However, asynchrony is what makes systems based on event logs robust: it allows a fault in one part of the system to be contained locally.

Stream processing allows changes in the input to be reflected in derived views with low delay, whereas batch processing allows large amounts of accumulated historical data to be reprocessed in order to derive new views onto an existing dataset.

Derived views allow _gradual_ evolution. If you want to restructure a dataset, you do not need to perform the migration as a sudden switch. Instead, you can maintain the old schema and the new schema side by side as two independent derived views onto the same underlying data, eventually you can drop the old view.

#### Lambda architecture

The whole idea behind lambda architecture is that incoming data should be recorded by appending immutable events to an always-growing dataset, similarly to event sourcing. From these events, read-optimised vuews are derived. Lambda architecture proposes running two different systems in parallel: a batch processing system such as Hadoop MapReduce, and a stream-processing system as Storm.

The stream processor produces an approximate update to the view: the batch processor produces a corrected version of the derived view.

The stream process can use fast approximation algorithms while the batch process uses slower exact algorithms.

### Unbundling databases

#### Creating an index

Batch and stream processors are like elaborate implementations of triggers, stored procedures, and materialised view maintenance routines. The derived data systems they maintain are like different index types.

There are two avenues by which different storate and processing tools can nevertheless be composed into a cohesive system:
* Federated databases: unifying reads. It is possible to provide a unified query interface to a wide variety of underlying storate engines and processing methods, this is known as _federated database_ or _polystore_. An example is PostgreSQL's _foreign data wrapper_.
* Unbundled databases: unifying writes. When we compose several storage systems, we need to ensure that all data changes end up in all the right places, even in the face of faults, it is like _unbundling_ a database's index-maintenance features in a way that can synchronise writes across disparate technologies.

Keeping the writes to several storage systems in sync is the harder engineering problem.

Synchronising writes requires distributed transactions across heterogeneous storage systems which may be the wrong solution. An asynchronous event log with idempotent writes is a much more robust and practical approach.

The big advantage is _loose coupling_ between various components:
1. Asynchronous event streams make the system as a whole more robust to outages or performance degradation of individual components.
2. Unbundling data systems allows different software components and services to be developed, improved and maintained independently from each other by different teams.

If there is a single technology that does everything you need, you're most likely best off simply using that product rather than trying to reimplement it yourself from lower-level components. The advantages of unbundling and composition only come into the picture when there is no single piece of software that satisfies all your requirements.

#### Separation of application code and state

It makes sense to have some parts of a system that specialise in durable data storage, and other parts that specialise in running application code. The two can interact while still remaining independent.

The trend has been to keep stateless application logic separate from state management (databases): not putting application logic in the database and not putting persistent state in the application.

#### Dataflow, interplay between state changes and application code

Instead of treating the database as a passive variable that is manipulated by the application, application code responds to state changes in one place by triggering state changes in another place.

#### Stream processors and services

A customer is purchasing an item that is priced in one currency but paid in another currency. In order to perform the currency conversion, you need to know the current exchange rate.

This could be implemented in two ways:
* Microservices approach, the code that processes the purchase would probably wuery an exchange-rate service or a database in order to obtain the current rate for a particular currency.
* Dataflow approach, the code that processes purchases would subscribe to a stream of exchange rate updates ahead of time, and record the current rate in a local database whenever it changes. When it comes to processing the purchase, it only needs to query the local database.

The dataflow is not only faster, but it is also more robust to the failure of another service.

#### Observing derived state

##### Materialised views and caching

A full-text search index is a good example: the write path updates the index, and the read path searches the index for keywords.

If you don't have an index, a search query would have to scan over all documents, which is very expensive. No index means less work on the write path (no index to update), but a lot more work on the read path.

Another option would be to precompute the search results for only a fixed set of the most common queries. The uncommon queries can still be served from the inxed. This is what we call a _cache_ although it could also be called a materialised view.

##### Read are events too

It is also possible to represent read requests as streams of events, and send both the read events and write events through a stream processor; the processor responds to read events by emiting the result of the read to an output stream.

It would allow you to reconstruct what the user saw before they made a particular decision.

Enables better tracking of casual dependencies.

### Aiming for correctness

If your application can tolerate occasionally corrupting or losing data in unpredictable ways, life is a lot simpler. If you need stronger assurances of correctness, the serializability and atomic commit are established approaches.

While traditional transaction approach is not going away, there are some ways of thinking about correctness in the context of dataflow architectures.

#### The end-to-end argument for databases

Bugs occur, and people make mistakes. Favour of immutable and append-only data, because it is easier to recover from such mistakes.

We've seen the idea of _exactly-once_ (or _effectively-once_) semantics. If something goes wrong while processing a message, you can either give up or try again. If you try again, there is the risk that it actually succeeded the first time, the message ends up being processed twice.

_Exactly-once_ means arranging the computation such that the final effect is the same as if no faults had occurred.

One of the most effective approaches is to make the operation _idempotent_, to ensure that it has the same effect, no matter whether it is executed once or multiple times. Idempotence requires some effort and care: you may need to maintain some additional metadata (operation IDs), and ensure fencing when failing over from one node to another.

Two-phase commit unfortunately is not sufficient to ensure that the transaction will only be executed once.

You need to consider _end-to-end_ flow of the request.

You can generate a unique identifier for an operation (such as a UUID) and include it as a hidden form field in the client application, or calculate a hash of all the relevant form fields to derive the operation ID. If the web browser submits the POST request twice, the two requests will have the same operation ID. You can then pass that operation ID all the way through to the database and check that you only ever execute one operation with a given ID. You can then save those requests to be processed, uniquely identified by the operation ID.

Is not enough to prevent a user from submitting a duplicate request if the first one times out. Solving the problem requires an end-to-end solution: a transaction indentifier that is passed all the way from the end-user client to the database.

Low-level reliability mechanisms such as those in TCP, work quite well, and so the remaining higher-level faults occur fairly rarely.

Transactions have long been seen as a good abstraction, they are useful but not enough.

It is worth exploring F=fault-tolerance abstractions that make it easy to provide application-specific end-to-end correctness properties, but also maintain good performance and good operational characteristics.

#### Enforcing constraints

##### Uniqueness constraints require consensus

The most common way of achieving consensus is to make a single node the leadder, and put it in charge of making all decisions. If you need to tolerate the leader failing, you're back at the consensus problem again.

Uniqueness checking can be scaled out by partitioning based on the value that needs to be unique. For example, if you need usernames to be unique, you can partition by hash or username.

Asynchronous multi-master replication is ruled out as different masters concurrently may accept conflicting writes, so values are no longer unique. If you want to be able to immediately reject any writes that would violate the constraint, synchronous coordination is unavoidable.

##### Uniqueness in log-based messaging

A stream processor consumes all the messages in a log partition sequentially on a single thread. A stream processor can unambiguously and deterministically decide which one of several conflicting operations came first.
1. Every request for a username is encoded as a message.
2. A stream processor sequentially reads the requests in the log. For every request for a username tht is available, it records the name as taken and emits a success message to an output stream. For every request for a username that is already taken, it emits a rejection message to an output stream.
3. The client waits for a success or rejection message corresponding to its request.

The approach works not only for uniqueness constraints, but also for many other kinds of constraints.

##### Multi-partition request processing

There are potentially three partitions: the one containing the request ID, the one containing the payee account, and one containing the payer account.

The traditional approach to databases, executing this transaction would require an atomic commit across all three partitions.

Equivalent correctness can be achieved with partitioned logs, and without an atomic commit.

1. The request to transfer money from account A to account B is given a unique request ID by the client, and appended to a log partition based on the request ID.
2. A stream processor reads the log of requests. For each request message it emits two messages to output streams: a debit instruction to the payer account A (partitioned by A), and a credit instruction to the payee account B (partitioned by B). The original request ID is included in those emitted messages.
3. Further processors consume the streams of credit and debit instructions, deduplicate by request ID, and apply the chagnes to the account balances.

#### Timeliness and integrity

Consumers of a log are asynchronous by design, so a sender does not wait until its message has been proccessed by consumers. However, it is possible for a client to wait for a message to appear on an output stream.

_Consistency_ conflates two different requirements:
* Timeliness: users observe the system in an up-to-date state.
* Integrity: Means absence of corruption. No data loss, no contradictory or false data. The derivation must be correct.

Violations of timeless are "eventual consistency" whereas violations of integrity are "perpetual inconsistency".

#### Correctness and dataflow systems

When processing event streams asynchronously, there is no guarantee of timeliness, unless you explicitly build consumers that wait for a message to arrive before returning. But integrity is in fact central to streaming systems.

_Exactly-once_ or _effectively-once_ semantics is a mechanism for preserving integrity. Fault-tolerant message delivery and duplicate supression are important for maintaining the integrity of a data system in the face of faults.

Stream processing systems can preserve integrity without requireing distributed transactions and an atomic commit protocol, which means they can potentially achieve comparable correctness with much better performance and operational robustness. Integrity can be achieved through a combination of mechanisms:
* Representing the content of the write operation as a single message, this fits well with event-sourcing
* Deriving all other state updates from that single message using deterministic derivation functions
* Passing a client-generated request ID, enabling end-to-end duplicate supression and idempotence
* Making messages immutable and allowing derived data to be reprocessed from time to time

In many businesses contexts, it is actually acceptable to temporarily violate a constraint and fix it up later apologising. The cost of the apology (money or reputation), it is often quite low.

#### Coordination-avoiding data-systems

1. Dataflow systems can maintain integrity guarantees on derived data without atomic commit, linearizability, or synchronous cross-partition coordination.
2. Although strict uniqueness constraints require timeliness and coordination, many applications are actually fine with loose constraints than may be temporarily violated and fixed up later.

Dataflow systems can provide the data management services for many applications without requiring coordination, while still giving strong integrity guarantees. _Coordination-avoiding_ data systems can achieve better performance and fault tolerance than systems that need to perform synchronous coordination.

#### Trust, but verify

Checking the integrity of data is know as _auditing_.

If you want to be sure that your data is still there, you have to actually read it and check. It is important to try restoring from your backups from time to time. Don't just blindly trust that it is working.

_Self-validating_ or _self-auditing_ systems continually check their own integrity.

ACID databases has led us toward developing applications on the basis of blindly trusting technology, neglecting any sort of auditability in the process.

By contrast, event-based systems can provide better auditability (like with event sourcing).

Cryptographic auditing and integrity checking often relies on _Merkle trees_. Outside of the hype for cryptocurrencies, _certificate transparency_ is a security technology that relies on Merkle trees to check the validity of TLS/SSL certificates.

### Doing the right thing

Many datasets are about people: their behaviour, their interests, their identity. We must treat such data with humanity and respect. Users are humans too, and human dignitity is paramount.

There are guidelines to navigate these issues such as ACM's Software Engineering Code of Ethics and Professional Practice

It is not sufficient for software engineers to focus exclusively on the technology and ignore its consequences: the ethical responsibility is ours to bear also.

In countries that respect human rights, the criminal justice system presumes innocence until proven guilty; on the other hand, automated systems can systematically and artbitrarily exclude a person from participating in society without any proof of guilt, and with little chance of appeal.

If there is a systematic bias in the input to an algorithm, the system will most likely learn and amplify bias in its output.

It seems ridiculous to believe that an algorithm could somehow take biased data as input and produce fair and impartial output from it. Yet this believe often seems to be implied by proponents of data-driven decision making.

If we want the future to be better than the past, moral imagination is required, and that's something only humans can provide. Data and models should be our tools, not our masters.

If a human makes a mistake, they can be held accountable. Algorithms make mistakes too, but who is accountable if they go wrong?

A credit score summarises "How did you behave in the past?" whereas predictive analytics usually work on the basis of "Who is similar to you, and how did people like you behave in the past?" Drawing parallels to others' behaviour implies stereotyping people.

We will also need to figure outhow to prevent data being used to harm people, and realise its positive potential instead, this power could be used to focus aid an support to help people who most need it.

When services become good at predicting what content users want to se, they may end up showing people only opinions they already agree with, leading to echo chambers in which stereotypes, misinformation and polaristaion can breed.

Many consequences can be predicted by thinking about the entire system (not just the computerised parts), an approach known as _systems thinking_.

#### Privacy and tracking

When a system only stores data that a user has explicitly entered, because they want the system to store and process it in a certain way, the system is performing a service for the user: the user is the customer.

But when a user's activity is tracked and logged as a side effect of other things they are doing, the relationship is less clear. The service no longer just does what the users tells it to do, but it takes on interests of its own, which may conflict with the user's interest.

If the service is funded through advertising, the advertirsers are the actual customers, and the users' interests take second place.

The user is given a free service and is coaxed into engaging with it as much as possible. The tracking of the user serves the needs of the advertirses who are funding the service. This is basically _surveillance_.

As a thougt experiment, try replacing the word _data_ with _surveillance_.

Even themost totalitarian and repressive regimes could only dream of putting a microphone in every room and forcing every person to constantly carry a device capable of tracking their location and movements. Yet we apparently voluntarily, even enthusiastically, throw ourselves into this world of total surveillance. The difference is just that the data is being collected by corporations rather than government agencies.

Perhaps you feel you have nothing to hide, you are totally in line with existing power structures, you are not a marginalised minority, and you needn't fear persecution. Not everyone is so fortunate.

Without understanding what happens to their data, users cannot give any meaningful consent. Often, data from one user also says things about other people who are not users of the service and who have not agreed to any terms.

For a user who does not consent to surveillance, the only real alternative is simply to not user the service. But this choice is not free either: if a service is so popular that it is "regarded by most people as essential for basic social participation", then it is not reasonable to expect people to opt out of this service. Especially when a service has network effects, there is a social cost to people choosing _not_ to use it.

Declining to use a service due to its tracking of users is only an option for the small number of people who are priviledged enough to have the time and knowledge to understand its privacy policy, and who can affort to potentially miss out on social participation or professional opportunities that may have arisen if they ahd participated in the service. For people in a less priviledged position, there is no meaningful freedom of choice: surveillance becomes inescapable.

Having privacy does not mean keeping everything secret; it means having the freedom to choose which things to reveal to whom, what to make public, and what to keep secret.

Companies that acquire data essentially say "trust us to do the right thing with your data" which means that the right to decide what to reveal and what to keep secret is transferred from the individual to the company.

Even if the service promises not to sell the data to third parties, it usually grants itself unrestricted rights to process and analyse the data internally, often going much further than what is overtly visible to users.

If targeted advertising is what pays for a service, then behavioral data about people is the service's core asset.

When collecting data, we need to consider not just today's political environment, but also future governments too. There is no guarantee that every government elected in the future will respect human rights and civil liberties, so "it is poor civic hygiene to install technologies that could someday facilitate a police state".

To scrutinise others while avoiding scrutiny oneself is one of the most important forms of power.

In the industrial revolution tt took a long time before safeguards were established, such as environmental protection regulations, safety protocols for workplaces, outlawing child labor, and health inspections for food. Undoubtedly the cost of doing business increased when factories could no longer dump their waste into rivers, sell tainted foods, or exploit workers. But society as a whole benefited hugely, and few of us would want to return to a time before those regulations.

We should stop regarding users as metrics to be optimised, and remember that they are humans who deserve respect, dignity, and agency. We should self-regulate our data collection and processing practices in order to establish an maintain the trust of the people who depend on our software. And we should take it upon ourselves to educate end users about how their data is used, rather than keeping them in the dark.

We should allow each individual to maintain their privacy, their control over their own data, and not steal that control from them through surveillance.

We should not retain data forever, but purge it as soon as it is no longer needed.

## Essential Technologies

### Caching
* Pros
  * faster reads (and writes?)
  * reduced load on database
* Cons
  * cache misses are expensive -- say our cache server is diff, and our application server makes a network call to just find out that our value is not there in cache. Then have to make disk seek. For instance in thrashing -- value that we want in cache keep getting replaced right before we want to read it since cache size is small
  * data consistency is complex, depends on how much you care

#### What do we cache
* database results
* computation done by application servers
* popular static content (CDN)

#### Client-side cache
* Like on mobile or web browser with some TTL

#### Server-local cache (seeing at low level)
* L1 is smallest (fastest) cache. Different for different CPU core
* L2 is bigger (slower than L1) cache. Can be shared or different for different CPU core
* L3 is biggest (slowest than L2) cache. Shared across different CPU core.
* First we check L1, if not found, L2, if not found L3, if not found RAM, if not found disk seek.
* ![Single computer cache](/metadata/single_computer_cache.png)

#### Server-local cache (seeing at high level) 
* cache on application server, database, message broker
* say if we want to see number of likes on our post, application server can cache that result. If we make call again, first we need to reach the same application server (consistent hashing on load balancer), and then we can fetch the result from cache on application server and not make network call to database.
* Pros: fewer network calls, very fast
* Cons: cache size is proportional to number of servers

#### Global caching layer
* dedicated cache server(s)
* hit application server, which makes network call to caching server, if not present, make network call to database server
* Pros: scale independently (replication + partitioning) from application server
* Cons: extra network call

#### Keeping Cache Updated on Writes
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
     * Don't do anything -- durability gone + eventual consistency
     * Distributed locking + replication: cache server holds a lock in order to modify the object. When another server wants to read the object, lock server (ZooKeeper) informs the cache server that it must release its lock on the object, and write back the modifications to the database before doing so.

#### Cache Eviction Policies
1. FIFO
   * implemented using queue
   * easy to understand, but has huge downside
2. Least Recently Used (LRU)
   * implemented using hashmap + doubly linked list
   * most used
3. Least Frequently Used (LFU)

#### (single node) Redis vs Memcached
| Memcached | Redis |
| --------- | ----- |
| Bare bones in memory | Feature rich: hashmaps, sorted sets, geo indexes, etc |
| Volatile; do not write to disk | Durable; writes to disk using WAL or checkpointing |
| Partitioned using consistent hashing | Partitioned using fixed number of partitions (16384) via gossip protocol |
| Multithreaded | Singlethreaded (allows transaction; actual serial execution) |
| LRU eviction (single policy) | | LRU, LFU, TTL, etc |
| Natively don't support replication; so no availability | Single leader replication |

#### Content Delivery Network (CDN)
* a lot of applications need to serve static content (music, video, image, scripts, etc), and these files are BIG.
* CDN are geographically distributed caches for static content, hence reduces latency
* Domain Name System (DNS) directs the request to the nearest CDN server based on the user’s location
* Two types
  * Pull CDN / Lazy Propagation: Content is fetched from the origin server on-demand and cached on the CDN's edge servers when requested by a user. Higher latency on first request, possible cache misses.
  * Push CDN / Eager Propagation: Content is manually uploaded to the CDN's edge servers ahead of time. Immediate availability, lower latency.
* Example: Akamai, Cloudfare

#### Object Stores
* Where do we store static content when we don't cache it?
* Can't use Hadoop
  * Hadoop clusters nodes have both compute and storage. Expensive! We just want more space and not compute.
* Object Stores
  * service offered by large cloud providers to store static content
  * Amazon S3, Azure Cloud Storage, Google Cloud Storage
  * handles scaling for you
  * handles replication for you
  * cheaper to run than a Hadoop cluster (1/10th of the price)
* Data Lake Paradigm -- bunch of different kinds of data (logs, metrics, images, etc) -- schemaless -- put in Object Stores
* Batch jobs on Object Stores? Object Stores lack the requisite compute power, need to transport the data from storage node to compute node! Expensive
### Search Indexes
