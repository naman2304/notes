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
* Redis, Memcached, Amazon DynamoDB


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
* Key-value stores, such as Memcached are intended for cache only, it's acceptable for data to be lost if the machine is restarted.
* Other in-memory databases aim for durability, with special hardware, writing a log of changes to disk, writing periodic snapshots to disk or by replicating in-memory sate to other machines. Redis provide weak durability by writing to disk asynchronously.
* In-memory databases can be faster because they can avoid the overheads of encoding in-memory data structures in a form that can be written to disk.
* Another interesting area is that in-memory databases may provide data models that are difficult to implement with disk-based indexes (sets, priority queues, etc)

### Transaction processing or analytics?

* A _transaction_ is a group of reads and writes that form a logical unit, this pattern became known as _online transaction processing_ (OLTP).
* _Data analytics_ has very different access patterns. A query would need to scan over a huge number of records, only reading a few columns per record, and calculates aggregate statistics. These queries are often written by business analysts, and fed into reports. This pattern became known for _online analytics processing_ (OLAP).


### Data warehousing
* A _data warehouse_ is a separate database that analysts can query to their heart's content without affecting OLTP operations. It contains read-only copy of the data in all various OLTP systems in the company. Data is extracted out of OLTP databases (through periodic data dump or a continuous stream of update), transformed into an analysis-friendly schema, cleaned up, and then loaded into the data warehouse (process _Extract-Transform-Load_ or ETL).
* A data warehouse is most commonly relational, but the internals of the systems can look quite different because of different access patterns.
* Amazon RedShift is hosted version of ParAccel. Apache Hive, Spark SQL, Cloudera Impala, Facebook Presto, Apache Tajo, and Apache Drill. Some of them are based on ideas from Google's Dremel.
* Data warehouses are used in fairly formulaic style known as a _star schema_.
    * Central table is fact table in which each row is a fact -- facts are captured as individual events, because this allows maximum flexibility of analysis later. The fact table can become extremely large.
    * Some of the columns in the fact are attributes, others are foreign key references to other tables called dimension tables. Dimensions represent the _who_, _what_, _where_, _when_, _how_ and _why_ of the event.
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
WHERE product_sk IN (30, 68, 69) // load the three bitmaps for 30, 68, 69 from product_sk column file and take bitwise OR.
WHERE product_sk = 30 AND store_sk = 3 // load bitmap for product_sk=30 and store_sk=3, and calculate bitwise AND.
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
