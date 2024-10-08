## Introduction

#### About Distributed Systems
* Concurrency on a single computer is also known as shared-memory concurrency, since multiple threads running in the same process have access to the same address space. Thus, data can easily be passed from one thread to another: a variable or pointer that is valid for one thread is also valid for another.
* In distributed systems, we don’t typically have shared memory, since each computer runs its own operating system with its own address space, using the memory built into that computer. Different computers can only communicate by sending each other messages over a network.
* In some supercomputers and research systems, there are technologies like remote direct memory access (RDMA) that allow computers to access each others’ memory over a network. 
* A distributed system is multiple computers communicating via a network trying to achieve some task together
* Why need for distributed systems?
  * It’s inherently distributed e.g. sending a message from your mobile phone to your friend’s phone
  * For better reliability: even if one node fails, the system as a whole keeps functioning
  * For better performance: get data from a nearby node rather than one halfway round the world
  * To solve bigger problems: e.g. huge amounts of data, can’t fit on one machine
* Downsides of distributed systems: network may fail, nodes may crash (can happen at any moment without warning)
* In a single computer, if one component fails (e.g. one of the RAM modules develops a fault), we normally don’t expect the computer to continue working nevertheless: it will probably just crash. Software does not need to be written in a way that explicitly deals with faulty RAM.
* However, in a distributed system we often do want to tolerate some parts of the system being broken, and for the rest to continue working. For example, if one node has crashed (a partial failure), the remaining nodes may still be able to continue providing the service.
* If one component of a system stops working, we call that a fault, and many distributed systems strive to provide fault tolerance: that is, the system as a whole continues functioning despite the fault.

#### Distributed Systems and computer networking
* From a distributed systems point of view, the method of delivering the message is not important: we only see an abstract communication channel with a certain latency (delay from the time a message is sent until it is received) and bandwidth (the volume of data that can be transferred per unit time).
* Latency: time until message arrives
  * In the same building/datacenter: ≈ 1 ms
  * One continent to another: ≈ 100 ms
* In the web there are two main types of nodes: servers host websites, and clients (web browsers) display them. When you load a web page, your web browser sends a HTTP request message to the appropriate server. On receiving that request, the web server sends a response message containing the page contents to the client that requested it.
* Since the requests and responses can be larger than we can fit in a single network packet, the HTTP protocol runs on top of TCP, which breaks down a large chunk of data into a stream of small network packets, and puts them back together again at the recipient.  HTTP also allows multiple requests and multiple responses to be sent over a single TCP connection.

#### Remote Procedure Calls (RPC)
* Online shop server will send a payment request over the Internet to a service that specialises in processing card payments.

```java
// Online shop handling customer's card details
Card card = new Card();
card.setCardNumber("1234 5678 8765 4321");
card.setExpiryDate("10/2024");
card.setCVC("123");

Result result = paymentsService.processPayment(card, 3.99, Currency.GBP);

if (result.isSuccess()) {
  fulfilOrder();
}
```

* Calling the processPayment function looks like calling any other function, but in fact, what is happening behind the scenes is that the shop is sending a request to the payments service, waiting for a response, and then returning the response it received.
* In Java, it is called Remote Method Invocation (RMI).
* The software that implements RPC is called an RPC framework or middleware. The RPC framework provides a stub in its place. The stub has the same type signature as the real function, but instead of executing the real function, it encodes the function arguments in a message and sends that message to the remote node, asking for that function to be called. The process of encoding the function arguments is known as marshalling. 
* The sending of the message from the RPC client to the RPC server may happen over HTTP (in which case this is also called a web service). The RPC framework unmarshals (decodes) the message and calls the desired function with the provided arguments. When the function returns, the same happens in reverse.
* In practice for RPC, many things can go wrong:
  * What if the service crashes during the function call?
  * What if a message is lost?
  * What if a message is delayed?
  * If something goes wrong, is it safe to retry?
* In practice the client will have to give up after some timeout.
* Today, the most common form of RPC is implemented using JSON data sent over HTTP. A popular set of design principles for such HTTP-based APIs is known as representational state transfer or REST
  * communication is stateless (each request is self-contained and independent from other requests)
  * resources (objects that can be inspected and manipulated) are represented by URLs, and
  * the state of a resource is updated by making a HTTP request with a standard method type, such as POST or PUT, to the appropriate URL.

```js
let args = {amount: 3.99, currency: 'GBP', /*...*/ };
let request = {
  method: 'POST',
  body: JSON.stringify(args),
  headers: {'Content-Type': 'application/json'}
};

fetch('https://example.com/payments', request)
  .then((response) => {
    if (response.ok) success(response.json());
    else failure(response.status); // server error
  })
  .catch((error) => {
    failure(error); // network error
  });
```

* RPC in enterprise systems (“Service-oriented architecture” (SOA) / “microservices”)
  * splitting a large software application into multiple services (on multiple nodes) that communicate via RPC.
  * Different services implemented in different languages:
    * interoperability: datatype conversions
    * Interface Definition Language (IDL)
  * When different programming languages are used, the RPC framework needs to convert datatypes such that the caller’s arguments are understood by the code being called, and likewise for the function’s return value. A typical solution is to use an Interface Definition Language (IDL) to provide language independent type signatures of the functions that are being made available over RPC.
  * gRPC is an implementation of RPC paradigm, and uses IDL called Protocol Buffers.


### Delivery order in broadcast protocols
#### Broadcast Protocol
 * message is sent to all nodes in the group
 * set of group members can be fixed (static) or dynamic
 * system model
   * Network assumptions: best effort (may drop messages) OR reliable (retransmit dropped messages)
   * Timing assumptions: Asynchronous/Partially synchronous
 * **Note**: After broadcast algorithm receives message from network, it may buffer/queue it before delivering to the application
 * ![Receiving versus delivering](/metadata/receiving_versus_delivering.png)

##### Forms of reliable broadcast
* all of these are reliable (network) + asynchronous (timing)
* four different type (they differ in terms of the order in which messages may be delivered at each node)

**FIFO broadcast**
* If m1 and m2 are broadcast by the same node, and broadcast(m1) → broadcast(m2), then m1 must be delivered before m2
* weakest form of broadcast
* when a node broadcasts a message, it is always possible to immediately deliver that message to itself
* Example:
  * m1 must be delivered before m3, since they were both sent by A. However, m2 can be delivered at any time before, between, or after m1 and m3.
  * ![FIFO Broadcast example](/metadata/fifo_broadcast_example.png)

**Causal broadcast**
* If broadcast(m1) → broadcast(m2) then m1 must be delivered before m2. If two messages are broadcast concurrently, a node may deliver them in either order.
* stricter than FIFO broadcast
* when a node broadcasts a message, it is always possible to immediately deliver that message to itself
* Example:
  * in above FIFO example, it violates causality, because node C delivers m2 before m1, even though B broadcast m2 after delivering m1.
  * ![Causal Broadcast example](/metadata/causal_broadcast_example.png)

**Total order broadcast** (aka atomic broadcast)
* FIFO and causal broadcast allow different nodes to deliver messages in different orders, total order broadcast enforces consistency across the nodes, ensuring that all nodes deliver messages in the same order. The precise delivery order is not defined, as long as it is the same on all nodes (If m1 is delivered before m2 on one node, then m1 must be delivered before m2 on all nodes)
* when a node broadcasts a message, it is NOT always possible to immediately deliver that message to itself
* Example:
  * ![Total Order Broadcast example 1](/metadata/total_order_broadcast_example_1.png)
  * ![Total Order Broadcast example 2](/metadata/total_order_broadcast_example_2.png)

**FIFO-total order broadcast**
* like total order broadcast, but with the additional FIFO requirement that any messages broadcast by the same node are delivered in the order they were sent

**Relationship between broadcast models**
* ![Relationshiop between broadcast models](/metadata/relationship_between_broadcast_models.png)
* For example, FIFO-total order broadcast is a strictly stronger model than causal broadcast; in other words, every valid FIFO-total order broadcast protocol is also a valid causal broadcast protocol (but not the opposite)

#### Broadcast Algorithms
Break down into two layers
1. Make best-effort broadcast reliable by retransmitting dropped messages
2. Enforce delivery order on top of reliable broadcast (which allows us to do either FIFO, Causal, Total Order or FIFO-Total Order)

We will first look at reliable broadcast algorithms.
* First Attempt
  * broadcasting node sends message directly to every other node
  * use reliable links (retry + deduplicate)
  * Problem: node may crash before all messages are delivered (A sends m1 to B and C. m1 gets delivered to B. m1 does not get delivered to C due to temporary network failure between A and C resulting in dropping of m1 reaching C. And then A crashes. How will A retransmit m1 to C?). Hence, not reliable
* Second Attempt
  * Eager reliable broadcast
  * Idea: **first time** a node receives a particular message, it re-broadcasts to each other node (via reliable links)
  * Problem: O(n^2) messages. Reliable, but uses too much bandwidth.
  * ![Eager Reliable Broadcast](/metadata/eager_reliable_broadcast.png)
* Third Attempt
  * Gossip Protocol aka epidemic protocol
  * Idea: when a node receives a message for the **first time**, forward it to 3 other nodes, chosen randomly.
  * A message reaches all nodes (with very high probability -- there is a chance that a message may not reach some nodes, but can be minimized by tuning the parameters of algorithm like sending to say 5 nodes)
  * Note: gossip protocol is also used in Cassandra and Riak to track partition assignment
 
Now, we will implement ordering on top of reliable broadcast
1. FIFO broadcast
   ```python
   on initialisation do
    // number of messages broadcast by this node
    sendSeq := 0;
    // vector of integers indicating how many messages from each sender that we have delivered on this node.
    delivered := <0, 0, . . . , 0>;
    // holdback queue of messages until they are delivered on this node
    buffer := {}
   end on
   
   on request to broadcast m at node Ni do
    // attaching node number (i) of sender
    send (i, sendSeq, m) via reliable broadcast
    sendSeq := sendSeq + 1
   end on
   
   on receiving msg from reliable broadcast at node Ni do
    // msg = {sender, sendSeq, m}
    buffer := buffer ∪ {msg}
    while ∃(sender, sendSeq, m) ∈ buffer such that sendSeq == delivered[sender] do 
      deliver m to application
      delivered[sender] := delivered[sender] + 1
      buffer := buffer - {(sender, sendSeq, m)}
    end while
   end on
   ```
2. Causal broadcast (also called vector clock algorithm)
   *  In traditional vector clocks, the vector elements count the number of events that have occurred at each node, while in the causal broadcast algorithm, the vector elements count the number of messages from each sender that have been delivered on this node

   ```python
   on initialisation do
    sendSeq := 0;
    delivered := <0, 0, . . . , 0>;
    buffer := {}
   end on
   
   on request to broadcast m at node Ni do
    deps := delivered;
    deps[i] := sendSeq
    send (i, deps, m) via reliable broadcast
    sendSeq := sendSeq + 1
   end on
   
   on receiving msg from reliable broadcast at node Ni do
    buffer := buffer ∪ {msg}
    while ∃(sender, deps, m) ∈ buffer such that deps ≤ delivered do
     deliver m to the application
     delivered[sender] := delivered[sender] + 1
     buffer := buffer - {(sender , deps, m)}
    end while
   end on
   ```
3. Total order broadcast
   * both total order and FIFO-total order are harder.
   * neither of below 2 algorithm is fault tolerant
   * Single leader approach
     * One node is designated as leader (sequencer; one who decides the unique sequence of messages that needs to be adhered to all nodes)
     * To broadcast message, send it to the leader; leader broadcasts it via FIFO broadcast
     * Problem: leader crashes ⇒ no more messages delivered
     * Changing the leader safely is difficult
   * Leaderless (lamport clocks)
     * Attach Lamport timestamp to every message
     * Deliver messages in _total order_ of timestamps
     * Problem: how do you know if you have seen all messages with timestamp < T? Need to use FIFO links and wait for message with timestamp ≥ T from every node


### Consensus

* Fault-tolerant total order broadcast
  * total order broadcast is very useful for stateful machine replication
  * can implement via single leader approach
  * Problem was what if the leader crashes or becomes unavailable? Can we choose the new leader automatically
* Consensus and total order broadcast
  * traditional formulation of consensus: several nodes want to come to agreement about a single value. One or more nodes may propose a value, and then the consensus algorithm will decide on one of those values
  * in context of total order broadcast: this value is the next message to deliver
  * once one node decides on a certain message order, all nodes will decide the same order
  * consensus and total order broadcast are formally equivalent
  * common consensus algorithms
    * Paxos: single value consensus
    * Multi Paxos: generalisation to FIFO-total order broadcast
    * Raft, Viewstamped Replication, ZAB (zookeeper atomic broadcast): FIFO-total order broadcast by default
* Consensus system models
  * Paxos, Raft, etc assume **fairloss links, crash recovery nodes and partially synchronous** system model
  * Why not asynchronous?
    * **FLP result** (Fischer, Lynch, Paterson) -- there is no deterministic consensus algorithm that is guaranteed to terminate in an asynchronous crash stop system model
    * Paxos, Raft, etc use clocks only used for timeouts/failure detector to ensure progress. Safety (correctness) does not depend on timing (boundness). Only thing that timing dictates is the time to "deliver" a message, so timing affects "liveliness" and not "safety" guarantee.
  * There are also consensus algorithms for Byzantine partially synchronous system models (blockchain)
* Leader election
  * Multi-Paxos, Raft, etc use a leader to sequence messages
    * Use a failure detector (timeout) to determine suspected crash or unavailability of leader
    * On suspected leader crash, elect a new one
    * Prevent two leaders at the same time (split brain)
  * Ensure <=1 leader per **term**
    * Term is just an integer that is incremented every time a leader election is started. 
    * A node can only vote once per term
    * Require a quorum of nodes to elect a leader in a term
* Can we guarantee there is only one leader?
  * If a leader is elected, the voting algorithm guarantees that that it is the only leader within that particular term. 
  * **Cannot** prevent having multiple leaders from different terms.
  * Example: node 1 is leader in term t, but due to network partition it can no longer communicate to node 2 & 3, and node 2 & 3 may elect a new leader in term t+1. Node 1 may not even know that a new leader has been elected.
* How to solve above?
  * Even after a node has been elected as leader, it must act carefully, since at any moment the system might contain be another leader with a later term that it has not yet heard about
    * first roundtrip, node is elected as leader thanks to votes from other nodes
    * second roundtrip, leader proposes next message to deliver, and the followers acknowledge that they do not know of any leader with a later term than t. **This is the trip that really solves the problem of split brain because if another leader has been elected, the old leader will find out from at least one of the acks**
    * third roundtrip, the leader actually delivers m and broadcasts this fact to the followers, so that they can do the same.
  * ![Checking if leader is voted out](/metadata/checking_if_leader_is_voted_out.png)


## Replica Consistency

Distributed Transactions
* Recall atomicity in the context of ACID txn
  * A txn either commits or aborts
  * If it commits, its updates are durable
  * If it aborts, it has no visible side effects
* If the txn updates data on multiple nodes, this implies
  * either all the nodes must commit, or all must abort
  * if any node crashes, all must abort
  * ensuring this is the **atomic commitment** problem
  * looks similar to consensus?

| Consensus                                | Atomic Commit                             |
| ---------                                | ----------------------------------------- |
| one or more node propose a value         | every node votes whether to commit or not |
| any one of the proposed value is decided | must commit if all nodes vote to commit; must abort if >=1 nodes vote to abort |
| crashed nodes can be tolerated, as long as the quorum is working | must abort if a participating node crashes |

### Two phase commit (2PC)
* 2PL ensures serializable isolation, 2PC ensures atomic commitment
* algorithm
  * client starts a regular single-node txn on each replica that is participating in the txn, and performs the usual reads and writes within those txns.
  * when the client is ready to commit the txn, it sends a commit request to the txn coordinator, a designated node that manages the 2PC protocol
  * **First phase** - coordinator first sends a **prepare** message to each replica participating in the txn, and each replica replies with a message indicating whether it is able to commit the txn. The replicas do not actually commit the txn yet, but they must ensure that they will definitely be able to commit the txn in the second phase if instructed by the coordinator. This means, in particular, that the replica must write all of the txn’s updates to disk and check any integrity constraints before replying ok to the prepare message, while continuing to hold any locks for the txn
  * **Second phase** - The coordinator collects the responses, and decides whether or not to actually commit the txn. If all nodes reply ok, the coordinator decides to commit; if any node wants to abort, or if any node fails to reply within some timeout, the coordinator decides to abort. The coordinator then sends its decision to each of the replicas, who all commit or abort as instructed. If the decision was to commit, each replica is guaranteed to be able to commit its txn because the previous prepare request laid the groundwork. If the decision was to abort, the replica rolls back the txn.
* ![Two phase commit](/metadata/two_phase_commit.png)
* Coordinator - SPOF - what if it crashes?
  * Coordinator writes its decision (commit/abort) to disk
  * When it recovers, read decision from disk and send it to replicas (or abort if no decision was made before crash)
  * Problem: if coordinator crashes after "prepare", but before broadcasting decision, other nodes do not know how it has decided ("in doubt txns")
  * Replicas participating in txn cannot commit or abort after responding “ok” to the prepare request (otherwise we risk violating atomicity)
  * Algorithm is blocked until coordinator recovers
  * Solution
    * use total order broadcast algorithm to disseminate each nodes' vote whether to commit or abort

```python
on initialisation for transaction T do
  # replica IDs of nodes that have voted in favour of commiting a txn T
  commitVotes[T] := {};
  # all replicas participating in txn T
  replicas[T] := {};
  # flag telling if we have decided or not.
  decided[T] := false
end on

# Coordinator work
on request to commit transaction T with participating nodes R do
  for each r ∈ R do send (Prepare, T, R) to r
end on

on receiving (Prepare, T, R) at node replicaId do
  replicas[T] := R
  ok = “is transaction T able to commit on this replica?”
  # every node that is participating in the txn uses total order broadcast (TOB) to disseminate its vote on whether to commit or abort.
  total order broadcast (Vote, T, replicaId, ok) to replicas[T]
end on

# if node A suspects that node B has failed (because no vote from B was received within some timeout), then A may try to vote to abort on behalf of B.
# this introduces a race condition: if node B is slow, it might be that node B broadcasts its own vote to commit around the same time that node A suspects B to have failed and votes abort on B’s behalf.
on a node suspects node replicaId to have crashed do
  for each transaction T in which replicaId participated do
    total order broadcast (Vote, T, replicaId, false) to replicas[T]
  end for
end on

# These votes are delivered to each node by TOB, and each recipient independently counts the votes.
# Hence, we count only the first vote from any given replica, and ignore any subsequent votes from the same replica.
# Since TOB guarantees the same delivery order on each node, all nodes will agree on whether the first delivered vote from a given replica was a commit vote or an abort vote, even in the case of a race condition b/w multiple nodes broadcasting contradictory votes for the same replica
# If a node observes that the first delivered vote from some replica is a vote to abort, then the transaction can immediately be aborted.
# Otherwise a node must wait until it has delivered at least one vote from each replica.
# Once these votes have been delivered, and none of the replicas vote to abort in their first delivered message, then the transaction can be committed.
# Thanks to TOB, all nodes are guaranteed to make the same decision on whether to abort or to commit, which preserves atomicity
on delivering (Vote, T, replicaId, ok) by total order broadcast do
  if replicaId ∈/ commitVotes[T] ∧ replicaId ∈ replicas[T] ∧ ¬decided[T] then
    if ok = true then
      commitVotes[T] := commitVotes[T] ∪ {replicaId}
      if commitVotes[T] = replicas[T] then
        decided[T] := true
        commit transaction T at this node
      end if
    else
      decided[T] := true
      abort transaction T at this node
    end if
  end if
end on
```

### Linearizability
* Atomic commitment preserves consistency across multiple replicas in face of faults, by ensuring that a transaction either commits or aborts on all participating replicas. However, when there are concurrent operations going on, we need to reason about that too.
* Linearizability (aka strong consistency) is the strongest consistency model.
* Relation to serializability
  * Not equal to serializability.
  * Serializability means transaction has the same effect as if they are executed in serial order, but it does not define what that order should be.
  * Linearizability defines the value that operations must return, depending on concurrency and relative ordering of those operations. If system is both, that combination is called strict serializability or one-copy serializability
* Properties
  * every operation takes affect **atomically** sometime after it started and before it finished.
  * all operations behave as if executed on a single copy of data
  * this is from perspective of client
  * when two operations overlap, then we don't necessarily know the order in which the operation takes effect, so either is fine
  * not equal to **happens before** relation -- which is based on message sent and received, so possible that two operations that do not overlap in time, but are still concurrent, because no commmunication has happened before those two operations.
  * On the other hand, linearizability is defined in terms of real time: that is, a hypothetical global observer who can instantaneously see the state of all nodes
* Quorum consistency
  * Quorum consistency is not equal to linearizability due to network delays
  * Can be made linearizable though
    * read operation(**get**) has to do read repair so that atleast quorum nodes (say n/2 + 1) have updated value, and only then return the value to application (this is called ABD algorithm)
    * write operation (**set**) -- first read and do read repair, and then write
  * ![ABD algorithm](/metadata/abd_algorithm.png)
  * Can we implement **cas** (compare and set) operation as linearizable -- YES, by using total order broadcast (TOB gives us this single threaded view such that all operations are delivered in the same order on all of the nodes, we have state machine replication)

```python
on request to perform get(x) do
 total order broadcast (get, x) and wait for delivery
end on

on request to perform CAS(x, old, new) do
 total order broadcast (CAS, x, old, new) and wait for delivery
end on

on delivering (get, x) by total order broadcast do
 return localState[x] as result of operation get(x)
end on

on delivering (CAS, x, old, new) by total order broadcast do
 success := false
 if localState[x] = old then
  localState[x] := new; success := true
 end if
 return success as result of operation CAS(x, old, new)
end on
```

### Eventual Consistency
* Linearizability
  * Advantages
    * Makes a distributed system behave as if it were non-distributed
    * Simple for applications to use
  * Disadvantages:
    * Performance cost: lots of messages and waiting for responses
    * Scalability limits: in algorithms where all updates need to be sequenced through a leader, such as Raft, the leader can become a bottleneck that limits the number of operations that can be processed per second
    * Availability problems: if you can’t contact a quorum of nodes, you can’t process any operations
* Hence, eventual consistency, which is a weak consistency model
  * if no new updates are made to an object, eventually all reads will return the last updated value (no guarantee on how long, also what if updates don't stop)
  * **Strong eventual consistency**
    * Eventual delivery: every update made to one non-faulty replica is eventually processed by every non-faulty replica.
    * Convergence: any two replicas that have processed the same set of updates are in the same state (even if updates were processed in a different order).
    * Properties:
      * Does not require waiting for network communication
      * Causal broadcast (or weaker) can disseminate updates
      * Concurrent updates ⇒ conflicts need to be resolved

| Problem                                       | Must wait for communication         | Requires synchrony  |
| --------------------------------------------- | ----------------------------------- | ------------------  |
| atomic commit                                 | all participating nodes             | partial synchronous |
| consensus, linearizable CAS, TOB              | quorum nodes                        | partial synchronous |
| linearizable get and set                      | quorum nodes                        | asynchronous        |
| eventual consistency, FIFO, Causal broadcast  | local replica only                  | asynchronous        |

