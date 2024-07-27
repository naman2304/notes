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
  // replica IDs of nodes that have voted in favour of commiting a txn T
  commitVotes[T] := {};
  // all replicas participating in txn T
  replicas[T] := {};
  // flag telling if we have decided or not.
  decided[T] := false
end on

// Coordinator work
on request to commit transaction T with participating nodes R do
  for each r ∈ R do send (Prepare, T, R) to r
end on

on receiving (Prepare, T, R) at node replicaId do
  replicas[T] := R
  ok = “is transaction T able to commit on this replica?”
  // every node that is participating in the txn uses total order broadcast (TOB) to disseminate its vote on whether to commit or abort.
  total order broadcast (Vote, T, replicaId, ok) to replicas[T]
end on

// if node A suspects that node B has failed (because no vote from B was received within some timeout), then A may try to vote to abort on behalf of B.
// this introduces a race condition: if node B is slow, it might be that node B broadcasts its own vote to commit around the same time that node A suspects B to have failed and votes abort on B’s behalf.
on a node suspects node replicaId to have crashed do
  for each transaction T in which replicaId participated do
    total order broadcast (Vote, T, replicaId, false) to replicas[T]
  end for
end on

// These votes are delivered to each node by TOB, and each recipient independently counts the votes.
// Hence, we count only the first vote from any given replica, and ignore any subsequent votes from the same replica.
// Since TOB guarantees the same delivery order on each node, all nodes will agree on whether the first delivered vote from a given replica was a commit vote or an abort vote, even in the case of a race condition b/w multiple nodes broadcasting contradictory votes for the same replica
// If a node observes that the first delivered vote from some replica is a vote to abort, then the transaction can immediately be aborted.
// Otherwise a node must wait until it has delivered at least one vote from each replica.
// Once these votes have been delivered, and none of the replicas vote to abort in their first delivered message, then the transaction can be committed.
// Thanks to TOB, all nodes are guaranteed to make the same decision on whether to abort or to commit, which preserves atomicity
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

### Eventual Consistency
