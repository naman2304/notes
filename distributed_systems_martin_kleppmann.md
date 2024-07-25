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
