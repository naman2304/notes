Way to approach a system design question
1. Requirement clarification
   * functional requirements -- scope out the problem i.e. what actual part of the application we are trying to build (create boundary!)
   * non functional requirements
     * load (# of users, # of ops like read/writes)
     * performance (latency, throughput)
   * capacity estimation (BOTEC Back Of The Envelope Calculations)
2. API definitions
3. Data Tables: schema of the data that you will store
   * replication based on BOTEC
   * partitioning based on BOTEC
4. Design (which meets functional requirements) and **always give WHY** (contrasting with other choices and tradeoffs)
   * load balancing
   * microservice design
   * database choice
   * replication
   * partitioning

Disk: 10TB per machine 
RAM: 128GB per machine

## TinyURL/PasteBin

