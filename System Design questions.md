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

BOTEC
* Disk: 10TB per machine 
* RAM: 128GB per machine

## TinyURL/PasteBin

#### Functional Requirements
* Generate expiring unique short URL from user provided URL
* Redirect users to correct website upon clicking short URL
* Logging clicks per short URL

#### BOTEC
* 100:1 read to write ratio
* 500M short URLs generated per month
  * this is 200 writes per second
  * which means 20k reads per second
* 500 bytes per URL
  * 500 bytes per URL * 500M URLs per month * 12 months per year * 5 years = 15TB storage for 5 years
* 80/20 rule. 20% of entries responsible for 80% of traffic. Use cache to make things faster and store 20% of data
  * 20% * 3 TB for a year = 50GB for a month

#### API Design
