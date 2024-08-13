Way to approach a system design question
1. Requirement clarification
   * functional requirements -- scope out the problem i.e. what actual part of the application we are trying to build (create boundary!)
   * non functional requirements
     * load (# of users, # of ops like read/writes)
     * performance (latency, throughput)
2. API definitions
3. BOTEC (Back Of The Envelope Calculations)
4. Data Model: schema of the data that you will store
   * replication based on BOTEC
   * partitioning based on BOTEC
5. Design (which meets functional requirements) 
   * load balancing
   * microservice design
   * database choice
   * replication
   * partitioning
6. Dig deeper into 2-3 components (**always give WHY** contrasting with other choices and tradeoffs)

BOTEC
* Disk: 10TB per machine 
* RAM: 128GB per machine

## TinyURL/PasteBin

#### Functional Requirements
* Generate expiring unique short URL from user provided URL
* Users should optionally be able to pick a custom short link for their URL.
* Redirect users to correct website upon clicking short URL
* Logging clicks per short URL

#### Non functional requirement
* Service should be highly available (both generating + redirection)
* Minimal latency on redirection
* shortened URL should not be guessable

#### BOTEC
* 100:1 read to write ratio
* 500M short URLs generated per month
  * this is 200 writes per second
  * which means 20k reads per second
* 500 bytes per URL
  * 500 bytes per URL * 500M URLs per month * 12 months per year * 5 years = 15TB storage for 5 years
* 80/20 rule. 20% of entries responsible for 80% of traffic. Use cache to make things faster and store 20% of data
  * 20% * 3 TB for a year = 50GB for a month
* shortUrl has A-Z and 0-9 chars, so 36 total
  * 6 length (36^6) = 2B unique Urls
  * 8 length (36^8) = 3T unique Urls

#### API Design

#### Database Tables

User Table
* userId: int
* email: string
* passwordHash: string
* creationTime: date

Link Table
* shortUrl: string [primary key]
* originalUrl: string
* userId: int
* expiration: date

Click Table
* shortUrl: string
* timeClicked: date
* clickerInfo: json

#### Choosing database
* Not many relations b/w the tables. Not much joining
* Access pattern is reading or writing to shortUrl

#### Design

How to generate shortUrl from originalUrl?
1. Hashing. 6 or 8 char are enough (not 96/128 bits because not full set of chars are used; only A-Z and 0-9), hashing algorithms give 128/160/256/32/64 bit. Problems
