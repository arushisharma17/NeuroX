---
# SYSTEM DESIGN
---
## Table of Contents
---
1. [Computer Architecture](#computer-architecture)
2. [Application Architecture](#application-architecture)
3. [Design Requirements](#design-requirements)
4. [Networking Basics](#networking-basics)
5. [TCP and UDP](#tcp-and-udp)
6. [Domain Name System (DNS)](#domain-name-system-dns)
---
## Computer Architecture
---
Understanding the building blocks of a computer is essential for designing efficient systems. This section explores the core components of computer architecture.

### Components

#### Disk
A disk is the primary storage device in a computer, providing persistent storage, meaning data remains stored even when the machine is powered off. Modern computers typically use disks with capacities measured in terabytes (TBs). Two common types of disks are Hard Disk Drives (HDDs) and Solid State Drives (SSDs). HDDs are mechanical devices with moving parts, which can wear down over time, while SSDs are faster and more reliable due to their lack of moving parts.

#### RAM
Random Access Memory (RAM) is a type of volatile memory, meaning data is lost when the computer is turned off. RAM is used for storing data that is actively being processed by the CPU. It is significantly faster than disk storage but is also more expensive and typically smaller in capacity, ranging from 1GB to 128GB. RAM speeds up data access times and is essential for running applications smoothly.

#### CPU
The Central Processing Unit (CPU) is often referred to as the brain of the computer. It performs computations and executes instructions stored in RAM. The CPU reads data from RAM, processes it, and writes results back to RAM or disk. CPUs also have a cache, which is extremely fast memory used to store frequently accessed data, reducing the time needed to access this data.

#### Cache
Cache memory is a small, fast type of volatile memory located on the same chip as the CPU. It stores copies of frequently accessed data from RAM, allowing the CPU to access this data more quickly. Most CPUs have multiple levels of cache (L1, L2, and L3), with L1 being the smallest and fastest and L3 being larger and slightly slower. Effective use of cache can significantly improve a system's performance.

#### Moore's Law
Moore's Law is an observation that the number of transistors on a CPU doubles approximately every two years, leading to exponential growth in computing power and reductions in cost. However, in recent years, this growth has started to plateau.

### Key Concepts
- **Latency**: The delay between a request and the response. It is a critical factor in the performance of both internal computer operations and network communications.
- **Throughput**: The number of operations or amount of data processed in a given amount of time. Higher throughput indicates better performance.

---
## Application Architecture
---

This section provides a high-level overview of a production-grade application architecture, detailing how various components interact to create a robust system.

### Developer's Perspective
From a developer's perspective, application architecture involves writing code that is deployed to a server. The server handles requests from clients and requires persistent storage for the application's data. Servers often use external storage systems like databases to manage large amounts of data efficiently.

### User's Perspective
Users interact with the application through a client, usually a web browser. When a user makes a request, the server responds with the necessary code (JavaScript, HTML, CSS) to render the requested feature. As the number of users grows, a single server may become a bottleneck, necessitating strategies for scaling.

### Scaling
- **Vertical Scaling**: Involves upgrading the existing server's hardware, such as adding more RAM or using a faster CPU. This method is limited by the server's maximum upgrade capacity.
- **Horizontal Scaling**: Involves adding more servers to distribute the load. This method is more complex but provides better performance and redundancy. A load balancer is used to evenly distribute incoming requests across multiple servers.

### Components
- **Load Balancer**: Distributes incoming requests among multiple servers to ensure no single server becomes a bottleneck. It also provides redundancy by rerouting traffic if a server goes down.
- **Logging and Metrics**: Servers log activities and performance metrics, which are essential for monitoring and diagnosing issues. Logs can be stored on the same server or an external server for better reliability.

### Alerts
Alerts are set up to notify developers of any unusual activity or performance issues. These alerts can be triggered based on predefined metrics, such as a drop in successful request rates.

### Key Concepts
- **Availability**: The percentage of time a system is operational and accessible. High availability is crucial for modern applications that require near 24/7 uptime.
- **Reliability**: The system's ability to function correctly and handle errors gracefully over time.
- **Fault Tolerance**: The system's ability to continue operating correctly even when some components fail. This is achieved through redundancy, such as having backup servers.
- **Redundancy**: Having additional components or systems that take over when the primary component fails. This can be active-active (both systems are operational) or active-passive (backup system is on standby).

---
## Design Requirements
---

Understanding the basics of designing a system and meeting quality standards for a large, effective distributed system is crucial. This section covers key aspects of system design.

### Key Aspects

#### Moving Data
Moving data efficiently between clients and servers, especially when they are geographically dispersed, is a significant challenge in system design. Ensuring low latency and high throughput in data movement is essential.

#### Storing Data
Different methods for storing data include databases, blob stores, and file systems. Choosing the right storage method depends on the specific use case and performance requirements. Efficient data storage solutions are critical for the overall system performance.

#### Transforming Data
Transforming data involves processing and manipulating data to derive meaningful insights. Examples include analyzing server logs to determine successful vs. failed requests or filtering medical records by specific criteria. Efficient data transformation is vital for providing valuable outputs.

### Good Design Principles

#### Availability
Availability is the measure of how often a system is operational and accessible. It is calculated as the ratio of uptime to the total time (uptime plus downtime). High availability is crucial for minimizing downtime and ensuring users have continuous access to the system.

#### Reliability, Fault Tolerance, and Redundancy
- **Reliability**: The likelihood that a system will perform its intended functions without failure.
- **Fault Tolerance**: The ability of a system to continue operating correctly even when some components fail.
- **Redundancy**: Having backup systems that can take over when the primary system fails. This ensures continuous operation and high fault tolerance.

#### Throughput
Throughput is the measure of how much data or how many operations a system can handle over a specific period. Improving throughput can be achieved through vertical scaling (upgrading a single server) or horizontal scaling (adding more servers).

#### Latency
Latency is the delay between a request and the response. Minimizing latency is crucial for improving the user experience and the overall performance of the system. Latency can be internal (within the computer's components) or external (in network communications).

### Closing Notes
Designing effective systems involves balancing multiple factors, including availability, reliability, throughput, and latency. The goal is to create systems that handle failures gracefully, have high performance, and meet the required quality standards. Future topics will delve deeper into optimizing these aspects and ensuring efficient system design.

---
## Networking Basics
---

This section introduces the fundamental concepts of networking, focusing on how devices communicate over a network.

### What is a Network?

A network is a collection of devices connected to share resources and data. Each device in a network is assigned a unique IP address, allowing it to send and receive data. We use the analogy of Alice and Bob, two characters representing devices on the network, to explain these concepts. Alice wants to send an invitation to Bob, and in networking terms, this invitation is akin to data being transmitted over a network.

### IP Address

An IP address is a unique identifier for a device on a network. There are two main types:
- **IPv4**: 32-bit address, expressed in the format `0.0.0.0` to `255.255.255.255`. IPv4 addresses are running out due to the exponential growth of the internet, which led to the development of IPv6.
- **IPv6**: 128-bit address, expressed in the format `xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx`. IPv6 provides a significantly larger address space, making address exhaustion highly improbable.

### Protocols of Sending Data Over a Network

Data is transferred over a network using data packets. These packets consist of a header (source and destination IP addresses), data (payload), and a trailer.

### TCP (Transmission Control Protocol)

TCP ensures reliable data transmission by establishing a connection through a 3-way handshake and retransmitting lost packets. It uses sequence numbers to ensure packets are reassembled correctly at the destination.

### Application Data

Application data can take various forms, such as HTTP POST or GET requests, and resides in the application segment of the packet. For example, in an HTTP POST request, the information we wish to transmit is in the application data portion of the packet.

### Network Layers

Protocols are organized into layers to create a hierarchical structure:
- **Network Layer**: Includes the IP protocol, which handles the routing of data.
- **Transport Layer**: Includes the TCP protocol, which ensures reliable data transmission.
- **Application Layer**: Includes the HTTP protocol, which enables client-server communication.

### Public vs Private Network

- **Public IP Address**: Unique identifier for a device on the internet, accessible globally.
- **Private IP Address**: Used within a local

 network, not accessible from the internet. Private IP addresses are used in home or office networks.

### Static vs Dynamic IP Addresses

- **Dynamic IP Address**: Temporarily assigned, changes with each connection, commonly used for clients.
- **Static IP Address**: Permanently assigned, does not change, often used for servers. Static IP addresses require manual configuration.

### Ports

Ports are numeric identifiers used to distinguish between different services running on the same device. For example, port 80 is typically used for HTTP, and port 443 for HTTPS. Ports allow multiple services to run on the same IP address without conflict.

---
## TCP and UDP
---

This section delves deeper into the characteristics and use cases of TCP and UDP.

### TCP (Transmission Control Protocol)

TCP is a connection-oriented protocol that ensures reliable data transmission. It establishes a connection through a 3-way handshake and retransmits lost packets. TCP is slower due to its reliability features but is essential for applications where data integrity is crucial.

### Use Cases for TCP

TCP is suitable for applications requiring reliable data delivery, such as web browsing, email, and file transfers. These applications need to ensure that all data is received accurately and in order.

### UDP (User Datagram Protocol)

UDP is a connectionless protocol that allows faster data transmission but does not guarantee delivery. It is suitable for applications where speed is more critical than reliability, such as online gaming and video streaming.

### Use Cases for UDP

UDP is preferred for real-time applications like gaming and streaming, where occasional data loss is acceptable to maintain speed and performance.

### Closing Notes
Understanding the differences between TCP and UDP and their appropriate use cases is crucial for designing network applications that meet specific performance and reliability requirements.

---
## Domain Name System (DNS)
---

This section explains the Domain Name System (DNS), which translates human-readable domain names into numerical IP addresses.

### What is DNS?

DNS is like the internet's phone book, converting domain names (e.g., google.com) into IP addresses (e.g., 142.251.211.238). This allows computers to route requests correctly on the internet.

### ICANN and Domain Name Registrars

ICANN (Internet Corporation for Assigned Names and Numbers) manages the overall coordination and security of DNS. Domain registrars, certified by ICANN, handle domain name registration and maintenance.

### DNS Records

DNS records store information about domains. The most common type, the A (Address) record, links a domain name to an IPv4 address. This ensures that requests to a domain are routed to the correct server.

### Anatomy of a URL

A URL consists of several parts:
- **Protocol (Scheme)**: Indicates the protocol used (e.g., HTTP, HTTPS).
- **Domain**: The primary domain and top-level domain (TLD).
- **Path**: Specifies a particular resource within the domain.
- **Ports**: Specifies the port number if different from the default (e.g., `localhost:8080`).

### Closing Notes

Understanding networking fundamentals, including TCP, UDP, and DNS, is essential for designing and maintaining efficient systems. These components ensure reliable, fast, and accurate communication over the internet.

---
