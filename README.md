

**Strategic &**

**Competitive Intelligence**

**Artificial Intelligence & Defence**

**Team**

**Teachers**

Antonella Martini

Filippo Chiarello

Alessandro Bonini

Carlo Alberto Carrucciu

Gian Maria Pandolfi





OUR TEAM

**Carlo Alberto Carrucciu**

**Alessandro Bonini**

**Gian Maria Pandolfi**

Background:

Management Engineering

Background:

Economics

Background:

Computer Engineering





RACI

MATRIX

**WEB**

**SCRAPING**

**TOPIC**

**MODELING**

**NETWORK**

**ANALYSIS**

**TREND**

**ANALYSIS**

Responsible

Responsible

Informed

Consulted

Responsible

Consulted

Responsible

Consulted

Consulted

Responsible

Responsible

Accountable

3





**Work Breakdown Structure**

4









Our research links artificial intelligence with its applications in the defence topic

Methodologies have been developed in order to find new key technologies and also answers the Asset+ challenge

promoted by EU, for which we have been selected.

The analysis started with web crawling and scraping on Scopus to collect scientific articles. Then using our query we

scraped Google Patents to collect the patents.

We applied text cleaning on the abstracts collected in order to prepare our dataset for the topic modelling.

Once we’ve collected the topics we did a trend analysis and we applied network analysis algorithms to each of the

topic in order to understand which are the documents with most relevance. So, we discovered new trend technologies

and technologies which are reaching the maturity.

Abstract





ARTIFICIAL INTELLIGENCE

DEFENCE





Context

The European Commission has declared

technology as the EU’s top priority for the next

five years.

More particularly, one key objective is the

mastery and ownership of key technologies in

Europe.

With this objective in mind we decided to

analyze the patent and scientific articles

landscape in order to evaluate the most

important technologies for Europe’s

sovereignty in defence and security.





**Which technologies and technology bricks**

**are particularly relevant for Europe’s**

**sovereignty in defence and security?**





**01**

Text Cleaning &

Topic Modeling

**Query Formulation**

**Data Retrieval**

**02**

**04**

Trend Analysis &

Network Analysis

Agenda

**03**

Top Technologies

Selection





("artificial intelligence" OR "machine learning" OR

"image recognition" OR "natural language processing"

OR "neural networks" OR "big data" OR "optical

recognition system" OR "synthetic intelligence" OR

"speech recognition" OR "machine vision" OR

"dialogue system" OR "machine translation" OR

"pattern recognition" OR "autonomous vehicle" OR

"machine perception")

We tried to focus our research on the opportunities

that the artificial intelligence can bring and the

dangers misinformation can leads to…

AND

( “cyber defence” OR “cyberdefence” OR "election" OR

"fake news" OR "malware" OR "terrorism" OR

"hacking" OR "sabotage" OR "propaganda" OR

"misinformation" OR "disinformation" OR "espionage")

Our Query!





\1. Retrieve Information

**STEP 3**

**STEP 2**

**STEP 1**

Data Cleaning and

Data Preparation

Documents

Collection and Data

Structuring

Querying Scopus

and Google to get

scientific articles

and patents

12









**DATASET STRUCTURE**

code

title

pub\_date citations abstract

class

US8370262B2

System and ...

2013-02-05

US10681025B2, ...

The System and ...

G06

. . . .

. . . .

. . . .

. . . .

. .

. .

. .

. .





**!**

Cleaning your data

is important

**Abstract**

**Citations**

A small part of the patents was without abstract,

and have been deleted. Abstracts are

fundamental for our analysis

Only the records that had at least one internal

citation have been condìsdiered

15





**22,573**

Patents retrieved from Google Patents

7,981

Scientific Papers retrieved from Scopus

17,457

Number of Documents after Data Cleaning





**Documents Analysis**

G06-COMPUTING;CALCULATING;COUNTING

17

H04-ELECTRICCOMMUNICATIONTECHNIQUE





\2. Topic modeling

**STEP 3**

**STEP 2**

**STEP 1**

Topic Analysis

and Understanding

Topic Modelling

Abstract Collection

and Preprocessing





pyLDAvis





Basic passages for text

cleaning?

01

Cleaning text to

obtain optimal result

during the next phase

**Remove punctuation**

**and case normalization**

02

**Tokenization**

**and stopword**

**removing**

03

**Lemmatization and**

**bigram costrupting**





Optimal

No. topics

Analysing **perplexity** and

**coherence** for different

numbers of topics, we

discovered that optimal

results are reached using 4

or 7 topics

21





Topics structure

22





**Topic Analysis**

[Topic](https://topicmodeling.000webhostapp.com/topics.html)[ ](https://topicmodeling.000webhostapp.com/topics.html)[Dashboard](https://topicmodeling.000webhostapp.com/topics.html)





**Topic Understanding**

TOPIC 2

**User**

**Authentication**

system, user,

sensors, engine,

processor,

TOPIC 0

**Cyber Security**

security, server

risk, blockchain

cloud,

TOPIC 1

**Image - Video**

image,

recognition,

camera, face,

picture,...

transaction,...

controller, ...

TOPIC 6

**Not Relevant**

first, second,

message,

document, dns,

der, und, die

TOPIC 3

**Computer**

**science**

TOPIC 4

**Machine**

**learning**

TOPIC 5

**Social - Web**

social,website,web

page, audio,

advertisement,

online, search,

page

file, computing,

computer,

application,

executable,

systems

model, learning,

training, vector,

neural, classifier,

clustering,

sequences





\3. Network analysis &

Trend analysis

Components

Trends

Topics

STEP 4

STEP 2 (a)

Community

Detection

Cores Documents

Detection

STEP 2 (b)

Topic

Behavioural

Analysis

STEP 3

Trend Analysis

STEP 1

Component

Analysis

Communities

Cores









Components :

network is

unconnected…

Giant Component

Only one big

component of 1084

nodes

01

02

385

Medium size

agglomerates

Interesting separated

groups, any high

clustered

Outliers

Unconnected group of

nodes that are not

considered





What about the two

components of

medium size?

**120 nodes**

**High clustering coeff.**

**68 nodes**

**Various topics**

Mobile

Content

System

Video

Content

Communication

Search

Media

28





Analysing the giant

component!





Cores

Structure

More **central nodes have**

**an high “core score”**

according to the core

algorithm… 15 concentric

cores sections are obtained

30





Communities

Detection

**Topics** do not create any

clusters because they are

not well separated.

So, we can **divide** the

network in clustered

**Louvain communities.**

31





**Trend Analysis**





Each topic has a core...

3





Topics Cores

Detection

Taking in consideration

only the **cores of each**

**topics**, they appear well

separated. We will focus

on these patents.

[Surf](https://topicmodeling.000webhostapp.com/#/)[ ](https://topicmodeling.000webhostapp.com/#/)[the](https://topicmodeling.000webhostapp.com/#/)[ ](https://topicmodeling.000webhostapp.com/#/)[network](https://topicmodeling.000webhostapp.com/#/)[ ](https://topicmodeling.000webhostapp.com/#/)[!](https://topicmodeling.000webhostapp.com/#/)

34





Cores Heatmap

Thanks to the heatmap we are able to select only

the most relevant documents given the

intersection between the topic cores and

communities core.

Heatmap communities and topics

Intersection cores communities and topics





\4. Final Evaluation

Network Security

User Authentication

Image - Video





Network Security

**Technologies**

Cloud based security monitoring using unsupervised

pattern recognition and deep learning, predicting

behavioral analysis for malware detection.

(US20190068627A1)

Method, system and equipment for deducing

malicious code rules based on deep learning method

(CN105975857A)

Profiling cyber threats detected in a target

environment and automatically generating one or

more rule bases for an expert system usable to profile

cyber threats detected in a target environment

(US20150163242A1)

**Maturity: Growing**

Identification of computerized bots and automated

cyber-attack modules

(US20160306974A1)





User Authentication

**Technologies**

System, device, and method of visual login

and stochastic cryptography

(US10032010B2)

Systems and methods for enabling biometric

authentication options (US9569605B1)

Device, system, and method of three-

dimensional spatial user authentication

(US20160300054A1)

Systems and methods for detecting security

threats based on user profiles (US9401925B1)

**Maturity: Emerging/Growing**





Image - Video

**Technologies**

Image Recognition

Hologram Reproducing and Recording

Face Identification

Maturity: **Emerging**





Deep Fake - Fake News

**Strengths:**

**Threat:**

\-

\-

Unlimited

\-

\-

Misinformation

Political and Social

Instability

Inexpensive

\-

Easily accessible

**Recognition Technologies:**

Deep Learning,

Convolutional Neural Networks,

Image Recognition and

Face Identification





Thank you for

your attention!

