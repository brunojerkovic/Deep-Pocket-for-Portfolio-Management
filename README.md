# Thesis_DeepPocket
Implementation of the research paper for portfolio optimization with Graph neural networks and deep reinforcement learning.

Farzan Soleymani, Eric Paquet,
Deep graph convolutional reinforcement learning for financial portfolio management – DeepPocket,
Expert Systems with Applications,
Volume 182,
2021,
115127,
ISSN 0957-4174,
https://doi.org/10.1016/j.eswa.2021.115127.
(https://www.sciencedirect.com/science/article/pii/S0957417421005686)
Abstract: Portfolio management aims at maximizing the return on investment while minimizing risk by continuously reallocating the assets forming the portfolio. These assets are not independent but correlated during a short time period. A graph convolutional reinforcement learning framework called DeepPocket is proposed whose objective is to exploit the time-varying interrelations between financial instruments. These interrelations are represented by a graph whose nodes correspond to the financial instruments while the edges correspond to a pair-wise correlation function in between assets. DeepPocket consists of a restricted, stacked autoencoder for feature extraction, a convolutional network to collect underlying local information shared among financial instruments and an actor-critic reinforcement learning agent. The actor-critic structure contains two convolutional networks in which the actor learns and enforces an investment policy which is, in turn, evaluated by the critic in order to determine the best course of action by constantly reallocating the various portfolio assets to optimize the expected return on investment. The agent is initially trained offline with online stochastic batching on historical data. As new data become available, it is trained online with a passive concept drift approach to handle unexpected changes in their distributions. DeepPocket is evaluated against five real-life datasets over three distinct investment periods, including during the Covid-19 crisis, and clearly outperformed market indexes.
Keywords: Portfolio management; Deep reinforcement learning; Restricted stacked autoencoder; Online leaning; Actor-critic; Graph convolutional network
