## What is reinforcement learning?
Reinforcement learning is a machine learning training method based on rewarding desired behaviors and punishing undesired ones. 
In general, a reinforcement learning agent -- the entity being trained -- is able to perceive and interpret its environment, take actions and learn through trial and error.
Reinforcement learning is one of several approaches developers use to train machine learning systems. What makes this approach important is that it empowers an agent,
whether it's a feature in a video game or a robot in an industrial setting, to learn to navigate the complexities of the environment it was created for.
Over time, through a feedback system that typically includes rewards and punishments, the agent learns from its environment and optimizes its behaviors.

## How does reinforcement learning work?
In reinforcement learning, developers devise a method of rewarding desired behaviors and punishing negative behaviors. This method assigns positive values to the desired actions to encourage the agent to use them, while negative values are assigned to undesired behaviors to discourage them. This programs the agent to seek long-term and maximum overall rewards to achieve an optimal solution.
These long-term goals help prevent the agent from getting stuck on less important goals. With time, the agent learns to avoid the negative and seek the positive. This learning method has been adopted in artificial intelligence (AI) as a way of directing unsupervised machine learning through rewards or positive reinforcement and penalties or negative reinforcement.

The Markov decision process serves as the basis for reinforcement learning systems. In this process, an agent exists in a specific state inside an environment; it must select the best possible action from multiple potential actions it can perform in its current state. Certain actions offer rewards for motivation.
When in its next state, new rewarding actions are available to it. Over time, the cumulative reward is the sum of rewards the agent receives from the actions it chooses to perform.

## Eg:
When reinforcement learning is used to train a logistics robot, the robot is the agent that functions in a warehouse environment.
It chooses various actions that are met with feedback, which includes rewards and information or observations from the environment. All the feedback helps the agent develop a strategy for future actions.
![image](https://github.com/DevJSter/AIML/assets/115056248/c560932e-910b-4a40-864c-5b9e962d3c30)

## Applications and examples of reinforcement learning
While reinforcement learning has been a topic of much interest in the field of AI, its widespread, real-world adoption and application remain limited. Noting this, however, research papers abound on theoretical applications, and there have been some successful use cases.
Current uses include but are not limited to the following:
1. Gaming.
2. Resource management.
3. Personalized recommendations.
4. Robotics.
5. Gaming is likely the most common use for reinforcement learning, as it can achieve superhuman performance in numerous games. A common example involves the game Pac-Man.

A learning algorithm playing Pac-Man might have the ability to move in one of four possible directions, barring obstruction. From pixel data, an agent might be given a numeric reward for the result of a unit of travel: 0 for empty spaces, 1 for pellets, 2 for fruit, 3 for power pellets, 4 for ghost post-power pellets, 5 for collecting all pellets to complete a level, and a 5-point deduction for collision with a ghost. The agent starts from randomized play and moves to more sophisticated play, learning the goal of getting all pellets to complete the level. 
Given time, an agent might even learn tactics such as conserving power pellets until needed for self-defense.
Reinforcement learning can operate in a situation as long as a clear reward can be applied. In enterprise resource management, reinforcement algorithms allocate limited resources to different tasks as long as there's an overall goal it is trying to achieve. A goal in this circumstance would be to save time or conserve resources.
In robotics, reinforcement learning has found its way into limited tests. This type of machine learning can provide robots with the ability to learn tasks a human teacher cannot demonstrate, to adapt a learned skill to a new task and to achieve optimization even when analytic formulation isn't available.
Reinforcement learning is also used in operations research, information theory, game theory, control theory, simulation-based optimization, multi-agent systems, swarm intelligence, statistics, genetic algorithms and ongoing industrial automation efforts.

## Challenges of applying reinforcement learning
Reinforcement learning, while high in potential, comes with some tradeoffs. It can be difficult to deploy and remains limited in its application. One of the barriers for deployment of this type of machine learning is its reliance on exploration of the environment.

For example, if you were to deploy a robot that was reliant on reinforcement learning to navigate a complex physical environment, it will seek new states and take different actions as it moves. With this type of reinforcement learning problem, however, it's difficult to consistently take the best actions in a real-world environment because of how frequently the environment changes.

The time required to ensure the learning is done properly through this method can limit its usefulness and be intensive on computing resources. As the training environment grows more complex, so too do the demands on time and compute resources.

Supervised learning can deliver faster, more efficient results than reinforcement learning to companies if the proper amount of data is available, as it can be employed with fewer resources.

## Common reinforcement learning algorithms
Rather than referring to a specific algorithm, the field of reinforcement learning is made up of several algorithms that take somewhat different approaches. The differences are mainly because of the different strategies they use to explore their environments:

1. State-action-reward-state-action. This reinforcement learning algorithm starts by giving the agent what's known as a policy. Determining the optimal policy-based approach requires looking at the probability of certain actions resulting in rewards, or beneficial states, to guide its decision-making.
2. Q-learning. This approach to reinforcement learning takes the opposite approach. The agent receives no policy and learns about an action's value based on exploration of its environment. This approach isn't model-based but instead is more self-directed. Real-world implementations of Q-learning are often written using Python programming.
3. Deep Q-networks. Combined with deep Q-learning, these algorithms use neural networks in addition to reinforcement learning techniques. They are also referred to as deep reinforcement learning and use reinforcement learning's self-directed environment exploration approach. As part of the learning process, these networks base future actions on a random sample of past beneficial actions.

![image](https://github.com/DevJSter/AIML/assets/115056248/eb9b9c30-606a-4d5c-a909-b296b94ddbc8)
#### A neural network consists of a set of algorithms that closely resembles the human brain. These algorithms are designed to recognize patterns.

## How is reinforcement learning different from supervised and unsupervised learning?
Reinforcement learning is considered its own branch of machine learning. However, it does have some similarities to other types of machine learning, which break down into the following four domains:

1. Supervised learning. In supervised learning, algorithms train on a body of labeled data. Supervised learning algorithms can only learn attributes that are specified in the data set. A common application of supervised learning is image recognition models. These models receive a set of labeled images and learn to distinguish common attributes of predefined forms.
2. Unsupervised learning. In unsupervised learning, developers turn algorithms loose on fully unlabeled data. The algorithms learn by cataloging their own observations about data features without being told what to look for.
3. Semisupervised learning. This method takes a middle-ground approach. Developers enter a relatively small set of labeled training data, as well as a larger corpus of unlabeled data. The algorithm is then instructed to extrapolate what it learns from the labeled data to the unlabeled data and draw conclusions from the set as a whole.
4. Reinforcement learning. This takes a different approach. It situates an agent in an environment with clear parameters defining beneficial activity and nonbeneficial activity and an overarching endgame to reach.
##### Reinforcement learning is similar to supervised learning in that developers must give algorithms specified goals and define reward functions and punishment functions. This means the level of explicit programming required is greater than in unsupervised learning. But, once these parameters are set, the algorithm operates on its own -- making it more self-directed than supervised learning algorithms. For this reason, people sometimes refer to reinforcement learning as a branch of semisupervised learning; in truth, though, it is most often acknowledged as its own type of machine learning.

![image](https://github.com/DevJSter/AIML/assets/115056248/379fc463-4c1e-445e-943e-979641d5d34f)
###### Reinforcement learning is one of four types of training approaches for machine learning models.

## The future of reinforcement learning
Reinforcement learning is projected to play a bigger role in the future of AI. The other approaches to training machine learning algorithms require large amounts of preexisting training data. Reinforcement learning agents, on the other hand, require the time to gradually learn how to operate via interactions with their environments. Despite the challenges, various industries are expected to continue exploring reinforcement learning's potential.
Reinforcement learning has already demonstrated promise in various areas. For example, marketing and advertising firms are using algorithms trained this way for recommendation engines. Manufacturers are using reinforcement learning to train their next-generation robotic systems.
Scientists at Alphabet's AI subsidiary, Google DeepMind, have proposed that reinforcement learning could bring the current state of AI -- often called narrow AI -- to its theoretical final form of artificial general intelligence. They believe machines that learn through reinforcement learning will eventually become sentient and operate independently of human supervision


