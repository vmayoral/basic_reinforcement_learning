# Reviewing Policy Gradient methods

>Continuous states and actions in high dimensional spaces cannot be treated by most off-the-shelf reinforcement learning approaches. Policy gradient methods differ significantly as they do not suffer from these problems in the same way. For example, uncertainty in the state might degrade the performance of the policy (if no additional state estimator is being used) but the optimization techniques for the policy do not need to be changed. Continuous states and actions can be dealt with in exactly the same way as discrete ones while, in addition, the learning performance is often increased. Convergence at least to a local optimum is guaranteed (**critical for robotics**).

>The advantages of policy gradient methods for real world applications are numerous. Among the most important ones are that the policy representations can be chosen so that it is meaningful for the task and can incorporate domain knowledge, that often fewer parameters are needed in the learning process than in value-function based approaches and that there is a variety of different algorithms for policy gradient estimation in the literature which have a rather strong theoretical underpinning. Additionally, policy gradient methods can be used either model-free or model-based as they are a generic formulation.

Policy gradient methods however,
> are by definition on-policy (note that tricks like importance sampling can slightly alleviate this problem) and need to forget data very fast in order to avoid the introduction of a bias to the gradient estimator. Hence, the use of sampled data is not very efficient. In tabular representations, value function methods are guaranteed to converge to a global maximum while policy gradients only converge to a local maximum and there may be many maxima in discrete problems. Policy gradient methods are often quite demanding to apply, mainly because one has to have considerable knowledge about the system one wants to control to make reasonable policy definitions. Finally, policy gradient methods always have an open parameter, the learning rate, which may decide over the order of magnitude of the speed of convergence, these have led to new approaches inspired by expectation-maximization (see, e.g., Vlassis et al., 2009; Kober & Peters, 2008).


### Notation:
The three main components of a Reinfocement Learning model for robotics include the state $s$ (also found in literature as $x$), the action $a$ (also found as $u$) and the reward denoted by $r$. The stochasticity of the environment gets accounted by using a probability distribution $\mathbf{x}_{k+1}\sim p\left(  \mathbf{x} _{k+1}\left\vert \mathbf{x}_{k},\mathbf{u}_{k}\right.  \right)$.




# Sources:
- http://www.scholarpedia.org/article/Policy_gradient_methods#Likelihood_Ratio_Methods_and_REINFORCE
- https://theneuralperspective.com/2016/10/27/gradient-topics/
- https://theneuralperspective.com/2016/11/25/reinforcement-learning-rl-policy-gradients-i/
- https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient
