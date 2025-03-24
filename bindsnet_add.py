from abc import ABC
from typing import Dict, Union, Optional, Sequence, Iterable

import torch
import numpy as np
import collections

from bindsnet.learning import LearningRule
from bindsnet.network.topology import (
    AbstractConnection,
    Connection,
    Conv2dConnection,
    LocalConnection,
)
from bindsnet.network.nodes import LIFNodes, AdaptiveLIFNodes
from bindsnet.network import Network
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input
from util import plot_performance, mv

class adam_optimizer():
    #def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-09):
    def __init__(self, learning_rate, beta_1=0.999, beta_2=0.99999, epsilon=1e-09):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self._cache = {}        
    
    def delta(self, grads, name="_", learning_rate=None):
        if name not in self._cache:
            self._cache[name] = [[torch.zeros_like(i).to(grads[0].device) for i in grads],
                                 [torch.zeros_like(i).to(grads[0].device) for i in grads],
                                 0]
        self._cache[name][2] += 1 
        t = self._cache[name][2]
        deltas = []
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        learning_rate = self.learning_rate if learning_rate is None else learning_rate        
        for n, g in enumerate(grads):                
            m = self._cache[name][0][n]
            v = self._cache[name][1][n]
            m = beta_1 * m + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * (g ** 2)
            self._cache[name][0][n] = m
            self._cache[name][1][n] = v            
            m_hat = m / (1 - np.power(beta_1, t).item())
            v_hat = v / (1 - np.power(beta_2, t).item())
            deltas.append(learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon))
        
        return deltas   

class FBTDSTDP(LearningRule):
    # language=rst
    """
    Feedback-modulated TD-STDP.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``FBTDSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP``
            learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        :param tc_e_trace: Time constant for eligibility trace
        :param tc_a_trace: Time constant for feedback-gated eligibility trace
        :param fb_gate: Boolean whether to gate by feedback signal. If not then just TD-STDP
        :param targ_firing_r: If larger than 0, then it will add target firing rate subtracting
            current firing rate as reward, scaled by targ_firing_r, for all post-synaptic unit
        :param targ_firing_rate: Target firing rate of post-synpatic units
        :param inh: Whether the target neuron is inhibitory; only affect entropy grad sign in updating
        :param adam: Whether to use adam optimizer
        :param adam_beta_1: Beta1 in adam optimizer
        :param adam_beta_2: Beta2 in adam optimizer
        :param gpu: Whether to use gpu
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )
        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 20.0))
        self.tc_a_trace = torch.tensor(kwargs.get("tc_a_trace", 20.0))
        self.fb_gate = kwargs.get("fb_gate", False)

        if self.fb_gate:
            self.num_actions = kwargs["num_actions"]
            assert self.target.n % self.num_actions == 0, "Number of node has to be divisible by number of actions"
            self.node_label = torch.repeat_interleave(torch.arange(start=0, end=self.num_actions), 
                self.target.n//self.num_actions).unsqueeze(0)     
            if kwargs["gpu"]: self.node_label = self.node_label.cuda()

        self.targ_firing_r = torch.tensor(kwargs.get("targ_firing_r", 0))        
        self.targ_firing_rate = torch.tensor(kwargs.get("targ_firing_rate", 0.2))        
        self.inh = kwargs.get("inh", False)
        self.adam = kwargs.get("adam", False)
        adam_beta_1 = kwargs.get("adam_beta_1", 0.999)
        adam_beta_2 = kwargs.get("adam_beta_2", 0.99999)
        self.weight_reg = kwargs.get("weight_reg", 0)
        self.gpu = kwargs.get("gpu", False)

        if self.adam: self.optimzier = adam_optimizer(nu[1], beta_1=adam_beta_1, beta_2=adam_beta_2)

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        FB-modulated TD-STDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size
        gpu = kwargs.get("gpu", False)

        self.at_decay = torch.exp(-self.connection.dt / self.tc_a_trace)
        self.et_decay = torch.exp(-self.connection.dt / self.tc_e_trace)
        self.ltp_decay = torch.exp(-self.connection.dt / self.tc_plus)
        self.ltd_decay = torch.exp(-self.connection.dt / self.tc_minus)

        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(batch_size, int(np.prod(self.source.shape)))
            if gpu: self.p_plus = self.p_plus.cuda()
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(batch_size, *self.target.shape)
            if gpu: self.p_minus = self.p_minus.cuda()
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(batch_size, *self.connection.w.shape)
            if gpu: self.eligibility = self.eligibility.cuda()
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(batch_size, *self.connection.w.shape)
            if gpu: self.eligibility_trace = self.eligibility_trace.cuda() 
        if self.fb_gate:     
            if not hasattr(self, "action_v"):
                self.action_v = torch.zeros(batch_size, *self.connection.w.shape)
                if gpu: self.action_v = self.action_v.cuda()            
            if not hasattr(self, "action_trace"):
                self.action_trace = torch.zeros(batch_size, *self.connection.w.shape)
                if gpu: self.action_trace = self.action_trace.cuda()

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()        
        reward = kwargs["reward"][self.connection]

        if self.fb_gate:   
            p = torch.repeat_interleave(kwargs["p"], self.target.n//self.num_actions, dim=-1).unsqueeze(1)            
            a = (kwargs["action"].unsqueeze(-1) == self.node_label).unsqueeze(1).float()     
            self.action_v = (a - p) * self.eligibility_trace
            self.action_trace *= self.at_decay
            self.action_trace += self.action_v / self.tc_a_trace
            update = reward * self.action_trace

            if kwargs.get("ent", None) is not None:
                ent = torch.repeat_interleave(kwargs["ent"], self.target.n//self.num_actions, dim=-1).unsqueeze(1)
                if self.inh: 
                    update -= ent*self.eligibility_trace
                else:
                    update += ent*self.eligibility_trace            
        else:
            update = reward * self.eligibility_trace

        if self.targ_firing_r != 0:
            targ_adj = (self.targ_firing_rate - torch.mean(self.target.x, dim=1)/self.target.tc_trace)*self.targ_firing_r
            update += targ_adj.unsqueeze(-1).unsqueeze(-1)*self.eligibility_trace
        
        if self.weight_reg != 0:
            update -= 0.5 * self.weight_reg * self.connection.w

        if self.adam: 
            update = self.optimzier.delta([self.reduction(update, dim=0)])[0]
        else:
            update = self.reduction(update, dim=0)
            
        self.connection.w += update

        # Update P^+ and P^- values.
        self.p_plus *= self.ltp_decay
        self.p_plus += self.nu[1] * source_s
        # Calculate point eligibility value.
        self.eligibility = torch.bmm(self.p_plus.unsqueeze(2), target_s.unsqueeze(1))

        if self.nu[0] != 0:
            self.p_minus *= self.ltd_decay
            self.p_minus += -self.nu[0] * target_s
            self.eligibility += torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))

        # Compute weight update based on the eligibility value of the past timestep.          
        self.eligibility_trace *= self.et_decay
        self.eligibility_trace += self.eligibility / self.tc_e_trace
        
        super().update()

    def reset_state_variables(self, index=slice(None)) -> None:
        #super().reset_state_variables()      
        if hasattr(self, "p_plus"):  self.p_plus[index] = 0
        if hasattr(self, "p_minus"):  self.p_minus[index] = 0
        if hasattr(self, "eligibility_trace"):  self.eligibility_trace[index] = 0
        if self.fb_gate and hasattr(self, "action_trace"): self.action_trace[index] = 0

    def set_batch_size(self, batch_size) -> None:
        if hasattr(self, "p_plus"): delattr(self, "p_plus")
        if hasattr(self, "p_minus"): delattr(self, "p_minus")
        if hasattr(self, "eligibility"): delattr(self, "eligibility")
        if hasattr(self, "eligibility_trace"): delattr(self, "eligibility_trace")
        if hasattr(self, "action_v"): delattr(self, "action_v")
        if hasattr(self, "action_trace"): delattr(self, "action_trace")

class Input_add(Input):
    def reset_state_variables(self, index=slice(None)) -> None:
        super().reset_state_variables()

class LIFNodes_add(LIFNodes):
    # Just LIF Node but with index reset

    def reset_state_variables(self, index=slice(None)) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        self.v[index] = self.rest  # Neuron voltages.
        self.refrac_count[index] = 0  # Refractory period counters.
        if self.traces: self.x[index] = 0
        if self.sum_input: self.summed[index] = 0  # Summed inputs.        

class Network_add(Network):

    def run(
        self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
    ) -> None:
        # language=rst
        """
        Same as original network but reward function is called after spike computation
        and reward function has to return a dictionary with connection as key and value
        as the reward to that connection. Network will be passed to reward function when
        calling.
        """

        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                self.layers[l].forward(x=current_inputs[l])

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[unclamp] = 0
                    else:
                        self.layers[l].s[unclamp[t]] = 0

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]
            
            
            # Compute reward.
            if self.reward_fn is not None:
                # Call reward function, passing self (Network class) and t (time step)
                kwargs["reward"] = self.reward_fn.compute(self, t, **kwargs)

            for c in self.connections:   
                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )

            # Get input to all layers.
            current_inputs.update(self._get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()



    def reset_state_variables(self, index=slice(None)) -> None:
        # language=rst
        """
        Index version of reset excluding monitor
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables(index=index)

        for connection in self.connections:
            self.connections[connection].update_rule.reset_state_variables(index=index)

        #for monitor in self.monitors:
        #    self.monitors[monitor].reset_state_variables()

def build_ac_network(input_shape, init_scale, num_critics, num_actors_pa,
                     num_actions, critic_layer_param, critic_fd_conn_param,
                     actor_layer_param, actor_fd_conn_param, actor_active=True, 
                     gpu=False):

    """
    layer_param: dict
    :param input_shape: shape of input
    :param init_scale: scale when used to initialize weight
    :param num_critics: (number of excitatory critics, number of inhibitory critics)
    :param num_actors_pa: (number of excitatory actors per action, number of inhibitory actors per action)
    :param num_actions: number of actions available
    :param critic_layer_param: dictionary passed to critic layer init.
    :param critic_fd_conn_param: dictionary passed to critic layer connection init.
    :param actor_layer_param: dictionary passed to actor layer
    :param actor_fd_conn_param: dictionary passed to actor layer connection init.
    :param dt: time step
    :param actor_active: whether to have actor. If false then only critic network remains

    """

    network = Network_add(dt=1)
    # Layers of neurons
    input_layer = Input_add(n=np.prod(input_shape), shape=(np.prod(input_shape),), traces=True, tc_trace=25.0)    
    network.add_layer(input_layer, name="H0")
    layers = [input_layer]
    in_n_neurons = int(np.prod(input_shape))
    
    if type(init_scale) != list: init_scale = [init_scale, init_scale]
    
    # Critic layers                

    critic_layer_param.update({"n": num_critics[0],
                               "shape": (num_critics[0],)})
    layers.append(LIFNodes_add(**critic_layer_param))    
    network.add_layer(layers[-1], name="CE")
    fd_w = torch.rand(in_n_neurons, num_critics[0]) * init_scale[0]
    critic_fd_conn_param.update({"source": input_layer,
                                 "target": layers[-1],
                                 "w": fd_w,
                                 "inh": False,
                                 "gpu": gpu})
    critic_fd_conn = Connection(**critic_fd_conn_param)
    network.add_connection(critic_fd_conn, source="H0", target="CE")         


    if num_critics[1] > 0:
        critic_layer_param.update({"n": num_critics[1],
                                   "shape": (num_critics[1],)})
        layers.append(LIFNodes_add(**critic_layer_param))
        network.add_layer(layers[-1], name="CN")
        fd_w = torch.rand(in_n_neurons, num_critics[1]) * init_scale[0]
        critic_fd_conn_param.update({"source": input_layer,
                                     "target": layers[-1],
                                     "w": fd_w,
                                     "inh": True,
                                     "gpu": gpu})
        critic_fd_conn = Connection(**critic_fd_conn_param)
        network.add_connection(critic_fd_conn, source="H0", target="CN")         

                      
    # Actor layers              

    if actor_active:                  
      actor_layer_param.update({"n": num_actors_pa[0]*num_actions,
                                "shape": (num_actors_pa[0]*num_actions,)})
      layers.append(LIFNodes_add(**actor_layer_param))      
      network.add_layer(layers[-1], name="AE")      
      fd_w = torch.rand(in_n_neurons, num_actors_pa[0]*num_actions) * init_scale[1]  
      actor_fd_conn_param.update({"source": input_layer,
                                   "target": layers[-1],
                                   "w": fd_w,
                                   "num_actions": num_actions,
                                   "inh": False,
                                   "gpu": gpu})
      actor_fd_conn = Connection(**actor_fd_conn_param)      
      network.add_connection(actor_fd_conn, source="H0", target="AE")

      if num_actors_pa[1] > 0:
        actor_layer_param.update({"n": num_actors_pa[1]*num_actions,
                                "shape": (num_actors_pa[1]*num_actions,)})
        layers.append(LIFNodes_add(**actor_layer_param))
        network.add_layer(layers[-1], name="AN") 
        fd_w = torch.rand(in_n_neurons, num_actors_pa[1]*num_actions) * init_scale[1]
        actor_fd_conn_param.update({"source": input_layer,
                                     "target": layers[-1],
                                     "w": fd_w,
                                     "num_actions": num_actions,
                                     "inh": True,
                                     "gpu": gpu})        
        actor_fd_conn = Connection(**actor_fd_conn_param)  
        network.add_connection(actor_fd_conn, source="H0", target="AN")
      
    if gpu: network.to("cuda")
    return network        


class Reward_fn():
  # Reward function to be called in network.run
  def __init__(self, tc, network_steps, isEnd_zero, last_zero_n, batch_size, value_m, value_b, gpu=False):
    self.gpu = gpu
    self.isEnd_zero = isEnd_zero
    self.last_zero_n = last_zero_n
    self.batch_size = batch_size
    self.value_m = value_m
    self.value_b = value_b

    self.v_prev = torch.zeros(self.batch_size, dtype=torch.float32)    
    self.tc = torch.tensor(tc).float()
    self.gamma = torch.tensor(np.exp(-1/tc)).float()    
    self.r_gamma = torch.sqrt(self.gamma)    
    self.network_steps = network_steps    
    
    if self.gpu: 
      self.tc = self.tc.float()
      self.v_prev = self.v_prev.cuda()
      self.gamma = self.gamma.cuda()
      self.r_gamma = self.r_gamma.cuda()
      
    # For record only
    self.v_prev_rec = torch.zeros(network_steps, dtype=torch.float32)    
    self.td_error_rec = torch.zeros(network_steps, dtype=torch.float32)    
    self.reward_rec = torch.zeros(network_steps, dtype=torch.float32)    
    self.isEnd_rec = torch.zeros(network_steps, dtype=torch.bool)    
    
  def compute(self, network, t, **kwargs):    
    # Requires also env_reward (environment reward), isEnd (whether eps ends)
    # and info which contains trunactedEnd (whether eps. is truncated) and 
    # StateCode (0 for normal, 1 for resting, 2 for resetting, 3 for warming up)
    # for computing TD Error to be broadcast to network
        
    env_reward = torch.from_numpy(kwargs.get("env_reward", None)).float()    
    stateCode = torch.from_numpy(kwargs["info"]["stateCode"]).float()
    truncatedEnd = torch.from_numpy(kwargs["info"]["truncatedEnd"])  
    isEnd = torch.from_numpy(kwargs.get("isEnd", None))
    isEnd = isEnd & ~truncatedEnd    
    critic_exc = network.layers["CE"]
    critic_inh = network.layers["CN"] if "CN" in network.layers else None
    
    if self.gpu: 
      stateCode = stateCode.cuda()
      env_reward = env_reward.cuda()
      isEnd = isEnd.cuda()
      
    env_reward /= self.network_steps  #TBR
    isLive = ~(stateCode == 1) if t < self.network_steps - self.last_zero_n else ~isEnd
    # here all V' is set to 0 if it is (i) last_zero_n time step in first time isEnd 
    # is True, or (ii) episodes enter resting period. Note resting period begins one
    # env. step AFTER the first env. step isEnd is true.
    
    #if t > 0: env_reward = 0    #TBR
    #isLive = ~isEnd #TBR
    
    v = self.value_b + torch.mean(critic_exc.x, dim=1)/critic_exc.tc_trace*self.value_m
    if critic_inh is not None:
      v -= torch.mean(critic_inh.x, dim=1)/critic_inh.tc_trace*self.value_m    
      
    td_error = torch.zeros(self.batch_size, dtype=torch.float32)    
    if self.isEnd_zero:
      td_error = (self.r_gamma * env_reward + self.gamma * v * isLive.float()) - self.v_prev    
    else:
      td_error = (self.r_gamma * env_reward + self.gamma * v ) - self.v_prev    
    td_error[stateCode >= 2] = 0.
    # no td error for reset period or warm up period    
    self.v_prev = v
    
    #if t > 0: td_error[isEnd] = 0.  #TBR
    
    td_error = td_error.unsqueeze(-1).unsqueeze(-1)    
    rewards = {}        
    rewards[network.connections[("H0","CE")]] = td_error
    if "CN" in network.layers: rewards[network.connections[("H0","CN")]] = -td_error
    if "AE" in network.layers: rewards[network.connections[("H0","AE")]] = td_error
    if "AN" in network.layers: rewards[network.connections[("H0","AN")]] = -td_error    
      
    self.td_error_rec[t] = td_error[0]
    self.v_prev_rec[t] = self.v_prev[0]
    self.reward_rec[t] = env_reward[0] if type(env_reward) == torch.Tensor else 0
    self.isEnd_rec[t]= isEnd[0]
      
    return rewards    

class Snn_actor():
  # Actor class that takes state as input and output action
  def __init__(self, network, batch_size, state_to_spike, augment_state, env_dt, reward_adj,
               actor_m, num_actions, num_actors_pa, actor_active, entropy_reg, freeze_action, 
               stat_rec=False, gpu=False):
        
    self.network = network    
    self.state_to_spike = state_to_spike
    self.augment_state = augment_state
    self.env_dt = env_dt
    self.reward_adj = reward_adj
    self.actor_m = actor_m
    self.num_actions = num_actions    
    self.num_actors_exc_pa = num_actors_pa[0]
    self.num_actors_inh_pa = num_actors_pa[1]
    self.actor_active = actor_active
    self.entropy_reg = entropy_reg
    self.freeze_action = freeze_action    
    self.stat_rec = stat_rec # Whether to record stat. on each call
    self.gpu = gpu
    
    self.p = torch.full((batch_size, num_actions), 1/num_actions)
    self.ent_eye = torch.eye(num_actions)      
    self.action = torch.distributions.Categorical(self.p).sample()
    self.freeze_timer = torch.zeros(batch_size)        
    
    if gpu: 
      self.p = self.p.cuda()
      self.ent_eye = self.ent_eye.cuda()
      self.action = self.action.cuda()
      self.freeze_timer = self.freeze_timer.cuda()
      
    if not actor_active:
      self.policy = uniform_policy(batch_size=batch_size, actions=[i for i in range(num_actions)])    

    qlen, dlen = int(3000), int(3000/env_dt)
    self.stat = {"est_v": collections.deque(maxlen=qlen),
                "td_error": collections.deque(maxlen=qlen),
                "reward": collections.deque(maxlen=qlen),
                "isEnd": collections.deque(maxlen=qlen),              
                "v_exc_rate": collections.deque(maxlen=dlen),
                }
    if "CN" in self.network.layers:
      self.stat.update({"v_inh_rate": collections.deque(maxlen=dlen)})
      
    if self.actor_active:
      self.stat.update({"a_exc_rate": [collections.deque(maxlen=dlen) for _ in range(num_actions)],
                       "p": [collections.deque(maxlen=dlen) for _ in range(num_actions)]})
      if "AN" in self.network.layers:
        self.stat.update({"a_inh_rate": [collections.deque(maxlen=dlen) for _ in range(num_actions)],})
     
      
  def __call__(self, state, reward, isEnd, info):    
    network = self.network
    
    stateCode = info["stateCode"]
    if np.any(stateCode==2):
      network.reset_state_variables(stateCode==2)    
    
    p = self.p
    ent = (-self.entropy_reg * torch.sum(((p * (torch.log(p) + 1))[:, np.newaxis, :] * (self.ent_eye - p[..., np.newaxis])), axis=-1)
           if self.entropy_reg > 0 else None)    
    augment = np.concatenate([isEnd[:, np.newaxis], (stateCode==1)[:,np.newaxis]], axis=-1) if self.augment_state else None    
    state_spike = self.state_to_spike(state, augment)  
    if self.gpu: state_spike = state_spike.cuda()
    
    network.run(inputs={"H0": state_spike.reshape(state_spike.shape[:2]+(-1,))}, 
                time=self.env_dt, 
                input_time_dim=1, 
                action=self.action, 
                p=self.p, 
                ent=ent, 
                env_reward=self.reward_adj*reward, 
                info=info,
                isEnd=isEnd, 
                gpu=self.gpu,
                )    
    if self.actor_active:
      actor_exc = network.layers["AE"]
      a_exc_rate = torch.mean(actor_exc.x.view(-1, self.num_actions, self.num_actors_exc_pa), dim=-1)/actor_exc.tc_trace
      if "AN" in network.layers:
        actor_inh = network.layers["AN"]
        a_inh_rate = torch.mean(actor_inh.x.view(-1, self.num_actions, self.num_actors_inh_pa), dim=-1)/actor_inh.tc_trace      
        a_rate = self.actor_m * (a_exc_rate - a_inh_rate)
      else:
        a_rate = self.actor_m * a_exc_rate
      self.p = torch.nn.functional.softmax(a_rate, dim=1)
      self.freeze_timer += self.env_dt
      new_b = self.freeze_timer >= self.freeze_action      
      if new_b.any().detach().cpu().numpy():        
        self.action[new_b] = torch.distributions.Categorical(logits=a_rate[new_b]).sample()       
        self.freeze_timer[new_b] = 0
    else:
      self.action = torch.from_numpy(self.policy(state))
    
    # Following are for recording statistic only (for first sample in batch)
    if self.stat_rec:
      self.stat["est_v"].extend(network.reward_fn.v_prev_rec.detach().cpu().numpy().tolist())
      self.stat["td_error"].extend(network.reward_fn.td_error_rec.detach().cpu().numpy().tolist())
      self.stat["reward"].extend(network.reward_fn.reward_rec.detach().cpu().numpy().tolist())
      self.stat["isEnd"].extend(network.reward_fn.isEnd_rec.detach().cpu().numpy().tolist())
      v_exc_rate_0 = torch.mean(network.layers["CE"].x[0], dim=-1)/network.layers["CE"].tc_trace
      self.stat["v_exc_rate"].append(v_exc_rate_0.detach().cpu().numpy())
      if "CN" in network.layers:
        v_inh_rate_0 = torch.mean(network.layers["CN"].x[0], dim=-1)/network.layers["CN"].tc_trace
        self.stat["v_inh_rate"].append(v_inh_rate_0.detach().cpu().numpy())              
      if self.actor_active:
        for i in range(self.num_actions): 
          self.stat["a_exc_rate"][i].append(a_exc_rate[0, i].detach().cpu().numpy())
          if "AN" in network.layers:
            self.stat["a_inh_rate"][i].append(a_inh_rate[0, i].detach().cpu().numpy())
          self.stat["p"][i].append(self.p[0, i].detach().cpu().numpy())             
          
    return (self.action.detach().cpu().numpy(), self.stat)
  
class Stat_plotter():  
  def __init__(self, len_adj):
    self.pers, self.axs = [None]*5, [None]*5
    self.len_adj = len_adj
  
  def __call__(self, stat, eps_ret, eps_len):
                          
    d = {"Return of Episode": np.array(eps_ret)}
    if len(eps_ret) >= 100: d.update({"Trailing avg return of Episode": 
                                     np.concatenate([np.full((99,), np.nan,), mv(eps_ret, 100)])})
    self.pers[0], self.axs[0] = plot_performance(d, 
                                                 title="Learning Curve", 
                                                 xlabel="Episodes", 
                                                 ylabel="Return of Episode", 
                                                 fig=self.pers[0], 
                                                 ax=self.axs[0])
    d = {"Length of Episode": np.array(eps_len)+self.len_adj}
    if len(eps_ret) >= 100: d.update({"Trailing avg length of Episode": 
                                     np.concatenate([np.full((99,), np.nan,), mv(eps_len, 100)])+self.len_adj})    
    self.pers[1], self.axs[1] = plot_performance(d, 
                                                 title="Learning Curve", 
                                                 xlabel="Episodes", 
                                                 ylabel="Lengths (Seconds)", 
                                                 fig=self.pers[1], 
                                                 ax=self.axs[1])   
      
    d = {"Exc Value Neuron": stat["v_exc_rate"]}           
    if "v_inh_rate" in stat: d.update({"Inh Value Neuron": stat["v_inh_rate"]})
    if "a_exc_rate" in stat: 
      for i in range(len(stat["a_exc_rate"])):
        d.update({"Exc Actor %d Neuron"% i: stat["a_exc_rate"][i]})
    if "a_inh_rate" in stat: 
      for i in range(len(stat["a_inh_rate"])):
        d.update({"Inh Actor %d Neuron"% i: stat["a_inh_rate"][i]})  
    self.pers[2], self.axs[2] = plot_performance(d, 
                                                 title="Firing Rate", 
                                                 xlabel="Step", 
                                                 ylabel="Firing Rate", 
                                                 fig=self.pers[2], 
                                                 ax=self.axs[2], 
                                                 ylim=[0, 1])
    if "p" in stat:
      d = {}
      for i in range(len(stat["p"])):
        d.update({"Action %d"% i: stat["p"][i]})                   
      self.pers[3], self.axs[3] = plot_performance(d, 
                                                   title="Probability of Action", 
                                                   xlabel="Step", 
                                                   ylabel="p", 
                                                   fig=self.pers[3], 
                                                   ax=self.axs[3], 
                                                   ylim=[0, 1])
    d = {"Value from Network": np.array(stat["est_v"]),          
         "TD Error": np.array(stat["td_error"]),
         "Reward": np.array(stat["reward"]),
          "isEnd": np.array(stat["isEnd"])*0.05
         }   
    
    self.pers[4], self.axs[4] = plot_performance(d, 
                                                  title="Value Stat.", 
                                                  xlabel="Step", 
                                                  ylabel="", 
                                                  fig=self.pers[4], 
                                                  ax=self.axs[4])