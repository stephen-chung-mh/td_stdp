import argparse, configparser
import os, sys, argparse, collections, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network import load
from util import *
from bindsnet_add import FBTDSTDP, build_ac_network, Snn_actor, Reward_fn, Stat_plotter

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=False, default="config_cp.ini",
   help="location of config file")
args = ap.parse_args()
f_name = os.path.join("config", "%s" % args.config) 
print("Loading config from %s" % f_name)

config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read(f_name)

gpu = config.getboolean("USER", "gpu")
min_eps = config.getint("USER", "min_eps")
max_eps = config.getint("USER", "max_eps")
n_run = config.getint("USER", "n_run")

# Input parameters
batch_size = config.getint("USER", "batch_size")
input_type = config.getint("USER", "input_type")
augment_state = config.getboolean("USER", "augment_state")
rep = config.getint("USER", "rep")
basis = config.getboolean("USER", "basis")

# Input parameters (Discrete bin or RBF only):
bin_num = config.getint("USER", "bin_num") 

# Input parameters (Fourier only):
forier_order = config.getint("USER", "forier_order") 
forier_cross_term = config.getboolean("USER", "forier_cross_term")
forier_double = config.getboolean("USER", "forier_double")

# Scale parameter
critic_lr = config.getfloat("USER", "critic_lr")
actor_lr = config.getfloat("USER", "actor_lr")
init_scale = config.getfloat("USER", "init_scale")
adam = config.getboolean("USER", "adam")
adam_beta_1 = config.getfloat("USER", "adam_beta_1")
adam_beta_2 = config.getfloat("USER", "adam_beta_2")

# Env parameter
env_name = config.get("USER", "env_name")
env_tc = config.getfloat("USER", "env_tc") 
env_dt = config.getint("USER", "env_dt")
reward_adj = config.getfloat("USER", "reward_adj")
freeze_action = config.getint("USER", "freeze_action") 
isEnd_zero = config.getboolean("USER", "isEnd_zero")
last_zero_n = config.getint("USER", "last_zero_n") 
rest_n = config.getint("USER", "rest_n") 
warm_n = config.getint("USER", "warm_n") 

# Network parameter
actor_active = config.getboolean("USER", "actor_active")
num_critics_exc = config.getint("USER", "num_critics_exc") 
num_critics_inh =  config.getint("USER", "num_critics_inh") 
num_actors_exc_pa = config.getint("USER", "num_actors_exc_pa") 
num_actors_inh_pa = config.getint("USER", "num_actors_inh_pa") 
entropy_reg = config.getfloat("USER", "entropy_reg")
weight_reg = config.getfloat("USER", "weight_reg")
value_m = config.getfloat("USER", "value_m")
value_b = config.getfloat("USER", "value_b")
actor_m = config.getfloat("USER", "actor_m")
tau_n = config.getfloat("USER", "tau_n")
tau_plus = config.getfloat("USER", "tau_plus")
tau_z = config.getfloat("USER", "tau_z")
tau_q = config.getfloat("USER", "tau_q")
tau_v = config.getfloat("USER", "tau_v")
targ_firing_r = config.getfloat("USER", "targ_firing_r")
targ_firing_rate = config.getfloat("USER", "targ_firing_rate")

name = config.get("USER", "name")
test = config.getboolean("USER", "test")
checkpoint = config.get("USER", "checkpoint")
test_eps = config.getint("USER", "test_eps") 
test_vis = config.getboolean("USER", "test_vis")

results = []
plot_stat = False
p_eps_n = 1 if test else 10
batch_size = 1 if test and test_vis else batch_size

state_lim = {"CartPole-v1":[np.array([-2.4, -3.2, -np.pi/12.0, -3.2]),
                         np.array([2.4, 3.2, np.pi/12.0, 3.2])],          
             "LunarLander-v2":[np.array([-1,-0.2,-1,-1,-1,-1, 0, 0]),
                               np.array([+1,+2,+1,+1,+1,+1, 1, 1])],}

solve_def = {"CartPole-v1": (100, 500),
             "LunarLander-v2": (100, 200)}   
                         
env = batch_envs(name=env_name, batch_size=batch_size, rest_n=rest_n, warm_n=warm_n)  
state_size = env.state.shape[1]
num_actions = env.action_space.n # Number of actions available in env
bin_min, bin_max = state_lim[env_name]
policy = uniform_policy(batch_size=batch_size, actions=np.arange(num_actions))

if input_type == 0:
  state_to_spike = State_to_spike_bin(env_dt, bin_min, bin_max, bin_num, 
                                      basis=basis, rep=rep)
elif input_type == 1:
  state_to_spike = State_to_spike_RBF(env_dt, bin_min, bin_max, bin_num, 
                                      basis=basis, rep=rep)  
elif input_type == 2:
  state_to_spike = State_to_spike_Fourier(env_dt, bin_min, bin_max,  
                                        k=forier_order, basis=basis, soft=False,
                                        cross_term=forier_cross_term, double=forier_double, rep=rep)

input_shape = state_to_spike.out_shape
if augment_state: input_shape = (rep, input_shape[1]+2)
  
print("Input shape:", input_shape)

critic_layer_param = {
   "traces":True,             
   "traces_additive":True,   
   "tc_trace":tau_n,           
   "refrac":0,
   "tc_decay":tau_v}

actor_layer_param = critic_layer_param.copy()
actor_layer_param.update({"tc_trace":tau_plus,})     
                  
critic_fd_conn_param = {
   "update_rule":FBTDSTDP,
   "nu":(0, critic_lr),
   "wmin":-np.inf,
   "wmax":np.inf,
   "norm":None,
   "tc_plus":tau_plus,
   "tc_e_trace":tau_z,
   "fb_gate": False,
   "adam": adam,
   "adam_beta_1": adam_beta_1,
   "adam_beta_2": adam_beta_2,
}
actor_fd_conn_param = critic_fd_conn_param.copy()
actor_fd_conn_param.update({"update_rule":FBTDSTDP,
                            "nu":(0, actor_lr), 
                            "tc_plus":tau_plus,
                            "tc_e_trace": tau_z,
                            "tc_a_trace": tau_q,
                            "fb_gate": True,
                            "targ_firing_r": targ_firing_r,
                            "targ_firing_rate": targ_firing_rate,
                            "adam": adam}
                             )     

stat_plotter = Stat_plotter(len_adj=-(rest_n-warm_n)*env_dt)
                         
for n in range(1 if test else n_run):  
  env.reset()
  if test:
    print("Loading checkpoint from %s.." % checkpoint)
    f_name = os.path.join("model", "%s" % checkpoint)  
    network = load(f_name)
    if gpu: network.to("cuda")    
    network.learning = False
  else:
    network = build_ac_network(input_shape=input_shape,
                            init_scale=init_scale,
                            num_critics=(num_critics_exc, num_critics_inh), 
                            num_actors_pa=(num_actors_exc_pa, num_actors_inh_pa),                               
                            num_actions=num_actions,
                            critic_layer_param=critic_layer_param, 
                            critic_fd_conn_param=critic_fd_conn_param,
                            actor_layer_param=actor_layer_param, 
                            actor_fd_conn_param=actor_fd_conn_param,    
                            actor_active=actor_active,
                            gpu=gpu)
    print("Initializing %s network.." % name)
  
  for k, v in network.layers.items(): v.set_batch_size(batch_size)
  for k, v in network.connections.items(): v.update_rule.set_batch_size(batch_size)

  network.reward_fn = Reward_fn(tc=env_tc,
                                network_steps=env_dt,
                                isEnd_zero=isEnd_zero, 
                                last_zero_n=last_zero_n,
                                batch_size=batch_size,
                                value_m=value_m,
                                value_b=value_b,
                                gpu=gpu)  
  snn_actor = Snn_actor(network=network, 
                        batch_size=batch_size, 
                        state_to_spike=state_to_spike, 
                        augment_state=augment_state, 
                        env_dt=env_dt, 
                        reward_adj=reward_adj, 
                        actor_m=actor_m, 
                        num_actions=num_actions, 
                        num_actors_pa=(num_actors_exc_pa, num_actors_inh_pa),
                        actor_active=actor_active, 
                        entropy_reg=entropy_reg, 
                        freeze_action=freeze_action, 
                        stat_rec=plot_stat, 
                        gpu=gpu)

  eps_ret, eps_len = [], []  
  c_eps_ret = np.zeros(batch_size)
  c_eps_len = np.zeros(batch_size)
  step, p_eps = 0, 0
  f_perfect, solved = False, False

  # Initialize Variable  

  state = env.reset()
  reward = env.reward
  isEnd = env.isEnd
  info = env.info
  
  while(True):
    step += 1
    action, stat = snn_actor(state, reward, isEnd, info)
    state, reward, _isEnd, info = env.step(action)    
    if test and test_vis: env._env[0].render(mode="rgb_array")
    stateCode = info['stateCode']

    c_eps_ret += reward
    c_eps_len += env_dt
    new_end = np.logical_and(isEnd == False, _isEnd==True)
    
    if np.any(new_end):
      eps_ret.extend(c_eps_ret[new_end].tolist())
      eps_len.extend(c_eps_len[new_end].tolist())      
      c_eps_ret[new_end] = 0.
      c_eps_len[new_end] = 0.    
        
       # Print eps return 
      while (len(eps_ret) >= p_eps + p_eps_n):      
        p_eps += p_eps_n
        print("%d: Return of Episode: %.2f; Last 100 Avg. Return of Episode: %.2f; Last 100 Avg. Length of Episode: %.2f Solved: %s" % (
                p_eps, eps_ret[p_eps-1], np.average(eps_ret[p_eps-100:p_eps]), np.average(eps_len[p_eps-100:p_eps]),
                "Y" if solved else "N"))            
        
      # Check if solved     
      if not test and env_name in solve_def:          
        avg_n = solve_def[env_name][0]
        p_score = solve_def[env_name][1]
        if not f_perfect and np.amax(eps_ret) >= p_score:
          r = np.argmax(np.array(eps_ret) >= p_score, axis=-1)+1
          print("%d: First perfect. Eps required: %d" % (n, r))
          f_perfect = True           
        if not solved and len(eps_ret) > avg_n and np.amax(mv(eps_ret, avg_n)) >= p_score:
          r = np.argmax(np.array(mv(eps_ret, avg_n)) >= p_score, axis=-1) + avg_n - 1          
          solved = True             
          f_name = os.path.join("model", "model_%s_%d.pt" %(name, n))  
          print("%d: Solved. Eps required: %d. Model saved to %s" % (n, r, f_name))
          network.save(f_name)            
        
      if test:
        if len(eps_ret) >= test_eps: break
      else:
        if (solved and len(eps_ret) >= min_eps) or len(eps_ret) >= max_eps: break

    isEnd = np.copy(_isEnd)
    
    if step % 50 == 0 and plot_stat: stat_plotter(stat, eps_ret, eps_len) 
  
  results.append([eps_ret, eps_len])
  print("Average return : %f (%d episodes)" % (np.average(eps_ret), len(eps_ret)))
  
  if not test:
    f_name = os.path.join("result", "rewards_%s_%d.pkl" %(name, n))
    with open(f_name, 'wb') as f:
      pickle.dump(results[-1], f)

# Save all reward stat
if not test:      
  f_name = os.path.join("result", "rewards_%s_all.pkl" %(name))
  with open(f_name, 'wb') as f:
    pickle.dump(results, f)    
    
# Print training result 

if not test and env_name in solve_def: 
  print("Stat on t_f (first eps. having perfect score) and t_s (eps. required to solve):")
  avg_n = solve_def[env_name][0]
  p_score = solve_def[env_name][1]
  t_f, t_s = [], []
  for i in results:    
    rets = np.array(i[0])
    if (rets >= p_score).any():
      t_f.append(np.argmax(rets >= p_score, axis=-1)+1)
    else:
      t_f.append(-1)
    if len(rets) >= avg_n and (mv(rets, avg_n) >= p_score).any():
      t_s.append(np.argmax(mv(rets, avg_n) >= p_score, axis=-1)+avg_n)
    else:
      t_s.append(-1)  
  def print_t(name, t):  
    t = np.array(t)
    print("%s: %d / %d achieved." % (name, np.sum(t>-1), len(t)))    
    t_filter = t[t>-1]
    if len(t_filter) > 0:
      print("%s: avg. %.2f median %.2f min %.2f max %.2f std %.2f"% (name,  np.average(t_filter),
                                                                           np.median(t_filter),
                                                                           np.amin(t_filter),
                                                                           np.amax(t_filter),
                                                                           np.std(t_filter))) 
    print("%s: "%name, t)
    
  print_t("t_f", t_f)
  print_t("t_s", t_s)

r = np.array([i[0][:min_eps] for i in results])
r_avg = np.average(r, axis=0)
r_std = np.std(r, axis=0)    
vis_len, sel = r.shape[1], 1 if r.shape[1] < 500 else 5

ind = (np.arange(vis_len)%sel)==0
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(np.arange(vis_len//sel)*sel, r_avg[ind], '-', color='blue', label=name)
plt.fill_between(np.arange(vis_len//sel)*sel, (r_avg - r_std)[ind], (r_avg + r_std)[ind], color='blue', alpha=0.2)
plt.xlabel('Episodes')
plt.ylabel('Return of Episode')
plt.legend(loc='lower right')
plt.grid(axis='y')
fig.tight_layout()
f_name = os.path.join("result", "fig_%s_%s.png" % (name, "test" if test else "train"))
fig.savefig(f_name) 
print("Plot saved to %s" % f_name)
plt.show()

    