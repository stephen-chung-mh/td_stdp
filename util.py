import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from torch.nn.modules.utils import _pair
from matplotlib.collections import PathCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.figure import Figure
from typing import Tuple, List, Optional, Sized, Dict, Union
import gym

def sigmoid(x):
  return 1/(1+np.exp(-x))
  
def softmax(x):
  e_x = np.exp(x - np.amax(x, axis=-1)[..., np.newaxis])
  return e_x / e_x.sum(axis=-1)[..., np.newaxis]

def relu(x):
  y = np.copy(x)
  y[y<0] = 0
  return y

def getl(x, n):
  return x[n] if type(x) == list else x
  
def from_one_hot(y):      
  return np.argmax(y, axis=-1)

def to_one_hot(a, size):
  oh = np.zeros((a.shape[0], size), np.int)
  oh[np.arange(a.shape[0]), a.astype(int)] = 1
  return oh

def shift_diag(m, n):
  w = np.zeros((m, n))  
  w[np.repeat(np.arange(m), m), np.arange(n)] = 1
  return torch.from_numpy(w).float()  

def block_diag(m, n, opp=False):
  a = np.ones((n,n))
  w = np.kron(np.eye(m//n), a)
  if opp:
    return torch.from_numpy(1-w).float()  
  np.fill_diagonal(w, 0)  
  return torch.from_numpy(w).float()  

def normalize(w, norm):
  w_abs_sum = w.abs().sum(0).unsqueeze(0)
  w_abs_sum[w_abs_sum == 0] = 1.0
  return w * norm / w_abs_sum

def mv(a, n=1000):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def real_to_bin(real, bin_min=0, bin_max=1, bin_num=10):  
  bin_index = np.array((real - bin_min)/(bin_max-bin_min) * bin_num).astype(np.int)
  bin_index -= 1
  bin_index[bin_index>=bin_num] = bin_num-1
  bin_index[bin_index<0] = 0
  return bin_index

def multinomial_rvs(p, n=1):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


class uniform_policy():
  def __init__(self, batch_size, actions):
    self.actions = np.array(actions, np.int)
    self.batch_size = batch_size
  
  def __call__(self, state):
    return np.random.choice(self.actions, size=self.batch_size)  
  
# State to spike converter class    

class State_to_spike_bin():
  # One hot encoding of state according to discrete bins
  def __init__(self, time, bin_min, bin_max, bin_num, basis=False, rep=1):    
    self.time = time
    self.bin_min = bin_min
    self.bin_max = bin_max
    self.bin_num = bin_num
    self.basis = basis    
    self.rep = rep
    self.out_shape = (rep, len(bin_min)*bin_num+(1 if self.basis else 0))
    
  def __call__(self, state, augment=None):  
    b = real_to_bin(real=state, bin_min=self.bin_min, bin_max=self.bin_max, bin_num=self.bin_num)
    bins = np.zeros(state.shape+(self.bin_num,), np.int)  
    bins[np.arange(state.shape[0])[..., np.newaxis],np.arange(state.shape[1]),  b] = 1        
    bins = np.reshape(bins, (state.shape[0], -1))
    if self.basis: bins = np.concatenate([bins, np.ones((state.shape[0], 1))], axis=-1)
    if augment is not None: bins = np.concatenate([bins, augment], axis=-1)
    bins = np.broadcast_to(bins, (self.time, self.rep) + bins.shape)
    bins = np.swapaxes(bins, 1, 2).reshape(self.time, state.shape[0], self.rep, -1)            
    return torch.from_numpy(bins).float()
  
class State_to_spike_RBF():
  def __init__(self, time, bin_min, bin_max, bin_num, basis=False, rep=1):
    self.time = time
    self.bin_min = bin_min
    self.bin_max = bin_max
    self.bin_size = (bin_max - bin_min)/bin_num
    self.rbf_mean = (bin_min[..., np.newaxis] + 
                self.bin_size[..., np.newaxis] * (np.arange(bin_num)+0.5))
    self.rbf_var = self.bin_size[..., np.newaxis]/2
    self.basis = basis    
    self.rep = rep
    self.out_shape = (rep, len(bin_min)*bin_num+(1 if self.basis else 0))
    
  def __call__(self, state, augment=None):  
    state = np.clip(state, self.bin_min+self.bin_size/2, self.bin_max-self.bin_size/2)
    rbf_output = np.exp(-np.abs(state[..., np.newaxis]-self.rbf_mean)/self.rbf_var).reshape(state.shape[0], -1)
    if self.basis: rbf_output = np.concatenate([rbf_output, np.ones((state.shape[0], 1))], axis=-1)
    if augment is not None: rbf_output = np.concatenate([rbf_output, augment], axis=-1)    
    rbf_spike = np.random.binomial(n=1,p=rbf_output, size=(self.time, self.rep,)+rbf_output.shape)      
    rbf_spike = np.swapaxes(rbf_spike, 1, 2).reshape(self.time, state.shape[0], self.rep, -1) 
    return torch.from_numpy(rbf_spike).float()

class State_to_spike_Fourier():  
  def __init__(self, time, bin_min, bin_max, k=2, basis=False, soft=False, cross_term=True, double=False, rep=1):
    n = len(bin_min)
    # create vector of coefficient
    if type(k) != np.ndarray:
      if type(k) == list:
        k = np.array(k)
      else:
        k = np.array([k]*n)
    if cross_term:
      self._c = np.zeros((n, np.prod(k+1)))
      for i in range(np.prod(k+1)):
          l = i
          for j in range(n):    
              self._c[j, i] = l % (k[j]+1)
              l //= (k[j]+1)
              if l == 0: break
    else:
      self._c = np.zeros((n, np.sum(k)))
      j, l = 0, 0
      for i in range(np.sum(k)):
        l += 1
        self._c[j, i] = l  
        if l >= k[j]:
          l = 0
          j += 1
    self._k = k
    self._n = n    
    self.mean_re = (bin_max + bin_min) / 2
    self.range_re = bin_max - bin_min
    self.time = time    
    self.out_shape = (rep, (self._c.shape[-1])*(2 if double else 1)+(1 if basis else 0))
    self.basis = basis
    self.soft = soft
    self.double = double
    self.rep = rep
      
  def __call__(self, state, augment=None):    
    norm_state = (state - self.mean_re) / self.range_re + 0.5  
    norm_state = np.clip(norm_state, 0, 1)  
    if self.double:
      f_basis_p = relu(np.cos(np.pi * np.dot(norm_state, self._c)))
      f_basis_n = relu(-np.cos(np.pi * np.dot(norm_state, self._c)))
      f_basis = np.concatenate([f_basis_p, f_basis_n], -1)      
    else:
      f_basis = (np.cos(np.pi * np.dot(norm_state, self._c))+1)/2 
    if self.basis: f_basis = np.concatenate([f_basis, np.ones((state.shape[0], 1))], axis=-1)
    if augment is not None: f_basis = np.concatenate([f_basis, augment], axis=-1)        
      
    if self.soft: 
      f_output = np.broadcast_to(f_basis[np.newaxis, np.newaxis, :, :], (self.time, self.rep) + f_basis.shape)      
    else:
      f_output = np.random.binomial(n=1,p=f_basis, size=(self.time, self.rep,)+f_basis.shape)      
    f_output = np.swapaxes(f_output, 1, 2).reshape(self.time, state.shape[0], self.rep, -1) 
    return torch.from_numpy(f_output).float()     

class batch_envs():
  
  def __init__(self, name, batch_size=1, rest_n=100, warm_n=100):
    self._batch_size = int(batch_size)
    self._action = None
    self._reward = np.zeros(batch_size)
    self._isEnd = np.ones(batch_size, bool)
    self._truncatedEnd = np.zeros(batch_size, bool)
    self._rest = np.zeros(batch_size)
    self._warm = np.zeros(batch_size)
    self._state = np.zeros((batch_size,)+gym.make(name).reset().shape)
    self._stateCode = np.zeros(batch_size, np.int)
    self._rest_n = rest_n
    self._warm_n = warm_n     
    self._env = [gym.make(name) for _ in range(batch_size)]    

  @property
  def name(self):
    return self._name

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def reward(self):
    return self._reward

  @property
  def action(self):
    return self._action

  @property
  def action_space(self):
    return self._env[0].action_space

  @property
  def isEnd(self):
    return self._isEnd

  @property
  def stateCode(self):
    return self._stateCode

  @property
  def state(self):
    return self._state    

  @property
  def info(self):
    return {"stateCode": self._stateCode, "truncatedEnd": self._truncatedEnd}
  
  def step(self, action):
    self._rest[self._isEnd] += 1
    self._warm += 1
    self.reset(self._rest>self._rest_n)      
    isLive = np.logical_and(self._warm>self._warm_n, ~self._isEnd)
    self._reward[~isLive] = 0
    for i in isLive.nonzero()[0]:   
      self._state[i], self._reward[i], self._isEnd[i], info = self._env[i].step(action[i])  
      self._truncatedEnd[i] = info['TimeLimit.truncated'] if 'TimeLimit.truncated' in info else False
      if self._truncatedEnd[i]: self._rest[i] = self._rest_n

    # Live  
    self._stateCode[isLive] = 0            
    # Rest
    self._stateCode[self._rest>=1] = 1
    # Warm up
    self._stateCode[self._warm<= self._warm_n] = 3    
    # Reset
    self._stateCode[self._warm==0] = 2

    return self.state, self.reward, self._isEnd, self.info

  def reset(self, index=None):    
    for i in range(self._batch_size) if index is None else index.nonzero()[0]:
      self._state[i] = self._env[i].reset()
      self._reward[i] = 0
    if index is None: index = slice(None)  
    self._rest[index] = 0
    self._warm[index] = 0
    self._truncatedEnd[index] = False
    self._isEnd[index] = False
    return self._state

# Plotting function are shown below; mostly copied from bindsnet with little amendment

def get_square_weights(
    weights: Union[Tensor, np.ndarray], n_sqrt: Union[int, Tuple[int, int]], side: Union[int, Tuple[int, int]]
) -> Tensor:
    # language=rst
    """
    Return a grid of a number of filters ``sqrt ** 2`` with side lengths ``side``.

    :param weights: Two-dimensional tensor of weights for two-dimensional data.
    :param n_sqrt: Square root of no. of filters.
    :param side: Side length(s) of filter.
    :return: Reshaped weights to square matrix of filters.
    """
    if isinstance(side, int):
        side = (side, side)

    if isinstance(n_sqrt, int):
        n_sqrt = (n_sqrt, n_sqrt) 

    if isinstance(weights, np.ndarray):
        square_weights = np.zeros((side[0] * n_sqrt[0], side[1] * n_sqrt[1]))
        n_size = weights.shape[1]
    else:
        square_weights = torch.zeros(side[0] * n_sqrt[0], side[1] * n_sqrt[1])
        n_size = weights.size(1)

    
    for i in range(n_sqrt[0]):
        for j in range(n_sqrt[1]):
            n = i * n_sqrt[1] + j

            if not n < n_size:
                break

            x = i * side[0]
            y = j * side[1]
            if isinstance(weights, np.ndarray):
                filter_ = weights[:, n].reshape(side)
            else:
                filter_ = weights[:, n].contiguous().view(*side)
            square_weights[x : x + side[0], y : y + side[1]] = filter_

    return square_weights

def plot_cartpole(state, ax=None, fig=None):
  x, theta = state[0], state[2]
  if fig is None:
    fig, ax = plt.subplots(figsize=(48,6))      
  ax.clear()
  ax.set_xlim(-3, 3)
  ax.set_ylim(-0.5, 3)     
  cart_w = 0.6
  rect = patches.Rectangle((x-cart_w/2, 0), cart_w, 0.5, fc='b')  
  elip1 = patches.Ellipse((x-0.25, 0), 0.04, 0.2, fc='k')  
  elip2 = patches.Ellipse((x+0.25, 0), 0.04, 0.2, fc='k')  
  x = [x, x + 2 * np.sin(theta)]
  y = [0.5, 0.5 + 2 * np.cos(theta)]
  line = mlines.Line2D(x, y, lw=5., c='b')
  ax.add_patch(rect)
  ax.add_line(line)
  ax.add_patch(elip1)
  ax.add_patch(elip2)
  fig.canvas.draw()
  fig.show()
  return fig, ax    


def plot_spikes(
    spikes: Dict[str, torch.Tensor],
    time: Optional[Tuple[int, int]] = None,
    n_neurons: Optional[Dict[str, Tuple[int, int]]] = None,
    ims: Optional[List[PathCollection]] = None,
    fig: Optional[Figure] = None,
    axes: Optional[Union[Axes, List[Axes]]] = None,    
    figsize: Tuple[float, float] = (8.0, 4.5),    
) -> Tuple[List[AxesImage], List[Axes]]:
    # language=rst
    """
    Plot spikes for any group(s) of neurons.

    :param spikes: Mapping from layer names to spiking data. Spike data has shape
        ``[time, n_1, ..., n_k]``, where ``[n_1, ..., n_k]`` is the shape of the
        recorded layer.
    :param time: Plot spiking activity of neurons in the given time range. Default is
        entire simulation time.
    :param n_neurons: Plot spiking activity of neurons in the given range of neurons.
        Default is all neurons.
    :param ims: Used for re-drawing the plots.
    :param axes: Used for re-drawing the plots.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: ``ims, axes``: Used for re-drawing the plots.
    """
    n_subplots = len(spikes.keys())
    if n_neurons is None:
        n_neurons = {}

    spikes = {k: v.view(v.size(0), -1) for (k, v) in spikes.items()}
    if time is None:
        # Set it for entire duration
        for key in spikes.keys():
            time = (0, spikes[key].shape[0])
            break

    # Use all neurons if no argument provided.
    for key, val in spikes.items():
        if key not in n_neurons.keys():
            n_neurons[key] = (0, val.shape[1])

    if ims is None:
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        fig.show()
        fig.canvas.draw()
        if n_subplots == 1:
            axes = [axes]

        ims = []
        for i, datum in enumerate(spikes.items()):
            spikes = (
                datum[1][
                    time[0] : time[1], n_neurons[datum[0]][0] : n_neurons[datum[0]][1]
                ]
                .detach()
                .clone()
                .cpu()
                .numpy()
            )
            ims.append(
                axes[i].scatter(
                    x=np.array(spikes.nonzero()).T[:, 0],
                    y=np.array(spikes.nonzero()).T[:, 1],
                    s=3,
                )
            )            
            args = (
                datum[0],
                n_neurons[datum[0]][0],
                n_neurons[datum[0]][1],
                time[0],
                time[1],
            )
            axes[i].set_title(
                "%s spikes for neurons (%d - %d) from t = %d to %d " % args
            )
            axes[i].set_xlim(time[0], time[1])
            axes[i].set_ylim(n_neurons[datum[0]][0], n_neurons[datum[0]][1])
        for ax in axes:
            ax.set_aspect("auto")

        plt.setp(
            axes, xticks=[], yticks=[], xlabel="Simulation time", ylabel="Neuron index"
        )
        plt.tight_layout()
    else:
        for i, datum in enumerate(spikes.items()):
            spikes = (
                datum[1][
                    time[0] : time[1], n_neurons[datum[0]][0] : n_neurons[datum[0]][1]
                ]
                .detach()
                .clone()
                .cpu()
                .numpy()
            )
            ims[i].set_offsets(np.array(spikes.nonzero()).T)
            args = (
                datum[0],
                n_neurons[datum[0]][0],
                n_neurons[datum[0]][1],
                time[0],
                time[1],
            )
            axes[i].set_title(
                "%s spikes for neurons (%d - %d) from t = %d to %d " % args
            )

    #plt.draw()
    fig.canvas.draw()
    return ims, fig, axes

def plot_voltages(
    voltages: Dict[str, torch.Tensor],
    ims: Optional[List[AxesImage]] = None,
    fig: Optional[Figure] = None,
    axes: Optional[List[Axes]] = None,    
    time: Tuple[int, int] = None,
    n_neurons: Optional[Dict[str, Tuple[int, int]]] = None,
    cmap: Optional[str] = "jet",
    plot_type: str = "color",
    thresholds: Dict[str, torch.Tensor] = None,
    figsize: Tuple[float, float] = (8.0, 4.5),    
) -> Tuple[List[AxesImage], List[Axes]]:
    # language=rst
    """
    Plot voltages for any group(s) of neurons.

    :param voltages: Contains voltage data by neuron layers.
    :param ims: Used for re-drawing the plots.
    :param axes: Used for re-drawing the plots.
    :param time: Plot voltages of neurons in given time range. Default is entire
        simulation time.
    :param n_neurons: Plot voltages of neurons in given range of neurons. Default is all
        neurons.
    :param cmap: Matplotlib colormap to use.
    :param figsize: Horizontal, vertical figure size in inches.
    :param plot_type: The way how to draw graph. 'color' for pcolormesh, 'line' for
        curved lines.
    :param thresholds: Thresholds of the neurons in each layer.
    :return: ``ims, axes``: Used for re-drawing the plots.
    """
    n_subplots = len(voltages.keys())

    for key in voltages.keys():
        voltages[key] = voltages[key].view(-1, voltages[key].size(-1))

    if time is None:
        for key in voltages.keys():
            time = (0, voltages[key].size(0))
            break

    if n_neurons is None:
        n_neurons = {}

    for key, val in voltages.items():
        if key not in n_neurons.keys():
            n_neurons[key] = (0, val.size(1))

    if not ims:
        fig, axes = plt.subplots(n_subplots, 1, figsize=figsize)
        ims = []
        if n_subplots == 1:  # Plotting only one image
            for v in voltages.items():
                if plot_type == "line":
                    ims.append(
                        axes.plot(
                            v[1]
                            .detach()
                            .clone()
                            .cpu()
                            .numpy()[                                
                                time[0] : time[1],
                                n_neurons[v[0]][0] : n_neurons[v[0]][1]
                            ]
                        )
                    )

                    if thresholds is not None and thresholds[v[0]].size() == torch.Size(
                        []
                    ):
                        ims.append(
                            axes.axhline(
                                y=thresholds[v[0]].item(), c="r", linestyle="--"
                            )
                        )
                else:
                    ims.append(
                        axes.pcolormesh(
                            v[1]
                            .cpu()
                            .numpy()[                                
                                time[0] : time[1],
                                n_neurons[v[0]][0] : n_neurons[v[0]][1]
                            ]
                            .T,
                            cmap=cmap,
                        )
                    )

                args = (v[0], n_neurons[v[0]][0], n_neurons[v[0]][1], time[0], time[1])
                plt.title("%s voltages for neurons (%d - %d) from t = %d to %d " % args)
                plt.xlabel("Time (ms)")

                if plot_type == "line":
                    plt.ylabel("Voltage")
                else:
                    plt.ylabel("Neuron index")

                axes.set_aspect("auto")

        else:  # Plot each layer at a time
            for i, v in enumerate(voltages.items()):
                if plot_type == "line":
                    ims.append(
                        axes[i].plot(
                            v[1]
                            .cpu()
                            .numpy()[                               
                                time[0] : time[1],
                                n_neurons[v[0]][0] : n_neurons[v[0]][1],
                            ]
                        )
                    )
                    if thresholds is not None and thresholds[v[0]].size() == torch.Size(
                        []
                    ):
                        ims.append(
                            axes[i].axhline(
                                y=thresholds[v[0]].item(), c="r", linestyle="--"
                            )
                        )
                else:
                    ims.append(
                        axes[i].matshow(
                            v[1]
                            .cpu()
                            .numpy()[                                
                                time[0] : time[1],
                                n_neurons[v[0]][0] : n_neurons[v[0]][1],
                            ]
                            .T,
                            cmap=cmap,
                        )
                    )
                args = (v[0], n_neurons[v[0]][0], n_neurons[v[0]][1], time[0], time[1])
                axes[i].set_title(
                    "%s voltages for neurons (%d - %d) from t = %d to %d " % args
                )

            for ax in axes:
                ax.set_aspect("auto")

        if plot_type == "color":
            plt.setp(axes, xlabel="Simulation time", ylabel="Neuron index")
        elif plot_type == "line":
            plt.setp(axes, xlabel="Simulation time", ylabel="Voltage")

        plt.tight_layout()

    else:
        # Plotting figure given
        if n_subplots == 1:  # Plotting only one image
            for v in voltages.items():
                axes.clear()
                if plot_type == "line":
                    axes.plot(
                        v[1]
                        .cpu()
                        .numpy()[
                             time[0] : time[1], n_neurons[v[0]][0] : n_neurons[v[0]][1]
                        ]
                    )
                    if thresholds is not None and thresholds[v[0]].size() == torch.Size(
                        []
                    ):
                        axes.axhline(y=thresholds[v[0]].item(), c="r", linestyle="--")
                else:
                    axes.matshow(
                        v[1]
                        .cpu()
                        .numpy()[
                            time[0] : time[1], n_neurons[v[0]][0] : n_neurons[v[0]][1]
                        ]
                        .T,
                        cmap=cmap,
                    )
                args = (v[0], n_neurons[v[0]][0], n_neurons[v[0]][1], time[0], time[1])
                axes.set_title(
                    "%s voltages for neurons (%d - %d) from t = %d to %d " % args
                )
                axes.set_aspect("auto")

        else:
            # Plot each layer at a time
            for i, v in enumerate(voltages.items()):
                axes[i].clear()
                if plot_type == "line":
                    axes[i].plot(
                        v[1]
                        .cpu()
                        .numpy()[
                            time[0] : time[1], n_neurons[v[0]][0] : n_neurons[v[0]][1]
                        ]
                    )
                    if thresholds is not None and thresholds[v[0]].size() == torch.Size(
                        []
                    ):
                        axes[i].axhline(
                            y=thresholds[v[0]].item(), c="r", linestyle="--"
                        )
                else:
                    axes[i].matshow(
                        v[1]
                        .cpu()
                        .numpy()[
                            time[0] : time[1], n_neurons[v[0]][0] : n_neurons[v[0]][1]
                        ]
                        .T,
                        cmap=cmap,
                    )
                args = (v[0], n_neurons[v[0]][0], n_neurons[v[0]][1], time[0], time[1])
                axes[i].set_title(
                    "%s voltages for neurons (%d - %d) from t = %d to %d " % args
                )

            for ax in axes:
                ax.set_aspect("auto")

        if plot_type == "color":
            plt.setp(axes, xlabel="Simulation time", ylabel="Neuron index")
        elif plot_type == "line":
            plt.setp(axes, xlabel="Simulation time", ylabel="Voltage")

        plt.tight_layout()
    fig.canvas.draw()
    return ims, fig, axes

def plot_weights(
    weights: torch.Tensor,
    wmin: Optional[float] = 0,
    wmax: Optional[float] = 1,
    im: Optional[AxesImage] = None,
    fig: Optional[Figure] = None,
    figsize: Tuple[int, int] = (5, 5),    
    cmap: str = "hot_r",
    title: str = "weight"
) -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix.

    :param weights: Weight matrix of ``Connection`` object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: ``AxesImage`` for re-drawing the weights plot.
    """
    local_weights = weights.detach().clone().cpu().numpy()
    if not im:
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(local_weights, cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        #ax.set_aspect("auto")
        plt.colorbar(im, cax=cax)
        plt.axvline(x=0,color='red')
        fig.tight_layout()
    else:
        im.set_data(local_weights)
    fig.canvas.draw()
    return im, fig

def plot_performance(
    performances: Dict[str, List[float]],    
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Estimated classification accuracy",
    xlabel: str = "No. of examples",
    ylabel: str = "Accuracy",
    ylim: Tuple[int, int] = None,
) -> Axes:
    # language=rst
    """
    Plot training accuracy curves.

    :param performances: Lists of training accuracy estimates per voting scheme.
    :param ax: Used for re-drawing the performance plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :return: Used for re-drawing the performance plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)         
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                       box.width, box.height * 0.9])      
    
    ax.clear()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
      ax.set_ylim(ylim)

    for scheme in performances:
        ax.plot(
            range(len(performances[scheme])),
            [p for p in performances[scheme]],
            label=scheme,
        )

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                fancybox=True, shadow=False, ncol=4)
    #ax.legend() 
    fig.canvas.draw()
    fig.show()
    return fig, ax
