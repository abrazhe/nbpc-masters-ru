---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Creating a large network of AdEx neurons connected with synapses

In this notebook we first investigate parameter space of the AdEx neuron models, then select candidates for future pyramidal and interneuron parameter sets.

Next we create a network consisting of PCs and interneurons, which we connect by synapses with plasticity mechanisms. 

```python
# check if notebook is running in Colab and install packages if it is
RunningInCOLAB = 'google.colab' in str(get_ipython())

if RunningInCOLAB:
  ! pip install brian2
  ! pip install pandas
  ! wget https://raw.githubusercontent.com/abrazhe/nbpc-masters-ru/master/notebooks/input_factory.py
  ! wget https://raw.githubusercontent.com/abrazhe/nbpc-masters-ru/master/notebooks/adex_params.csv
```

```python
%pylab inline
```

```python
#style.use(('ggplot','seaborn-muted'))
# Крупные шрифты в рисунках
rc('xtick', labelsize=16) 
rc('ytick', labelsize=16)
rc('axes',labelsize=16, grid=True)
rc('font',size=16)
```

```python
import pandas as pd
```

```python
from brian2 import *
```

```python
prefs.codegen.target = 'numpy'  # use the Python fallback
```

```python
import input_factory as inpf
```

```python
def beautify_spikes(statemon,spikemon,neuron_id):
    "helper function to make voltage trajectories of IF neurons to appear with spikes"
    vm = statemon[neuron_id].v[:]
    offset = statemon.t[0]#/defaultclock.dt
    spike_times = spikemon.t[spikemon.i == neuron_id]
    for t in spike_times:
        i = int((t-offset) / defaultclock.dt)
        vm[i] = 20*mV
    return vm
```

```python
def convert_table_cell(col_name):
    units = col_name.split(' ')[1][1:-1]

def convert_from_table(row):
    return dict(
        a = float(row['a [nS]'])*nS,
        b = float(row['b [pA]'])*pA,
        tau_w = float(row['tau_u [ms]'])*ms,
        tau_m = float(row['tau_m [ms]'])*ms,
        Vreset = float(row['Vreset [mV]'])*mV,
        C = float(row['tau_m [ms]'])*ms*gl,
        Ibias = 0*pA
    )
```

```python
def sample_from_class(parameters,sigma=0.01):
    if iterable(parameters['a']):
        N = len(parameters['a'])
        k = randint(N)
        params = {key:parameters[key][k] for key in ['a','b','tau_m','tau_w', 'Vreset','Ibias']}
    else:
        params = parameters
    multipliers = 1 + sigma*randn(10)
    return dict(a = params['a']*multipliers[0],
                b = params['b']*multipliers[1],
                tau_m = params['tau_m']*multipliers[2],
                tau_w = params['tau_w']*multipliers[3],
                Vreset = params['Vreset']*multipliers[4],
                Ibias = params['Ibias']*multipliers[5]
               )

```

```python
def update_neuron_parameters(neurons, k, pars):
    neurons.a[k] = pars['a']
    neurons.b[k] = pars['b']
    neurons.tau_m[k] = pars['tau_m']
    neurons.tau_w[k] = pars['tau_w']
    neurons.Vreset[k] = pars['Vreset']
    neurons.Ibias[k] = pars['Ibias']
```

```python
def raster_spikes(spikemon,Nexc=None,ax=None,colors='rb'):
    if ax is None:
        f,ax = subplots(1,1)
    if Nexc is None:
        Nexc = max(spikemon.i)
    exc_spikes = spikemon.i<Nexc
    if any(exc_spikes):
        ax.plot(spikemon.t[exc_spikes]/ms, spikemon.i[exc_spikes],',',c=colors[0])
    if any(~exc_spikes):
        ax.plot(spikemon.t[~exc_spikes]/ms, spikemon.i[~exc_spikes],',',c=colors[1])
```

## AdEx neuron

```python
## Parameters that are shared by all neurons

# Neurons
Vth = -50*mV   # rheobase threshold
El = -70*mV     # resting membrane potential
Vcut = -20*mV    # spike detection threshold
deltaT = 2*mV  # spike initiation sharpness
Rin = 500*Mohm  # input resistance of a neuron at rest
gl = 1/Rin



# Synapses
E_e = 0*mV     # Excitatory synaptic reversal potential (AMPA and NMDA receptors)
E_i = -80*mV   # Inhibitory synaptic reversal potential (GABAA receptors)
tau_e = 5*ms   # time scale of excitatory synaptic conductance
tau_i = 10*ms  # time scale of excitatory synaptic conductance



AdEx_equations = Equations('''
dv/dt = (-(v-El) + deltaT*exp((v-Vth)/deltaT) + Rin*(Isyn + Ibias + Iapp - w))/tau_m : volt 
dw/dt = (a*(v-El) - w)/tau_w: amp  # adaptation variable
a : siemens
b : amp
tau_m : second
tau_w : second
Vreset: volt
stim_amp : 1
Ibias : amp
Iapp = stim_amp*input_current(t,i): amp
''')

# Synaptic input
synaptic_equations = Equations("""
Isyn =  - g_e*(v-E_e) - g_i*(v-E_i) : amp
dg_e/dt = -g_e/tau_e : siemens
dg_i/dt = -g_i/tau_i : siemens
""")
```

```python
adex_params = pd.read_csv('adex_params.csv',index_col='type')
adex_params
```

```python
tonic_pars = convert_from_table(adex_params.loc['tonic'])
adapting_pars = convert_from_table(adex_params.loc['adapting'])
bursting_pars = convert_from_table(adex_params.loc['bursting'])
initial_burst_pars = convert_from_table(adex_params.loc['init. burst'])
irregular_pars = convert_from_table(adex_params.loc['irregular'])
transient_pars = convert_from_table(adex_params.loc['transient'])
delayed_pars = convert_from_table(adex_params.loc['delayed'])
```

```python
tonic_pars
```

```python
transient_pars
```

```python
sample_from_class(transient_pars)
```

Here we create a source of "natural" drive to our future network in form of Poisson spike trains from 100 neurons

```python
#set_device('cpp_standalone')
```

```python
start_scope()
P = PoissonGroup(25,  'rand()*25*Hz')
Sp = SpikeMonitor(P)
```

```python
run(5*second)
```

```python
raster_spikes(Sp,100,colors='gg')
```

## Now create a new network with 80% pyramidal cells and 20% interneurons

Below we will create a model network with many principal (excitatory) neurons and a smaller amount of interneurons (inhibitory). A random subset of principal neurons will receive "sensory" input in form of Poisson spike trains.

Neurons within the network will be connected with synapses having both short-time plasticity (Tsodyks & Markram model) and spike-timing dependent plasticity (STDP).

```python
seed(4022)
```

```python
start_scope()


Ninh = 500
Nexc = 4*Ninh


Nneurons = Nexc + Ninh

defaultclock.dt = 0.1*ms

G = NeuronGroup(Nneurons, AdEx_equations+synaptic_equations,threshold='v>Vcut', 
                reset='v=Vreset; w += b',
                method='exponential_euler')

G.set_states(dict(v=El,w=0))

M = StateMonitor(G, ['v','g_e'], record=True,)
S = SpikeMonitor(G,)
```

```python
for k in range(Nneurons):
    #exemplar_parameters = pyramidal_pars if k < Nexc else interneuron_pars
    #exemplar_parameters = pyramidal_pars# if k < Nexc else interneuron_pars
    exemplar_parameters = adapting_pars if k < Nexc else tonic_pars
    sigma = 0.05 if k < Nexc else 0.01
    p = sample_from_class(exemplar_parameters,sigma)
    #update_neuron_parameters(G,k,adapting_pars)
    update_neuron_parameters(G,k,p)
```

```python
P = PoissonGroup(25,  'rand()*20*Hz')
Sp = SpikeMonitor(P)
```

```python

```

```python

```

### Setting up synaptic connections

```python
tau_ps = 0.8*second   # facilitation timescale (seconds, really?)
tau_ns = 1.5*second   # replenishing timescale
tau_stdp = 20*ms     # STDP time constant

p_s0 = 0.6            # ground-state probability of release
epsilon = 0.05        # sparsity synaptic connectivity

Apre0 = 0.01
Apost0 = -Apre0*1.05

w_e = 20*0.05*nS
w_i = 20*1*nS
pool_size=10

stdp_value = 0.05 # change between 0 and 1

tm_plasticity_model = Equations('''
dp_s/dt = (p_s0-p_s)/tau_ps : 1 (event-driven)    # release probability
dn_s/dt = (1-n_s)/tau_ns   : 1    (event-driven)    # fraction of resources available
''')

stdp_model=Equations('''
dApre/dt = -Apre/tau_stdp : 1 (event-driven)    # STDP
dApost/dt = -Apost/tau_stdp : 1(event-driven)   # STDP
w_syn: 1
''')

plasticity_action_pre='''
p_s += p_s0*(1-p_s) # facilitation
r_s = p_s*n_s       # probability of release
n_s -= r_s          # depletion
'''

plasticity_action_pre_s='''
p_s += p_s0*(1-p_s) # facilitation
r_s = p_s*n_s       # probability of release
will_release = (rand() < r_s)
n_s = clip(n_s-will_release/pool_size,0,1)          # depletion
'''

stdp_action_pre='''
Apre += Apre0
'''

stdp_action_post='''
Apost += Apost0
'''


pre_actions_e = '''
w_syn = clip(w_syn+Apost, (1-stdp_value)*w_e/nS, w_e/nS) 
g_e_post += w_syn*r_s*nS
'''

pre_actions_e_s = '''
w_syn = clip(w_syn+Apost, (1-stdp_value)*w_e/nS, w_e/nS) 
g_e_post += w_syn*will_release*nS
'''


pre_actions_i = '''
w_syn = clip(w_syn+Apost, (1-stdp_value)*w_i/nS, w_i/nS) 
g_i_post += w_syn*r_s*nS
'''

pre_actions_i_s = '''
w_syn = clip(w_syn+Apost, (1-stdp_value)*w_i/nS, w_i/nS) 
g_i_post += w_syn*will_release*nS
'''


post_actions_e='''
w_syn = clip(w_syn+Apre,(1-stdp_value)*w_e/nS,w_e/nS)
'''

post_actions_i='''
w_syn = clip(w_syn+Apre,(1-stdp_value)*w_i/nS,w_i/nS)
'''
```

```python

```

```python
S_exc = Synapses(G[:Nexc],G, model=tm_plasticity_model+stdp_model,
                 on_pre=plasticity_action_pre+stdp_action_pre+pre_actions_e,
                 on_post=stdp_action_post+post_actions_e)

                
S_inh = Synapses(G[Nexc:],G, model=tm_plasticity_model+stdp_model,
                 on_pre=plasticity_action_pre+stdp_action_pre+pre_actions_i,
                 on_post=stdp_action_post+post_actions_i)


S_input = Synapses(P,G[:Nexc],model=tm_plasticity_model,
                 on_pre=plasticity_action_pre+'g_e_post += 10*w_e*r_s')
```

```python
%%time 

S_input.connect(p=0.01)

S_exc.connect(p=epsilon)
S_exc.delay = '10*ms + 0.1*randn()*ms'

S_inh.connect(p=2*epsilon)
S_inh.delay = '10*ms + 0.1*randn()*ms'
```

```python
S_input.active=True
S_exc.active=False
S_inh.active=False
```

```python
net = Network(G,P,M,S,Sp,S_exc,S_inh,S_input)
```

```python
for s in [S_input, S_exc, S_inh]:
    #s.delay = "average_delay-0.5*delay_jitter + delay_jitter*rand()"  # delays in synapses.
    s.p_s = p_s0
    s.n_s = 1.0
```

```python
net.store()
```

```python

```

```python
input_current = inpf.get_step_current(1000, 6000, 1*ms, 1.0*pA,Nneurons=Nneurons)
```

### No spike input to network

```python
net.restore()
S_input.active=False
S_exc.active=False
S_inh.active=False
G.stim_amp[:Nexc//2] = 35
```

```python
%time net.run(10*second,report='text')
```

```python
f,axs = subplots(2,1,sharex=True,figsize=(16,8),gridspec_kw=dict(height_ratios=(5,1)))
raster_spikes(S,Nexc,axs[0])
axs[0].set_ylim(0,Nneurons)
raster_spikes(Sp,100,axs[1],colors='gg')
xlim(M.t[0]/ms,M.t[-1]/ms)
```

### Turn on Poisson input

```python
net.restore()
S_input.active=True
S_exc.active=False
S_inh.active=False
G.stim_amp[:Nexc//2] = 35
```

```python
%time net.run(10*second,report='text')
```

```python
f,axs = subplots(2,1,sharex=True,figsize=(16,8),gridspec_kw=dict(height_ratios=(5,1)))
raster_spikes(S,Nexc,axs[0])
axs[0].set_ylim(0,Nneurons)
raster_spikes(Sp,100,axs[1],colors='gg')
#xlim(M.t[0]/ms,M.t[-1]/ms)
#xlim(500, 8000)
```

```python
k = 450
figure(figsize=(14,4))
plot(M.t/ms, beautify_spikes(M,S,k)/mV)
```

```python
#k = randint(Nexc)
figure(figsize=(14,4))
plot(M.t/second, M.g_e[k]/nS)
xlabel('time, s'); ylabel('g_e, nS')
```

### Add excitatory synapses

```python
net.restore()
stdp_value = 0.1 # change between 0 and 1
```

```python
S_input.active=True
S_exc.active=True
S_inh.active=False
G.stim_amp[:Nexc//2] = 35
```

```python
%time net.run(10*second,report='text')
```

```python
f,axs = subplots(2,1,sharex=True,figsize=(16,8),gridspec_kw=dict(height_ratios=(5,1)))

raster_spikes(S,Nexc,axs[0])
raster_spikes(Sp,100,axs[1],colors='gg')

axs[0].set_ylim(0,Nneurons)

xlim(M.t[0]/ms,M.t[-1]/ms)
axs[0].set_ylabel('# neuron')
xlabel('time')
ylabel('# Poisson generator')
```

```python
print(k)
figure(figsize=(14,4))

plot(M.t/second, beautify_spikes(M,S,k)/mV)
xlabel('time, s'); ylabel('Vm, mV')
```

```python
figure(figsize=(14,4))

plot(M.t/ms, M.g_e[k]/nS)
```

### Add inhibitory synapses

```python
net.restore()
stdp_value = 0.1 # change between 0 and 1
```

```python
S_input.active=True
S_exc.active=True
S_inh.active=True
G.stim_amp[:Nexc//2] = 35
```

```python

```

```python
%time net.run(10*second,report='text')
```

```python
f,axs = subplots(2,1,sharex=True,figsize=(16,8),gridspec_kw=dict(height_ratios=(5,1)))

raster_spikes(S,Nexc,axs[0])
raster_spikes(Sp,100,axs[1],colors='gg')

axs[0].set_ylim(0,Nneurons)

xlim(M.t[0]/ms,M.t[-1]/ms)
axs[0].set_ylabel('# neuron')
xlabel('time')
ylabel('# Poisson generator')
```

```python

```

```python


```

```python jupyter={"outputs_hidden": true}

```

------------------

### Optional: Selecting exemplars of interneurons and PCs by Monte-Carlo

Here we create a group of 10^4 neurons with random parameter values. Next we select only neurons with admissible behavior. 

```python
start_scope()

Nneurons = 10000
#Nneurons = int(1e4)

defaultclock.dt = 0.1*ms

G = NeuronGroup(Nneurons, AdEx_equations+synaptic_equations,threshold='v>Vcut', reset='v=Vreset; w += b',
                method='exponential_euler')

G.set_states(dict(v=El,w=0))

M = StateMonitor(G, ['v','w'], record=True)
S = SpikeMonitor(G,)
```

```python
G.a = uniform(-1,3,Nneurons)*nsiemens
G.b = uniform(5,50,Nneurons)*pA
G.tau_m = uniform(5,200,Nneurons)*ms
G.tau_w = uniform(30,100,Nneurons)*ms
G.Vreset = uniform(-60,-45,Nneurons)*mV
G.Ibias = 'randn()*5*pA'

# G.a = -0.5*nS
# G.b = 7*pA
# G.tau_m = 5*ms
# G.tau_w = 100*ms
# G.Vreset = -46*mV
```

```python
input_current = inpf.get_step_current(1000, 2000, 1*ms, 1.0*pA,Nneurons=Nneurons)
```

```python
store()
```

```python
restore()
G.stim_amp = 100
%time run(5*second)
```

```python
figure(figsize=(16,6))
plot(S.t/ms, S.i, 'k,',lw=0.5)
xlabel('time, ms')
ylabel('neuron #')
title('Raster plot for the neuron group during step current')
```

```python
k = randint(Nneurons)
plot(M.t, beautify_spikes(M,S,k)/mV)
xlabel('time, s')
ylabel('Vm, mV')
title('membrane voltage of a randomly chosen neuron')
```

```python
array([1/mean(diff((S.t[S.i==k])))])
```

```python
def validate_neurons(statemon,spikes,
                     min_spiking_rate = 5*Hz,
                     max_spiking_rate = 300*Hz,
                     reject_restless_prob=0.5):
    N = len(statemon.v)
    invalid = any(abs(statemon.v[:,:])>200*mV,axis=1) + isnan(statemon.v.sum(1))
    late_spikes = array([any(spikes.t[spikes.i==i]>3*second) for i in range(N)])
    
    rates = array([1/mean(diff(spikes.t[spikes.i==i]))/second if len(spikes.t[spikes.i==i])>2 else 0*Hz 
                   for i in range(N)])/second
    invalid += (rates < min_spiking_rate)
    invalid += (rates > max_spiking_rate)
    invalid += (late_spikes)&(rand(N)<reject_restless_prob)
    return ~invalid
    
```

```python
%time valid = validate_neurons(M,S)
print(sum(valid)/Nneurons)
```

```python
k = where(valid)[0][randint(sum(valid))]
```

```python
k,valid[k]
```

```python
figure()
plot(M.t/ms,beautify_spikes(M,S,k)/mV)
xlabel('time , ms')
ylabel('Vm, mV')
```

```python
def is_good_interneuron(statemon,spikes,k,min_rate=50*Hz,min_omega=5*Hz):
    rate = 1/mean(diff(spikes.t[spikes.i==k])) if len(spikes.t[spikes.i==k])>2 else 0*Hz
    aR = G.a[k]*Rin
    tautau = G.tau_m[k]/G.tau_w[k]
    omega = 4*(aR - 2*(1-tautau**2)/tautau)/G.tau_w[k]
    return(rate>min_rate) and (omega>0.5*min_rate)

def is_good_pyramidal(statemon,spikes,k,max_rate=25*Hz):
    rate = 1/mean(diff(spikes.t[spikes.i==k])) if len(spikes.t[spikes.i==k])>2 else 0*Hz
    aR = G.a[k]*Rin
    tautau = G.tau_m[k]/G.tau_w[k]
    omega = 4*(aR - 2*(1-tautau**2)/tautau)/G.tau_w[k]
    return(rate<=max_rate) and (omega <= 0)
```

```python
%time interneurons = array([is_good_interneuron(M,S,k) for k in range(Nneurons)])
```

```python
%time pyramidals = array([is_good_pyramidal(M,S,k) for k in range(Nneurons)])
```

```python
acc = zeros((Nneurons,2))
for k in range(Nneurons):
    if valid[k]:
        aR = G.a[k]*Rin
        tautau = G.tau_m[k]/G.tau_w[k]
        acc[k] = tautau,aR

```

```python
plot(acc[interneurons][:,0],acc[interneurons][:,1],'b.',label='inter')
plot(acc[pyramidals][:,0],acc[pyramidals][:,1],'r.',alpha=0.1, label='pyr')
legend()
xlabel('$\\tau_m/\\tau_w$')
ylabel('aR')
```

```python
sum(interneurons)/Nneurons, sum(pyramidals)/Nneurons
```

```python
interneuron_pars = dict(a=G.a[interneurons], b=G.b[interneurons],
                        tau_w=G.tau_w[interneurons],tau_m=G.tau_m[interneurons],
                        Ibias=G.Ibias[interneurons],
                        Vreset = G.Vreset[interneurons])
pyramidal_pars = dict(a=G.a[pyramidals], b=G.b[pyramidals],
                        tau_w=G.tau_w[pyramidals],tau_m=G.tau_m[pyramidals],
                        Ibias = G.Ibias[pyramidals],
                        Vreset = G.Vreset[pyramidals])
```

```python
len(interneuron_pars['a'])
```

```python
iterable(tonic_pars['a'])
```

```python

```
