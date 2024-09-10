---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Simplified/Spiking neuronal models


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
import pandas as pd
```

```python
#style.use('ggplot')           # more stylish plots
#style.use('seaborn-muted')    # better default line colors

rc('axes',labelsize=12, grid=True)
rc('figure', dpi=150, figsize=(9,9*0.618))
```

```python
from brian2 import *
```

```python
import input_factory as inpf
```

```python
def beautify_spikes(statemon,spikemon,neuron_id):
    vm = statemon[neuron_id].v[:]
    offset = statemon.t[0]#/defaultclock.dt
    spike_times = spikemon.t[spikemon.i == neuron_id]
    for t in spike_times:
        i = int((t-offset) / defaultclock.dt)
        vm[i] = 20*mV
    return vm
```

```python
#import ipywidgets as ipw
```

## Adaptive threshold model
### Single timescale

```python
start_scope()

N = 100
tau = 10*ms
vr = -70*mV
vt0 = -50*mV
delta_vt0 = 5*mV
tau_t = 100*ms
sigma = 0.5*(vt0-vr)
v_drive = 2*(vt0-vr)
duration = 500*ms

eqs = '''
dv/dt = (v_drive+vr-v)/tau + sigma*xi*tau**-0.5 : volt
dvt/dt = (vt0-vt)/tau_t : volt
'''


reset1 = '''
vt += delta_vt0
'''

reset2 = '''
v = vr
vt += delta_vt0
'''


G = NeuronGroup(N, eqs, threshold='v>vt', reset=reset1, refractory=5*ms, method='euler')
spikemon = SpikeMonitor(G)
vmon =  StateMonitor(G,('v', 'vt'),record=True)

G.v = 'rand()*(vt0-vr)+vr'
G.vt = vt0

run(duration)

_ = hist(spikemon.t/ms, 100, histtype='stepfilled', facecolor='k', weights=ones(len(spikemon))/(N*defaultclock.dt))
xlabel('Time (ms)')
ylabel('Instantaneous firing rate (sp/s)');
```

```python
vx = beautify_spikes(vmon, spikemon,5)
figure(figsize=(24,3)); plot(vx/mV)
xlabel('time, s'); ylabel('Vm, mV')
title('Single-timescale adaptive threshold model')
```

### Two timescales

```python
alpha1 = 1.1*mV
alpha2 = 2*mV
tau_t1 = 10*ms
tau_t2 = 200*ms

eqs_mat = '''
dv/dt = (v_drive+vr-v)/tau + sigma*xi*tau**-0.5 : volt
dvt1/dt =  -vt1/tau_t1 : volt
dvt2/dt =  -vt2/tau_t2 : volt
'''

reset_mat = '''
vt1 += alpha1
vt2 += alpha2
'''
```

```python
G = NeuronGroup(N, eqs_mat, threshold='v > vt0 + vt1 + vt2', reset=reset_mat, refractory=5*ms, method='euler')
spikemon = SpikeMonitor(G)
vmon =  StateMonitor(G,('v', 'vt1'),record=True)

G.v = 'rand()*(vt0-vr)+vr'
G.vt1 = 0
G.vt2 =  0

run(duration)

_ = hist(spikemon.t/ms, 100, histtype='stepfilled', facecolor='k', weights=ones(len(spikemon))/(N*defaultclock.dt))
xlabel('Time (ms)')
ylabel('Instantaneous firing rate (sp/s)');
```

```python
vx = beautify_spikes(vmon, spikemon,5)
figure(figsize=(24,3)); plot(vx/mV)
xlabel('time, s'); ylabel('Vm, mV')
title('Single-timescale adaptive threshold model')
```

## Adaptive exponential IF model


Итак, модель AdEx:

\begin{align}
C\frac{dv}{dt} &= I_{stim} - g_L(v-E_L) + g_L\Delta T e^{\frac{v-V_{\theta}}{\Delta T}} - u + I_{syn}\\
\tau_u\frac{du}{dt} &= a(v-E_L) - u
\end{align}

Если $v$ превышает пороговое значение $V_{cut}$, то интегрирование системы останавливается, и переменным $v$ и $u$ присваиваются новые значения:
\begin{align}
v &\leftarrow v_{reset}\\
u &\leftarrow u + b\,,
\end{align}
после чего, интегрирование продолжается снова.

<!-- #region -->
### AdEx model in Destexhe's formulation


\begin{align}
C_m\frac{dV}{dt} & =   -g_l(V-E_l) + g_l\Delta V\exp(\frac{V-V_T}{\Delta V}) - w/S\\
\tau_w\frac{dw}{dt} & =  a(V-E_l) - w
\end{align}

<!-- #endregion -->

```python
## Parameters that are shared by all neurons

# Neurons
Vth = -50*mV   # rheobase threshold
El = -70*mV     # resting membrane potential
Vcut = -0.1*mV    # spike detection threshold
deltaT = 2*mV  # spike initiation sharpness
Rin = 500*Mohm  # input resistance of a neuron at rest
gl = 1/Rin
Ena = 50*mV
Ek = -75*mV



# Synapses
E_e = 0*mV     # Excitatory synaptic reversal potential (AMPA and NMDA receptors)
E_i = -80*mV   # Inhibitory synaptic reversal potential (GABAA receptors)
tau_e = 5*ms   # time scale of excitatory synaptic conductance
tau_i = 10*ms  # time scale of excitatory synaptic conductance
```

```python
AdEx_equations = Equations('''

dv/dt = (-gl*(v-El) + activation_curr - u + Ibias + Iapp)/C : volt 
du/dt = (a*(v-El) - u)/tau_u: amp  # adaptation variable

activation_curr = gl*deltaT*exp((v-Vth)/deltaT) : amp
stim_amp : 1
Ibias : amp
Iapp = stim_amp*input_current(t,i): amp
''')
```

```python
adex_params = pd.read_csv('adex_params.csv',index_col='type')
adex_params
```

```python
adex_params.loc['adapting']
```

```python
def convert_table_cell(col_name):
    units = col_name.split(' ')[1][1:-1]

def convert_from_table(row):
    return dict(
        a = float(row['a [nS]'])*nS,
        b = float(row['b [pA]'])*pA,
        tau_u = float(row['tau_u [ms]'])*ms,
        Vreset = float(row['Vreset [mV]'])*mV,
        C = float(row['tau_m [ms]'])*ms*gl,
    )
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

### Nullclines

```python
def v_nullcline(v,Ibias=0*pA):
    return Ibias - gl*(v - El) + gl*deltaT*exp((v-Vth)/deltaT)

def u_nullcline(v,pars):
    return pars['a']*(v-El)
```

```python
vv = linspace(-85, -40, 200)*mV
plot(vv/mV,v_nullcline(vv)/nA)
#plot(vv/mV, u_nullcline(vv,bursting_pars)/nA)
plot(vv/mV, u_nullcline(vv,bursting_pars)/nA)
xlabel('membrane potential [mV]')
ylabel('adaptation current [nA]')
title('Nullclines of the bursting AdEx neuron')
```

```python
start_scope()

Nneurons = 10

defaultclock.dt = 0.1*ms

G = NeuronGroup(Nneurons, AdEx_equations,threshold='v>Vcut', reset='v=Vreset; u += b',
                namespace=tonic_pars,
                method='exponential_euler')

G.set_states(dict(v=El,u=0))

G.stim_amp = linspace(0,0.5,Nneurons)
G.stim_amp[1] = 0.065
G.v = -70*mV
M = StateMonitor(G, ['v','u'], record=True)
S = SpikeMonitor(G,)
```

```python
input_current = inpf.get_step_current(200, 1500, 1*ms, 1.0*nA,Nneurons=Nneurons)
```

```python
G.stim_amp[1]*nA
```

```python
store()
```

```python
restore()
```

```python
%time run(2*second)
```

```python
plot(M.t/ms, M.v[-1]/mV)
xlim(200,250)
```

```python
k = 2

f,axs = subplots(2,1,sharex=True, figsize=(15,5))
vx = beautify_spikes(M,S,k)/mV
axs[0].plot(M.t/ms,vx)
axs[1].plot(M.t/ms, G.stim_amp[k]*input_current(M.t,k)/nA,c='orange')
xlim(0,2000)
```

```python
f,axs = subplots(2,1,sharex=True, figsize=(15,5))
vx = beautify_spikes(M,S,k)/mV
axs[0].plot(M.t/ms,vx)
axs[1].plot(M.t/ms, G.stim_amp[k]*input_current(M.t,k),c='orange')
xlim(250,500)
```

```python
figure(figsize=(10,10))

vv = linspace(-85, -40, 200)*mV

plot(vv/mV,v_nullcline(vv,0)/nA,ls='--',c='blue',label='V nullcline before stim')
plot(vv/mV,v_nullcline(vv,G.stim_amp[k]*nA)/nA,ls='-',label='V nullcline during stim')
plot(vv/mV, u_nullcline(vv,tonic_pars, )/nA,label='u nullcline')

# trajectory
plot(vx[M.t<250*ms],M.u[k][M.t<250*ms]/nA,color='gray')
plot(vx[0],M.u[k][0]/nA,'ms')

axis([-72,-40,-0.1,0.1])
legend()

xlabel('membrane potential [mV]')
ylabel('adaptation current [nA]')
title('Nullclines and trajectory of the tonic AdEx neuron')
```

### Bursting

```python
start_scope()

Nneurons = 10

defaultclock.dt = 0.1*ms

G = NeuronGroup(Nneurons, AdEx_equations,threshold='v>Vcut', reset='v=Vreset; u += b',
                namespace=bursting_pars,
                method='exponential_euler')

G.set_states(dict(v=El,u=0))

G.stim_amp = linspace(0,0.5,Nneurons)
G.stim_amp[1] = 0.065
G.v = -70*mV
M = StateMonitor(G, ['v','u'], record=True)
S = SpikeMonitor(G,)
```

```python
input_current = inpf.get_step_current(200, 1500, 1*ms, 1.0*nA,Nneurons=Nneurons)
```

```python
store()
```

```python
restore()

%time run(2*second)
```
```python

```


```python
k = 2

f,axs = subplots(2,1,sharex=True, figsize=(15,5))
vx = beautify_spikes(M,S,k)/mV
axs[0].plot(M.t/ms,vx)
axs[1].plot(M.t/ms, G.stim_amp[k]*input_current(M.t,k)/nA,c='orange')
xlim(0,2000)
```

```python
f,axs = subplots(2,1,sharex=True, figsize=(15,5))
vx = beautify_spikes(M,S,k)/mV
axs[0].plot(M.t/ms,vx)
axs[1].plot(M.t/ms, G.stim_amp[k]*input_current(M.t,k),c='orange')
xlim(250,500)
```

```python
figure(figsize=(10,10))

vv = linspace(-85, -40, 200)*mV

plot(vv/mV,v_nullcline(vv,0)/nA,ls='--',c='blue',label='V nullcline before stim')
plot(vv/mV,v_nullcline(vv,G.stim_amp[k]*nA)/nA,ls='-',label='V nullcline during stim')
plot(vv/mV, u_nullcline(vv,tonic_pars, )/nA,label='u nullcline')

# trajectory
plot(vx[M.t<250*ms],M.u[k][M.t<250*ms]/nA,color='gray')
plot(vx[0],M.u[k][0]/nA,'ms')

axis([-72,-40,-0.1,0.1])
legend(fontsize=16)

xlabel('membrane potential [mV]', fontsize=23)
ylabel('adaptation current [nA]', fontsize=23)
title('Nullclines and trajectory of the tonic AdEx neuron', fontsize=23)
```

<!-- #region -->
## CAdEx model

В этой модели адаптационный ток $w$ заменяется на адаптационную **проводимость** с добавлением новых параметров:


\begin{align}
C_m\frac{dV}{dt} & =   -g_l(V-E_l) + g_l\Delta V\exp(\frac{V-V_T}{\Delta V}) + g_A(E_A - V) + I_s\\
\tau_A\frac{dg_A}{dt} & =  \frac{\bar{g}_A}{1 + \exp(\frac{V_A-V}{\Delta_A})} - g_A
\end{align}

<!-- #endregion -->

https://github.com/neural-decoder/cadex/blob/master/Cadex_min.py

```python
start_scope()

#integration step
dt=0.001
defaultclock.dt = dt*ms


#simulation duration
TotTime=500
duration = TotTime*ms

#number of neuron
N1=1
```

```python
eqs='''
dv/dt = (-gl*(v-El)+ gl*Dt*exp((v-Vt)/Dt)-ga*(v-Ea) + Is)/Cm : volt (unless refractory)
dga/dt = (ga_max/(1.0+exp((-Va-v)/Da))-ga)/tau_a : siemens

Is:ampere
Cm:farad
gl:siemens
El:volt
ga_max:siemens
tau_a:second
Dt:volt
Vt:volt
Va:volt
Da:volt
Ea:volt
'''
```

```python
Dga = 3.0*nS

G1 = NeuronGroup(N1, eqs, threshold='v > -40.*mV', reset='v = -65*mV; ga += Dga', refractory='5*ms', method='heun')
#init:
G1.v = -65*mV
G1.ga = 0.0*nS

#parameters
G1.Cm = 200.*pF
G1.gl = 10.*nS
G1.El = -60.*mV
G1.Vt = -50.*mV
G1.Dt = 2.*mV
G1.tau_a = 500.0*ms
G1.Va = 65.*mV
G1.Da = 5.*mV
G1.ga_max = 0.0*nS
G1.Ea = -70.*mV
G1.Is = 1.0*nA
```

```python
# record variables
Mon_v  = StateMonitor(G1, 'v', record=range(N1))
Mon_ga = StateMonitor(G1, 'ga', record=range(N1))
S = SpikeMonitor(G1,)

run(duration, report='stdout')
```

```python
#plot
fig,axs =subplots(2,1, sharex=True, figsize=(15,6))
fig.suptitle('CAdEx')

axs[0].set_title("V")
axs[1].set_title("ga")
axs[0].plot(Mon_v.t/ms, beautify_spikes(Mon_v,S,0))
axs[1].plot(Mon_ga.t/ms, Mon_ga[0].ga/nS)

plt.show()
```
```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

