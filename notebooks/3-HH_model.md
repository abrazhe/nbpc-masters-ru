---
jupyter:
  jupytext:
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

```python
# check if notebook is running in Colab and install packages if it is
if RunningInCOLAB:
  ! pip install brian2
  ! pip install pandas
  !wget https://raw.githubusercontent.com/abrazhe/nbpc-masters-ru/master/notebooks/input_factory.py
```

```python
%pylab inline
```

```python
rc('axes',labelsize=12, grid=True)
rc('figure', dpi=150, figsize=(9,9*0.618))
```

```python
from brian2 import *
```

```python
from brian2.units.constants import zero_celsius, gas_constant as R, faraday_constant as F
```

```python
R*F
```

```python
import input_factory as inpf
```

## The original HH model

```python
T = 6*kelvin + zero_celsius
```

```python
def Nernst(Ci,Co,z=1):
    return R*T*log(Co/Ci)/F/z
```

```python

```

```python
radius = 0.25*mm
length = 2*cm
area = 2*pi*radius*length
area
```

```python
def malpha_hh(V):
     return 0.1*(V+40*mV)/(1 - exp(-(V+40*mV)/10./mV))/mV/ms
```

```python
defaultclock.dt = 0.01*ms
El = -54.5*mV
Nai,Nao = 20*mM,155*mM
Ki,Ko = 75*mM,3*mM


C = 1*uF/cm2*area

gna = 120*mS/cm2*area
gk = 36*mS/cm2*area
gl = 0.3*mS/cm2*area

# leak current
ileak_eqs = Equations("ileak = gl*(V-El) : amp")

# potassium current
Ek = Nernst(Ki,Ko)
ik_eqs = Equations("""
ik = gk*n**4*(V - Ek) : amp
# -- gating -- 
dn/dt = nalpha*(1-n) - nbeta*n : 1
nalpha = 0.01*(V + 55.0*mV)/(1 - exp(-(V + 55.0*mV)/10.0/mV))/mV/ms : 1/second
nbeta = 0.125*exp(-(V + 65.0*mV)/80.0/mV)/ms: Hz
""")

# sodium current
Ena = Nernst(Nai,Nao)
ina_eqs = Equations("""
ina = gna*m**3*h*(V-Ena) : amp
# -- gating -- 
dm/dt = malpha*(1-m) - mbeta*m : 1
dh/dt = halpha*(1-h) - hbeta*h : 1
# -- activation gate rates -- 
malpha = 0.1*(V+40.0*mV)/(1 - exp(-(V+40*mV)/10./mV))/mV/ms : 1/second
mbeta = 4*exp(-(V+65*mV)/18/mV)/ms : 1/second
# -- inactivation -- 
halpha = 0.07*exp(-(V+65.0*mV)/20.0/mV)/ms : 1/second
hbeta = 1/(1.0 + exp(-(V+35.0*mV)/10.0/mV))/ms : 1/second
""")

# Full set of equations
hh_eqs = Equations("""
dV/dt = (I_stim - ileak - ina - ik)/C : volt
stim_amp : 1
I_stim =  stim_amp*input_current(t,i): amp
""") + ik_eqs + ina_eqs + ileak_eqs
```

```python
input_current = inpf.get_step_current(300, 800, 1.0*ms, 1.0*uA)
```

```python
print("Membrane capacitance: {:0.3e}".format(C))
print("Max Na conductance: {:0.3e}".format(gna))
```

```python
print("Sodium reversal potential Ena = {:2.1f} mV".format(Ena/mV))
print("Potassium reversal potential EK = {:2.1f} mV".format(Ek/mV))
```

```python
start_scope()

G = NeuronGroup(1, hh_eqs, method='exponential_euler')
G.set_states(dict(V=-65.2*mV, m=0.05,h=0.6,n=0.3))
```

```python
G.stim_amp = 0 
run(20*ms)
#states = G.get_states()
stateM = StateMonitor(G, variables=True,record=True)
store() # Store the initialized state of the model
#store()
```

### Excitability properties

```python
restore() # restore the initialized state and run new experiment
G.stim_amp = 2.1 # try to find threshold for (1) single spike, (2) sustained firing
run(1*second)
f,axs = subplots(2,1,sharex=True)
axs[0].plot(stateM.t/ms, stateM.V[0]/mV,label='membrane potential [mV]')
axs[1].plot(stateM.t/ms, stateM.I_stim[0]/uA,c='tomato',label='I_stim')
axs[0].legend(); axs[1].legend()
xlabel('time [ms]'); 
```

### Rebound spike

```python
restore() # restore the initialized state and run new experiment
G.stim_amp = -2.1
run(1*second)
f,axs = subplots(2,1,sharex=True)
axs[0].plot(stateM.t/ms, stateM.V[0]/mV,label='membrane potential [mV]')
axs[1].plot(stateM.t/ms, stateM.I_stim[0]/uA,c='tomato',label='I_stim')
axs[0].legend(); axs[1].legend()
xlabel('time [ms]'); 
```

```python
start_scope()

Ndummy = 200
Vv = linspace(-90,40,Ndummy)*mV

dummy_eqs = """dV/dt = (V0-V)/ms: volt
               V0:volt
            """

dummyG = NeuronGroup(Ndummy,
                     Equations(dummy_eqs)+ik_eqs+ina_eqs,method='euler')
dummyG.V0 = Vv
dummyG.V = Vv
M = StateMonitor(dummyG,variables=True,record=True)
run(defaultclock.dt)
```

```python
f,axs = subplots(1,3,sharex=True,figsize=(16,4))
axs[0].plot(Vv/mV, M.malpha,label=r'$\alpha_m$')
axs[0].plot(Vv/mV, M.mbeta,label=r'$\beta_m$')
axs[0].legend()

axs[1].plot(Vv/mV, M.malpha/(M.malpha+M.mbeta),label=r'$m_\infty$')
axs[1].legend()

axs[2].plot(Vv/mV, 1/(M.malpha+M.mbeta)/ms,label=r'$\tau_m$')
axs[2].legend()


xlabel('V [mV]')

```

**Exercise:**
 - [ ] Make graphs like the one above, but for $n$ and $h$ gating variables


  ## Neurons of the VCN (Rothman & Manis *J Neurophysiol* 2003)

```python
import pandas as pd
```

```python
soma_diameter = 21*um
soma_area = pi*soma_diameter**2
C = 0.9*uF/cm2*soma_area
C
```

```python
Ek = -70*mV
Ena = +55*mV
Eh = -43*mV
Elk = -65*mV
```

### Sodium current

```python
# sodium current
ina_eqs_vcn = Equations("""
ina_vcn = gnabar*m**3*h*(V-Ena) : amp
# -- gating -- 
dm/dt = (minf-m)/mtau : 1
dh/dt = (hinf-h)/htau : 1

minf= 1/(1 + exp(-(V+38*mV)/7./mV)) : 1
hinf = 1/(1 + exp((V+65*mV)/6./mV)) : 1
mtau = 10*ms/(5*exp((V+60*mV)/18./mV) + 36*exp(-(V+60*mV)/25./mV)) + 0.04*ms : second
htau = 100*ms/(7*exp((V+60*mV)/11./mV) + 10*exp(-(V+60*mV)/25./mV)) + 0.6*ms : second
""")
```

### Kht (high-threshold К current)


```python
ikht_eqs = Equations("""
ikht = gkhtbar*(phi*n**2 + (1-phi)*p)*(V-Ek) : amp
phi =  0.85 : 1

dn/dt = (ninf - n)/ntau: 1
dp/dt = (pinf-p)/ptau : 1

ninf = 1/(1 + exp(-(V/mV+15)/5.))**0.5 : 1
pinf = 1/(1 + exp(-(V/mV+23)/6.)) : 1
ntau = (100/(11*exp((V/mV+60.0)/24.0) + 21.0*exp(-(V/mV+60.0)/23.0)) + 0.7)*ms : second
ptau = (100/(4*exp((V/mV+60.0)/32.0) + 5.0*exp(-(V/mV+60.0)/22.0)) + 5.0)*ms   : second
""")
```

### Klt (low-threshold К current)

```python
iklt_eqs = Equations("""
iklt = gkltbar*(w**4)*z*(V-Ek) : amp

dw/dt = (winf-w)/wtau : 1
dz/dt = (zinf-z)/ztau : 1

winf = (1 + exp(-(V/mV+48)/6.0))**-0.25 : 1
wtau = (100/(6.0*exp((V/mV+60.0)/6.0) + 16.0*exp(-(V/mV+60.0)/45.0)) + 1.5)*ms: second

zinf = (1-0.5)*(1 + exp((V/mV+71)/10.0))**-1 + 0.5 : 1
ztau = (1000/(exp((V/mV+60)/20.0) + exp(-(V/mV+60)/8.0)) + 50)*ms : second
""")
```

### Ih (hyperpolarization-activated current)

```python
ih_eqs = Equations("""
ih = ghbar*r*(V-Eh) : amp
dr/dt = (rinf-r)/rtau : 1
rinf = 1/(1 + exp((V/mV+76)/7.)) : 1
rtau = (10**5/(237*exp((V/mV+60)/12.) + 17*exp(-(V/mV+60)/14.)) + 25)*ms : second
""")

```

### Full vcn equations:

```python
vcn_equations = Equations("""
dV/dt = (I_stim - ileak - ina_vcn - ikht - iklt - ih)/C : volt
ileak = glk*(V-Elk) : amp
stim_amp : 1
I_stim =  stim_amp*input_current(t,i): amp
""") + ina_eqs_vcn + ikht_eqs + iklt_eqs + ih_eqs
```

### Conductances in nS

```python
rm_params = pd.read_csv("Rothman-Manis-2003-table1.csv", index_col=0)
rm_params
```

```python
def convert_rm_table_units(key):
    return {k:v*nS for k,v in dict(rm_params[key]).items()}
```

## Setting up the neurons

```python
start_scope()

G_typeIc =  NeuronGroup(1, vcn_equations, namespace=convert_rm_table_units('Ic'), method='exponential_euler')
```

```python
input_current = inpf.get_step_current(300, 450, 1*ms, 1.0*pA)
```

```python
run(200*ms)
MIc = StateMonitor(G_typeIc, variables=True, record=True)
store()

```

```python
restore()
G_typeIc.stim_amp = 50
run(0.5*second)
vplus = MIc.V[0][:]
#plot(MIc.t/ms, MIc.V[0]/mV)
```

```python
restore()
G_typeIc.stim_amp = -50
run(0.5*second)
plot(MIc.t/ms, vplus/mV)
plot(MIc.t/ms, MIc.V[0]/mV)
legend(('response to +I', 'response to -I '))
```

```python
start_scope()
G_typeII =  NeuronGroup(1, vcn_equations, namespace=convert_rm_table_units('II'), method='exponential_euler')
```

```python
run(200*ms)
MII = StateMonitor(G_typeII, variables=True, record=True)
store()
```

```python
restore()
G_typeII.stim_amp = 500
run(0.5*second)
plot(MII.t/ms, MII.V[0]/mV, label='+500pA')

restore()
G_typeII.stim_amp = -400
run(0.5*second)
plot(MII.t/ms, MII.V[0]/mV,label='-400pA')
xlabel('time [ms]')
ylabel('membrane voltage [mV]')

legend()
```

**Exercise:**
   - Explain, why the membrane voltage sags during hyperpolarizing stimulus
   - why does it sag more in the typeII model?
   - test if typeI neuron can encode input current amplitude in spiking frequency
   - which current is the slowest?


**Themes for projects (may want to choose one)**
   - Investigate bifurcations of the resting state of the typeI and typeII neurons
   - Investigate response of typeI and typeII models to stimulation with noisy injected current or injected conductance. You may want to read Brian2 documentation on how to do this kind of simulation

```python

```

```python

```
