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



# Bifurcations in the Ina,p+Ik model

```python
#! pip install brian2
```

```python
#! pip install pandas
```

```python
#!wget https://raw.githubusercontent.com/abrazhe/nbpc-masters-ru/master/notebooks/input_factory.py
#!wget https://raw.githubusercontent.com/abrazhe/nbpc-masters-ru/master/notebooks/Rothman-Manis-2003-table1.csv
```

```python
%pylab inline

#style.use('ggplot')
#style.use('seaborn-muted')
```

```python
rc('figure', dpi=150, figsize=(8,5))
```

```python
from numba import jit
```

```python
def ensure_dir(f):
        import os
        d = os.path.dirname(f)
        if not os.path.exists(d):
                os.makedirs(d)
        return f
```

```python
# This set of parameters corresponds to SNIC
napk_pset = dict(
    I_bias = 0.0,
    El = -80.0,
    Ena = 60.0,
    Ek = -90.0,
    gl = 8.0,
    gna = 20.0,
    gk = 10.0,
    ntau = 1.0,
    minf_vhalf = -20.,
    minfk = 15.,
    ninf_vhalf = -25.,
    ninfk = 5.0
)

SN_off = napk_pset.copy()   # Saddle-Node off limit cycle
superAH = napk_pset.copy()  # Supercritical AH
superAH2 = napk_pset.copy() # Supercritical AH v2
subAH = napk_pset.copy()    # subcritical AH
subAH2 = napk_pset.copy()   # subcritical AH (idential to Izhikevich 2007)


SN_off.update(ntau=0.16)
superAH.update(ninf_vhalf=-44,minf_k=10.)
superAH2.update(ninf_vhalf=-44, gl=7, gna=19, ninfk=7, minf_k=10)
subAH.update(gl=1.0, gna=4., gk=4., minf_vhalf=-30.,ninf_vhalf=-40., minfk=7.0)
subAH2.update(gl=1.0, gna=4., gk=4., minf_vhalf=-30., minfk=7., ninf_vhalf=-45., El=-78.0,)
```

```python
@jit
def boltzman(v,vhalf,k):
    return 1./(1.0 + exp((vhalf-v)/k))

def locmax(v,th=-20):
    allmax = where(diff(sign(diff(v)))<=-1)[0]+1
    allmax = [a for a in allmax if v[a]>=th]
    return array(allmax)
```

```python
from scipy.interpolate import UnivariateSpline as usp
```

```python
def adams_bashforth(rhs, init_state, dt=0.025, tstart=0, tstop=500,  fnkwargs=None):
    #f_buff = deque()
    if fnkwargs is None:
        fnkwargs = {}
        
    ndim = len(init_state)
    tv = arange(tstart,tstop,dt)
    xout = zeros((len(tv), ndim))
    xout[0] = init_state
    fprev = array(rhs(xout[0], tv[0], **fnkwargs))
    xnew = xout[0] + dt*fprev
    xout[1] = xnew
    for k,t in enumerate(tv[1:-1]):
        xprev,xcurr = xout[k:k+2]
        fnew = array(rhs(xcurr, t,**fnkwargs))
        xnew = xcurr + dt*(3*fnew/2.0 - fprev/2.0)
        fprev = fnew
        xout[k+2] = xnew
    return tv, xout
```

```python
def I_pulse(tx, amp, start, stop,k=10):
    return amp*0.5*(1 + tanh(k*(tx-start)) * tanh(k*(-tx+stop)))

def I_ramp(tx, k, start=50,stop=1000):
    #return  k*(tx-start)*I_pulse(tx, 1, start, stop)
    return usp(tx, k*(tx-start)*I_pulse(tx, 1, start, stop),s=0)


def make_pulses(tv, npulses=5, period=5, start=100, amp=5, width=1,steep=10.):
    onsets = arange(start, start+npulses*period,period)
    y = np.sum([I_pulse(tv, amp, on, on+width,k=steep) for on in onsets],0)
    return usp(tv, y, s=0)

def combine_interpolations(tv, signals):
    y = np.sum([s(tv) for s in signals],0)
    return usp(tv, y, s=0)

class InapkNeuron:
    def __init__(self, **params):
        self.__dict__.update(params)
        self.I_bias = 0
    def minf(self, v):
        return boltzman(v, self.minf_vhalf, self.minfk)
    def ninf(self, v):
        return boltzman(v, self.ninf_vhalf, self.ninfk)
    def ina(self, v):
        return self.gna*self.minf(v)*(v-self.Ena)
    def ik(self, v, n):
        return self.gk*n*(v-self.Ek)
    def ileak(self,v):
        return self.gl*(v-self.El)
    def vnullcline(self, v):
        return (self.I_bias - self.ina(v) - self.ileak(v))/(self.gk*(v-self.Ek))
    def nnullcline(self, v, I=0):
        return self.ninf(v)
                                                  
    def __call__(self, state, t, I_dyn=0):
        v,n = state
        I_app = self.I_bias
        if I_dyn:
            I_app += I_dyn(t)
        dV = I_app - (self.ina(v) + self.ileak(v) + self.ik(v,n))
        dn = (self.ninf(v)-n)/self.ntau
        return np.array([dV, dn])

def inapk_model(state, t, 
                El = -80.0, 
                I_bias = 0,
                I_dyn = 0, 
                Ena = 60.0, 
                Ek = -90.0,
                gl = 8.0,
                gna = 20.0,
                gk = 10.0,
                ntau = 1.0,
                minf_vhalf = -20.,
                minfk = 15.,
                ninf_vhalf = -25.,ninfk = 5.0):
    v,n = state
    minf = boltzman(v, minf_vhalf, minfk)
    ninf = boltzman(v, ninf_vhalf, ninfk)
    dn = (ninf-n)/ntau
    I_app = I_bias #+ I_ramp(t, I_kramp)
    if I_dyn:
        I_app += I_dyn(t)
    dV = I_app - (gna*minf*(v-Ena) + gk*n*(v-Ek) + gl*(v-El))
    return np.array([dV,dn])
```

```python
tx = arange(0,1000,0.1)
Ipulses = make_pulses(tx,npulses=5,period=5,width=1)
Ipulses2 = make_pulses(tx,npulses=5,period=2,width=1,start=150)

Iapp =combine_interpolations(tx, (Ipulses, Ipulses2))

plot(tx, Iapp(tx))
xlim(90,160)
xlabel('time, ms'); ylabel('Istim, a.u.')
```

```python
Ibias_bif = 4.514
Vrest = -63.8
nrest = boltzman(Vrest, napk_pset['ninf_vhalf'],napk_pset['ninfk'])

#pars = dict(I_bias=4,I_kramp=0.005)

Ipulses = combine_interpolations(tx, 
                                 (make_pulses(tx,amp=10,npulses=10,period=p,start=100+50*k) 
                                  for k,p in enumerate([1.5, 2, 2.5, 3])))

Ipulses2 = combine_interpolations(tx, 
                                 (make_pulses(tx,amp=6,npulses=5,period=p,start=100+(p*5+k*50)) 
                                  for k,p in enumerate([2,  3, 7.4,])))

Ipulses3 = combine_interpolations(tx, [Ipulses2, 
                                       make_pulses(tx, amp=6, start=50,npulses=1),
                                       make_pulses(tx, amp=-6, start=350,npulses=1),
                                       make_pulses(tx, amp=-6, start=300,npulses=1),
                                       make_pulses(tx, amp=6, start=355,npulses=1)])


                                      
pars=dict(I_dyn=Ipulses3)

snic_neuron = InapkNeuron(**napk_pset)

snoff_neuron = InapkNeuron(**SN_off)
supAH_neuron = InapkNeuron(**superAH)

subAH_neuron = InapkNeuron(**subAH)
subAH2_neuron = InapkNeuron(**subAH2)
```

## Threshold manifolds

```python
Vv = arange(-80, 10)

snic_neuron.I_bias = 0

plot(Vv, snic_neuron.vnullcline(Vv),lw=2)
plot(Vv, snic_neuron.nnullcline(Vv),lw=2)
```

```python
def test_init_state(neuron, state,tstop=100):
    tvx, outx = adams_bashforth(neuron, state,tstop=tstop)
    return np.max(outx[:,0])

def test_init_state2(neuron, state,tstop=200):
    tvx, outx = adams_bashforth(neuron, state,tstop=tstop)
    tail = (outx[:,0][tvx>100])
    return amax(tail)-amin(tail)
```

```python
def prep_phase_portrait(neuron, ax=None):
    if ax is None:
        f,ax = subplots(1,1)
    Vv = arange(-80, 10)
    ax.plot(Vv, neuron.vnullcline(Vv),lw=2,color='#ff6600')
    ax.plot(Vv, neuron.nnullcline(Vv),lw=2,color='#2a7fff')
    setp(ax, ylabel = 'Ik activation', xlabel='membrane potential [mV]')
    ax.axis([-80,10,-0.1,1])
    return ax
    
    

def map_trajectories(neuron, 
                     voltages=linspace(-80,-30,10),
                     activations =  linspace(0, 0.5, 10)):
    ax = prep_phase_portrait(neuron)
    for v in voltages:
        for n in activations:
            _, traj = adams_bashforth(neuron, array([v,n]), tstop=100)
            ax.plot(traj[:,0],traj[:,1], color='k',alpha=0.1)
            ax.plot(traj[0,0],traj[0,1],marker='s',ms=1,mfc='y',mec='y')
            ax.plot(traj[-1,0],traj[-1,1],color='m',marker='.')
            
    
```

```python

%time map_trajectories(snic_neuron)
title('phase trajectories near SNIC bifurcation')
```

```python
snoff_neuron.I_bias = 3.
%time map_trajectories(snoff_neuron)
title('phase trajectories near SN bifurcation')
```

```python
superAH
```

```python
supAH2_neuron = InapkNeuron(**superAH2)


supAH2_neuron.I_bias=33


#t,traj = adams_bashforth(supAH2_neuron, [-80,0.1],tstop=200)
#plot(t, traj[:,0])

#prep_phase_portrait(supAH2_neuron)
```

```python
#supAH_neuron.I_bias = 23


%time map_trajectories(supAH2_neuron)
title('phase trajectories near supAH bifurcation')
```

```python
subAH2_neuron.I_bias = 43

%time map_trajectories(subAH2_neuron)

title('phase trajectories near subAH bifurcation')
```

## Response to short pulse batches

```python
ensure_dir('outputs/')
```

```python
%%time 

init_state = [-63.8, snic_neuron.ninf(-63.8)]
init_state2=[-53.9,  subAH_neuron.ninf(-53.9)]

snic_neuron.I_bias = 3
subAH_neuron.I_bias = 19

tvx, outx = adams_bashforth(snic_neuron, init_state, fnkwargs=pars,tstop=400)
tvx2, outx2 = adams_bashforth(subAH_neuron, init_state2, fnkwargs=pars,tstop=400)

f,axs = subplots(3,1,sharex=True,figsize=(16,9))

axs[0].plot(tvx, outx[:,0],color='orange',label='integrator (SNIC)')
axs[1].plot(tvx2, outx2[:,0],color='royalblue',label='resonator (subAH)')

axs[0].legend(loc='upper left')
axs[1].legend(loc='upper left')


axs[2].plot(tvx2, Ipulses3(tvx2),color='gray',)
#xlim(330,360)
ylim(-6.5, 6.5)
axs[0].set_ylabel('Vm [mv]')
axs[1].set_ylabel('Vm [mv]')

ylabel('Iapp [a.u.]')
xlabel('time [ms]')
#axs[1].set_ylim(-60,-50)
savefig('outputs/integrator-vs-resonator-inapk.svg')
#savefig('outputs/integrator-vs-resonator-inapk.p')
```

# Slow modulation

```python
snic_neuron.I_bias = 0
```

```python
i_stim =   make_pulses(tx, 8, amp=5, period=100, width=50,steep=0.2)
```

```python
plot(tx, i_stim(tx))
xlabel('time, ms'); ylabel('Stim current, a.u.')
```

```python
init_state = [-60, 0.01]
```

```python
def plot_with_i_dyn(neuron, init_tate, stim,tstop=1000,title=""):
    t,traj = adams_bashforth(neuron, init_state, tstop=tstop, fnkwargs=dict(I_dyn=stim))

    figure(figsize=(14,6))

    ax1 = subplot2grid((4,1), (0,0), rowspan=3)
    ax2 = subplot2grid((4,1), (3,0))

    ax1.plot(t, traj[:,0],'royalblue')
    setp(ax1, ylim=(-80,15), xticklabels=[], ylabel='membrane potential [mV]')
    
    ystim = stim(t) + neuron.I_bias
    ax2.plot(t, ystim,'gray')
    
    stim_range = abs(amax(ystim)-amin(ystim))
    ylim = (amin(ystim)-0.1*stim_range, amax(ystim)+0.1*stim_range)
    setp(ax2, ylim=ylim, xlabel='time [ms]', ylabel='I stim [a.u.]')

    ax1.set_title(title)
    #ax1.set_title()
```

```python
snic_neuron.I_bias=3
plot_with_i_dyn(snic_neuron, init_state, i_stim, title="Forced bursting in Ina,p+Ik model near SNIC")
```

```python
snic_neuron.I_bias=3
plot_with_i_dyn(snic_neuron, init_state, i_stim,title="",tstop=500)
for ax in gcf().axes:
    ax.set_xlim(280,380)
    
for ax in gcf().axes:
    ax.set_xlim(280,380)
    
setp(gcf(), size_inches=(8,8))
```

```python
snoff_neuron.I_bias=0
plot_with_i_dyn(snoff_neuron, init_state, i_stim,
                title="Forced bursting in Ina,p+Ik model near SN off limit cycle")
```

```python
snoff_neuron.I_bias=0
plot_with_i_dyn(snoff_neuron, init_state, i_stim,
                title="")

for ax in gcf().axes:
    ax.set_xlim(280,380)
    
setp(gcf(), size_inches=(8,8))
```

```python
supAH2_neuron.I_bias=10

_,traj = adams_bashforth(subAH2_neuron, [-50.6,0.2459],tstop=200)
init_state_supAH = traj[-1]
print( init_state_supAH)

plot_with_i_dyn(supAH2_neuron, init_state_supAH, 
                usp(tx,7*i_stim(tx)+0.05*randn(len(tx)),s=0),
                title="Forced bursting in Ina,p+Ik model near supAH")
```

```python
plot_with_i_dyn(supAH2_neuron, init_state_supAH, 
                usp(tx,7*i_stim(tx)+0.05*randn(len(tx)),s=0),
                title="")

for ax in gcf().axes:
    ax.set_xlim(280,380)
    
setp(gcf(), size_inches=(8,8))
```

```python
subAH2_neuron.I_bias=40

_,traj = adams_bashforth(subAH2_neuron, [-50.6,0.2459],tstop=200)
init_state_subAH = traj[-1]
print( init_state_subAH)

plot_with_i_dyn(subAH2_neuron, init_state_subAH, 
                usp(tx,5*i_stim(tx)+0.05*randn(len(tx)),s=0),
                title="Forced bursting in Ina,p+Ik model near subAH")
```

```python
plot_with_i_dyn(subAH2_neuron, init_state_subAH, 
                usp(tx,5*i_stim(tx)+0.05*randn(len(tx)),s=0),
                title="")

for ax in gcf().axes:
    ax.set_xlim(280,380)
    
setp(gcf(), size_inches=(8,8))
```

# Response to noise

```python
tx[-1]
```

```python
tx = arange(0,100*1000,0.1)
i_stim_noise = usp(tx[::10],randn(len(tx[::10])),s=0)
```

```python
snoff_neuron.I_bias = 3.9
_,traj = adams_bashforth(snoff_neuron, init_state,tstop=200)
init_state_snoff = traj[-1]


print (init_state_snoff)
```

```python

plot_with_i_dyn(snoff_neuron, init_state_snoff, i_stim_noise,tstop=1e4-1,
                title='noise response near SN (off limit cycle)')
#gcf().axes[1].set_ylim(2,6)
#for ax in gcf().axes:
#    ax.set_xlim(350,450)

```

```python
%%time 

snic_neuron.I_bias=3.9

_,traj = adams_bashforth(snic_neuron, init_state,tstop=200)
init_state_snic = traj[-1]


plot_with_i_dyn(snic_neuron, init_state_snic, i_stim_noise,tstop=10000-1,
                title='noise response near SNIC')
#gcf().axes[1].set_ylim(2,6)
```

```python
%time t,traj = adams_bashforth(snic_neuron, init_state,fnkwargs=dict(I_dyn=i_stim_noise),tstop=1e5-1)

dt = t[1]-t[0]
spikes = locmax(traj[:,0])*dt
isi = diff(spikes)
```

```python
figure(figsize=(16,3))
plot(t,traj[:,0],color='gray')
plot(spikes, ones(len(spikes)), 'r|',markersize=10)
xlim(4e4,5e4)
```

```python
amax(isi)
```

```python
_ = hist(isi, 100, range=(0,800))
```

```python
subAH_neuron.I_bias=21.2

_,traj = adams_bashforth(subAH_neuron, init_state,tstop=200)
init_state_subAH = traj[-1]
print( init_state_subAH)

plot_with_i_dyn(subAH_neuron, init_state_subAH, i_stim_noise,tstop=1e4-1,
                title='noise response near subAH [?]')
#gcf().axes[1].set_ylim(2,6)
```

```python
subAH2_neuron.I_bias=45.

_,traj = adams_bashforth(subAH2_neuron, [-50.6,0.2459],tstop=200)
init_state_subAH = traj[-1]
print (init_state_subAH)

plot_with_i_dyn(subAH2_neuron, init_state_subAH, i_stim_noise,tstop=1e4-1,
                title='noise response near subAH')
#gcf().axes[1].set_ylim(2,6)
```

```python
supAH2_neuron.I_bias=38

_,traj = adams_bashforth(supAH2_neuron, init_state,tstop=200)
init_state_supAH = traj[-1]
print( init_state_supAH)


plot_with_i_dyn(supAH2_neuron, init_state_supAH, i_stim_noise,tstop=1e4-1,
                title='noise response near supAH')
```

**Exercise:** Build interspike interval historgrams for all bifurcation types.


## Response to steps

```python
#from imfun import ui
```

```python
def calc_steps(neuron, modifier = None):
    acc = []
    Vrest = -65
    if modifier is None:
        modifier={}
    iamps = Ibias_bif*1.01 + logspace(0, 6,5,base=2)-1
    
    ib = neuron.I_bias
    neuron.I_bias = 0
    

    for iamp in iamps:
        #print iamp
        Iapp = usp(tx, I_pulse(tx, iamp, 50, 550),s=0)
        pars = dict(I_dyn=Iapp)
        nrest = neuron.ninf(Vrest)
        st_init = array([Vrest, nrest])
        tvx, outx = adams_bashforth(neuron, st_init, fnkwargs=pars, tstop=600)
        acc.append(outx[:,0])
    
    neuron.I_bias = ib
    return tvx,array(acc),iamps

    
               
```

```python
def plot_coll(vecs,x=None,sep=None,positions=None,colors=None,
              ax = None,
              figsize=None,
              frame_on=False,
              labels = None,
              xshift=0,
              fill_alpha=0.85,
              line_color='w',
              do_fill = False,
              **kwargs):


    if sep is None:
        mean_range = np.mean([np.max(v)-np.min(v) for v in vecs])
        sep = 0.05*mean_range

    if colors is None: colors = 'b'
    if labels is None: labels = [None]*len(vecs)
    if isinstance(colors, str):
        c = colors
        colors = (c for i in range(int(1e6)))
    if positions is None:
        prevpos,positions = 0,[0]
        ranges = [(v.min(),v.max()) for v in vecs]
        for r,rnext,v in zip(ranges, ranges[1:],vecs):
            pos = prevpos + r[1] + sep -np.min(rnext[0])
            positions.append(pos)
            prevpos = pos
    Lmin = np.min(list(map(len, vecs)))
    Lmax = np.max(list(map(len, vecs)))
    if x is None:
        x = np.arange(Lmax)
    else:
        if len(x) > Lmax:
            x = x[:Lmax]
        else:
            x = np.pad(x, (0, Lmax-len(x)), mode='linear_ramp')
    if ax is None:
        f,ax = plt.subplots(1,1,figsize=figsize)

    zorder = 0
    for v,p,c,l in zip(vecs,positions[::-1],colors,labels):
        zorder += 1
        if len(v) < Lmax:
            vpadded = np.pad(v, (0, Lmax-len(v)), mode='constant')
        else:
            vpadded = v
        ax.plot(x + xshift*zorder, vpadded+p, color=line_color, label=l,zorder=zorder, **kwargs)
        if do_fill:
            ax.fill_between(x + xshift*zorder, p, vpadded+p, color=c, alpha=fill_alpha,zorder=zorder )
        #a.axhline(p, color='b')
    plt.setp(ax, yticks=[],frame_on=frame_on)
    ax.axis('tight')
    return ax, positions

```

```python
tvx,resp_1,iamps = calc_steps(snic_neuron)
tvx,resp_2,iamsp = calc_steps(supAH_neuron)
```

```python
#f, axs = subplots(2,1,sharex=True,figsize=(16,12))

figure(figsize=(16,9))
ax1 = plt.subplot2grid((4,2), (0,0), rowspan=3)
ax2 = plt.subplot2grid((4,2),(3,0),)

ax3 = plt.subplot2grid((4,2), (0,1), rowspan=3)
ax4 = plt.subplot2grid((4,2),(3,1),)


labels=['%1.1f'%i for i in iamps]

plot_coll(resp_1[::-1], x=tvx,ax=ax1,frame_on=True,line_color='orange',sep=50, do_fill=False)
plot_coll(resp_2[::-1], x=tvx,ax=ax3,frame_on=True,line_color='royalblue',sep=70, do_fill=False)

setp(ax1, ylim = (-150, 600), ylabel='Vm [mV]', title='Class 1 response')
setp(ax3, ylim = (-150, 500), title='Class 2 response')

setp(ax2, xlabel='time [ms]', ylabel = 'Iapp [a.u.]')

#ax1.legend()

for line,label in zip(ax1.lines,labels[::-1]):
    yl = line.get_ydata()[0]
    ax1.text(5, yl+10, label)

for line,label in zip(ax3.lines,labels[::-1]):
    yl = line.get_ydata()[0]
    ax3.text(5, yl+10, label)
    

for i in iamps:
    y = usp(tx, I_pulse(tx, i, 50, 559),s=0)
    
    ax2.plot(tvx, y(tvx),color=(0.1,0.1,0.1))
    ax4.plot(tvx, y(tvx),color=(0.1,0.1,0.1))
#ylim(-1,55)

savefig('outputs/class1-2-responses-inapk.png')
savefig('outputs/class1-2-responses-inapk.svg')
```

```python
def find_spiking_freq(neuron,Iv,T_cut = 500):
    acc = []
    _,traj = adams_bashforth(neuron, [-60,0.5], tstop=100)
    init_state = traj[-1]
    ib = neuron.I_bias
    for i in Iv:
        neuron.I_bias = i
        tvx_, outx_ = adams_bashforth(neuron, init_state,tstop=1000)
        v = outx_[:,0]
        kk = locmax(v)
        tlocs = array([tvx_[k] for k in kk])
        
        if len(tlocs) and any(tlocs>T_cut):
            ff = 1000./mean(diff(tlocs[tlocs>T_cut]))
        else:
            ff = 0
        #print i, len(tlocs), min(tlocs),max(tlocs)
        acc.append(ff)
    neuron.I_bias = ib
    return array(acc)
    
```

```python
tx = linspace(0,1000,200)
Iv = 4 + 0.005*(tx-50)*I_pulse(tx, 1, 50, 4000)
%time ffv = find_spiking_freq(snic_neuron, Iv)
```

```python
plot(Iv, ffv)
```

```python
snic_neuron.I_bias = 0
```

```python
f, axs = subplots(3,1,figsize=(12,6), sharex=True)


tvx, outx = adams_bashforth(snic_neuron, [-65, 0.5],tstop=100)
init_state = outx[-1]

tvx, outx = adams_bashforth(snic_neuron, init_state, tstop=1000, fnkwargs=dict(I_dyn=usp(tx, Iv,s=0)))

V = outx[:,0]
axs[0].plot(tvx, V, color='black')
axs[1].plot(tx,Iv)

tlocs = [tvx[k] for k in locmax(V)]
ffv2 = 1000./diff(tlocs)


axs[2].plot(tlocs[:-1], ffv2,'-',color='skyblue',lw=2)
ylim(0,150)
```

```python

```

```python
plot(Iv, ffv,'.',color='skyblue',)
#axs[2].plot(tlocs[:-1], ffv2,'.',color='skyblue',)
ylim(0,300)
xlabel('applied current')
ylabel('spiking frequency')
title("F-I curve, type 1 excitability")
savefig('outputs/F-I_curve_class1-inapk.svg')
```

```python
#pars = dict(I_bias=6,I_kramp=0.05)

Iv2 = 4 + 0.05*(tx-10)*I_pulse(tx, 1, 10, 4000)

supAH_neuron.I_bias = 0

stim = usp(tx, Iv2+0.05*randn(len(tx)),s=0)

tvx1, outx1 = adams_bashforth(snic_neuron, init_state, fnkwargs=dict(I_dyn=stim),tstop=1000)
tvx2, outx2 = adams_bashforth(supAH_neuron, init_state, fnkwargs=dict(I_dyn=stim),tstop=1000)
```

```python
f, axs = subplots(3,1,figsize=(12,6), sharex=True)

V = outx2[:,0]
axs[0].plot(tvx2, V, color='black')
axs[1].plot(tx,Iv2)

axs[0].set_ylabel('V [mV]')
axs[1].set_ylabel('I app, a.u.')
axs[2].set_ylabel('F [Hz]')



tlocs = [tvx2[k] for k in locmax(V,-30)]
print (len(tlocs))
ffvx = 1000./diff(tlocs)


axs[2].plot(tlocs[:-1], ffvx,'-',color='skyblue',lw=2)
ylim(0,300)
```

```python
%time ffv2 = find_spiking_freq(supAH_neuron, Iv2)
%time ffv3 = find_spiking_freq(snic_neuron, Iv2)
```

```python
plot(Iv2, ffv2,'.',color='skyblue',)
#axs[2].plot(tlocs[:-1], ffv2,'.',color='skyblue',)
ylim(0,300)
xlabel('applied current')
ylabel('spiking frequency')
title("F-I curve, type 2 excitability")
savefig('outputs/F-I_curve_class2-inapk.svg')
```

```python
f, axs = subplots(1,2,sharey=True,sharex=True,figsize=(12,6))

axs[0].plot(Iv2, ffv3,'.',color='skyblue',)
axs[1].plot(Iv2, ffv2,'.',color='skyblue',)
#axs[2].plot(tlocs[:-1], ffv2,'.',color='skyblue',)
ylim(0,300)
axs[0].set_xlabel('applied current [a.u.]')
axs[1].set_xlabel('applied current [a.u.]')

axs[0].set_ylabel('spiking frequency [Hz]')
axs[0].set_title("class 1 excitability")
axs[1].set_title("class 2 excitability")
savefig('outputs/F-I_curves_both-inapk.svg')
```

```python

```

```python

```

```python

```
