---
jupyter:
  jupytext:
    formats: ipynb,md
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

<!-- #region id="mnbSq-yJWeEB" -->


# Введение в среду моделирования Brian2
Основная часть скопирована из туториала Brian2.

Сначала нам нужно выяснить, в колабе ли запущен ноутбук, и если в колабе, то нужно доустановить некоторые библиотеки.

[идея и другие варианты — отсюда](https://stackoverflow.com/questions/53581278/test-if-notebook-is-running-on-google-colab)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="S9A_YFGwV_GA" outputId="60c0d76f-d78c-4f91-c325-ac0fef435a31"
RunningInCOLAB = 'google.colab' in str(get_ipython())
RunningInCOLAB
```

```python colab={"base_uri": "https://localhost:8080/"} id="4e9277cb" outputId="ad277350-3e38-463f-b690-97d85205c43e"
if RunningInCOLAB:
  ! pip install brian2
  ! pip install pandas
  !wget https://raw.githubusercontent.com/abrazhe/nbpc-masters-ru/master/notebooks/input_factory.py
```

<!-- #region heading_collapsed=false level=1 id="c3ace824" -->
# Introduction to Brian part 1: Neurons
## (Взято из документации к Brian)
<!-- #endregion -->

<!-- #region id="d4227dcf" -->
All Brian scripts start with the following. If you're trying this notebook out in the Jupyter notebook, you should start by running this cell.
<!-- #endregion -->

```python id="4b471351"
from brian2 import *
```

<!-- #region id="078f908a" -->
Later we'll do some plotting in the notebook, so we activate inline plotting in the notebook by doing this:
<!-- #endregion -->

```python id="9b433d6a"
%matplotlib inline
```

```python id="358c71e0"
rc('figure', dpi=150)
```

<!-- #region id="01a56e34" -->
If you are not using the Jupyter notebook to run this example (e.g. you are using a standard Python terminal, or you copy&paste these example into an editor and run them as a script), then plots will not automatically be displayed. In this case, call the ``show()`` command explicitly after the plotting commands.
<!-- #endregion -->

<!-- #region heading_collapsed=false level=2 id="b281fae8" -->
## Units system

Brian has a system for using quantities with physical dimensions:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 37} id="f0593d13" outputId="abba93ad-c5ae-48b5-bcde-baf799d119dc"
siemens
```

```python colab={"base_uri": "https://localhost:8080/", "height": 37} id="3e267e86" outputId="4e05ebc3-b69d-4bfc-ade1-bf904aed79bf"
20*volt
```

<!-- #region id="07fd2961" -->
All of the basic SI units can be used (volt, amp, etc.) along with all the standard prefixes (m=milli, p=pico, etc.), as well as a few special abbreviations like ``mV`` for millivolt, ``pF`` for picofarad, etc.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 37} id="25001b9f" outputId="4ee25856-d508-4526-8590-30e546190d4b"
1000*amp
```

```python colab={"base_uri": "https://localhost:8080/", "height": 37} id="58b04859" outputId="1b00ed4f-e69b-45fe-8036-93e33b86f39b"
1e6*volt
```

```python colab={"base_uri": "https://localhost:8080/", "height": 38} id="edab1f7b" outputId="2a060ad7-188c-4329-d9b9-70e79ec83d91"
1000*namp
```

<!-- #region id="6e710f46" -->
Also note that combinations of units with work as expected:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 37} id="3761b7fe" outputId="1889f0cd-c408-475d-a143-93159ec50eac"
10*nA*5*Mohm
```

<!-- #region id="ef347ba7" -->
And if you try to do something wrong like adding amps and volts, what happens?
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 304} id="8d1a0a0b" outputId="10066f12-76cd-41a4-afe2-c5f12be86d14"
5*amp+10*volt
```

<!-- #region id="f3680516" -->
If you haven't see an error message in Python before that can look a bit overwhelming, but it's actually quite simple and it's important to know how to read these because you'll probably see them quite often.

You should start at the bottom and work up. The last line gives the error type ``DimensionMismatchError`` along with a more specific message (in this case, you were trying to add together two quantities with different SI units, which is impossible).

Working upwards, each of the sections starts with a filename (e.g. ``C:\Users\Dan\...``) with possibly the name of a function, and then a few lines surrounding the line where the error occurred (which is identified with an arrow).

The last of these sections shows the place in the function where the error actually happened. The section above it shows the function that called that function, and so on until the first section will be the script that you actually run. This sequence of sections is called a traceback, and is helpful in debugging.

If you see a traceback, what you want to do is start at the bottom and scan up the sections until you find your own file because that's most likely where the problem is. (Of course, your code might be correct and Brian may have a bug in which case, please let us know on the email support list.)
<!-- #endregion -->

<!-- #region heading_collapsed=false level=2 id="abf2bf04" -->
## A simple model

Let's start by defining a simple neuron model. In Brian, all models are defined by systems of differential equations. Here's a simple example of what that looks like:
<!-- #endregion -->

```python id="290bb961"
tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''
```

<!-- #region id="9ae2e818" -->
In Python, the notation ``'''`` is used to begin and end a multi-line string. So the equations are just a string with one line per equation. The equations are formatted with standard mathematical notation, with one addition. At the end of a line you write ``: unit`` where ``unit`` is the SI unit of that variable.
Note that this is not the unit of the two sides of the equation (which would be ``1/second``), but the unit of the *variable* defined by the equation, i.e. in this case $v$.

Now let's use this definition to create a neuron.
<!-- #endregion -->

```python id="7394533b"
G = NeuronGroup(1, eqs)
```

<!-- #region id="c36d1b22" -->
In Brian, you only create groups of neurons, using the class ``NeuronGroup``. The first two arguments when you create one of these objects are the number of neurons (in this case, 1) and the defining differential equations.

Let's see what happens if we didn't put the variable ``tau`` in the equation:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 446} id="52c6def0" outputId="f93b8de0-53cc-40d4-ce92-d48bcedee67c"
eqs = '''
dv/dt = 1-v : 1
'''
G = NeuronGroup(1, eqs)
run(100*ms)
```

<!-- #region id="4700f945" -->
An error is raised, but why? The reason is that the differential equation is now dimensionally inconsistent. The left hand side ``dv/dt`` has units of ``1/second`` but the right hand side ``1-v`` is dimensionless. People often find this behaviour of Brian confusing because this sort of equation is very common in mathematics. However, for quantities with physical dimensions it is incorrect because the results would change depending on the unit you measured it in. For time, if you measured it in seconds the same equation would behave differently to how it would if you measured time in milliseconds. To avoid this, we insist that you always specify dimensionally consistent equations.

Now let's go back to the good equations and actually run the simulation.
<!-- #endregion -->

```python id="f6636e02"
start_scope()

tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''

G = NeuronGroup(1, eqs)
run(100*ms)
```

<!-- #region level=7 id="423215a3" -->
First off, ignore that ``start_scope()`` at the top of the cell. You'll see that in each cell in this tutorial where we run a simulation. All it does is make sure that any Brian objects created before the function is called aren't included in the next run of the simulation.

Secondly, you'll see that there is an "INFO" message about not specifying the numerical integration method. This is harmless and just to let you know what method we chose, but we'll fix it in the next cell by specifying the method explicitly.

So, what has happened here? Well, the command ``run(100*ms)`` runs the simulation for 100 ms. We can see that this has worked by printing the value of the variable ``v`` before and after the simulation.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="cdd92355" outputId="55dadcc2-e234-4683-e1fb-0b85695784eb"
start_scope()

G = NeuronGroup(1, eqs, method='linear')
print('Before v = %s' % G.v[0])
run(100*ms)
print('After v = %s' % G.v[0])
```

<!-- #region id="17d2726a" -->
By default, all variables start with the value 0. Since the differential equation is ``dv/dt=(1-v)/tau`` we would expect after a while that ``v`` would tend towards the value 1, which is just what we see. Specifically, we'd expect ``v`` to have the value ``1-exp(-t/tau)``. Let's see if that's right.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="f88010b7" outputId="12678f8c-26e0-49d5-dde5-76225befd39e"
print('Expected value of v = %s' % (1-exp(-100*ms/tau)))
```

<!-- #region id="3508e5fb" -->
Good news, the simulation gives the value we'd expect!

Now let's take a look at a graph of how the variable ``v`` evolves over time.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 664} id="d305740a" outputId="a5178db8-2d95-4255-9e17-20cf536c38d0"
start_scope()

G = NeuronGroup(1, eqs, method='linear')
M = StateMonitor(G, 'v', record=True)

run(30*ms)

plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v');
```

<!-- #region id="7e9e5f9e" -->
This time we only ran the simulation for 30 ms so that we can see the behaviour better. It looks like it's behaving as expected, but let's just check that analytically by plotting the expected behaviour on top.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 664} id="faca38ad" outputId="6a1d5aa7-b4eb-4e0a-f6a3-68ca9d4bfd9a"
start_scope()

G = NeuronGroup(1, eqs, method='linear')
M = StateMonitor(G, 'v', record=0)

run(30*ms)

plot(M.t/ms, M.v[0], 'C0', label='Brian')
plot(M.t/ms, 1-exp(-M.t/tau), 'C1--',label='Analytic')
xlabel('Time (ms)')
ylabel('v')
legend();
```

<!-- #region id="d93f3a2f" -->
As you can see, the blue (Brian) and dashed orange (analytic solution) lines coincide.

In this example, we used the object ``StateMonitor`` object. This is used to record the values of a neuron variable while the simulation runs. The first two arguments are the group to record from, and the variable you want to record from. We also specify ``record=0``. This means that we record all values for neuron 0. We have to specify which neurons we want to record because in large simulations with many neurons it usually uses up too much RAM to record the values of all neurons.

Now try modifying the equations and parameters and see what happens in the cell below.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 701} id="05a3fccd" outputId="4e666292-9f9d-462c-da78-5a822973252d"
start_scope()

tau = 10*ms
eqs = '''
dv/dt = (sin(2*pi*100*Hz*t)-v)/tau : 1
'''

# Change to Euler method because exact integrator doesn't work here
G = NeuronGroup(1, eqs, method='euler')
M = StateMonitor(G, 'v', record=0)

G.v = 5 # initial value

run(60*ms)

plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v');
```

<!-- #region heading_collapsed=false level=2 id="af8c22cf" -->
## Adding spikes

So far we haven't done anything neuronal, just played around with differential equations. Now let's start adding spiking behaviour.
<!-- #endregion -->

```python id="fa1fb122"
NeuronGroup?
```

```python colab={"base_uri": "https://localhost:8080/", "height": 664} id="e809bee7" outputId="f273b3d1-55b5-4f74-c995-a61dbd784f38"
start_scope()

tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1
'''

G = NeuronGroup(1, eqs, threshold='v>0.85', reset='v = 0', method='linear')

M = StateMonitor(G, 'v', record=0)
run(50*ms)
plot(M.t/ms, M.v[0])
xlabel('Time (ms)')
ylabel('v');
```

<!-- #region id="492c6a8d" -->
We've added two new keywords to the ``NeuronGroup`` declaration: ``threshold='v>0.8'`` and ``reset='v = 0'``. What this means is that when ``v>0.8`` we fire a spike, and immediately reset ``v = 0`` after the spike. We can put any expression and series of statements as these strings.

As you can see, at the beginning the behaviour is the same as before until ``v`` crosses the threshold ``v>0.8`` at which point you see it reset to 0. You can't see it in this figure, but internally Brian has registered this event as a spike. Let's have a look at that.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="7c6324cd" outputId="c3b24094-b19b-4381-c1ea-0757b144e1b9"
start_scope()

G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', method='linear')

spikemon = SpikeMonitor(G)

run(50*ms)

print('Spike times: %s' % spikemon.t[:])
```

<!-- #region id="b0ec3d11" -->
The ``SpikeMonitor`` object takes the group whose spikes you want to record as its argument and stores the spike times in the variable ``t``. Let's plot those spikes on top of the other figure to see that it's getting it right.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 664} id="3c65b0ba" outputId="9e740627-c6c7-4e76-f62e-3c1587110638"
start_scope()

G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', method='linear')

statemon = StateMonitor(G, 'v', record=0)
spikemon = SpikeMonitor(G)

run(50*ms)

plot(statemon.t/ms, statemon.v[0])
for ti in spikemon.t:
    axvline(ti/ms, ls='--', lw=3,color='C3')
xlabel('Time (ms)')
ylabel('v');
```

<!-- #region id="d95f6d4f" -->
Here we've used the ``axvline`` command from ``matplotlib`` to draw an orange, dashed vertical line at the time of each spike recorded by the ``SpikeMonitor``.

Now try changing the strings for ``threshold`` and ``reset`` in the cell above to see what happens.
<!-- #endregion -->

<!-- #region heading_collapsed=false level=2 id="65750e42" -->
## Refractoriness

A common feature of neuron models is refractoriness. This means that after the neuron fires a spike it becomes refractory for a certain duration and cannot fire another spike until this period is over. Here's how we do that in Brian.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 664} id="7e8f9d76" outputId="9740cb91-ecf6-48cb-eff6-418c06037013"
start_scope()

tau = 10*ms
eqs = '''
dv/dt = (1-v)/tau : 1 (unless refractory)
'''

G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', refractory=5*ms, method='linear')

statemon = StateMonitor(G, 'v', record=0)
spikemon = SpikeMonitor(G)

run(50*ms)

plot(statemon.t/ms, statemon.v[0])
for t in spikemon.t:
    axvline(t/ms, ls='--', c='C1', lw=3)
xlabel('Time (ms)')
ylabel('v');
```

<!-- #region id="d5d89288" -->
As you can see in this figure, after the first spike, ``v`` stays at 0 for around 5 ms before it resumes its normal behaviour. To do this, we've done two things. Firstly, we've added the keyword ``refractory=5*ms`` to the ``NeuronGroup`` declaration. On its own, this only means that the neuron cannot spike in this period (see below), but doesn't change how ``v`` behaves. In order to make ``v`` stay constant during the refractory period, we have to add ``(unless refractory)`` to the end of the definition of ``v`` in the differential equations. What this means is that the differential equation determines the behaviour of ``v`` unless it's refractory in which case it is switched off.

Here's what would happen if we didn't include ``(unless refractory)``. Note that we've also decreased the value of ``tau`` and increased the length of the refractory period to make the behaviour clearer.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 681} id="33899d14" outputId="721ebd85-7572-406c-c83d-debc4d61b1aa"
start_scope()

tau = 5*ms
eqs = '''
dv/dt = (1-v)/tau : 1 (unless refractory)
'''

G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', refractory=2*ms, method='linear')

statemon = StateMonitor(G, 'v', record=0)
spikemon = SpikeMonitor(G)

run(50*ms)

plot(statemon.t/ms, statemon.v[0])
for t in spikemon.t:
    axvline(t/ms, ls='--', c='C1', lw=3)
axhline(0.8, ls=':', c='C2', lw=3)
xlabel('Time (ms)')
ylabel('v')
print("Spike times: %s" % spikemon.t[:])
```

<!-- #region id="565b63df" -->
So what's going on here? The behaviour for the first spike is the same: ``v`` rises to 0.8 and then the neuron fires a spike at time 8 ms before immediately resetting to 0. Since the refractory period is now 15 ms this means that the neuron won't be able to spike again until time 8 + 15 = 23 ms. Immediately after the first spike, the value of ``v`` now instantly starts to rise because we didn't specify ``(unless refractory)`` in the definition of ``dv/dt``. However, once it reaches the value 0.8 (the dashed green line) at time roughly 8 ms it doesn't fire a spike even though the threshold is ``v>0.8``. This is because the neuron is still refractory until time 23 ms, at which point it fires a spike.

Note that you can do more complicated and interesting things with refractoriness. See the full documentation for more details about how it works.
<!-- #endregion -->

<!-- #region heading_collapsed=false level=2 id="08ba033e" -->
## Multiple neurons

So far we've only been working with a single neuron. Let's do something interesting with multiple neurons.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 664} id="5e344c80" outputId="eff99a4f-d5e6-4fc0-abdb-3cdee53d293a"
start_scope()

N = 100
tau = 10*ms
eqs = '''
dv/dt = (2-v)/tau : 1
'''

G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='linear')
G.v = 'rand()'

spikemon = SpikeMonitor(G)

run(50*ms)

plot(spikemon.t/ms, spikemon.i, '|k')
xlabel('Time (ms)')
ylabel('Neuron index');
```

<!-- #region id="f4579269" -->
This shows a few changes. Firstly, we've got a new variable ``N`` determining the number of neurons. Secondly, we added the statement ``G.v = 'rand()'`` before the run. What this does is initialise each neuron with a different uniform random value between 0 and 1. We've done this just so each neuron will do something a bit different. The other big change is how we plot the data in the end.

As well as the variable ``spikemon.t`` with the times of all the spikes, we've also used the variable ``spikemon.i`` which gives the corresponding neuron index for each spike, and plotted a single black dot with time on the x-axis and neuron index on the y-value. This is the standard "raster plot" used in neuroscience.
<!-- #endregion -->

<!-- #region heading_collapsed=false level=2 id="5f5ae2ce" -->
## Parameters

To make these multiple neurons do something more interesting, let's introduce per-neuron parameters that don't have a differential equation attached to them.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 477} id="edd8d478" outputId="511f7f8f-3ad2-45d1-bceb-9bc7a7d1e964"
start_scope()

N = 100
tau = 10*ms
v0_max = 3.
duration = 1000*ms

eqs = '''
dv/dt = (v0-v)/tau : 1 (unless refractory)
v0 : 1
'''

G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=5*ms, method='linear')
M = SpikeMonitor(G)

G.v0 = 'i*v0_max/(N-1)'

run(duration)

figure(figsize=(12,4))
subplot(121)
plot(M.t/ms, M.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
subplot(122)
plot(G.v0, M.count/duration)
xlabel('v0')
ylabel('Firing rate (sp/s)');
```

<!-- #region id="40c9cb48" -->
The line ``v0 : 1`` declares a new per-neuron parameter ``v0`` with units ``1`` (i.e. dimensionless).

The line ``G.v0 = 'i*v0_max/(N-1)'`` initialises the value of v0 for each neuron varying from 0 up to ``v0_max``. The symbol ``i`` when it appears in strings like this refers to the neuron index.

So in this example, we're driving the neuron towards the value ``v0`` exponentially, but when ``v`` crosses ``v>1``, it fires a spike and resets. The effect is that the rate at which it fires spikes will be related to the value of ``v0``. For ``v0<1`` it will never fire a spike, and as ``v0`` gets larger it will fire spikes at a higher rate. The right hand plot shows the firing rate as a function of the value of ``v0``. This is the I-f curve of this neuron model.

Note that in the plot we've used the ``count`` variable of the ``SpikeMonitor``: this is an array of the number of spikes each neuron in the group fired. Dividing this by the duration of the run gives the firing rate.
<!-- #endregion -->

<!-- #region heading_collapsed=false level=2 id="289ee295" -->
## Stochastic neurons

Often when making models of neurons, we include a random element to model the effect of various forms of neural noise. In Brian, we can do this by using the symbol ``xi`` in differential equations. Strictly speaking, this symbol is a "stochastic differential" but you can sort of thinking of it as just a Gaussian random variable with mean 0 and standard deviation 1. We do have to take into account the way stochastic differentials scale with time, which is why we multiply it by ``tau**-0.5`` in the equations below (see a textbook on stochastic differential equations for more details).
Note that we also changed the ``method`` keyword argument to use ``'euler'`` (which stands for the [Euler-Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method)); the ``'linear'`` method that we used earlier is not applicable to stochastic differential equations.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 477} id="56f343f6" outputId="d1973c93-2a13-43be-e362-8a0eeb7203ad"
start_scope()

N = 100
tau = 10*ms
v0_max = 3.
duration = 1000*ms
sigma = 0.2

eqs = '''
dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
v0 : 1
'''

G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', refractory=5*ms, method='euler')
M = SpikeMonitor(G)

G.v0 = 'i*v0_max/(N-1)'

run(duration)

figure(figsize=(12,4))
subplot(121)
plot(M.t/ms, M.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
subplot(122)
plot(G.v0, M.count/duration)
xlabel('v0')
ylabel('Firing rate (sp/s)');
```

<!-- #region id="26642034" -->
That's the same figure as in the previous section but with some noise added. Note how the curve has changed shape: instead of a sharp jump from firing at rate 0 to firing at a positive rate, it now increases in a sigmoidal fashion. This is because no matter how small the driving force the randomness may cause it to fire a spike.
<!-- #endregion -->

<!-- #region heading_collapsed=false level=2 id="8dc0f29f" -->
## End of tutorial

That's the end of this part of the tutorial. The cell below has another example. See if you can work out what it is doing and why. Try adding a ``StateMonitor`` to record the values of the variables for one of the neurons to help you understand it.

You could also try out the things you've learned in this cell.

Once you're done with that you can move on to the next tutorial on Synapses.
<!-- #endregion -->

<!-- #region id="3b5b5a50" -->
# Adaptive threshold model

Based on the acquired knowledge,let's try to implement an IF neuron adaptive threshold.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 664} id="f60c1db2" outputId="d89d4374-3da4-410b-e471-b90341aaf8bc"
start_scope()

N = 1000
tau = 10*ms
vr = -70*mV
vt0 = -50*mV
delta_vt0 = 5*mV
tau_t = 100*ms
sigma = 0.5*(vt0-vr)
v_drive = 2*(vt0-vr)
duration = 100*ms

eqs = '''
dv/dt = (v_drive+vr-v)/tau + sigma*xi*tau**-0.5 : volt
dvt/dt = (vt0-vt)/tau_t : volt
'''

reset = '''
v = vr
vt += delta_vt0
'''

G = NeuronGroup(N, eqs, threshold='v>vt', reset=reset, refractory=5*ms, method='euler')
spikemon = SpikeMonitor(G)
stm = StateMonitor(G, ['v','vt'], record=True)

G.v = 'rand()*(vt0-vr)+vr'
G.vt = vt0

run(duration)

_ = hist(spikemon.t/ms, 100, histtype='stepfilled', facecolor='k', weights=ones(len(spikemon))/(N*defaultclock.dt))
xlabel('Time (ms)')
ylabel('Instantaneous firing rate (sp/s)');
```

```python colab={"base_uri": "https://localhost:8080/", "height": 512} id="2057f54d" outputId="cef6a0f6-c55a-4adf-dcbb-b882001377c1"
k = 100

f,(ax1,ax2)  = subplots(2,1, sharex=True, figsize=(12,4))

ax1.plot(stm.t/ms, stm.v[k]/mV, label='Vm')
ax2.plot(stm.t/ms, stm.vt[k]/mV)


tspikes = spikemon.t[spikemon.i == k]

Nspikes = len(tspikes)

ax1.plot(tspikes/ms, ones(Nspikes)*-35, 'r|', markersize=10, label='spikes')


ax1.set_title('%dth neuron'%k)
ax1.legend()

ax1.set_ylabel('Membrane\n potential, mV')
ax2.set_ylabel(u'Threshold\n potential')
xlabel('time, ms')
```
