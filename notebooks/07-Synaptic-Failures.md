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

# Соотношения доли переданной информации и энергозатраты (Монте-Карло)


По материалам статьи:

![image.png](attachment:495626ad-561b-4677-aa86-757f3b612579.png)

```python
%matplotlib inline
```

```python
from matplotlib import pyplot as plt
import matplotlib as mpl

from matplotlib.pyplot import colorbar, close, figure, gca, gcf, imshow, plot, setp, subplots
from matplotlib.pyplot import title, suptitle, ylabel, xlabel, xlim, ylim
from matplotlib.pyplot import legend

```

```python jupyter={"outputs_hidden": false}
plt.style.use(['ggplot','seaborn-v0_8-muted'])
mpl.rc('figure', figsize=(9,9*0.618), dpi=150)
```

```python

import numpy as np
from numpy import linalg


# import most often used functions
from numpy import array, arange, clip, ones, percentile, where, zeros
from numpy import exp, tanh, log, log2, linspace

from numpy.linalg import svd
from numpy.random import permutation, rand
```

```python
from numba import jit
```

<!-- #region -->
Взаимная информация:
\begin{equation}
I_m = -\sum_rP[s]\log_2P [s] + \sum_{r,s}P[r]P[s|r]\log_2 P[s|r]
\end{equation}


\begin{equation}
I_m = \sum_{s,r}P[r,s]\log_2\frac{P[r,s]}{P[r]P[s]}
\end{equation}
<!-- #endregion -->

## Один синапс на нейрон

```python jupyter={"outputs_hidden": false}
@jit(nopython=True)
def model_Hnoise(s,r,Ntrials=1e6):
    responses = zeros((2,2))
    for i in range(int(Ntrials)):
        if rand() < s: # spike arrived
            row = 1
            col = (rand() < r) and 1 or 0
        else :
            row,col = 0,0
        responses[row, col] +=1
    
    probs =  responses/Ntrials
    Prs = np.sum(probs, 0)
    Pss = np.sum(probs, 1)
    acc = 0
    for row in [0,1]:
        for col in [0,1]:
            Prs_joint = probs[row,col]
            # skipping zero probabilities to avoid numerical errors...
            if Prs_joint*Prs[col]*Pss[row] > 0:
                acc += Prs_joint*log2(Prs_joint/(Prs[col]*Pss[row]))
    Iinp = -sum(Pss*log2(Pss))
    return acc/Iinp # доля переданной информации!
            
```

```python jupyter={"outputs_hidden": false}
%time model_Hnoise(0.5,0.9,Ntrials=1e7)
```

```python jupyter={"outputs_hidden": false}
sx = 0.01
rv = linspace(0.01,1.0)
sv = [0.01, 0.1, 0.5, 0.95, 0.99]
%time yx = [[model_Hnoise(s,r, Ntrials=1e7) for r in rv] for s in sv]
```

```python jupyter={"outputs_hidden": false}
fig, axs = plt.subplots(1,2, figsize=(10,5))
for y,s in zip(yx,sv):
    axs[0].plot(rv, y, color='deepskyblue')
    axs[1].semilogy(rv, y/(s*rv), color='deepskyblue')


axs[0].text(0.4, 0.5, "s=0.01")
axs[0].text(0.7, 0.15, "s=0.99")


axs[1].text(0.2, 65, "s=0.01")
axs[1].text(0.2, 0.15, "s=0.99")


axs[0].set_title(u"Доля переданной информации",size=12)
axs[1].set_title(u"Доля переданной информации/энергия",size=12)
axs[0].set_xlabel(u'вероятность срабатывания синапса, r')
axs[1].set_xlabel(u'вероятность срабатывания синапса, r')
plt.suptitle(u"Моносинаптическая связь", size=16)
```

## Полисинаптическая связь со спонтанным выбросом

```python
@jit(nopython=True)
def model_Hnoise_poly(s,r,Nsyn=10, spont=0.003, Ntrials=1e6):
    "Здесь наличие хотя бы одного ВПСП рассматривается как передача события"
    responses = zeros((2,2))
    for i in range(int(Ntrials)):
        row, col = 0,0
        if rand() < s: # spike arrived
            row = 1
            # probability of at least one release:
            if rand() >= (1-r)**Nsyn:
                col = 1
        # handle spontaneous release
        if rand() >= (1-spont)**Nsyn:
            col = 1 
        responses[row, col] +=1
    #
    probs =  responses/Ntrials
    Prs = np.sum(probs, 0)
    Pss = np.sum(probs, 1)
    acc = 0
    for row in [0,1]:
        for col in [0,1]:
            Prs_joint = probs[row,col]
            # skipping zero probabilities to avoid numerical errors...
            if Prs_joint*Prs[col]*Pss[row] > 0:
                acc += Prs_joint*log2(Prs_joint/(Prs[col]*Pss[row]))
    Iinp = -sum(Pss*log2(Pss))
    return acc/Iinp # доля переданной информации!
```

```python jupyter={"outputs_hidden": false}
%time yx10 = [[model_Hnoise_poly(0.01,r,n,Ntrials=1e7) for r in rv] for n in [1,2,4,10]]
```

```python jupyter={"outputs_hidden": false}
fig, axs = plt.subplots(1,2, figsize=(10,6))
s = 0.01
for y,n in zip(yx10, [1,2,4,10]):
    axs[0].plot(rv, y, color='m')
    axs[1].plot(rv, y/(s*n*rv), color='m')

#axs[0].text(0.1, 0.85, "N=10")
axs[0].text(0.7, 0.45, "N=1")
    
axs[1].text(0.8, 85, "N=1")
axs[1].text(0.8, 2, "N=10")

axs[0].set_title(u"Доля переданной информации",size=12)
axs[1].set_title(u"Доля переданной информации/энергия",size=12)
axs[0].set_xlabel(u'вероятность срабатывания синапса, r')
axs[1].set_xlabel(u'вероятность срабатывания синапса, r')
plt.suptitle(u"Полисинаптическая связь и спонтанный выброс", size=16)
```

```python

```
