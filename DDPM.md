---
tags: []
parent: ""
collections:
    - DDPM
$version: 156
$libraryID: 1
$itemKey: SUGE2IKA

---
# Denoising Diffusion Probabilistic Model

## **前向加噪过程（Forward Process）**

扩散模型的前向过程，本质上是在做一件事：**把一张干净图像** $x_0$ **逐步加噪**，直到最后变成接近标准高斯噪声的 $x_T$。 。

### **（1）前提假设**

前向加噪过程是一个**马尔可夫链**：当前状态只依赖上一个状态，而不依赖更早的历史。这一特性可以写成

$$
q\left(x_t\mid x_{t-1},x_{t-2},\ldots,x_0\right)
= q\left(x_t\mid x_{t-1}\right)
$$

所以得到从 $x_0$ 加噪到任意状态 $x_t$ 的过程：

$$
\begin{aligned}
q\left(x_{1:T}\mid x_0\right)
&= \prod_{t=1}^{T} q\left(x_t\mid x_{t-1}\right),\\
q\left(x_t\mid x_{t-1}\right)
&= \mathcal{N}\left(x_t;\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right).
\end{aligned}
$$

第一个公式是分布的连乘，表示在已知原始图片 $x_0$ 的情况下，逐渐增加噪声，得到任意 $x_t$ 的过程。

第二个公式非常重要，它给出了每一步的加噪方式（**状态转移函数**）：

$$
x_t \sim \mathcal{N}\left(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right)
$$

它等价的采样写法是

$$
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\beta_t\epsilon_t,
\qquad \epsilon_t\sim\mathcal{N}(0,I)
$$

理解为：从 $x_{t-1}$ 到 $x_t$，先乘一个小于 1 的系数弱化原图，同时加入方差为 $\beta_t$ 的高斯噪声。

$\beta_t$ 是一个预先定义的噪声方差系数。不同 $t$ 的 $\beta_t$ 由时间 $1\sim T$ 逐渐递增，可以采用 Linear、Cosine 等调度，并满足

$$
\beta_1<\beta_2<\cdots<\beta_T.
$$

> **问题 ：** 为什么 $\beta_t$ 要逐渐增大？也就是说，噪声强度为什么要随时间逐渐增大？

简单理解：前期少加噪，让结构慢慢破坏；后期多加噪，让图像快速接近纯噪声。这样能保证初始阶段图像信息还比较完整，后面逐渐失真，最终到标准高斯。如果一开始就加太多噪声，信息会瞬间丢失，反向过程很难学。

> 不管原始图像 \(x_0\) 是什么，经过足够多步加噪后，得到的 \(x_T\) 都近似是标准高斯噪声。

这一步非常重要，因为它意味着最终噪声分布**不再依赖具体是哪一张原图**，最终加噪后的图像分布是一个标准正态分布，为后续生成图像做基础。

### **（2）前向过程的闭式表达式**

定义$\alpha_t=1-\beta_t,\alpha_t=\prod_{i=1}^{t}\alpha_i$，经过推导，可以得到 $x_t$ 与 $x_0$ 的关系：

$$
q\left(x_t\mid x_0\right)
= \mathcal{N}\left(x_t;\sqrt{\bar\alpha_t}\,x_0,\left(1-\bar\alpha_t\right)I\right).
\tag{1}
$$

随机取一个 $t$，然后直接用公式构造

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon.
\tag{1}
$$

这样就不需要逐步前向加噪。训练时也不需要模拟完整链条，只要知道某个 $t$ 对应的 $x_t$ 即可。

\*\*一步加噪，直接从$x_t$到 $x_0$ \*\*

### **（3）详细推导过程**

先定义$\alpha_t=1-\beta_t$，于是公式每一步可以写成

$$
x_t=\sqrt{\alpha_t}\,x_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t
$$

第一步：

$$
x_1=\sqrt{\alpha_1}\,x_0+\sqrt{1-\alpha_1}\,\epsilon_1
$$

第二步，展开：

$$
\begin{aligned}
x_2
&= \sqrt{\alpha_2}\,x_1+\sqrt{1-\alpha_2}\,\epsilon_2\\
&= \sqrt{\alpha_2}\left(\sqrt{\alpha_1}\,x_0+\sqrt{1-\alpha_1}\,\epsilon_1\right)
   +\sqrt{1-\alpha_2}\,\epsilon_2\\
&= \sqrt{\alpha_1\alpha_2}\,x_0
   +\alpha_2\sqrt{1-\alpha_1}\,\epsilon_1
   +\sqrt{1-\alpha_2}\,\epsilon_2.
\end{aligned}
$$

继续往前推会发现，最终都可以写成

$$
x_t=\left(\prod_{i=1}^{t}\sqrt{\alpha_i}\right)x_0
+(\text{若干噪声项的线性组合})
$$

定义$\bar\alpha_t=\prod_{i=1}^{t}\alpha_i$，后，则可以得到

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\text{noise 的线性组合}
$$

因为每一步加入的噪声 $\epsilon_t$ 都是独立标准高斯，而高斯的线性组合仍然是高斯，所以若干噪声项加起来仍然可以写成一个高斯项：

$$
\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I)
$$

通过上面的公式，就得到重参数化形式：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon\tag{1}
$$

所以，在给定条件 $x_0$ 时，$x_t$ 的分布是

$$
q\left(x_t\mid x_0\right)
= \mathcal{N}\left(x_t;\sqrt{\bar\alpha_t}\,x_0,\left(1-\bar\alpha_t\right)I\right).
\tag{1}
$$

## 反向去噪过程（Reverse Process）

### **（1）定义模型**

反向过程想做的事情，就是从噪声一步步还原出原图。前向过程是

$$
x_0\rightarrow x_1\rightarrow x_2\rightarrow\cdots\rightarrow x_T
$$

而反向过程则是

$$
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0
$$

反向过程中，我们希望学习一个参数化模型（极大似然估计），在标准高斯分布的基础上，一步一步去噪，还原任意时间步的图像。因此定义反向生成模型

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
$$

**思考过程：**

如果仅仅给一个当前噪声状态 $x_t$ ，我怎么来求状态 $x_{t-1}$ 呢，这个就是来求$q(x_{t-1}\mid x_t)$，但是这个可以算出来吗，怎么算？答案是不行，没法算。因为有那么多种可能，你怎么保证我想要的那张图呢，所以这个不可求，本质是因为你从状态 $x_{t-1}$ 到状态 $x_t$ 用概率算出来的，你想要逆过去，这样就没法求了，概率不可逆。那么怎么求呢？**用神经网络，用神经网络去拟合这个分布。**

但是我们有假设这个反向过程也是一个马尔科夫链，并且反向过程中的每一步都服从一个高斯分布，根据这俩个我们可以知道什么呢？由于是一个马尔科夫链，所以$q(x_{t-1}\mid x_t)$是等价于$q(x_{t-1}\mid x_t,x_0)$，那这个可以求吗？答案是可以。经过数学推导可以证明，如果知道状态 $x_t$ 和 $x_0$ ，那 $x_{t-1}$ 就是一个确定的分布，而且是高斯分布。

### **（2）反向去噪过程**

**理想后验**

理解反向去噪，最关键的是先看一个“理想后验”：

$$
q(x_{t-1}\mid x_t,x_0)
$$

它表示：已知当前噪声状态 $x_t$ 和原图 $x_0$ 时，上一步 $x_{t-1}$ 的真实分布是什么。这个分布可以推导出来，而且仍然是高斯。

> 这一步“理想后验”是我理解过程中花时间最长、也感觉最难的一步。理解之后，才感受到反向去噪的绝妙之处。

用大白话讲，“理想后验”就是：当前时刻已知当前图像 $x_t$ 和原始图像 $x_0$，竟然可以直接通过公式计算出下一步图像理想情况下应该被还原成什么样。而这个理想的图像 $q(x_{t-1}\mid x_t,x_0)$，在训练过程中被作为真值使用。

理想后验的表达式是

$$
q(x_{t-1}\mid x_t,x_0)
= \mathcal{N}\left(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I\right),
\tag{2}
$$

其中

$$
\tilde\mu_t(x_t,x_0)
= \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0
+ \frac{\sqrt{\alpha_t}\left(1-\bar\alpha_{t-1}\right)}{1-\bar\alpha_t}x_t
$$

$$
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

### **（3）推导过程**

**（3.1）**

首先，我们要推导的是 $q(x_{t-1}\mid x_t,x_0)$，也就是在已知当前噪声状态 $x_t$、原始干净样本 $x_0$ 的情况下，上一时刻 $x_{t-1}$ 的分布。

前向链路中，每个环节都是在前一张图的基础上加噪声，区别只是噪声的多少。我们可以用下面的公式表达如何从 $x_{t-1}$ 得到 $x_t$：

$$
q(x_t\mid x_{t-1})
= \mathcal{N}\left(x_t;\sqrt{\alpha_t}\,x_{t-1},\beta_t I\right).
\tag{2.1}
$$

与此同时，累计形式可以直接写成

$$
q(x_{t-1}\mid x_0)
= \mathcal{N}\left(x_{t-1};\sqrt{\bar\alpha_{t-1}}\,x_0,\left(1-\bar\alpha_{t-1}\right)I\right).
\tag{2.2}
$$

其中$\bar\alpha_t=\prod_{i=1}^{t}\alpha_i，\alpha_t=1-\beta_t$，根据贝叶斯公式，

$$
q(x_{t-1}\mid x_t,x_0)
\propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).
\tag{2.3}
$$

这里“$\propto$”表示忽略与 $x_{t-1}$ 无关的归一化常数。

所以我们只需要把公式（2.1）和（2.2）这两个高斯分布相乘，再把结果整理成关于 $x_{t-1}$ 的高斯。这样就能根据待定系数法得到 $q(x_{t-1}\mid x_t,x_0)$ 的表达式，剩下的其实只是函数展开的工作。

**（3.2）**

**整体推导：**

$$
q(x_{t-1}\mid x_t,x_0)
= \frac{\frac{q(x_{t-1},x_t,x_0)}{q(x_0)}}{\frac{q(x_t,x_0)}{q(x_0)}}
= \frac{q(x_{t-1},x_t\mid x_0)}{q(x_t\mid x_0)}
= \frac{q(x_t\mid x_{t-1},x_0)q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
= \frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

> 一切好像就是这么简单。
>
> 然而，如果没有搞懂公式（2.3）为什么成立，那就相当于一点都没搞懂 DDPM。

下面用最简单易懂的方式说明公式（2.3），确保和我一样概率学只有小学水平的人也能看懂接下来的推导。

**贝叶斯公式复习**

要理解公式（2.3）中的贝叶斯公式，先从最基础的条件概率开始。对于任意两个随机变量 $A$、$B$，有$p(A\mid B)=\frac{p(A,B)}{p(B)}$

意思是：“已知 $B$ 的情况下，$A$ 出现的概率”等于“$A$ 和 $B$ 同时出现的概率”除以“$B$ 出现的概率”。

现在把$A=x_{t-1},B=(x_t,x_0)$，代入条件概率定义，得到

$$
q(x_{t-1}\mid x_t,x_0)
= \frac{q(x_{t-1},x_t,x_0)}{q(x_t,x_0)}
= \frac{\frac{q(x_{t-1},x_t,x_0)}{q(x_0)}}{\frac{q(x_t,x_0)}{q(x_0)}}
= \frac{q(x_{t-1},x_t\mid x_0)}{q(x_t\mid x_0)}
$$

意思很简单：在“已知 $B$（$x_0$ 和 $x_t$ 都出现）的情况下，$A$（$x_{t-1}$ 出现）的概率”等于“$A$ 和 $B$ 同时（$x_0$、$x_t$ 和 $x_{t-1}$ 都出现）的概率”除以“$B$ 出现的概率”。

所以得到：

$$
q(x_{t-1}\mid x_t,x_0)
= \frac{q(x_{t-1},x_t\mid x_0)}{q(x_t\mid x_0)}.
\tag{2.4}
$$

这里的分子 $q(x_{t-1},x_t\mid x_0)$ 表示已知 $x_0$ 时，$x_{t-1}$ 和 $x_t$ 同时发生的概率。

概率乘法公式告诉我们，对于两个事件 $A$、$B$，有$p(A,B)=p(B\mid A)p(A)$

这句话的意思是：

> “$A$ 和 $B$ 同时发生的概率”
>
> 等于“先发生 $A$ 的概率”乘以“在 $A$ 已经发生的前提下 $B$ 发生的概率”。

换成条件概率版本：如果已知条件 $C$，那么公式变成$p(A,B\mid C)=p(B\mid A,C)p(A\mid C)$

意思是：

> 在已知 $C$ 的情况下，$A$ 和 $B$ 同时发生的概率
>
> 等于先在 $C$ 下发生 $A$ 的概率，乘以在 $A$ 和 $C$ 都已知的情况下发生 $B$ 的概率。

所以，$q(x_{t-1},x_t\mid x_0)$ 可以在已知 $x_0$ 的情况下，表示 $x_{t-1}$、$x_t$ 同时发生的概率，即

$$
q(x_{t-1},x_t\mid x_0)
= q(x_t\mid x_{t-1},x_0)q(x_{t-1}\mid x_0)
$$

把公式代入：

$$
q(x_{t-1}\mid x_t,x_0)
= \frac{q(x_t\mid x_{t-1},x_0)q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

而根据前向的马尔可夫特性，$x_t$ 只被 $x_{t-1}$ 影响，与更早的 $x_0$ 无关，所以

$$
q(x_t\mid x_{t-1},x_0)=q(x_t\mid x_{t-1})
$$

于是公式可以写成

$$
q(x_{t-1}\mid x_t,x_0)
= \frac{q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)}{q(x_t\mid x_0)}
$$

因为分母 $q(x_t\mid x_0)$ 不依赖 $x_{t-1}$。对于推导 $x_{t-1}$ 的分布来说，分母只是一个归一化常数，所以（36）常写成

$$
q(x_{t-1}\mid x_t,x_0)
\propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)
$$

它就是公式（2.3）所表示的贝叶斯公式。一旦理解了公式（2.3），接下来理想后验的推导就变成了中学算术题。

**（3.3）**

**后验方差与均值**

公式写成指数形式：

$$
\begin{aligned}
q(x_t\mid x_{t-1})
&\propto \exp\left(
-\frac{1}{2\beta_t}\left\|x_t-\sqrt{\alpha_t}\,x_{t-1}\right\|^2
\right),\\
q(x_{t-1}\mid x_0)
&\propto \exp\left(
-\frac{1}{2(1-\bar\alpha_{t-1})}
\left\|x_{t-1}-\sqrt{\bar\alpha_{t-1}}\,x_0\right\|^2
\right).
\end{aligned}
\tag{38}
$$

因此后验满足

$$
\begin{aligned}
q(x_{t-1}\mid x_t,x_0)
\propto \exp\Bigg[
&-\frac{1}{2\beta_t}\left\|x_t-\sqrt{\alpha_t}\,x_{t-1}\right\|^2\\
&-\frac{1}{2(1-\bar\alpha_{t-1})}
\left\|x_{t-1}-\sqrt{\bar\alpha_{t-1}}\,x_0\right\|^2
\Bigg].
\end{aligned}
\tag{39}
$$

展开平方项，合并成关于 $x_{t-1}$ 的二次型。

第一项拆开为

$$
|x_t|^2-2\sqrt{\alpha_t}\,x_t^\top x_{t-1}
+\alpha_t|x_{t-1}|^2.
$$

乘上 $\frac{1}{2\beta_t}$ 后，与 $x_{t-1}$ 有关的部分是

$$
\frac{\alpha_t}{2\beta_t}\|x_{t-1}\|^2
-\frac{\sqrt{\alpha_t}}{\beta_t}x_t^\top x_{t-1}.
\tag{40}
$$

第二项拆开为

$$
|x_{t-1}|^2-2\sqrt{\bar\alpha_{t-1}}\,x_0^\top x_{t-1}
+\bar\alpha_{t-1}|x_0|^2.
$$

乘上 $\frac{1}{2(1-\bar\alpha_{t-1})}$ 后，与 $x_{t-1}$ 有关的部分是

$$
\frac{1}{2(1-\bar\alpha_{t-1})}\|x_{t-1}\|^2
-\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0^\top x_{t-1}.
\tag{41}
$$

把两部分加起来，后验指数中关于 $x_{t-1}$ 的项变成

$$
-\frac{1}{2}\left[
\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)\|x_{t-1}\|^2
-2\left(\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0\right)^\top x_{t-1}
\right].
\tag{42}
$$

这已经是一个标准高斯的二次型结构。标准高斯分布的一般形式可以写成

$$
\mathcal{N}(x;\mu,\Sigma)
\propto \exp\left(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\right).
\tag{43}
$$

展开后：

$$
-\frac{1}{2}x^\top\Sigma^{-1}x
+x^\top\Sigma^{-1}\mu+\text{常数}.
\tag{44}
$$

对比公式（42）的结果，可以识别出

$$
\Sigma^{-1}
=\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)I.
\tag{45}
$$

因为协方差是 $\tilde\beta_t I$，所以

$$
\tilde\beta_t
=\left(\frac{\alpha_t}{\beta_t}+\frac{1}{1-\bar\alpha_{t-1}}\right)^{-1}.
\tag{46}
$$

将 $\alpha_t=1-\beta_t$ 和 $\bar\alpha_t=\alpha_t\bar\alpha_{t-1}$ 代入，可以得到简化后的后验方差：

$$
\tilde\beta_t
=\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}.
\tag{47}
$$

均值满足

$$
\Sigma^{-1}\mu
=\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0.
\tag{48}
$$

所以

$$
\tilde\mu_t
=\tilde\beta_t\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0
\right).
\tag{49}
$$

把 $\tilde\beta_t$ 代入：

$$
\tilde\mu_t
=\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
\left(
\frac{\sqrt{\alpha_t}}{\beta_t}x_t
+\frac{\sqrt{\bar\alpha_{t-1}}}{1-\bar\alpha_{t-1}}x_0
\right).
\tag{50}
$$

展开得到后验均值公式：

$$
\tilde\mu_t
=\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t
+\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0.
\tag{51}
$$

即

$$
\tilde\mu_t(x_t,x_0)
=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
\tag{52}
$$

推导到这一步，基本上就把后验均值公式的前因后果弄清楚了。最终的后验分布可以再完整写一遍：

$$
\begin{aligned}
q(x_{t-1}\mid x_t,x_0)
&=\mathcal{N}\left(x_{t-1};\tilde\mu_t,\tilde\beta_t I\right),\\
\tilde\beta_t
&=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t,\\
\tilde\mu_t(x_t,x_0)
&=\frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t.
\end{aligned}
\tag{53}
$$

**如何理解后验均值**

这个均值是 $x_0$ 和 $x_t$ 的加权平均，但不是简单平均，而是由噪声强度决定权重：

*   如果当前步噪声很大， $x_t$  更不可靠，后验会更依赖  $x_0$ ；
*   如果噪声很小， $x_t$  本身就接近真实结构，后验会更依赖  $x_t$ 。

这符合“贝叶斯更新”的思想。后验均值本质上是在结合：

*   先验信息  $q(x_{t-1}\mid x_0)$ ；
*   观测信息  $q(x_t\mid x_{t-1})$ 。

从而做出最合理的估计。

训练时我们知道 $x_0$，也能从前向过程采样出 $x_t$，于是可以通过理想后验公式构造监督信号，通过训练参数，使模型 $p_\theta(x_{t-1}\mid x_t)$ 逼近这个理想后验，所以后续的 loss 设计也是基于该公式。

***

## 损失函数Loss

### **（1）训练 loss 的公式推导**

现在已经搞懂了理想后验。接下来最关键的问题是：为什么从理想后验最后会得到扩散模型的训练损失？在理解训练损失之前，需要先思考：从理想后验到学习目标，我们到底想学什么？

把前面的逻辑再梳理一遍。在前向扩散过程中，分布是固定的、已知的：

$$
q(x_{1:T}\mid x_0)=\prod_{t=1}^{T}q(x_t\mid x_{t-1}).
\tag{54}
$$

它会把真实数据 $x_0$ 一步一步加噪成 $x_T$，并且当 $T$ 足够大时，$x_T\approx\mathcal{N}(0,I)$。

现在我们希望反过来做：从纯噪声 $x_T\approx\mathcal{N}(0,I)$ 出发，一步步采样出 $x_{T-1},x_{T-2},\ldots,x_0$。因此定义反向生成模型

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t).
\tag{55}
$$

其中 $p(x_T)=\mathcal{N}(0,I)$。每一步的反向条件分布 $p_\theta(x_{t-1}\mid x_t)$ 都由神经网络参数 $\theta$ 控制。简单来说，我们希望通过训练神经网络参数 $\theta$，得到一个能够预测分布 $p_\theta(x_{t-1}\mid x_t)$ 的网络。这样只要输入 $x_t$，就能预测 $x_{t-1}$，进而一步步得到原始图像的分布。

前面已经推导出理想后验：

$$
q(x_{t-1}\mid x_t,x_0)
=\mathcal{N}\left(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t I\right).
\tag{56}
$$

这个分布表示：如果知道当前状态 $x_t$，又知道原始干净样本 $x_0$，那么上一步 $x_{t-1}$ 的真实条件分布理想情况下应该是什么。训练的目的，就是利用理想后验给模型提供监督，让模型去近似这个“理想后验”。我可以通过训练参数，得到一个参数模型给出的分布 $p_\theta(x_{t-1}\mid x_t)$。那怎么去逼近这个理想后验呢，那我去预测x0不就行了吗？x0怎么预测呢，可以预测一步加噪的噪声，再通过x_t就可以得到x0，这样再通过理想后验进行计算。在推理生成时，虽然不知道 $x_0$，但知道 $x_t$。通过 $x_t$ 和 $p_\theta(x_{t-1}\mid x_t)$，就可以推理出 $x_{t-1}$ 是什么样的。

### **（2）从最大似然出发**

生成模型训练的标准目标是最大化数据似然：

$$
\max_\theta \log p_\theta(x_0).
\tag{57}
$$

或者等价地最小化负对数似然：

$$
\min_\theta -\log p_\theta(x_0).
\tag{58}
$$

这里的问题是

$$
p_\theta(x_0)=\int p_\theta(x_{0:T})\,dx_{1:T}.
\tag{59}
$$

由于中间有很多隐变量 $(x_1,\ldots,x_T)$，这个积分非常难直接计算，所以需要引入变分推断。

$p_\theta(x_0)$是所有能从噪声生成出 (x\_0) 的中间路径的概率总和。

**补充：如何理解数据似然（Likelihood）**

结合 GPT 的回答来理解：生成模型里，我们希望学到一个概率分布 $p_\theta(x)$，它表示在参数为 $\theta$ 的模型下，一个样本 $x$ 出现的可能性有多大。如果我们手里有一个真实样本 $x_0$，那么这个样本在模型下的概率 $p_\theta(x_0)$ 就叫做它的**似然**。所以：

*   **似然** = 模型给某个真实数据分配的概率；
*   **数据似然** = 模型对整个数据集的概率分配情况。

如果有数据集 $x_1,x_2,\ldots,x_N$，那么整个数据集在模型下的概率通常写成

$$
\prod_{i=1}^{N}p_\theta(x_i).
\tag{60}
$$

这就是整个数据集的似然。通常取对数，变成

$$
\sum_{i=1}^{N}\log p_\theta(x_i).
\tag{61}
$$

这就是训练时常说的“最大化数据似然”。

**更直观地理解数据似然**

你可以把模型想成一个“概率分配器”。它要对所有可能的样本分配概率，总概率加起来是 1。训练的目标，就是让这个分配器更偏向真实数据。

如果真实数据是一张猫的图片，那么：

*   好模型会认为这张图“很像自己要生成的东西”，给它较高概率；
*   差模型会认为这张图“不太像”，给它很低概率。

所以，数据似然本质上就是**模型对真实数据的认可程度**。最大化数据似然，就是“调整模型参数，让模型尽量给真实数据分配更高概率，从而更好地拟合真实数据分布”。

用大白话讲，数据似然就是模型对数据样本的拟合程度；最大化数据似然，就是尽可能提高拟合精度。

### （3）引入 ELBO

**Jensen 不等式方法：**

**另外还有KL方法，详见另一文档**

理解最大化数据似然之后，就可以回到公式（59）。直接求最大数据似然（likelihood）显然很难做到，所以问题转化为：推导数据似然的下界，然后对下界做最大化。相当于要提高一个班级的总体学习成绩，只要提高班级每次考试的最低分就可以。

我们选取前向扩散分布 $q(x_{1:T}\mid x_0)$ 作为辅助分布，对 $\log p_\theta(x_0)$ 做下界。先写

$$
\log p_\theta(x_0)
=\log\int p_\theta(x_{0:T})\,dx_{1:T}.
\tag{62}
$$

乘上再除以 $q(x_{1:T}\mid x_0)$：

$$
\log p_\theta(x_0)
=\log\int q(x_{1:T}\mid x_0)
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}\,dx_{1:T}.
\tag{63}
$$

写成期望：

$$
\log p_\theta(x_0)
=\log\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].
\tag{64}
$$

对数作用在期望外面，不好计算，于是使用 Jensen 不等式：

$$
\log\mathbb{E}_q[z]\ge\mathbb{E}_q[\log z].
\tag{65}
$$

得到

$$
\log p_\theta(x_0)
\ge\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}\mid x_0)}
\right].
\tag{66}
$$

数据似然的下界就这样确定出来了，这就是 ELBO。训练时最大化 ELBO，等价于尽量提高真实数据的对数似然下界。

$$
\mathcal{L}_{\text{ELBO}}
=\mathbb{E}_q\left[
\log p_\theta(x_{0:T})-\log q(x_{1:T}\mid x_0)
\right].
\tag{67}
$$

**补充：数学期望的定义**

从公式（63）到（64）用到了数学期望的定义。对于连续随机变量 $z$，如果它服从分布 $q(z)$，那么任意函数 $f(z)$ 的期望定义为

$$
\mathbb{E}_{q(z)}[f(z)]=\int q(z)f(z)\,dz.
\tag{68}
$$

> **公式（67）中的两项分别是什么意思？**
>
> 这个比值表达的是：模型的反向生成路径，与前向加噪路径相比，到底有多“匹配”。
>
> *   **第一项**是模型认为这整条“从噪声到数据”的反向路径有多合理。也就是说，模型不仅要解释最后的  $x_0$ ，还要解释中间每一步。直观理解，我们希望模型学到的中间环节中，每个变量都有很高的确定性，属于增益。
> *   **第二项**是模型认为从  $x_0$  一步一步加噪成  $x_T$  的“路径概率”，也就是这条噪声轨迹在前向随机过程下有多可能。个人理解，前向过程是我们自己定义的噪声，从  $x_1$  到  $x_T$  中，每个环节都充满不确定性，存在负增益。但一旦模型反向学习得非常好，就可以认为前面过程每个环节的图像都定义得很好，负增益很小。
>
> GPT 的说法是：如果模型的反向过程学得很好，那么它应该能很好地解释前向加噪所产生的中间状态，于是这个差值就会变大。

**展开 ELBO**

$$
\begin{aligned}
\log p_\theta(x_0)
&\ge
\mathbb{E}_{q(x_{1:T}\mid x_0)}
\left[
\log
\frac{p_\theta(x_{0:T})}
{q(x_{1:T}\mid x_0)}
\right] \\
&=
\mathbb{E}_{q}
\left[
\log
\frac{
p(x_T)
\prod_{t=1}^{T}p_\theta(x_{t-1}\mid x_t)
}{
\prod_{t=1}^{T}q(x_t\mid x_{t-1})
}
\right] \\
&=
\mathbb{E}_{q}
\left[
\log
\frac{
p(x_T)p_\theta(x_0\mid x_1)
\prod_{t=2}^{T}p_\theta(x_{t-1}\mid x_t)
}{
q(x_1\mid x_0)
\prod_{t=2}^{T}q(x_t\mid x_{t-1},x_0)
}
\right] \\
&=
\mathbb{E}_{q}
\left[
\log
\frac{
p(x_T)p_\theta(x_0\mid x_1)
}{
q(x_1\mid x_0)
}
+
\log
\prod_{t=2}^{T}
\frac{
p_\theta(x_{t-1}\mid x_t)
}{
q(x_t\mid x_{t-1},x_0)
}
\right] \\
&=
\mathbb{E}_{q}
\left[
\log
\frac{
p(x_T)p_\theta(x_0\mid x_1)
}{
q(x_1\mid x_0)
}
+
\log
\prod_{t=2}^{T}
\frac{
p_\theta(x_{t-1}\mid x_t)
q(x_{t-1}\mid x_0)
}{
q(x_{t-1}\mid x_t,x_0)
q(x_t\mid x_0)
}
\right] \\
&=
\mathbb{E}_{q}
\left[
\log
\frac{
p(x_T)p_\theta(x_0\mid x_1)
}{
q(x_1\mid x_0)
}
+
\log
\frac{
q(x_1\mid x_0)
}{
q(x_T\mid x_0)
}
+
\log
\prod_{t=2}^{T}
\frac{
p_\theta(x_{t-1}\mid x_t)
}{
q(x_{t-1}\mid x_t,x_0)
}
\right] \\
&=
\mathbb{E}_{q}
\left[
\log
\frac{
p(x_T)p_\theta(x_0\mid x_1)
}{
q(x_T\mid x_0)
}
+
\sum_{t=2}^{T}
\log
\frac{
p_\theta(x_{t-1}\mid x_t)
}{
q(x_{t-1}\mid x_t,x_0)
}
\right] \\
&=
\mathbb{E}_{q(x_1\mid x_0)}
\left[
\log p_\theta(x_0\mid x_1)
\right] \\
&\quad
+
\mathbb{E}_{q(x_T\mid x_0)}
\left[
\log
\frac{p(x_T)}
{q(x_T\mid x_0)}
\right] \\
&\quad
+
\sum_{t=2}^{T}
\mathbb{E}_{q(x_t,x_{t-1}\mid x_0)}
\left[
\log
\frac{
p_\theta(x_{t-1}\mid x_t)
}{
q(x_{t-1}\mid x_t,x_0)
}
\right] \\
&=
\mathbb{E}_{q(x_1\mid x_0)}
\left[
\log p_\theta(x_0\mid x_1)
\right] \\
&\quad
-
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)
\middle\|
p(x_T)
\right) \\
&\quad
-
\sum_{t=2}^{T}
\mathbb{E}_{q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\middle\|
p_\theta(x_{t-1}\mid x_t)
\right)
\right].
\end{aligned}
$$

**最终得到的是 ELBO 的分解：**

$$
\mathrm{ELBO}
=
\underbrace{
\mathbb{E}_{q(x_1\mid x_0)}
\left[
\log p_\theta(x_0\mid x_1)
\right]
}_{\text{reconstruction}}
-
\underbrace{
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)\|p(x_T)
\right)
}_{\text{prior matching}}
-
\underbrace{
\sum_{t=2}^{T}
\mathbb{E}_{q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)
\right)
\right]
}_{\text{denoising matching}}.
$$

训练时通常最小化负 ELBO：

$$
\mathcal{L}
=
-
\mathrm{ELBO}.
$$

因此三项的符号会反过来：

$$
\mathcal{L}
=
\underbrace{
-
\mathbb{E}_{q(x_1\mid x_0)}
\left[
\log p_\theta(x_0\mid x_1)
\right]
}_{\text{重建损失}}
+
\underbrace{
D_{\mathrm{KL}}
\left(
q(x_T\mid x_0)\|p(x_T)
\right)
}_{\text{先验匹配损失}}
+
\underbrace{
\sum_{t=2}^{T}
\mathbb{E}_{q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t)
\right)
\right]
}_{\text{去噪匹配损失}}.
$$

在 DDPM 的常见简化训练目标中，先验匹配项通常忽略，重建项和去噪匹配项进一步化简，最终得到预测噪声的均方误差形式。

***

### **（4）合并**

$$
\text{minimize}
\sum_{t=1}^{T}
\mathbb{E}_{q(x_t\mid x_0)}
\left[
D_{\mathrm{KL}}
\left(
q(x_{t-1}\mid x_t,x_0)
\middle\|
p_\theta(x_{t-1}\mid x_t)
\right)
\right]
$$

**所以这个说的就是$q(x_{t-1}\mid x_t,x_0)$与$p_\theta(x_{t-1}\mid x_t)$之间的KL散度。**

**对于两个高斯分布的 KL 散度**

假设有 2 个高斯分布：

$$
x\sim\mathcal{N}(\mu_1,\sigma_1)
\qquad
y\sim\mathcal{N}(\mu_2,\sigma_2)
$$

它们的 KL 散度：

$$
\mathrm{KL}
\left(
\mathcal{N}(x\mid\mu_1,\Sigma_1)
\middle\|
\mathcal{N}(x\mid\mu_2,\Sigma_2)
\right)
=
\frac{1}{2}
\left[
\log\frac{|\Sigma_2|}{|\Sigma_1|}
-
K
+
\mathrm{tr}
\left(
\Sigma_2^{-1}\Sigma_1
\right)
+
(\mu_1-\mu_2)^\top
\Sigma_2^{-1}
(\mu_1-\mu_2)
\right]
$$

**$q(x_{t-1}\mid x_t,x_0)$来说：**

根据先前推导的公式（1）代入到公式（2）中（把x\_0用x\_t替换掉）：

$$
q(x_{t-1}\mid x_t,x_0)
=
\mathcal{N}
\left(
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}
\epsilon
\right),
\frac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
\right)
$$

$$
\begin{cases}
\mu
=
\dfrac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\dfrac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}
\epsilon
\right)
\\[6pt]
\sigma^2
=
\dfrac{\beta_t(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}
\end{cases}
$$

**$p_\theta(x_{t-1}\mid x_t)$：假设方差为固定值，因为实验证明对结果无影响，均值为网络输出 $\mu$**

所以U-Net网络做的就是去拟合$q(x_{t-1}\mid x_t,x_0)$分布，在具体一点，去拟合均值，再次优化，去拟合$\epsilon$





**网络不是在训练时把所有可能性强行压成一个固定答案，而是在学习一个“给定 \(x_t\) 时，各种 \(x_0\) 可能性应当如何分布”的条件分布**



## Algorithm

## Algorithm 1 Training

**1:** **repeat**

**2:** $\mathbf{x}_0\sim q(\mathbf{x}_0)$

**3:** $t\sim\mathrm{Uniform}(\{1,\ldots,T\})$

**4:** $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$

**5:** Take gradient descent step on

$$
\nabla_\theta
\left\|
\boldsymbol{\epsilon}
-
\boldsymbol{\epsilon}_\theta
\left(
\sqrt{\bar\alpha_t}\mathbf{x}_0
+
\sqrt{1-\bar\alpha_t}\boldsymbol{\epsilon},
t
\right)
\right\|^2
$$

**6:** **until converged**

***

## Algorithm 2 Sampling

**1:** $\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})$

**2:** **for** $t=T,\ldots,1$ **do**

**3:** $\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ if $t>1$, else $\mathbf{z}=\mathbf{0}$

**4:**

$$
\mathbf{x}_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
\mathbf{x}_t
-
\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)
\right)
+
\sigma_t\mathbf{z}
$$

**5:** **end for**

**6:** **return** $\mathbf{x}_0$
