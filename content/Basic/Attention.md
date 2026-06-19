---
title: Attention
publish: true
tags:
  - 注意力机制_attention
  - 深度学习_deep-learning
  - transformer
  - 核回归_kernel-regression
---
# 1. Query & Key & Value

在注意力机制中，原始数据转换成三个对象：Query, Key, Value. 
- $Q$: 查询条件, 表示当前要找什么.
- $K$: 匹配特征, 用来判断数据是否符合查询.
- $V$: 信息内容, 表示匹配后实际取出的数据.
而 Attention 就是在根据 ==Query 和 Key 的匹配程度==获得权重，给 Value 分配权重，然后做加权求和。

***Example. 1*** 一个简单例子
<div style="display: grid; grid-template-columns: 180px 100px minmax(360px, 560px); grid-template-rows: repeat(3, auto); gap: 10px 0; align-items: stretch; max-width: 860px; margin: 8px auto 14px auto;">
  <div style="grid-row: 1 / 4; border: 1px solid var(--background-modifier-border); border-radius: 10px; padding: 12px 14px; text-align: center; line-height: 1.7; background: var(--background-primary); display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <span style="background: rgba(255, 235, 59, 0.55); border-radius: 5px; padding: 1px 4px;">原始数据</span><br>
    例: 银行账单
  </div>
  <svg viewBox="0 0 100 100" preserveAspectRatio="none" style="grid-column: 2; grid-row: 1 / 4; width: 100px; height: 100%; min-height: 176px; overflow: visible; color: var(--text-muted);">
    <defs>
      <marker id="arrowhead-attention-qkv" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M 0 0 L 8 4 L 0 8 Z" fill="currentColor"></path>
      </marker>
    </defs>
    <path d="M 0 50 C 34 50, 58 16, 96 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" marker-end="url(#arrowhead-attention-qkv)"></path>
    <path d="M 0 50 C 34 50, 58 50, 96 50" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" marker-end="url(#arrowhead-attention-qkv)"></path>
    <path d="M 0 50 C 34 50, 58 84, 96 84" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" marker-end="url(#arrowhead-attention-qkv)"></path>
  </svg>
  <div style="grid-column: 3; grid-row: 1; margin-left: 8px; border: 1px solid var(--background-modifier-border); border-radius: 10px; padding: 10px 14px; line-height: 1.7; background: var(--background-primary);">
    <span style="background: rgba(255, 235, 59, 0.55); border-radius: 5px; padding: 1px 4px;"><strong>Query</strong>: 查询条件</span>, 例: 大于 60 岁的人
  </div>
  <div style="grid-column: 3; grid-row: 2; margin-left: 8px; border: 1px solid var(--background-modifier-border); border-radius: 10px; padding: 10px 14px; line-height: 1.7; background: var(--background-primary);">
    <span style="background: rgba(255, 235, 59, 0.55); border-radius: 5px; padding: 1px 4px;"><strong>Key</strong>: 原始数据的特征标签</span>, 例: {姓, 名, 年龄}
  </div>
  <div style="grid-column: 3; grid-row: 3; margin-left: 8px; border: 1px solid var(--background-modifier-border); border-radius: 10px; padding: 10px 14px; line-height: 1.7; background: var(--background-primary);">
    <span style="background: rgba(255, 235, 59, 0.55); border-radius: 5px; padding: 1px 4px;"><strong>Value</strong>: 原始数据中我们想要的信息</span>, 例: 每个人的电费
  </div>
</div>

<table style="border-collapse: collapse; width: auto; max-width: 820px; margin: 0 auto; table-layout: auto;">
  <caption style="caption-side: bottom; padding-top: 6px; font-size: 0.95em; color: var(--text-muted); text-align: center;">Table: 固定 Q 时, 不同 Key 下的注意力权重.</caption>
  <thead>
    <tr>
      <th style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;"><em>Q</em> \ <em>K</em></th>
      <th style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;"><em>k</em><sub>1</sub>={Li,Hua, 60}</th>
      <th style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;"><em>k</em><sub>2</sub>={Wu,Ming, 20}</th>
      <th style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;"><em>k</em><sub>3</sub>={Li,Meng, 40}</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: left; white-space: nowrap;">姓为 "Wu"</th>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">0</td>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">1</td>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">0</td>
    </tr>
    <tr>
      <th style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: left; white-space: nowrap;">年龄较大</th>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">0.7</td>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">0.05</td>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">0.25</td>
    </tr>
    <tr>
      <th style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: left; white-space: nowrap;">性别为 "男"</th>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">unknown / 0.2</td>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">unknown / 0.4</td>
      <td style="border: 1px solid var(--background-modifier-border); padding: 6px 10px; text-align: center; white-space: nowrap;">unknown / 0.2</td>
    </tr>
  </tbody>
</table>

- 当问到姓为 "Wu" 的时候, 注意力权重集中于第二个 Key $k_2$ , 这时候我们主要关心 Wu 姓用户的电费 (Value).
- 当问到年龄较大的时候, 注意力权重在年纪较大的 Key 大, 在年纪较小的 Key 小, 这时我们主要关心年纪大的用户的电费 (Value).
- 当问到性别为男时, Key 并不包含性别 tag, 故注意力权重溃散到 $k_1,k_2,k_3$.

当固定 $Q$ 为 "年龄较大" 时, 注意力为
$$\mathrm{Attention}=0.7\times v_1+0.05\times v_2+0.25\times v_3.$$

> [!definition] Definition. Attention pool
> - Attention 的本质：根据 Query 和 Key 的相关性求权重，然后做加权汇总 Value.
> - 给定一个 Query $q$ 和一组 Key-Value 对 $\{(k_i,v_i)\}_{i=1}^n$, 注意力
> $$\mathrm{Attention}(q,\{(k_i,v_i)\}_{i=1}^n)=\sum_{i=1}^n \alpha(q,k_i)v_i,$$
> $\alpha(q,k_i)$ 是 query $q$ 分配给 key $k_i$ 的注意力权重，需满足:
> $$\alpha(q,k_i)\ge 0,\quad \sum_{i=1}^n \alpha(q,k_i)=1.$$

接下来我们分析一个经典例子和现代最常见的例子:
- 来自 *Theory of Probability & Its Applications* ([[#^ref-nadaraya-1964|Nadaraya (1964)]]) 的 Nadaraya-Watson regression
- 来自 *Attention is all you need* ([[#^ref-vaswani-2017|Vaswani et al. (2017)]]) 的 scaled dot-product attention

# 2. N-W 核回归 Nadaraya-Watson regression
***Step 1***
对由
$$y_i=2\sin(x_i)+x_i^{0.8}+\epsilon_i,\quad \epsilon_i \sim \mathcal{N}(0,0.5^2).$$
生成的数据集 $\{(x_1,y_1),\ldots,(x_{50},y_{50})\}$ 做回归.

***Step 2***
最简单的预测方法是对输出值取平均：
$$f(x)=\frac{1}{n}\sum_{i=1}^n y_i=\sum_{i=1}^n \frac{1}{n} y_i.$$
此时 Query 为 $x$, Key 为 $x_i$, Value 为 $y_i$. 注意力权重为 $\alpha(q,k_i)=\alpha(x,x_i)=\frac 1 n$. 但如下图, 拟合效果并不好, 因为我们没有有效的使用 Query 和 Key.
![[Attention-1781763908411.webp]]

***Step 3***
[[#^ref-nadaraya-1964|Nadaraya (1964)]] 和 [[#^ref-watson-1964|Watson (1964)]] 提出了 Nadaraya-Watson kernel regression 方法：
$$\begin{aligned} f(x)&=\sum_{i=1}^n \frac{K(x-x_i)}{\sum_{j=1}^n K(x-x_j)}y_i,\quad K\text{ is kernel} \\ &=\sum_{i=1}^n \alpha(x,x_i)y_i. \end{aligned}$$
这里根据 Query $x$ 和 Key $x_i$ 的匹配程度，给 Value $y_i$ 分配权重，然后做加权求和。

***Step 4***
When $K$ is Gaussian kernel
$$K(u)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{u^2}{2}\right),$$
we have
$$\begin{aligned} f(x)&=\sum_{i=1}^n \alpha(x,x_i)y_i \\ &=\sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x-x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x-x_j)^2\right)}y_i \\ &=\sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x-x_i)^2\right)y_i. \end{aligned}$$
可以看到，新模型的预测曲线更加平滑，并且比平均汇聚更接近真实函数.
![[Attention-1781763938713.webp]]

# 3. 缩放内积注意力 scaled dot-product attention
It is a part of Transformer from *Attention is all you need* ([[#^ref-vaswani-2017|Vaswani et al. (2017)]])
$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

## References

- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability & Its Applications*, 9(1), 141–142. ^ref-nadaraya-1964
- Watson, G. S. (1964). Smooth regression analysis. *Sankhyā: The Indian Journal of Statistics, Series A*, 359–372. ^ref-watson-1964
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30. ^ref-vaswani-2017
