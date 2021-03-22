
<!DOCTYPE html>
<html lang="en-us">
  
  <head>
  <meta charset="UTF-8">
  <title>Graph Attention Networks</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#157878">
  <link rel="stylesheet" href="/GAT/css/normalize.css">
  <link href='https://fonts.googleapis.com/css?family=Open+Sans:400,700' rel='stylesheet' type='text/css'>
  <link rel="stylesheet" href="/GAT/css/cayman.css">
  <script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
  </script>
</head>


  <body>
    <section class="page-header"> 
  <img src="/GAT/assets/cam.png">
  <img src="/GAT/assets/cvc.svg"> <br>
  <img src="/GAT/assets/mila_logo.png">
  <img src="/GAT/assets/logo-2.png">
  <h1 class="project-name">Graph Attention Networks</h1>
  <h2 class="project-tagline">Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò and Yoshua Bengio</h2>
  <a href="https://arxiv.org/abs/1710.10903" class="btn">Read on arXiv</a>
  <a href="https://github.com/PetarV-/GAT" class="btn">View on GitHub</a>
  <a href="https://github.com/PetarV-/GAT/zipball/master" class="btn">Download .zip</a>
  <a href="https://github.com/PetarV-/GAT/tarball/master" class="btn">Download .tar.gz</a>
</section>


    <section class="main-content">
      
      <h1 id="overview">Overview</h1>

<p>A multitude of important real-world datasets come together with some form of graph structure: social networks, citation networks, protein-protein interactions, brain connectome data, etc. Extending neural networks to be able to properly deal with this kind of data is therefore a very important direction for machine learning research, but one that has received comparatively rather low levels of attention until very recently.</p>

<table>
  <thead>
    <tr>
      <th style="text-align: center"><img src="https://www.dropbox.com/s/uip789ds97jvoak/graphs.png?raw=1" alt="" /></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><em>Motivating examples of graph-structured inputs: molecular networks, transportation networks, social networks and brain connectome networks.</em></td>
    </tr>
  </tbody>
</table>

<p>Here we will present our ICLR 2018 work on <a href="https://arxiv.org/abs/1710.10903">Graph Attention Networks</a> (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers (<a href="https://arxiv.org/abs/1706.03762">Vaswani <em>et al.</em>, 2017</a>) to address the shortcomings of prior methods based on graph convolutions or their approximations (including, but not limited to: <a href="https://arxiv.org/abs/1312.6203">Bruna <em>et al.</em>, 2014</a>; <a href="https://arxiv.org/abs/1509.09292">Duvenaud <em>et al.</em>, 2015</a>; <a href="https://arxiv.org/abs/1511.05493">Li <em>et al.</em>, 2016</a>; <a href="https://arxiv.org/abs/1606.09375">Defferrard <em>et al.</em>, 2016</a>; <a href="https://arxiv.org/abs/1704.01212">Gilmer <em>et al.</em>, 2017</a>; <a href="https://arxiv.org/abs/1609.02907">Kipf and Welling, 2017</a>; <a href="https://arxiv.org/abs/1611.08402">Monti <em>et al.</em>, 2017</a>; <a href="https://arxiv.org/abs/1706.02216">Hamilton <em>et al.</em>, 2017</a>).</p>

<h1 id="motivation-for-graph-convolutions">Motivation for graph convolutions</h1>

<p>We can think of graphs as encoding a form of <em>irregular spatial structure</em>—and therefore, it would be highly appropriate if we could somehow generalise the <strong>convolutional</strong> operator (as used in CNNs) to operate on arbitrary graphs!</p>

<p>CNNs are a major workforce when it comes to working with image data. They exploit the fact that images have a highly rigid and regular connectivity pattern (each pixel “connected” to its eight neighbouring pixels), making such an operator trivial to deploy (as a small kernel matrix which is slid across the image).</p>

<table>
  <thead>
    <tr>
      <th style="text-align: center"><img src="https://camo.githubusercontent.com/3309220c48ab22c9a5dfe7656c3f1639b6b1755d/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f6e3134713930677a386138726278622f32645f636f6e766f6c7574696f6e2e706e673f7261773d31" alt="" /></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><em>2D convolutional operator as applied to a grid-structured input (e.g. image).</em></td>
    </tr>
  </tbody>
</table>

<p>Arbitrary graphs are a <strong>much harder</strong> challenge! Ideally, we would like to aggregate information across each of the nodes’ neighbourhoods in a principled manner, but we are no longer guaranteed such rigidity of structure.</p>

<table>
  <thead>
    <tr>
      <th style="text-align: center"><img src="https://www.dropbox.com/s/zgj3a0hqabe4e4i/graph_conv.png?raw=1" alt="" /></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><em>A desirable form of a graph convolutional operator.</em></td>
    </tr>
  </tbody>
</table>

<p>Enumerating the desirable traits of image convolutions, we arrive at the following properties we would ideally like our graph convolutional layer to have:</p>
<ul>
  <li><strong>Computational and storage efficiency</strong> (requiring no more than \(O(V+E)\) time and memory);</li>
  <li><strong>Fixed</strong> number of parameters (independent of input graph size);</li>
  <li><strong>Localisation</strong> (acting on a <em>local neighbourhood</em> of a node);</li>
  <li>Ability to specify <strong>arbitrary importances</strong> to different neighbours;</li>
  <li>Applicability to <strong>inductive problems</strong> (arbitrary, unseen graph structures).</li>
</ul>

<p>Satisfying all of the above at once has proved to be quite challenging, and indeed, none of the prior techniques have been successful at achieving them simultaneously.</p>

<h2 id="towards-a-viable-graph-convolution">Towards a viable graph convolution</h2>

<p>Consider a graph of \(n\) nodes, specified as a set of node features, \((\vec{h}_1, \vec{h}_2, \dots, \vec{h}_n)\), and an adjacency matrix \(\bf A\), such that \({\bf A}_{ij} = 1\) if \(i\) and \(j\) are connected, and \(0\) otherwise<sup id="fnref:1" role="doc-noteref"><a href="#fn:1" class="footnote">1</a></sup>. A <strong>graph convolutional layer</strong> then computes a set of new node features, \((\vec{h}’_1, \vec{h}’_2, \dots, \vec{h}’_n)\), based on the input features as well as the graph structure.</p>

<p>Every graph convolutional layer starts off with a shared node-wise feature transformation (in order to achieve a higher-level representation), specified by a weight matrix \({\bf W}\). This transforms the feature vectors into \(\vec{g}_i = {\bf W}\vec{h}_i\). After this, the vectors \(\vec{g}_i\) are typically recombined in some way at each node.</p>

<p>In general, to satisfy the localisation property, we will define a graph convolutional operator as an aggregation of features across neighbourhoods; defining \(\mathcal{N}_i\) as the neighbourhood of node \(i\) (typically consisting of all first-order neighbours of \(i\), including \(i\) itself), we can define the output features of node \(i\) as:</p>

\[\vec{h}'_i = \sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\vec{g}_j\right)\]

<p>where \(\sigma\) is an activation function, and \(\alpha_{ij}\) specifies the <em>weighting factor</em> (importance) of node \(j\)’s features to node \(i\).</p>

<p>Most prior work defines \(\alpha_{ij}\) <em>explicitly</em><sup id="fnref:2" role="doc-noteref"><a href="#fn:2" class="footnote">2</a></sup> (either based on the structural properties of the graph, or as a learnable weight); this requires compromising at least one other desirable property.</p>

<h1 id="graph-attention-networks">Graph Attention Networks</h1>

<p>We instead decide to let \(\alpha_{ij}\) be <strong>implicitly</strong> defined, employing <em>self-attention</em> over the node features to do so. This choice was not without motivation, as self-attention has previously been shown to be self-sufficient for state-of-the-art-level results on machine translation, as demonstrated by the Transformer architecture (<a href="https://arxiv.org/abs/1706.03762">Vaswani <em>et al.</em>, 2017</a>).</p>

<p>Generally, we let \(\alpha_{ij}\) be computed as a byproduct of an <em>attentional mechanism</em>, \(a : \mathbb{R}^N \times \mathbb{R}^N \rightarrow \mathbb{R}\), which computes unnormalised coefficients \(e_{ij}\) across pairs of nodes \(i, j\), based on their features:</p>

\[e_{ij} = a(\vec{h}_i, \vec{h}_j)\]

<p>We inject the graph structure by only allowing node \(i\) to attend over nodes in its neighbourhood, \(j \in \mathcal{N}_i\). These coefficients are then typically normalised using the softmax function, in order to be comparable across different neighbourhoods:</p>

\[\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}_i}\exp(e_{ik})}\]

<p>Our framework is agnostic to the choice of attentional mechanism \(a\): in our experiments, we employed a simple single-layer neural network. The parameters of the mechanism are trained jointly with the rest of the network in an end-to-end fashion.</p>

<h2 id="regularisation">Regularisation</h2>

<p>To stabilise the learning process of self-attention, we have found <em>multi-head attention</em> to be very beneficial (as was the case in <a href="https://arxiv.org/abs/1706.03762">Vaswani <em>et al.</em>, 2017</a>). Namely, the operations of the layer are independently replicated \(K\) times (each replica with different parameters), and outputs are featurewise aggregated (typically by concatenating or adding).</p>

\[\vec{h}'_i = {\LARGE \|}_{k=1}^K \sigma\left(\sum_{j\in\mathcal{N}_i}\alpha_{ij}^k{\bf W}^k\vec{h}_j\right)\]

<p>where \(\alpha_{ij}^k\) are the attention coefficients derived by the \(k\)-th replica, and \({\bf W}^k\) the weight matrix specifying the linear transformation of the \(k\)-th replica.</p>

<p>With the setup of the preceding sections, this fully specifies a <a href="https://arxiv.org/abs/1710.10903">Graph Attention Network</a> (GAT) layer!</p>

<table>
  <thead>
    <tr>
      <th style="text-align: center"><img src="https://camo.githubusercontent.com/4fe1a90e67d17a2330d7cfcddc930d5f7501750c/68747470733a2f2f7777772e64726f70626f782e636f6d2f732f71327a703170366b37396a6a6431352f6761745f6c617965722e706e673f7261773d31" alt="" /></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><em>A GAT layer with multi-head attention. Every neighbour \(i\) of node \(1\) sends its own vector of attentional coefficients, \(\vec{\alpha}_{1i}\) one per each attention head \(\alpha_{1i}^k\). These are used to compute \(K\) separate linear combinations of neighbours’ features \(\vec{h}_i\), which are then aggregated (typically by concatenation or averaging) to obtain the next-level features of node \(1\), \(\vec{h}’_1\).</em></td>
    </tr>
  </tbody>
</table>

<p>Furthermore, we have found that applying <em>dropout</em> (<a href="http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf">Srivastava <em>et al.</em>, 2014</a>) to the attentional coefficients \(\alpha_{ij}\) was a highly beneficial regulariser, especially for small training datasets. This effectively exposes nodes to <em>stochastically sampled neighbourhoods</em> during training, in a manner reminiscent of the (concurrently published) FastGCN method (<a href="https://arxiv.org/abs/1801.10247">Chen <em>et al.</em>, 2018</a>).</p>

<h2 id="properties">Properties</h2>

<p>A brief analysis of the properties of this layer reveals that it satisfies all of the desirable properties for a graph convolution:</p>

<ul>
  <li><strong>Computationally efficient</strong>: the computation of attentional coefficients can be parallelised across all edges of the graph, and the aggregation may be parallelised across all nodes;</li>
  <li><strong>Storage efficient</strong>: It is possible to implement a GAT layer using sparse matrix operations only, requiring no more than \(O(V+E)\) entries to be stored anywhere;</li>
  <li><strong>Fixed</strong> number of parameters, irrespective of the graph’s node or edge count;</li>
  <li>Trivially <strong>localised</strong>, as we only attend over neighbourhoods;</li>
  <li>Allows for (implicitly) specifying <strong>different importances</strong> to <strong>different neighbours</strong>;</li>
  <li>Readily applicable to <strong>inductive problems</strong>, as it is a shared <em>edge-wise</em> mechanism and therefore does not depend on the global graph structure!</li>
</ul>

<p>To the best of our knowledge, this is the <em>first</em> proposed graph convolution layer to do so.</p>

<p>These theoretical properties have been further validated within <a href="https://arxiv.org/abs/1710.10903">our paper</a> by matching or exceeding state-of-the-art performance across four challenging transductive and inductive node classification benchmarks (Cora, Citeseer, PubMed and PPI). t-SNE visualisations on the Cora dataset further demonstrate that our model is capable of effectively discriminating between its target classes.</p>

<table>
  <thead>
    <tr>
      <th style="text-align: center"><img src="https://raw.githubusercontent.com/PetarV-/GAT/gh-pages/assets/t-sne.png" alt="" /></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><em>t-SNE + Attentional coefficients of a pre-trained GAT model, visualised on the Cora citation network dataset.</em></td>
    </tr>
  </tbody>
</table>

<h1 id="applications">Applications</h1>

<p>Following the publication of the GAT paper, we have been delighted to witness (as well as contribute to) several new lines of research in which GAT-like architectures have been leveraged to solve challenging problems. Here, we will highlight two subsequent contributions that some of us have developed, and outline some works that have been released by others.</p>

<h2 id="mesh-based-parcellation-of-the-cerebral-cortex">Mesh-based parcellation of the cerebral cortex</h2>

<p>In this work (done in collaboration with the University of Cambridge Department of Psychiatry and the Montréal Neurological Institute), we have considered the task of <em>cortical mesh segmentation</em> (predicting functional regions for locations on a human brain mesh). To do this, we have leveraged functional MRI data from the Human Connectome Project (HCP). We found that graph convolutional methods (such as GCNs and GATs) are capable of setting state-of-the-art results, exploiting the underlying structure of a brain mesh better than all prior approaches, enabling more informed decisions.</p>

<table>
  <thead>
    <tr>
      <th style="text-align: center"><img src="https://www.dropbox.com/s/f4362wl6uxpyj4d/parcellation.png?raw=1" alt="" /></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><em>Area parcellation qualitative results for several methods on a test subject. The approach of <a href="https://www.ncbi.nlm.nih.gov/pubmed/27693796">Jakobsen et al. (2016)</a> is the prior state-of-the-art.</em></td>
    </tr>
  </tbody>
</table>

<p>For more details, please refer to our MIDL publication (<a href="https://openreview.net/forum?id=rkKvBAiiz">Cucurull <em>et al.</em>, 2018</a>).</p>

<h2 id="neural-paratope-prediction">Neural paratope prediction</h2>

<p>Antibodies are a critical part of the immune system, having the function of directly neutralising or tagging undesirable objects (the antigens) for future destruction. Here we consider the task of <em>paratope prediction</em>: predicting the amino acids of an antibody that participate in binding to a target antigen. A viable paratope predictor is a significant facilitator to antibody design, which in turn will contribute to the development of personalised medicine.</p>

<p>In this work, we build on Parapred, the previous state of the art approach of <a href="https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty305/4972995">Liberis <em>et al.</em> (2018)</a>, substituting its convolutional and recurrent layers with à trous convolutional and attentional layers, respectively. These layers have been shown to perform more favourably, and allowed us to, for the first time, positively exploit antigen data. We do this through <em>cross-modal attention</em>, allowing amino acids of the antibody to attend over the amino acids of the antigen (which could be seen as a GAT-like model applied to a bipartite antibody-antigen graph). This allowed us to set new state-of-the-art results on this task, along with obtaining insightful interpretations about the model’s mechanism of action.</p>

<table>
  <thead>
    <tr>
      <th style="text-align: center"><img src="https://www.dropbox.com/s/hkgy68zjbh8afdi/try2.png?raw=1" alt="" /></th>
      <th><img src="https://www.dropbox.com/s/7zo2co53iaw6ifp/agvis.png?raw=1" alt="" /></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center"><em>Antibody amino acid binding probabilities to the antigen (in gold) assigned by our model for a test antibody-antigen complex. Warmer colours indicate higher probabilities.</em></td>
      <td><em>Normalised antigen attention coefficients for a single (binding) antibody amino acid (in red). Warmer colours indicate higher coefficients.</em></td>
    </tr>
  </tbody>
</table>

<p>For more details, please refer to our ICML WCB publication (<a href="https://arxiv.org/abs/1806.04398">Deac <em>et al.</em>, 2018</a>).</p>

<h2 id="additional-related-work">Additional related work</h2>

<p>Finally, we outline four interesting relevant pieces of work that leverage or further build on GAT(-like) models. The list is by no means exhaustive!</p>

<ul>
  <li><em>Gated Attention Networks</em> (GaAN) (<a href="https://arxiv.org/abs/1803.07294">Zhang <em>et al.</em>, 2018</a>), where gating mechanisms are inserted into the multi-head attention system of GATs, in order to give different value to different heads’ computations. Several challenging baselines are outperformed on both inductive node classification tasks (Reddit, PPI) and a traffic speed forecasting task (METR-LA).</li>
  <li><em>DeepInf</em> (<a href="https://www.haoma.io/pdf/deepinf.pdf">Qiu <em>et al.</em>, 2018</a>), leveraging graph convolutional layers to modelling <em>influence locality</em> (predicting whether a node will perform a particular action, given the action statuses of its \(r\)-hop neighbours at a particular point in time). Notably, this is the first study where attentional mechanisms (GAT) appear to be necessary for surpassing baseline approaches (such as SVMs or logistic regression), given the heterogeneity of the edges. Furthermore, a very nice qualitative analysis is performed on the action mechanism of the various attention heads employed by the GAT model.</li>
  <li><em>Attention Solves Your TSP</em> (<a href="https://arxiv.org/abs/1803.08475">Kool and Welling, 2018</a>), where GAT-like layers (using the Transformer-style attention mechanism) have been successfully applied to solving <em>combinatorial optimisation</em> problems, specifically the Travelling Salesman Problem (TSP).</li>
  <li><em>PeerNets</em> (<a href="https://arxiv.org/abs/1806.00088">Svoboda <em>et al.</em>, 2018</a>), which augment a standard convolutional neural network architecture for image classification with GAT-like layers over a graph of “neighbouring” feature maps from related images in a training dataset. This makes the learnt feature maps substantially more robust, providing a strong defence against a variety of adversarial attacks while sacrificing almost no test accuracy.</li>
</ul>

<h1 id="conclusions">Conclusions</h1>

<p>We have presented graph attention networks (GATs), novel convolution-style neural networks that operate on graph-structured data, leveraging masked self-attentional layers. The graph attentional layer utilised throughout these networks is computationally efficient (does not require costly matrix operations, and is parallelisable across all nodes in the graph), allows for (implicitly) assigning different importances to different nodes within a neighborhood while dealing with different sized neighborhoods, and does not depend on knowing the entire graph structure upfront—thus addressing many of the theoretical issues with approaches. Results, both within our work and the numerous subsequently published work, highlight the importance of such architectures towards building principled graph convolutional neural networks.</p>

<h2 id="citation">Citation</h2>

<p>If you make advantage of the GAT model in your research, please cite the following:</p>

<div class="language-plaintext highlighter-rouge"><div class="highlight"><pre class="highlight"><code>@article{
  velickovic2018graph,
  title="{Graph Attention Networks}",
  author={Veli{\v{c}}kovi{\'{c}}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`{o}}, Pietro and Bengio, Yoshua},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rJXMpikCZ},
}
</code></pre></div></div>

<h2 id="acknowledgements">Acknowledgements</h2>

<p>We would like to thank Andreea Deac, Edgar Liberis and Thomas Kipf for their helpful feedback on prior iterations of this blog post!</p>

<hr />

<div class="footnotes" role="doc-endnotes">
  <ol>
    <li id="fn:1" role="doc-endnote">
      <p>In general, the adjacency matrix may be weighted, contain edges of different types, or the edges may even have features of their own. We do not consider these cases for simplicity, but the GAT model may be trivially extended to handle them, as was done by EAGCN (<a href="https://arxiv.org/abs/1802.04944v1">Shang <em>et al.</em>, 2018</a>). <a href="#fnref:1" class="reversefootnote" role="doc-backlink">&#8617;</a></p>
    </li>
    <li id="fn:2" role="doc-endnote">
      <p>A notable exception is the MoNet framework (<a href="https://arxiv.org/abs/1611.08402">Monti <em>et al.</em>, 2017</a>) which our work can be considered an instance of. However, in all of the experiments presented in this paper, the weights were implicitly specified based on the graph’s structural properties, meaning that two nodes with same local topologies would always necessarily receive the same (learned) weighting—thus still violating one of the desirable properties for a graph convolution. <a href="#fnref:2" class="reversefootnote" role="doc-backlink">&#8617;</a></p>
    </li>
  </ol>
</div>


      <footer class="site-footer">
  <span class="site-footer-owner"><a href="http://petar-v.com/GAT/">Graph Attention Networks</a> is maintained by <a href="https://petar-v.com">Petar Veličković</a>.</span>
  <span class="site-footer-credits">This page was generated by <a href="https://pages.github.com">GitHub Pages</a> using the <a href="https://github.com/pietromenna/jekyll-cayman-theme">Cayman theme</a> by <a href="http://github.com/jasonlong">Jason Long</a>.</span>
</footer>


    </section>

  </body>
</html>

