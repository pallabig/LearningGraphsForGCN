<html lang="en-us">
  
  <head>
  <meta charset="UTF-8">
  <title>Learning Graphs for Knowledge Transfer with Limited Labels</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#157878">
  <link rel="stylesheet" href='https://petar-v.com/GAT/css/normalize.css'>
  <link href='https://fonts.googleapis.com/css?family=Open+Sans:400,700' rel='stylesheet' type='text/css'>
  <link rel="stylesheet" href='https://petar-v.com/GAT/css/cayman.css'>
  <script type="text/javascript" async
    src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
  </script>
</head>
<body>
    <section class="main-content">
      
      <h1 id="overview">Abstract</h1>

<p>Fixed input graphs are a mainstay in approaches that utilize Graph Convolution Networks (GCNs) for knowledge transfer. The standard paradigm is to utilize relationships in the input graph to transfer information using GCNs from training to testing nodes in the graph; for example, the semi-supervised, zero-shot, and few-shot learning setups. We propose a generalized framework for learning and improving the input graph as part of the standard GCN-based learning setup. Moreover, we use additional constraints between similar and dissimilar neighbors for each node in the graph by applying triplet loss on the intermediate layer output. We present results of semi-supervised learning on Citeseer, Cora, and Pubmed benchmarking datasets, and zero/few-shot action recognition on UCF101 and HMDB51 datasets, significantly outperforming current approaches. We also present qualitative results visualizing the graph connections that our approach learns to update.</p>

    </section>

    <section class="main-content">
      
      <h1 id="overview">Additional details and code is coming soon!</h1>

    </section>

  </body>
</html>


