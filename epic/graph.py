import numpy as np
import networkx as nx

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def stretchgrid(I1, I2):
    """Create a mesh the size of the first image that spans the second image

    Args:
        I1, I2: The two images (ndarray) or image shapes (tuple). The output
            meshes x,y are the same shape as I1, and numerically span the
            shape of I2.
    """
    M1,N1 = I1[:2] if isinstance(I1, tuple) else I1.shape[:2]
    M2,N2 = I2[:2] if isinstance(I2, tuple) else I2.shape[:2]
    x,y   = np.meshgrid(np.linspace(0,N2-1,N1), np.linspace(0,M2-1,M1))
    return x,y


# ----------------------------------------------------------------------------
# Construct the Image Graph
# ----------------------------------------------------------------------------
def graph(A, images=None, dropout=0.9, directed=True):
    """Construct an image graph from an Affinity matrix

    Args:
        A: The affinity matrix representing the graph edges

    Keyword Args:
        images: The image to add to the nodes
        dropout: Drop connections that have a [0.0 1.0] normalized weight less
            than the threshold
    """

    # normalize and threshold the affinity
    A = A / A.max()
    A[A < dropout] = 0

    # remove self-cycles
    np.fill_diagonal(A, 0)
    prototype = nx.DiGraph() if directed else nx.Graph()
    G = nx.from_numpy_matrix(A, create_using=prototype)

    for n in G.nodes():
        G.node[n] = {'image': images[n]} if images else {}

    # remove nodes with no out edges
    dangling = lambda: [k for k,v in (G.out_degree() if directed else G.degree()).iteritems() if v == 0]
    while dangling():
        G.remove_nodes_from(dangling())
    return G


def mapnodes(f, G, key=None, inkey=None, outkey=None):
    """Map a function across the nodes of the graph from a field to another"""

    # set the input and output fields
    inkey = key if key else inkey
    outkey = outkey if outkey else inkey

    for n in G.nodes_iter():
        G.node[n][outkey] = f(G.node[n][inkey])

def maperror(objective_function, G):
    """Compute the objective function across the graph"""
    for u,v in G.edges_iter():
        edge  = G.edge[u][v]
        F1,F2 = G.node[u]['features'], G.node[v]['features']
        fx,fy = edge['fx'], edge['fy']
        obj   = objective_function(F1,F2,fx,fy)
        edge['obj'] = obj


# ----------------------------------------------------------------------------
# Compute Flow Across the Graph
# ----------------------------------------------------------------------------
def flow(objective, G):
    """Compute the flow between nodes in the graph

    Args:
        objective (callable): The flow objective which takes two inputs,
            the feature images F1 and F2, and produces two outputs fx,fy
        G (graph): The directed graph
    """
    for u,v in G.edges_iter():
        print('computing flow from {u} to {v}...'.format(u=u, v=v))
        F1,F2 = G.node[u]['features'], G.node[v]['features']
        fx,fy = objective(F1, F2)
        G.edge[u][v].update({'fx': fx, 'fy': fy})

def distributed_flow(objective, G):
    """Compute the flow between nodes in the graph in a distributed manner

    Args:
        objective (callable): The flow objective which takes two inputs,
            the feature images F1 and F2, and produces two outputs fx,fy
        G (graph): The directed graph
    """
    import gridengine

    # create the jobs
    jobs = []
    for u,v in G.edges_iter():
        F1,F2 = G.node[u]['features'], G.node[v]['features']
        job = gridengine.Job(target=objective, args=(F1,F2))
        jobs.append(job)

    # run the jobs in parallel
    dispatcher = gridengine.JobDispatcher()
    dispatcher.dispatch(jobs)
    results = dispatcher.join()

    # insert the results into the graph
    for (u,v),(fx,fy) in zip(G.edges_iter(), results):
        G.edge[u][v].update({'fx': fx, 'fy': fy})


# ----------------------------------------------------------------------------
# Propagate Flow
# ----------------------------------------------------------------------------
def propagate_flow(G, source, target, cutoff=3):
    """Propagate flows through the graph by finding all short paths from
    source --> target. Internally uses nx.all_simple_paths.

    Args:
        G: The Graph
        from_node: The node to start at
        to_node: The node to finish at

    Keyword Args:
        cutoff: The maximum path length to consider (default: 3)
    """

    # get the shape
    Ms,Ns = G.node[source]['features'].shape[:2]
    Mt,Nt = G.node[target]['features'].shape[:2]
    uxt,uyt = stretchgrid((Ms,Ns), (Mt,Nt))
    fx,fy = [], []

    # propagate the flow
    for path in nx.all_simple_paths(G, source, target, cutoff=cutoff):

        x,y = np.meshgrid(np.arange(float(Ns)), np.arange(float(Ms)))
        for u,v in zip(path, path[1:]):

            # compute the baseline stretching between the images
            ux,uy = stretchgrid(G.node[u]['features'], G.node[v]['features'])

            # get the position estimates across the edge
            edge  = G.edge[u][v]
            px,py = ux+edge['fx'], uy+edge['fy']

            # get the valid pixels
            M2,N2 = px.shape
            xi,yi = np.round(x).astype(int), np.round(y).astype(int)
            mask = (xi >= 0) * (xi < N2) * (yi >= 0) * (yi < M2)
            x[mask],y[mask] = px[yi[mask],xi[mask]], py[yi[mask],xi[mask]]
            x[~mask] = -1
            y[~mask] = -1

        # add the flow estimate to the collection
        fxuv = x - uxt
        fyuv = y - uyt
        fxuv[~mask] = 1e9
        fyuv[~mask] = 1e9
        fx.append(fxuv)
        fy.append(fyuv)

    # return the estimates
    return fx,fy
